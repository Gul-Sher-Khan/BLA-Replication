#!/usr/bin/env python3
"""
Cluster-guided anonymization for JIT defect prediction datasets.

Sensitive attributes: la, ld (churn split)
We preserve quasi-identifier context using deterministic cluster assignment, then synthesize la/ld.

Methods (choose via --method):
  1) cluster_random : weakest baseline (cluster-level triangular churn + beta ratio; no bug split)
  2) gmm_simple     : single GMM per cluster (log1p(churn)) + beta ratio; no bug split
  3) gmm_2          : final method (two GMMs per cluster: bug/non-bug) + beta ratio; bug split
    4) gmm_no_cluster : gmm_simple logic without clustering (single global cluster)
    5) gmm_2_no_cluster : gmm_2 logic without clustering (single global cluster)

Output keeps all columns, but replaces la/ld (and updates churn).

Usage:
  python anon_stats.py --input-csv in.csv --output-csv out.csv --method cluster_random
  python anon_stats.py --input-csv in.csv --output-csv out.csv --method gmm_simple
  python anon_stats.py --input-csv in.csv --output-csv out.csv --method gmm_2
    python anon_stats.py --input-csv in.csv --output-csv out.csv --method gmm_no_cluster
    python anon_stats.py --input-csv in.csv --output-csv out.csv --method gmm_2_no_cluster
"""

import os
import time
import argparse
import logging
import hashlib
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import beta as betadist

# ==============================
# Config
# ==============================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("cluster-anon")

ID_FIELD = "commit_id"
DATE_FIELD = "author_date"

# Quasi-identifiers used for clustering
QIDS = ["nf", "nd", "ns", "ent", "ndev", "nuc", "age", "aexp", "asexp", "arexp"]

# For clipping
CHURN_Q_INTERVALS = [0.05, 0.25, 0.50, 0.75, 0.99]

DEFAULT_NUM_BINS = 20

METHOD_ALIASES = {
    "gmm_no_cluster": "gmm_simple",
    "gmm_2_no_cluster": "gmm_2",
}


# ==============================
# Deterministic RNG helpers
# ==============================
def _hash32(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def _rng_from_key(key: str) -> np.random.Generator:
    seed = _hash32(key) & 0xFFFFFFFF
    return np.random.default_rng(seed)


# ==============================
# Small utilities
# ==============================
def _safe_quantiles(x: np.ndarray, qs: List[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"q{int(q*100)}": 0.0 for q in qs}
    vals = np.quantile(x, qs)
    return {f"q{int(q*100)}": float(v) for q, v in zip(qs, vals)}


def _clip_int(x: float, lo: int, hi: int) -> int:
    if hi < lo:
        hi = lo
    return int(max(lo, min(hi, int(round(x)))))


def _candidate_component_counts(
    sample_size: int, fixed_gmm_components: Optional[int]
) -> List[int]:
    max_k = min(3, int(sample_size))
    if max_k <= 0:
        return [1]
    if fixed_gmm_components is None:
        return list(range(1, max_k + 1))
    return [max(1, min(int(fixed_gmm_components), max_k))]


def _resolve_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def _is_no_cluster_method(method: str) -> bool:
    return method in METHOD_ALIASES


def assign_global_cluster(df: pd.DataFrame) -> pd.Series:
    return pd.Series([("global", 0)] * len(df), index=df.index, name="cluster_key")


# ==============================
# Binning & clustering
# ==============================
def build_bins(df: pd.DataFrame, cols: List[str], q: int) -> Dict[str, np.ndarray]:
    """
    For each QID column, build quantile bins (qcut).
    If not enough unique values, fall back to [-inf, inf].
    """
    bins_map: Dict[str, np.ndarray] = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        s_non_na = s.dropna()
        if s_non_na.nunique() < 2:
            bins = np.array([-np.inf, np.inf], dtype=float)
        else:
            _, bins = pd.qcut(s_non_na, q, duplicates="drop", retbins=True)
        bins_map[c] = bins
    return bins_map


def bin_index(value: Any, bins: np.ndarray) -> int:
    try:
        v = float(value)
        i = np.digitize([v], bins, right=True)[0] - 1
        if 0 <= i < len(bins) - 1:
            return int(i)
    except Exception:
        pass
    return -1


def pick_primary_qid(commit_id: str, qids: List[str] = QIDS) -> str:
    """
    Deterministically pick which QID drives the cluster for this row.
    (Matches your approach: one QID per row, chosen by hash(commit_id).)
    """
    idx = _hash32(commit_id) % len(qids)
    return qids[idx]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    la = pd.to_numeric(out["la"], errors="coerce").fillna(0)
    ld = pd.to_numeric(out["ld"], errors="coerce").fillna(0)
    out["churn"] = (la + ld).astype(int)
    return out


def assign_clusters(df: pd.DataFrame, bins_map: Dict[str, np.ndarray]) -> pd.Series:
    """
    cluster_key = (qid_name, qid_bin_index)
    """
    keys = []
    for _, row in df.iterrows():
        cid = str(row[ID_FIELD])
        qid = pick_primary_qid(cid)
        b = bin_index(row.get(qid, np.nan), bins_map[qid])
        keys.append((qid, b))
    return pd.Series(keys, index=df.index, name="cluster_key")


# ==============================
# Ratio modeling (Beta)
# ==============================
def _fit_beta_for_ratio(
    ratio: np.ndarray,
    mu_fallback: float,
    k_strength: float = 50.0,
    min_mu_n: int = 5,
) -> Tuple[float, float]:
    """
    Fit Beta(alpha,beta) for ratio in [0,1].

    - If enough data + spread: attempt MLE fit.
    - Otherwise: preserve mean using alpha=mu*k, beta=(1-mu)*k.

    This keeps the method "script-based stats" and stable.
    """
    ratio = np.asarray(ratio, dtype=float)
    ratio = ratio[np.isfinite(ratio)]
    ratio = np.clip(ratio, 0.0, 1.0)

    if ratio.size >= 20 and float(np.std(ratio)) >= 1e-6:
        try:
            a, b, _, _ = betadist.fit(ratio, floc=0, fscale=1)
            a = max(float(a), 1e-3)
            b = max(float(b), 1e-3)
            return a, b
        except Exception:
            pass

    if ratio.size < min_mu_n:
        mu = float(mu_fallback)
    else:
        mu = float(np.mean(ratio))

    if not np.isfinite(mu):
        mu = float(mu_fallback)

    mu = float(np.clip(mu, 1e-3, 1 - 1e-3))
    k = float(k_strength)
    alpha = max(mu * k, 1e-3)
    beta = max((1.0 - mu) * k, 1e-3)
    return alpha, beta


# ==============================
# Parameter builders
# ==============================
def _compute_ratio_arrays(
    la: np.ndarray, ld: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    churn = (la + ld).astype(float)
    ratio = np.zeros_like(churn, dtype=float)
    m = churn > 0
    ratio[m] = la[m] / churn[m]
    ratio = np.clip(ratio, 0.0, 1.0)
    return churn, ratio


def _fit_ratio_tertiles(
    churn: np.ndarray, ratio: np.ndarray, mu_fallback: float
) -> Dict[str, Any]:
    churn_pos = churn[churn > 0]
    if churn_pos.size >= 10:
        q33 = float(np.quantile(churn_pos, 0.33))
        q66 = float(np.quantile(churn_pos, 0.66))
    else:
        q33, q66 = 0.0, 0.0

    low_mask = (churn > 0) & (churn <= q33)
    mid_mask = (churn > q33) & (churn <= q66)
    high_mask = churn > q66

    ratio_pos = ratio[churn > 0]
    mu_here = float(np.mean(ratio_pos)) if ratio_pos.size else mu_fallback
    if not np.isfinite(mu_here):
        mu_here = mu_fallback
    mu_here = float(np.clip(mu_here, 1e-3, 1 - 1e-3))

    aL, bL = _fit_beta_for_ratio(ratio[low_mask], mu_fallback=mu_here)
    aM, bM = _fit_beta_for_ratio(ratio[mid_mask], mu_fallback=mu_here)
    aH, bH = _fit_beta_for_ratio(ratio[high_mask], mu_fallback=mu_here)

    return {
        "churn_q33": q33,
        "churn_q66": q66,
        "ratio_low_alpha": aL,
        "ratio_low_beta": bL,
        "ratio_mid_alpha": aM,
        "ratio_mid_beta": bM,
        "ratio_high_alpha": aH,
        "ratio_high_beta": bH,
    }


def build_cluster_params_cluster_random(
    df: pd.DataFrame, cluster_keys: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """
    Weakest baseline:
      - per-cluster churn sampled from triangular(q5,q50,q99) (no bug split)
      - per-cluster ratio sampled from beta conditioned on churn tertiles (no bug split)
      - la split by Binomial(churn, ratio)

    This is "cluster-level random" and is intentionally weaker than any GMM.
    """
    df = df.copy()
    df["cluster_key"] = cluster_keys
    grouped = df.groupby("cluster_key", sort=False)

    # global fallback mean ratio
    la_all = pd.to_numeric(df["la"], errors="coerce").fillna(0).to_numpy(dtype=float)
    ld_all = pd.to_numeric(df["ld"], errors="coerce").fillna(0).to_numpy(dtype=float)
    churn_all, ratio_all = _compute_ratio_arrays(la_all, ld_all)
    ratio_all = ratio_all[np.isfinite(ratio_all)]
    mu_global = float(np.mean(ratio_all)) if ratio_all.size else 0.5
    if not np.isfinite(mu_global):
        mu_global = 0.5
    mu_global = float(np.clip(mu_global, 1e-3, 1 - 1e-3))

    params: Dict[str, Dict[str, Any]] = {}
    for key, group in grouped:
        cluster_id = f"{key[0]}|{key[1]}"

        la = pd.to_numeric(group["la"], errors="coerce").fillna(0).to_numpy(dtype=float)
        ld = pd.to_numeric(group["ld"], errors="coerce").fillna(0).to_numpy(dtype=float)
        churn, ratio = _compute_ratio_arrays(la, ld)

        p: Dict[str, Any] = {"n": int(churn.size)}

        churn_q = _safe_quantiles(churn, CHURN_Q_INTERVALS)
        q5 = int(round(churn_q["q5"]))
        q25 = int(round(churn_q["q25"]))
        q50 = int(round(churn_q["q50"]))
        q75 = int(round(churn_q["q75"]))
        q99 = int(round(churn_q["q99"]))
        q5, q25, q50, q75, q99 = sorted([q5, q25, q50, q75, q99])

        p["churn_q"] = {"q5": q5, "q25": q25, "q50": q50, "q75": q75, "q99": q99}
        p["churn_clip_min"] = max(0, q5)
        p["churn_clip_max"] = max(p["churn_clip_min"], q99)

        # For triangular sampling we use q5,q50,q99 (cluster-level)
        p["rand_churn_q5"] = int(max(0, q5))
        p["rand_churn_q50"] = int(max(p["rand_churn_q5"], q50))
        p["rand_churn_q99"] = int(max(p["rand_churn_q50"], q99))

        # Zero mass (important in JIT churn)
        p["p_zero"] = float(np.mean(churn == 0)) if churn.size else 1.0

        # Ratio model (tertiles)
        rparams = _fit_ratio_tertiles(churn, ratio, mu_fallback=mu_global)
        p.update(rparams)

        params[cluster_id] = p

    return params


def build_cluster_params_gmm_simple(
    df: pd.DataFrame,
    cluster_keys: pd.Series,
    fixed_gmm_components: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Old/simple GMM method:
      - one GMM per cluster on log1p(churn) (uses BIC to choose k<=3)
      - beta ratio conditioned on churn tertiles (no bug split)
    """
    df = df.copy()
    df["cluster_key"] = cluster_keys
    grouped = df.groupby("cluster_key", sort=False)

    # global fallback mean ratio
    la_all = pd.to_numeric(df["la"], errors="coerce").fillna(0).to_numpy(dtype=float)
    ld_all = pd.to_numeric(df["ld"], errors="coerce").fillna(0).to_numpy(dtype=float)
    churn_all, ratio_all = _compute_ratio_arrays(la_all, ld_all)
    ratio_all = ratio_all[np.isfinite(ratio_all)]
    mu_global = float(np.mean(ratio_all)) if ratio_all.size else 0.5
    if not np.isfinite(mu_global):
        mu_global = 0.5
    mu_global = float(np.clip(mu_global, 1e-3, 1 - 1e-3))

    params: Dict[str, Dict[str, Any]] = {}
    for key, group in grouped:
        cluster_id = f"{key[0]}|{key[1]}"

        la = pd.to_numeric(group["la"], errors="coerce").fillna(0).to_numpy(dtype=float)
        ld = pd.to_numeric(group["ld"], errors="coerce").fillna(0).to_numpy(dtype=float)
        churn, ratio = _compute_ratio_arrays(la, ld)

        p: Dict[str, Any] = {"n": int(churn.size)}

        churn_q = _safe_quantiles(churn, CHURN_Q_INTERVALS)
        q5 = int(round(churn_q["q5"]))
        q25 = int(round(churn_q["q25"]))
        q50 = int(round(churn_q["q50"]))
        q75 = int(round(churn_q["q75"]))
        q99 = int(round(churn_q["q99"]))
        q5, q25, q50, q75, q99 = sorted([q5, q25, q50, q75, q99])

        p["churn_q"] = {"q5": q5, "q25": q25, "q50": q50, "q75": q75, "q99": q99}
        p["churn_clip_min"] = max(0, q5)
        p["churn_clip_max"] = max(p["churn_clip_min"], q99)

        # zero mass
        p["p_zero"] = float(np.mean(churn == 0)) if churn.size else 1.0

        # Ratio params
        rparams = _fit_ratio_tertiles(churn, ratio, mu_fallback=mu_global)
        p.update(rparams)

        # GMM on log1p(churn) for churn > 0
        churn_fit = churn[churn > 0]
        if churn_fit.size < 2:
            mean = float(np.log1p(churn_fit[0])) if churn_fit.size == 1 else 0.0
            p["gmm_log_weights"] = [1.0]
            p["gmm_log_means"] = [mean]
            p["gmm_log_vars"] = [1.0]
        else:
            z = np.log1p(churn_fit).reshape(-1, 1)
            best_gmm, best_bic = None, float("inf")
            seed = _hash32(cluster_id)

            for k in _candidate_component_counts(churn_fit.size, fixed_gmm_components):
                g = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    random_state=seed,
                    reg_covar=1e-6,
                )
                g.fit(z)
                bic = g.bic(z)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = g

            gmm = best_gmm
            w = (gmm.weights_ / gmm.weights_.sum()).tolist()
            means = gmm.means_[:, 0].astype(float).tolist()
            covs = np.asarray(gmm.covariances_, dtype=float).reshape(-1)
            vars_ = np.clip(covs, 1e-6, None).astype(float).tolist()

            p["gmm_log_weights"] = w
            p["gmm_log_means"] = means
            p["gmm_log_vars"] = vars_

        params[cluster_id] = p

    return params


def build_cluster_params_gmm_2(
    df: pd.DataFrame,
    cluster_keys: pd.Series,
    fixed_gmm_components: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Final/best method:
      - per cluster, fit separate churn model for bug=0 and bug=1
      - per cluster, fit separate ratio betas (tertiles) for bug=0 and bug=1
      - churn model is a GMM on log1p(churn) for churn>0, plus p_zero mass
      - also stores triangular quantiles for 'random' sampling if you ever need it
    """
    df = df.copy()
    df["cluster_key"] = cluster_keys
    grouped = df.groupby("cluster_key", sort=False)

    params: Dict[str, Dict[str, Any]] = {}

    for key, group in grouped:
        cluster_id = f"{key[0]}|{key[1]}"

        la = pd.to_numeric(group["la"], errors="coerce").fillna(0).to_numpy(dtype=float)
        ld = pd.to_numeric(group["ld"], errors="coerce").fillna(0).to_numpy(dtype=float)
        churn, ratio = _compute_ratio_arrays(la, ld)

        bug_raw = (
            pd.to_numeric(group.get("bugcount", 0), errors="coerce")
            .fillna(0)
            .to_numpy()
        )
        bug = (bug_raw > 0).astype(int)

        p: Dict[str, Any] = {"n": int(churn.size)}

        churn_q = _safe_quantiles(churn, CHURN_Q_INTERVALS)
        q5 = int(round(churn_q["q5"]))
        q25 = int(round(churn_q["q25"]))
        q50 = int(round(churn_q["q50"]))
        q75 = int(round(churn_q["q75"]))
        q99 = int(round(churn_q["q99"]))
        q5, q25, q50, q75, q99 = sorted([q5, q25, q50, q75, q99])

        p["churn_q"] = {"q5": q5, "q25": q25, "q50": q50, "q75": q75, "q99": q99}
        p["churn_clip_min"] = max(0, q5)
        p["churn_clip_max"] = max(p["churn_clip_min"], q99)

        def _fit_for_bug_subset(
            churn_sub: np.ndarray, ratio_sub: np.ndarray, b: int
        ) -> Dict[str, Any]:
            churn_pos = churn_sub[churn_sub > 0]

            # quantiles for triangular random (kept for completeness)
            churn_qb = _safe_quantiles(churn_sub, [0.05, 0.50, 0.99])
            rq5 = int(round(churn_qb["q5"]))
            rq50 = int(round(churn_qb["q50"]))
            rq99 = int(round(churn_qb["q99"]))
            rq5, rq50, rq99 = sorted([rq5, rq50, rq99])

            # tertiles
            if churn_pos.size >= 10:
                q33 = float(np.quantile(churn_pos, 0.33))
                q66 = float(np.quantile(churn_pos, 0.66))
            else:
                q33, q66 = 0.0, 0.0

            low_mask = (churn_sub > 0) & (churn_sub <= q33)
            mid_mask = (churn_sub > q33) & (churn_sub <= q66)
            high_mask = churn_sub > q66

            ratio_pos = ratio_sub[churn_sub > 0]
            mu_fb = float(np.mean(ratio_pos)) if ratio_pos.size else 0.5
            if not np.isfinite(mu_fb):
                mu_fb = 0.5
            mu_fb = float(np.clip(mu_fb, 1e-3, 1 - 1e-3))

            aL, bL = _fit_beta_for_ratio(ratio_sub[low_mask], mu_fallback=mu_fb)
            aM, bM = _fit_beta_for_ratio(ratio_sub[mid_mask], mu_fallback=mu_fb)
            aH, bH = _fit_beta_for_ratio(ratio_sub[high_mask], mu_fallback=mu_fb)

            p_zero = float(np.mean(churn_sub == 0)) if churn_sub.size else 1.0

            churn_fit = churn_sub[churn_sub > 0]
            if churn_fit.size < 2:
                mean = float(np.log1p(churn_fit[0])) if churn_fit.size == 1 else 0.0
                w = [1.0]
                means = [mean]
                vars_ = [1.0]
            else:
                z = np.log1p(churn_fit).reshape(-1, 1)
                best_gmm, best_bic = None, float("inf")
                seed = _hash32(f"{cluster_id}|bug{b}")

                for k in _candidate_component_counts(
                    churn_fit.size, fixed_gmm_components
                ):
                    g = GaussianMixture(
                        n_components=k,
                        covariance_type="full",
                        random_state=seed,
                        reg_covar=1e-6,
                    )
                    g.fit(z)
                    bic = g.bic(z)
                    if bic < best_bic:
                        best_bic = bic
                        best_gmm = g

                gmm = best_gmm
                w = (gmm.weights_ / gmm.weights_.sum()).tolist()
                means = gmm.means_[:, 0].astype(float).tolist()
                covs = np.asarray(gmm.covariances_, dtype=float).reshape(-1)
                vars_ = np.clip(covs, 1e-6, None).astype(float).tolist()

            return {
                f"churn_q33_bug{b}": q33,
                f"churn_q66_bug{b}": q66,
                f"rand_churn_q5_bug{b}": rq5,
                f"rand_churn_q50_bug{b}": rq50,
                f"rand_churn_q99_bug{b}": rq99,
                f"ratio_low_alpha_bug{b}": aL,
                f"ratio_low_beta_bug{b}": bL,
                f"ratio_mid_alpha_bug{b}": aM,
                f"ratio_mid_beta_bug{b}": bM,
                f"ratio_high_alpha_bug{b}": aH,
                f"ratio_high_beta_bug{b}": bH,
                f"p_zero_bug{b}": p_zero,
                f"gmm_log_weights_bug{b}": w,
                f"gmm_log_means_bug{b}": means,
                f"gmm_log_vars_bug{b}": vars_,
            }

        for b in [0, 1]:
            mask_b = bug == b
            p.update(_fit_for_bug_subset(churn[mask_b], ratio[mask_b], b))

        params[cluster_id] = p

    return params


# ==============================
# Sampling helpers
# ==============================
def _sample_ratio_from_tertiles(
    rng: np.random.Generator, churn: int, p: Dict[str, Any]
) -> float:
    q33 = float(p.get("churn_q33", 0.0))
    q66 = float(p.get("churn_q66", 0.0))
    if churn <= 0:
        return 0.0
    if churn <= q33:
        r = float(rng.beta(float(p["ratio_low_alpha"]), float(p["ratio_low_beta"])))
    elif churn <= q66:
        r = float(rng.beta(float(p["ratio_mid_alpha"]), float(p["ratio_mid_beta"])))
    else:
        r = float(rng.beta(float(p["ratio_high_alpha"]), float(p["ratio_high_beta"])))
    return float(np.clip(r, 0.0, 1.0))


def _sample_ratio_from_tertiles_bug(
    rng: np.random.Generator, churn: int, p: Dict[str, Any], bug: int
) -> float:
    q33 = float(p.get(f"churn_q33_bug{bug}", 0.0))
    q66 = float(p.get(f"churn_q66_bug{bug}", 0.0))
    if churn <= 0:
        return 0.0
    if churn <= q33:
        r = float(
            rng.beta(
                float(p[f"ratio_low_alpha_bug{bug}"]),
                float(p[f"ratio_low_beta_bug{bug}"]),
            )
        )
    elif churn <= q66:
        r = float(
            rng.beta(
                float(p[f"ratio_mid_alpha_bug{bug}"]),
                float(p[f"ratio_mid_beta_bug{bug}"]),
            )
        )
    else:
        r = float(
            rng.beta(
                float(p[f"ratio_high_alpha_bug{bug}"]),
                float(p[f"ratio_high_beta_bug{bug}"]),
            )
        )
    return float(np.clip(r, 0.0, 1.0))


def _sample_churn_cluster_random(rng: np.random.Generator, p: Dict[str, Any]) -> int:
    p_zero = float(p.get("p_zero", 0.0))
    if rng.random() < p_zero:
        return 0
    q5 = int(p.get("rand_churn_q5", p.get("churn_clip_min", 0)))
    q50 = int(p.get("rand_churn_q50", max(q5, 1)))
    q99 = int(p.get("rand_churn_q99", p.get("churn_clip_max", max(q5, 1))))
    q5 = max(0, q5)
    q99 = max(q5, q99)
    q50 = min(max(q5, q50), q99)
    churn_f = rng.triangular(left=q5, mode=q50, right=q99) if q99 > q5 else float(q5)
    churn = max(0, int(round(churn_f)))
    return _clip_int(churn, int(p["churn_clip_min"]), int(p["churn_clip_max"]))


def _sample_churn_gmm_simple(rng: np.random.Generator, p: Dict[str, Any]) -> int:
    p_zero = float(p.get("p_zero", 0.0))
    if rng.random() < p_zero:
        return 0

    w = np.asarray(p["gmm_log_weights"], dtype=float)
    w = w / (w.sum() if w.sum() > 0 else 1.0)
    means = np.asarray(p["gmm_log_means"], dtype=float)
    vars_ = np.asarray(p["gmm_log_vars"], dtype=float)
    stds = np.sqrt(np.clip(vars_, 1e-6, None))

    comp = int(rng.choice(len(w), p=w))
    z = float(rng.normal(loc=means[comp], scale=stds[comp]))
    churn_f = float(np.expm1(z))
    churn = int(round(churn_f))
    churn = max(0, churn)
    return _clip_int(churn, int(p["churn_clip_min"]), int(p["churn_clip_max"]))


def _sample_churn_gmm_2(rng: np.random.Generator, p: Dict[str, Any], bug: int) -> int:
    p_zero = float(p.get(f"p_zero_bug{bug}", 0.0))
    if rng.random() < p_zero:
        return 0

    w = np.asarray(p[f"gmm_log_weights_bug{bug}"], dtype=float)
    w = w / (w.sum() if w.sum() > 0 else 1.0)
    means = np.asarray(p[f"gmm_log_means_bug{bug}"], dtype=float)
    vars_ = np.asarray(p[f"gmm_log_vars_bug{bug}"], dtype=float)
    stds = np.sqrt(np.clip(vars_, 1e-6, None))

    comp = int(rng.choice(len(w), p=w))
    z = float(rng.normal(loc=means[comp], scale=stds[comp]))
    churn_f = float(np.expm1(z))
    churn = int(round(churn_f))
    churn = max(0, churn)
    return _clip_int(churn, int(p["churn_clip_min"]), int(p["churn_clip_max"]))


def _split_la_ld(rng: np.random.Generator, churn: int, ratio: float) -> Tuple[int, int]:
    if churn <= 0:
        return 0, 0
    ratio = float(np.clip(ratio, 0.0, 1.0))
    la = int(rng.binomial(int(churn), ratio))
    ld = int(churn - la)
    return max(0, la), max(0, ld)


# ==============================
# Synthesis (per row)
# ==============================
def synthesize_la_ld(
    commit_id: str, p: Dict[str, Any], method: str, bug: Optional[int] = None
) -> Tuple[int, int]:
    method = _resolve_method(method)
    rng = _rng_from_key(commit_id)

    if method == "cluster_random":
        churn = _sample_churn_cluster_random(rng, p)
        ratio = _sample_ratio_from_tertiles(rng, churn, p)
        return _split_la_ld(rng, churn, ratio)

    if method == "gmm_simple":
        churn = _sample_churn_gmm_simple(rng, p)
        ratio = _sample_ratio_from_tertiles(rng, churn, p)
        return _split_la_ld(rng, churn, ratio)

    if method == "gmm_2":
        if bug is None:
            bug = 0
        churn = _sample_churn_gmm_2(rng, p, int(bug))
        ratio = _sample_ratio_from_tertiles_bug(rng, churn, p, int(bug))
        return _split_la_ld(rng, churn, ratio)

    raise ValueError(f"Unknown method: {method}")


# ==============================
# Main anonymization
# ==============================
def anonymize(
    df: pd.DataFrame,
    cluster_keys: pd.Series,
    params: Dict[str, Dict[str, Any]],
    method: str,
) -> pd.DataFrame:
    out_rows = []
    for idx, row in df.iterrows():
        cid = str(row[ID_FIELD])
        qid, bidx = cluster_keys.loc[idx]
        cluster_id = f"{qid}|{bidx}"
        p = params.get(cluster_id)
        if p is None:
            raise KeyError(f"Missing params for cluster {cluster_id}")

        bug = int(pd.to_numeric(row.get("bugcount", 0), errors="coerce") > 0)

        la_p, ld_p = synthesize_la_ld(cid, p, method=method, bug=bug)

        row2 = row.copy()
        row2["la"] = int(la_p)
        row2["ld"] = int(ld_p)
        row2["churn"] = int(la_p + ld_p)
        out_rows.append(row2)

    return pd.DataFrame(out_rows)


# ==============================
# CLI
# ==============================
def parse_cli_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Statistical anonymization for la/ld")
    ap.add_argument("--input-csv", required=True, help="Input CSV path")
    ap.add_argument("--output-csv", required=True, help="Output CSV path")
    ap.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_NUM_BINS,
        help="Quantile bins for clustering",
    )
    ap.add_argument(
        "--method",
        choices=[
            "cluster_random",
            "gmm_simple",
            "gmm_2",
            "gmm_no_cluster",
            "gmm_2_no_cluster",
        ],
        default="gmm_2",
        help="Anonymization method",
    )
    ap.add_argument(
        "--gmm-components",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Force a fixed GMM component count instead of per-cluster BIC selection",
    )
    return ap.parse_args()


def main():
    args = parse_cli_args()

    logger.info("Starting anonymization")
    logger.info(
        "Input: %s | Output: %s | bins=%d | method=%s | gmm_components=%s",
        args.input_csv,
        args.output_csv,
        args.bins,
        args.method,
        args.gmm_components if args.gmm_components is not None else "bic",
    )

    df = pd.read_csv(args.input_csv)

    # Required columns
    for col in [ID_FIELD, "la", "ld"]:
        if col not in df.columns:
            raise ValueError(
                f"Input must include '{ID_FIELD}', 'la', 'ld' (missing: {col})"
            )

    # Optional date sorting (keeps determinism in case you care about stable ordering)
    if DATE_FIELD in df.columns:
        df[DATE_FIELD] = pd.to_datetime(df[DATE_FIELD], unit="s", errors="coerce")
        df = df.sort_values(by=DATE_FIELD).reset_index(drop=True)

    df = add_derived(df)

    if _is_no_cluster_method(args.method):
        logger.info("Using single global cluster for method=%s", args.method)
        cluster_keys = assign_global_cluster(df)
    else:
        logger.info("Building %d quantile bins...", args.bins)
        bins_map = build_bins(df, QIDS, q=args.bins)

        logger.info("Assigning clusters...")
        cluster_keys = assign_clusters(df, bins_map)

    # Build params depending on method
    logger.info("Building per-cluster parameters...")
    t0 = time.time()
    resolved_method = _resolve_method(args.method)
    if resolved_method == "cluster_random":
        params = build_cluster_params_cluster_random(df, cluster_keys)
    elif resolved_method == "gmm_simple":
        params = build_cluster_params_gmm_simple(
            df, cluster_keys, fixed_gmm_components=args.gmm_components
        )
    elif resolved_method == "gmm_2":
        params = build_cluster_params_gmm_2(
            df, cluster_keys, fixed_gmm_components=args.gmm_components
        )
    else:
        raise ValueError(f"Unknown method: {resolved_method}")

    logger.info("Params built in %.2fs", time.time() - t0)

    logger.info("Anonymizing rows...")
    t1 = time.time()
    anon_df = anonymize(df, cluster_keys, params, method=args.method)
    logger.info("Anonymization done in %.2fs", time.time() - t1)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    anon_df.to_csv(args.output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(anon_df), args.output_csv)


if __name__ == "__main__":
    main()
