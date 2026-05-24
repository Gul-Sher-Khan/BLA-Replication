#!/usr/bin/env python3

import argparse
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("gmm-joint-anon")

ID_FIELD = "commit_id"
BUGCOUNT_FIELD = "bugcount"
GLOBAL_METHOD_NAME = "gmm_joint_2_global"
CLUSTER_METHOD_NAME = "gmm_joint_2_cluster"
METHOD_NAMES = [GLOBAL_METHOD_NAME, CLUSTER_METHOD_NAME]

COUNT_FEATURES = ["la", "ld", "nf", "nd", "ns", "ndev", "nuc"]
CONTINUOUS_FEATURES = ["ent", "age", "aexp", "asexp", "arexp"]
JOINT_FEATURES = COUNT_FEATURES + CONTINUOUS_FEATURES
QIDS = ["nf", "nd", "ns", "ent", "ndev", "nuc", "age", "aexp", "asexp", "arexp"]
DEFAULT_NUM_BINS = 20
MAX_COMPONENTS = 5
GLOBAL_REG_COVAR = 1e-4
CLUSTER_REG_COVAR = 1e-3


def _hash32(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def _rng_from_key(key: str) -> np.random.Generator:
    seed = _hash32(key) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def _clip_int(x: float, lo: int, hi: int) -> int:
    if hi < lo:
        hi = lo
    return int(max(lo, min(hi, int(round(x)))))


@dataclass
class FeatureTransform:
    name: str
    mean: float
    std: float
    observed_min: float
    observed_max: float
    use_log1p: bool


class JointGaussianSampler:
    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        self.weights = np.asarray(spec["weights"], dtype=float)
        total = float(self.weights.sum())
        self.weights = self.weights / (total if total > 0 else 1.0)
        self.means = np.asarray(spec["means"], dtype=float)
        self.covariance_type = str(spec["covariance_type"])
        if self.covariance_type == "full":
            self.covariances = np.asarray(spec["covariances"], dtype=float)
        else:
            self.covariances = np.asarray(spec["covariances"], dtype=float)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        comp = int(rng.choice(len(self.weights), p=self.weights))
        mean = self.means[comp]
        if self.covariance_type == "full":
            return rng.multivariate_normal(mean=mean, cov=self.covariances[comp])
        std = np.sqrt(np.clip(self.covariances[comp], 1e-12, None))
        return rng.normal(loc=mean, scale=std)


def build_bins(df: pd.DataFrame, cols: List[str], q: int) -> Dict[str, np.ndarray]:
    bins_map: Dict[str, np.ndarray] = {}
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce")
        series_non_na = series.dropna()
        if series_non_na.nunique() < 2:
            bins = np.array([-np.inf, np.inf], dtype=float)
        else:
            _, bins = pd.qcut(series_non_na, q, duplicates="drop", retbins=True)
        bins_map[col] = bins
    return bins_map


def bin_index(value: Any, bins: np.ndarray) -> int:
    try:
        numeric_value = float(value)
        index = np.digitize([numeric_value], bins, right=True)[0] - 1
        if 0 <= index < len(bins) - 1:
            return int(index)
    except Exception:
        pass
    return -1


def pick_primary_qid(commit_id: str, qids: List[str] = QIDS) -> str:
    index = _hash32(commit_id) % len(qids)
    return qids[index]


def assign_clusters(df: pd.DataFrame, bins_map: Dict[str, np.ndarray]) -> pd.Series:
    keys = []
    for _, row in df.iterrows():
        commit_id = str(row[ID_FIELD])
        qid = pick_primary_qid(commit_id)
        bidx = bin_index(row.get(qid, np.nan), bins_map[qid])
        keys.append((qid, bidx))
    return pd.Series(keys, index=df.index, name="cluster_key")


def _project_name_from_input(path: str) -> str:
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return parent or os.path.splitext(os.path.basename(path))[0]


def _finite_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _clean_feature(
    series: pd.Series, use_log1p: bool
) -> tuple[np.ndarray, FeatureTransform]:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        median = 0.0
        observed_min = 0.0
        observed_max = 0.0
    else:
        median = float(np.median(finite))
        observed_min = float(np.min(finite))
        observed_max = float(np.max(finite))

    clean = numeric.copy()
    clean[~np.isfinite(clean)] = median
    if use_log1p:
        clean = np.clip(clean, 0.0, None)
        observed_min = max(0.0, observed_min)
        observed_max = max(observed_min, observed_max)

    transformed = np.log1p(clean) if use_log1p else clean
    mean = float(np.mean(transformed)) if transformed.size else 0.0
    std = float(np.std(transformed)) if transformed.size else 1.0
    if not np.isfinite(std) or std <= 0.0:
        std = 1.0

    spec = FeatureTransform(
        name=str(series.name),
        mean=mean,
        std=std,
        observed_min=observed_min,
        observed_max=observed_max,
        use_log1p=use_log1p,
    )
    return clean, spec


def _build_transform_matrix(
    df: pd.DataFrame,
) -> tuple[np.ndarray, Dict[str, FeatureTransform]]:
    columns = []
    transforms: Dict[str, FeatureTransform] = {}
    for feature in JOINT_FEATURES:
        if feature not in df.columns:
            raise ValueError(f"Input CSV is missing required feature: {feature}")
        clean, spec = _clean_feature(df[feature], use_log1p=feature in COUNT_FEATURES)
        transformed = np.log1p(clean) if spec.use_log1p else clean
        columns.append((transformed - spec.mean) / spec.std)
        transforms[feature] = spec
    matrix = (
        np.column_stack(columns) if columns else np.empty((len(df), 0), dtype=float)
    )
    return matrix, transforms


def _prepare_joint_data(
    df: pd.DataFrame,
) -> tuple[np.ndarray, Dict[str, FeatureTransform], np.ndarray]:
    matrix, transforms = _build_transform_matrix(df)
    bug_labels = (
        (pd.to_numeric(df[BUGCOUNT_FIELD], errors="coerce").fillna(0) > 0)
        .astype(int)
        .to_numpy()
    )
    return matrix, transforms, bug_labels


def _gmm_spec_from_model(
    model: GaussianMixture, covariance_type: str
) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "weights": model.weights_.astype(float).tolist(),
        "means": model.means_.astype(float).tolist(),
        "covariance_type": covariance_type,
        "n_components": int(model.n_components),
    }
    if covariance_type == "full":
        spec["covariances"] = np.asarray(model.covariances_, dtype=float).tolist()
    else:
        spec["covariances"] = np.asarray(model.covariances_, dtype=float).tolist()
    return spec


def _fit_candidate_gmm(
    data: np.ndarray,
    n_components: int,
    seed: int,
    covariance_type: str,
    reg_covar: float,
) -> Optional[GaussianMixture]:
    try:
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=seed,
            reg_covar=reg_covar,
        )
        model.fit(data)
        score = float(model.score(data))
        bic = float(model.bic(data))
        if not np.isfinite(score) or not np.isfinite(bic):
            raise FloatingPointError("non-finite model score or BIC")
        if covariance_type == "full":
            covariances = np.asarray(model.covariances_, dtype=float)
            for cov in covariances:
                eigvals = np.linalg.eigvalsh(cov)
                if np.min(eigvals) <= 0.0:
                    raise np.linalg.LinAlgError("non-positive covariance")
        else:
            covariances = np.asarray(model.covariances_, dtype=float)
            if np.any(covariances <= 0.0):
                raise np.linalg.LinAlgError("non-positive diagonal covariance")
        return model
    except Exception:
        return None


def _fit_best_gmm(
    data: np.ndarray,
    seed_key: str,
    scope_label: str,
    bug_value: int,
    force_single_component: bool,
    fixed_gmm_components: Optional[int],
    reg_covar: float,
) -> Dict[str, Any]:
    seed = _hash32(seed_key)
    n_samples, n_features = data.shape
    if fixed_gmm_components is not None:
        component_grid = [
            max(1, min(int(fixed_gmm_components), min(MAX_COMPONENTS, n_samples)))
        ]
    elif force_single_component:
        component_grid = [1]
    else:
        component_grid = list(range(1, min(MAX_COMPONENTS, n_samples) + 1))

    best_model: Optional[GaussianMixture] = None
    best_bic = float("inf")
    chosen_covariance = "full"

    for k in component_grid:
        model = _fit_candidate_gmm(
            data,
            k,
            seed,
            covariance_type="full",
            reg_covar=reg_covar,
        )
        if model is None:
            continue
        bic = float(model.bic(data))
        if bic < best_bic:
            best_bic = bic
            best_model = model
            chosen_covariance = "full"

    if best_model is None:
        logger.warning(
            "%s bug=%d full-covariance fit failed for all k in %s; falling back to diagonal covariance",
            scope_label,
            bug_value,
            component_grid,
        )
        for k in component_grid:
            model = _fit_candidate_gmm(
                data,
                k,
                seed,
                covariance_type="diag",
                reg_covar=reg_covar,
            )
            if model is None:
                continue
            bic = float(model.bic(data))
            if bic < best_bic:
                best_bic = bic
                best_model = model
                chosen_covariance = "diag"

    if best_model is None:
        raise RuntimeError(
            f"Could not fit any GaussianMixture for bug={bug_value} with n={n_samples}, d={n_features}"
        )

    spec = _gmm_spec_from_model(best_model, chosen_covariance)
    spec["bic"] = best_bic
    logger.info(
        "Fit %s bug=%d: n_samples=%d, covariance=%s, k=%d, bic=%.3f",
        scope_label,
        bug_value,
        n_samples,
        chosen_covariance,
        spec["n_components"],
        best_bic,
    )
    return spec


def _fit_bug_models(
    matrix: np.ndarray,
    bug_labels: np.ndarray,
    project_name: str,
    fixed_gmm_components: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    n_features = matrix.shape[1]
    class_rows = {bug_value: matrix[bug_labels == bug_value] for bug_value in (0, 1)}
    class_counts = {bug_value: rows.shape[0] for bug_value, rows in class_rows.items()}

    deferred: List[int] = []
    models: Dict[int, Dict[str, Any]] = {}

    for bug_value in sorted(
        class_rows.keys(), key=lambda val: class_counts[val], reverse=True
    ):
        rows = class_rows[bug_value]
        n_samples = rows.shape[0]
        logger.info("Preparing bug=%d subset with n_samples=%d", bug_value, n_samples)

        if n_samples == 0:
            deferred.append(bug_value)
            continue

        if n_samples < n_features:
            logger.warning(
                "bug=%d has %d rows, fewer than n_features=%d; will reuse the other class model if available",
                bug_value,
                n_samples,
                n_features,
            )
            deferred.append(bug_value)
            continue

        force_single_component = n_samples < 50 or n_samples < (n_features * 2)
        if force_single_component:
            logger.warning(
                "bug=%d has a small subset (n_samples=%d, n_features=%d); forcing k=1",
                bug_value,
                n_samples,
                n_features,
            )
        models[bug_value] = _fit_best_gmm(
            rows,
            seed_key=f"{project_name}|bug{bug_value}",
            scope_label="global",
            bug_value=bug_value,
            force_single_component=force_single_component,
            fixed_gmm_components=fixed_gmm_components,
            reg_covar=GLOBAL_REG_COVAR,
        )

    for bug_value in deferred:
        other_bug = 1 - bug_value
        if other_bug in models:
            logger.warning(
                "bug=%d is reusing bug=%d model because its subset is too small",
                bug_value,
                other_bug,
            )
            models[bug_value] = dict(models[other_bug])
            models[bug_value]["reused_from_bug"] = other_bug
            continue

        rows = class_rows[bug_value]
        if rows.shape[0] == 0:
            rows = matrix
        logger.warning(
            "bug=%d could not reuse the other class model; fitting a single-component diagonal fallback on %d rows",
            bug_value,
            rows.shape[0],
        )
        models[bug_value] = _fit_best_gmm(
            rows,
            seed_key=f"{project_name}|bug{bug_value}|fallback",
            scope_label="global-fallback",
            bug_value=bug_value,
            force_single_component=True,
            fixed_gmm_components=None,
            reg_covar=GLOBAL_REG_COVAR,
        )

    return models


def _cluster_label(cluster_key: Any) -> str:
    if isinstance(cluster_key, tuple) and len(cluster_key) == 2:
        return f"{cluster_key[0]}|{cluster_key[1]}"
    return str(cluster_key)


def _fit_cluster_bug_samplers(
    matrix: np.ndarray,
    bug_labels: np.ndarray,
    cluster_keys: pd.Series,
    project_name: str,
    global_samplers: Dict[int, JointGaussianSampler],
    fixed_gmm_components: Optional[int] = None,
) -> Dict[str, Dict[int, JointGaussianSampler]]:
    n_features = matrix.shape[1]
    cluster_samplers: Dict[str, Dict[int, JointGaussianSampler]] = {}

    for cluster_key in pd.unique(cluster_keys):
        cluster_name = _cluster_label(cluster_key)
        cluster_mask = (cluster_keys == cluster_key).to_numpy()
        cluster_matrix = matrix[cluster_mask]
        cluster_bug_labels = bug_labels[cluster_mask]
        bug_samplers: Dict[int, JointGaussianSampler] = {}

        for bug_value in (0, 1):
            subset = cluster_matrix[cluster_bug_labels == bug_value]
            n_samples = subset.shape[0]
            if n_samples < n_features:
                logger.warning(
                    "cluster=%s bug=%d fallback=global_model reason=n_samples(%d)<n_features(%d)",
                    cluster_name,
                    bug_value,
                    n_samples,
                    n_features,
                )
                bug_samplers[bug_value] = global_samplers[bug_value]
                continue

            force_single_component = n_samples < (n_features * 2)
            if force_single_component:
                logger.warning(
                    "cluster=%s bug=%d fallback=k1 reason=n_samples(%d)<2xn_features(%d)",
                    cluster_name,
                    bug_value,
                    n_samples,
                    n_features * 2,
                )

            try:
                spec = _fit_best_gmm(
                    subset,
                    seed_key=f"{project_name}|{cluster_name}|bug{bug_value}",
                    scope_label=f"cluster={cluster_name}",
                    bug_value=bug_value,
                    force_single_component=force_single_component,
                    fixed_gmm_components=fixed_gmm_components,
                    reg_covar=CLUSTER_REG_COVAR,
                )
                bug_samplers[bug_value] = JointGaussianSampler(spec)
            except Exception as exc:
                logger.warning(
                    "cluster=%s bug=%d fallback=global_model reason=fit_failed error=%s",
                    cluster_name,
                    bug_value,
                    exc,
                )
                bug_samplers[bug_value] = global_samplers[bug_value]

        cluster_samplers[cluster_name] = bug_samplers

    return cluster_samplers


def _inverse_transform_sample(
    sample: np.ndarray, transforms: Dict[str, FeatureTransform]
) -> Dict[str, Any]:
    restored: Dict[str, Any] = {}
    for idx, feature in enumerate(JOINT_FEATURES):
        spec = transforms[feature]
        value = float(sample[idx] * spec.std + spec.mean)
        if spec.use_log1p:
            value = float(np.expm1(value))
            value = float(np.clip(value, 0.0, spec.observed_max))
            restored[feature] = _clip_int(value, 0, int(round(spec.observed_max)))
            continue
        value = float(np.clip(value, spec.observed_min, spec.observed_max))
        restored[feature] = value
    return restored


def fit_global_models(
    matrix: np.ndarray,
    bug_labels: np.ndarray,
    project_name: str,
    fixed_gmm_components: Optional[int] = None,
) -> Dict[int, JointGaussianSampler]:
    models = _fit_bug_models(
        matrix,
        bug_labels,
        project_name=project_name,
        fixed_gmm_components=fixed_gmm_components,
    )
    return {bug_value: JointGaussianSampler(spec) for bug_value, spec in models.items()}


def anonymize_global(
    df: pd.DataFrame,
    transforms: Dict[str, FeatureTransform],
    samplers: Dict[int, JointGaussianSampler],
) -> pd.DataFrame:
    out_rows = []
    for _, row in df.iterrows():
        bug_value = int(pd.to_numeric(row.get(BUGCOUNT_FIELD, 0), errors="coerce") > 0)
        rng = _rng_from_key(str(row[ID_FIELD]))
        sample = samplers[bug_value].sample(rng)
        restored = _inverse_transform_sample(sample, transforms)

        updated = row.copy()
        for feature, value in restored.items():
            updated[feature] = value
        updated["churn"] = int(updated["la"] + updated["ld"])
        out_rows.append(updated)

    anon_df = pd.DataFrame(out_rows)
    if "churn" not in df.columns:
        return anon_df[df.columns.tolist() + ["churn"]]
    return anon_df[df.columns.tolist()]


def anonymize_clustered(
    df: pd.DataFrame,
    cluster_keys: pd.Series,
    transforms: Dict[str, FeatureTransform],
    cluster_samplers: Dict[str, Dict[int, JointGaussianSampler]],
) -> pd.DataFrame:
    out_rows = []
    for idx, row in df.iterrows():
        bug_value = int(pd.to_numeric(row.get(BUGCOUNT_FIELD, 0), errors="coerce") > 0)
        cluster_name = _cluster_label(cluster_keys.loc[idx])
        samplers = cluster_samplers.get(cluster_name)
        if samplers is None:
            raise KeyError(f"Missing samplers for cluster {cluster_name}")
        rng = _rng_from_key(str(row[ID_FIELD]))
        sample = samplers[bug_value].sample(rng)
        restored = _inverse_transform_sample(sample, transforms)

        updated = row.copy()
        for feature, value in restored.items():
            updated[feature] = value
        updated["churn"] = int(updated["la"] + updated["ld"])
        out_rows.append(updated)

    anon_df = pd.DataFrame(out_rows)
    if "churn" not in df.columns:
        return anon_df[df.columns.tolist() + ["churn"]]
    return anon_df[df.columns.tolist()]


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint multivariate bug-aware GMM anonymization for JIT process metrics."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV path")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_NUM_BINS,
        help="Quantile bins for QID-based clustering",
    )
    parser.add_argument(
        "--method",
        choices=METHOD_NAMES,
        default=GLOBAL_METHOD_NAME,
        help="Anonymization method",
    )
    parser.add_argument(
        "--gmm-components",
        type=int,
        choices=list(range(1, MAX_COMPONENTS + 1)),
        default=None,
        help="Force a fixed GMM component count instead of BIC-based selection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    project_name = _project_name_from_input(args.input_csv)

    logger.info(
        "Starting joint anonymization | input=%s | output=%s | method=%s | project=%s | bins=%d | gmm_components=%s",
        args.input_csv,
        args.output_csv,
        args.method,
        project_name,
        args.bins,
        args.gmm_components if args.gmm_components is not None else "bic",
    )

    df = pd.read_csv(args.input_csv)
    required_columns = [ID_FIELD, BUGCOUNT_FIELD, *JOINT_FEATURES]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: {0}".format(", ".join(missing))
        )

    fit_start = time.time()
    matrix, transforms, bug_labels = _prepare_joint_data(df)
    global_samplers = fit_global_models(
        matrix,
        bug_labels,
        project_name=project_name,
        fixed_gmm_components=args.gmm_components,
    )

    cluster_keys = None
    cluster_samplers = None
    if args.method == CLUSTER_METHOD_NAME:
        bins_map = build_bins(df, QIDS, q=args.bins)
        cluster_keys = assign_clusters(df, bins_map)
        cluster_samplers = _fit_cluster_bug_samplers(
            matrix,
            bug_labels,
            cluster_keys,
            project_name=project_name,
            global_samplers=global_samplers,
            fixed_gmm_components=args.gmm_components,
        )

    logger.info("Model fitting completed in %.2fs", time.time() - fit_start)

    sample_start = time.time()
    if args.method == CLUSTER_METHOD_NAME:
        anon_df = anonymize_clustered(
            df,
            cluster_keys=cluster_keys,
            transforms=transforms,
            cluster_samplers=cluster_samplers,
        )
    else:
        anon_df = anonymize_global(df, transforms=transforms, samplers=global_samplers)
    logger.info("Sampling completed in %.2fs", time.time() - sample_start)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    anon_df.to_csv(args.output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(anon_df), args.output_csv)


if __name__ == "__main__":
    main()
