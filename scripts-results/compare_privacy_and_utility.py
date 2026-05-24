#!/usr/bin/env python3
import argparse, json, os, time
import pandas as pd
import numpy as np
from rename_methods_and_visualize import METHOD_RENAMES

from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    accuracy_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

# ====================== Author-like RF config ======================
METRICS_ALL = [
    "la",
    "ld",
    "nd",
    "ns",
    "nf",
    "ent",
    "ndev",
    "nuc",
    "age",
    "aexp",
    "asexp",
    "arexp",
]
METRICS_LA_LD = ["la", "ld"]
METRICS_QIDS_ONLY = [
    "nd",
    "ns",
    "nf",
    "ent",
    "ndev",
    "nuc",
    "age",
    "aexp",
    "asexp",
    "arexp",
]
METRICS_SETS = {
    "all": METRICS_ALL,
    "la_ld": METRICS_LA_LD,
    "qids_only": METRICS_QIDS_ONLY,
}

ALLOWED_VARIANT_METHODS = {
    method for method in METHOD_RENAMES.keys() if method != "Original_Baseline"
}


def _display_method_name(name: str) -> str:
    return METHOD_RENAMES.get(name, name)


def _prep(df: pd.DataFrame, metrics):
    df = df.copy()
    if "commit_id" not in df.columns or "bugcount" not in df.columns:
        raise ValueError("CSV must include commit_id and bugcount")
    df["bugcount"] = pd.to_numeric(df["bugcount"], errors="coerce").fillna(0)
    df["defect"] = (df["bugcount"] > 0).astype(int)
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce") / 3600 / 24
    feats = df.copy()
    for col in ["tcmt", "author_date", "defect"]:
        if col in feats.columns:
            feats.drop(columns=[col], inplace=True)
    X = feats[metrics].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = (pd.to_numeric(df["bugcount"], errors="coerce").fillna(0) > 0).astype(int)
    return X, y


def _rf_author(n_jobs=None):
    # Same as author’s rf_predictor.py
    return RandomForestClassifier(
        bootstrap=True,
        ccp_alpha=0.0,
        class_weight=None,
        criterion="gini",
        max_depth=100,
        max_features="sqrt",
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=1400,
        n_jobs=n_jobs,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False,
    )


def rf_eval_author(train_df: pd.DataFrame, test_df: pd.DataFrame, metrics, rf_n_jobs=None):
    X_tr, y_tr = _prep(train_df, metrics)
    X_te, y_te = _prep(test_df, metrics)
    X_tr_res, y_tr_res = SMOTEENN(random_state=42).fit_resample(X_tr, y_tr)

    clf = _rf_author(n_jobs=rf_n_jobs)
    clf.fit(X_tr_res, y_tr_res)

    pred = clf.predict(X_te)
    if hasattr(clf, "predict_proba"):
        auc_input = clf.predict_proba(X_te)[:, 1]
    else:
        auc_input = pred
    res = {
        "auc": float(roc_auc_score(y_te, auc_input)),
        "f1": float(f1_score(y_te, pred)),
        "pre": float(precision_score(y_te, pred)),
        "acc": float(accuracy_score(y_te, pred)),
        "rec": float(recall_score(y_te, pred)),
    }
    fi = dict(
        sorted(
            {m: float(w) for m, w in zip(metrics, clf.feature_importances_)}.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    return res, fi


def rf_runs_with_bootstrap(
    df_anon, df_base, boot_train_csv, boot_test_csv, metrics, rf_n_jobs=None
):
    bt_train = pd.read_csv(boot_train_csv)
    bt_test = pd.read_csv(boot_test_csv)
    runs = []
    for bid in bt_train["bootstrap_id"].unique():
        tr_ids = set(bt_train[bt_train["bootstrap_id"] == bid]["commit_id"].astype(str))
        te_ids = set(bt_test[bt_test["bootstrap_id"] == bid]["commit_id"].astype(str))
        tr = df_anon[df_anon["commit_id"].astype(str).isin(tr_ids)]
        te = df_base[df_base["commit_id"].astype(str).isin(te_ids)]
        if tr.empty or te.empty:
            continue
        res, fi = rf_eval_author(tr, te, metrics, rf_n_jobs=rf_n_jobs)
        runs.append({"bootstrap_id": int(bid), "results": res, "feat_importance": fi})
    return runs


def rf_single_split(
    df_anon, df_base, metrics, test_size=0.2, seed=42, rf_n_jobs=None
):
    """Backward-compatible single split (used if no multi-run requested)."""
    y_all = (
        (pd.to_numeric(df_base["bugcount"], errors="coerce").fillna(0) > 0)
        .astype(int)
        .values
    )
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_tr, idx_te = next(sss.split(np.zeros(len(y_all)), y_all))
    tr_ids = set(df_base.iloc[idx_tr]["commit_id"].astype(str))
    te_ids = set(df_base.iloc[idx_te]["commit_id"].astype(str))
    tr = df_anon[df_anon["commit_id"].astype(str).isin(tr_ids)]
    te = df_base[df_base["commit_id"].astype(str).isin(te_ids)]
    if tr.empty or te.empty:
        return []
    res, fi = rf_eval_author(tr, te, metrics, rf_n_jobs=rf_n_jobs)
    return [{"run_id": 0, "results": res, "feat_importance": fi}]


def generate_stratified_splits(
    df_base: pd.DataFrame, n_runs: int, test_size: float = 0.2, seed: int = 42
):
    """Generate stratified splits returning list of dicts with train/test commit id sets."""
    y_all = (
        (pd.to_numeric(df_base["bugcount"], errors="coerce").fillna(0) > 0)
        .astype(int)
        .values
    )
    sss = StratifiedShuffleSplit(
        n_splits=n_runs, test_size=test_size, random_state=seed
    )
    splits = []
    for i, (idx_tr, idx_te) in enumerate(sss.split(np.zeros(len(y_all)), y_all)):
        tr_ids = set(df_base.iloc[idx_tr]["commit_id"].astype(str))
        te_ids = set(df_base.iloc[idx_te]["commit_id"].astype(str))
        splits.append({"run_id": i, "train_ids": tr_ids, "test_ids": te_ids})
    return splits


def evaluate_variant_runs(
    df_anon: pd.DataFrame,
    df_base: pd.DataFrame,
    splits,
    metrics,
    rf_n_jobs=None,
    progress_label=None,
):
    runs = []
    total = len(splits)
    for index, s in enumerate(splits, start=1):
        run_start = time.time()
        tr = df_anon[df_anon["commit_id"].astype(str).isin(s["train_ids"])].copy()
        te = df_base[df_base["commit_id"].astype(str).isin(s["test_ids"])].copy()
        if tr.empty or te.empty:
            continue
        res, fi = rf_eval_author(tr, te, metrics, rf_n_jobs=rf_n_jobs)
        runs.append({"run_id": s["run_id"], "results": res, "feat_importance": fi})
        if progress_label and (index == 1 or index == total or index % 5 == 0):
            print(
                "      RF {0}: run {1}/{2} completed in {3:.1f}s".format(
                    progress_label, index, total, time.time() - run_start
                ),
                flush=True,
            )
    return runs


def evaluate_baseline_runs(df_base: pd.DataFrame, splits, metrics, rf_n_jobs=None):
    runs = []
    for s in splits:
        tr = df_base[df_base["commit_id"].astype(str).isin(s["train_ids"])].copy()
        te = df_base[df_base["commit_id"].astype(str).isin(s["test_ids"])].copy()
        if tr.empty or te.empty:
            continue
        res, fi = rf_eval_author(tr, te, metrics, rf_n_jobs=rf_n_jobs)
        runs.append({"run_id": s["run_id"], "results": res, "feat_importance": fi})
    return runs


def average_rf_runs(runs):
    if not runs:
        return {}, {}
    avg, fi_avg = {}, {}
    for r in runs:
        for k, v in r["results"].items():
            avg[k] = avg.get(k, 0.0) + float(v)
        for k, v in r["feat_importance"].items():
            fi_avg[k] = fi_avg.get(k, 0.0) + float(v)
    n = len(runs)
    return {k: v / n for k, v in avg.items()}, {k: v / n for k, v in fi_avg.items()}


def compute_median_decrease(baseline_runs, variant_runs):
    """Compute median decrease (baseline - variant) per metric across intersecting run_ids."""
    if not baseline_runs or not variant_runs:
        return {}

    def _rid(r):
        return r.get("run_id", r.get("bootstrap_id"))

    b_map = {_rid(r): r for r in baseline_runs if _rid(r) is not None}
    v_map = {_rid(r): r for r in variant_runs if _rid(r) is not None}
    common = sorted(set(b_map.keys()).intersection(v_map.keys()))
    if not common:
        return {}
    metrics = ["auc", "f1", "pre", "rec", "acc"]
    med = {}
    for m in metrics:
        diffs = []
        for rid in common:
            b_val = float(b_map[rid]["results"].get(m, 0.0))
            v_val = float(v_map[rid]["results"].get(m, 0.0))
            diffs.append(b_val - v_val)
        if diffs:
            med[m] = float(np.median(diffs))
    return med


# ====================== Privacy (robust) ======================
class PrivacyMeasurer:
    def __init__(
        self,
        attributes_for_queries,
        hidden_attributes,
        path_to_base_file,
        configuration_dict,
    ):
        self.attributes_for_queries = attributes_for_queries
        self.hidden_attributes = hidden_attributes
        self.path_to_base_file = path_to_base_file
        self.bin_atts = attributes_for_queries + hidden_attributes
        self._bins_map = {}
        self._query_dict = self._build_queries(configuration_dict)

    def _bin_col(self, s: pd.Series):
        s = pd.to_numeric(s, errors="coerce")
        return pd.qcut(s, 20, duplicates="drop", retbins=True)

    def _build_queries(self, configuration_dict):
        df = pd.read_csv(self.path_to_base_file)
        for att in self.bin_atts:
            if att not in df.columns:
                raise KeyError(f"Baseline missing column: {att}")
            _, bins = self._bin_col(df[att])
            self._bins_map[att] = bins
            df[att + "_binned"] = pd.cut(
                pd.to_numeric(df[att], errors="coerce"), bins=bins
            )
            print(f"Bins for {att} are {len(bins)}")
        all_queries = []
        rng = np.random.default_rng(42)
        for keys_len, total in configuration_dict.items():
            for _ in range(total):
                key_vals = {}
                sub = df.copy()
                candidates = self.attributes_for_queries.copy()
                for _k in range(keys_len):
                    if not candidates:
                        break
                    attr = rng.choice(candidates)
                    col = attr + "_binned"
                    choices = [v for v in sub[col].dropna().unique().tolist()]
                    if not choices:
                        candidates.remove(attr)
                        continue
                    val = rng.choice(choices)
                    sub = sub[sub[col] == val]
                    key_vals[col] = val
                    candidates.remove(attr)
                all_queries.append(key_vals)
        return all_queries

    def measure_privacy(self, anon_file_path):
        base = pd.read_csv(self.path_to_base_file)
        anon = pd.read_csv(anon_file_path)
        for att in self.bin_atts:
            bins = self._bins_map[att]
            base[att + "_binned"] = pd.cut(
                pd.to_numeric(base[att], errors="coerce"), bins=bins
            )
            anon[att + "_binned"] = pd.cut(
                pd.to_numeric(anon[att], errors="coerce"), bins=bins
            )

        breach = total = 0
        my_breach = my_total = 0
        orig_len = found_len = 0

        for key_vals in self._query_dict:
            b_sub = base.copy()
            a_sub = anon.copy()
            for k, interval in key_vals.items():
                raw = k.replace("_binned", "")
                left, right = interval.left, interval.right
                b_sub = b_sub[
                    pd.to_numeric(b_sub[raw], errors="coerce").between(
                        left, right, inclusive="right"
                    )
                ]
                a_sub = a_sub[
                    pd.to_numeric(a_sub[raw], errors="coerce").between(
                        left, right, inclusive="right"
                    )
                ]
            if b_sub.empty:
                continue

            if "commit_id" in a_sub.columns and "commit_id" in b_sub.columns:
                found = a_sub[
                    a_sub["commit_id"].astype(str).isin(b_sub["commit_id"].astype(str))
                ]
            else:
                found = a_sub.iloc[0:0]

            orig_len += len(b_sub)
            found_len += len(found)

            for sens in self.hidden_attributes:
                col = sens + "_binned"
                vc_a = (
                    a_sub[col].value_counts()
                    if col in a_sub.columns
                    else pd.Series(dtype="int64")
                )
                vc_b = (
                    b_sub[col].value_counts()
                    if col in b_sub.columns
                    else pd.Series(dtype="int64")
                )

                for value in vc_a.index:
                    a_n = int(vc_a.get(value, 0))
                    b_n = int(vc_b.get(value, 0))
                    my_total += 1
                    if a_n - b_n == 0:
                        my_breach += 1

                if len(vc_a) and len(vc_b):
                    max_a = vc_a.sort_values(ascending=True).index[-1]
                    max_b = vc_b.sort_values(ascending=True).index[-1]
                    if max_a == max_b:
                        breach += 1
                    total += 1

        total = max(total, 1)
        my_total = max(my_total, 1)
        orig_len = max(orig_len, 1)

        return {
            "tim_breaches": round(breach, 6),
            "tim_total": round(total, 6),
            "tim_ipr": round(1 - breach / total, 6),
            "my_breach_metric": round(my_breach, 6),
            "my_metric_total": round(my_total, 6),
            "my_ipr_metric": round(my_breach / my_total, 6),
            "original_length": round(orig_len, 6),
            "found_length": round(found_len, 6),
            "found_percentage": round(1 - found_len / orig_len, 6),
        }


# ====================== Runner ======================
def run_eval_for_variant(
    name,
    anon_csv,
    baseline_csv,
    metrics,
    bootstrap_prefix=None,
    splits=None,
    privacy_measurer=None,
    rf_n_jobs=None,
):
    print(f"\n=== {name} ===")
    # Privacy
    if privacy_measurer is None:
        privacy_measurer = PrivacyMeasurer(
            ["nf", "nd", "ns", "ent", "ndev", "nuc", "age", "aexp", "asexp", "arexp"],
            ["la", "ld"],
            baseline_csv,
            {1: 100, 2: 1000, 4: 1000},
        )
    privacy_start = time.time()
    privacy = privacy_measurer.measure_privacy(anon_csv)
    print(
        "      Privacy {0}: completed in {1:.1f}s".format(
            name, time.time() - privacy_start
        ),
        flush=True,
    )

    # Utility (RF)
    rf_start = time.time()
    df_anon = pd.read_csv(anon_csv).dropna(subset=["commit_id"])
    df_base = pd.read_csv(baseline_csv).dropna(subset=["commit_id"])

    # Align on common commit_ids
    common = set(df_anon["commit_id"].astype(str)).intersection(
        set(df_base["commit_id"].astype(str))
    )
    df_anon = df_anon[df_anon["commit_id"].astype(str).isin(common)].copy()
    df_base = df_base[df_base["commit_id"].astype(str).isin(common)].copy()

    if (
        bootstrap_prefix
        and os.path.isfile(f"{bootstrap_prefix}_train_bootstrap.csv")
        and os.path.isfile(f"{bootstrap_prefix}_test_bootstrap.csv")
    ):
        runs = rf_runs_with_bootstrap(
            df_anon,
            df_base,
            f"{bootstrap_prefix}_train_bootstrap.csv",
            f"{bootstrap_prefix}_test_bootstrap.csv",
            metrics,
            rf_n_jobs=rf_n_jobs,
        )
    elif splits is not None:
        runs = evaluate_variant_runs(
            df_anon,
            df_base,
            splits,
            metrics,
            rf_n_jobs=rf_n_jobs,
            progress_label=name,
        )
    else:
        runs = rf_single_split(
            df_anon, df_base, metrics, test_size=0.2, seed=42, rf_n_jobs=rf_n_jobs
        )
    print(
        "      RF {0}: {1} runs completed in {2:.1f}s".format(
            name, len(runs), time.time() - rf_start
        ),
        flush=True,
    )

    rf_avg, fi_avg = average_rf_runs(runs)
    return {
        "name": name,
        "anon_csv": os.path.abspath(anon_csv),
        "baseline_csv": os.path.abspath(baseline_csv),
        "privacy": privacy,
        "rf_avg": rf_avg,
        "rf_feat_imp_avg": fi_avg,
        "rf_runs": runs,
        "n_runs": len(runs),
    }


def to_excel_report(out_xlsx, summary, baseline_result, author_result, llm_result):
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        # Summary sheet
        summary_rows = [
            [
                "Variant",
                "IPR (Tim)",
                "My IPR",
                "AUC",
                "F1",
                "Precision",
                "Recall",
                "Accuracy",
                "RF Runs",
            ]
        ]
        # Baseline (no privacy) placeholders
        summary_rows.append(
            [
                baseline_result["name"],
                None,
                None,
                baseline_result["rf_avg"].get("auc"),
                baseline_result["rf_avg"].get("f1"),
                baseline_result["rf_avg"].get("pre"),
                baseline_result["rf_avg"].get("rec"),
                baseline_result["rf_avg"].get("acc"),
                baseline_result["n_runs"],
            ]
        )
        summary_rows.append(
            [
                author_result["name"],
                author_result["privacy"]["tim_ipr"],
                author_result["privacy"]["my_ipr_metric"],
                author_result["rf_avg"].get("auc"),
                author_result["rf_avg"].get("f1"),
                author_result["rf_avg"].get("pre"),
                author_result["rf_avg"].get("rec"),
                author_result["rf_avg"].get("acc"),
                author_result["n_runs"],
            ]
        )
        summary_rows.append(
            [
                llm_result["name"],
                llm_result["privacy"]["tim_ipr"],
                llm_result["privacy"]["my_ipr_metric"],
                llm_result["rf_avg"].get("auc"),
                llm_result["rf_avg"].get("f1"),
                llm_result["rf_avg"].get("pre"),
                llm_result["rf_avg"].get("rec"),
                llm_result["rf_avg"].get("acc"),
                llm_result["n_runs"],
            ]
        )
        sm = pd.DataFrame(summary_rows)
        sm.columns = sm.iloc[0]
        sm = sm.drop(0)
        sm.to_excel(xw, sheet_name="Summary", index=False)

        # Privacy sheets
        pd.DataFrame(
            author_result["privacy"].items(), columns=["metric", "value"]
        ).to_excel(xw, sheet_name="Privacy_Author", index=False)
        pd.DataFrame(
            llm_result["privacy"].items(), columns=["metric", "value"]
        ).to_excel(xw, sheet_name="Privacy_LLM", index=False)

        # RF averages + feature importance
        pd.DataFrame(
            author_result["rf_avg"].items(), columns=["metric", "value"]
        ).to_excel(xw, sheet_name="RF_Author_Avg", index=False)
        pd.DataFrame(
            author_result["rf_feat_imp_avg"].items(), columns=["feature", "importance"]
        ).to_excel(xw, sheet_name="RF_Author_Features", index=False)

        pd.DataFrame(
            llm_result["rf_avg"].items(), columns=["metric", "value"]
        ).to_excel(xw, sheet_name="RF_LLM_Avg", index=False)
        pd.DataFrame(
            llm_result["rf_feat_imp_avg"].items(), columns=["feature", "importance"]
        ).to_excel(xw, sheet_name="RF_LLM_Features", index=False)

        # All RF runs
        def runs_to_df(runs):
            rows = []
            for r in runs:
                rid = r.get("run_id", r.get("bootstrap_id", None))
                row = {"run_id": rid}
                row.update(r["results"])
                rows.append(row)
            return pd.DataFrame(rows)

        runs_to_df(baseline_result["rf_runs"]).to_excel(
            xw, sheet_name="RF_Baseline_Runs", index=False
        )
        runs_to_df(author_result["rf_runs"]).to_excel(
            xw, sheet_name="RF_Author_Runs", index=False
        )
        runs_to_df(llm_result["rf_runs"]).to_excel(
            xw, sheet_name="RF_LLM_Runs", index=False
        )

        # Median decreases sheet
        dec_rows = []
        for variant in [author_result, llm_result]:
            md = variant.get("median_decrease_vs_original", {})
            for k, v in md.items():
                dec_rows.append(
                    {"variant": variant["name"], "metric": k, "median_decrease": v}
                )
        if dec_rows:
            pd.DataFrame(dec_rows).to_excel(
                xw, sheet_name="Median_Decreases", index=False
            )

        # Metadata
        pd.DataFrame([summary]).to_excel(xw, sheet_name="Meta", index=False)


def _parse_variant_arg(raw: str):
    if not raw or "=" not in raw:
        raise ValueError(f"Invalid --variant '{raw}'. Use name=path")
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip().strip('"')
    if not name or not path:
        raise ValueError(f"Invalid --variant '{raw}'. Use name=path")
    return {"name": name, "path": path}


def _load_variants(variants_args, variants_json_path):
    variants = []
    if variants_args:
        for raw in variants_args:
            variants.append(_parse_variant_arg(raw))
    if variants_json_path:
        with open(variants_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k, v in data.items():
                variants.append({"name": str(k), "path": str(v)})
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "name" in item and "path" in item:
                    variants.append(
                        {"name": str(item["name"]), "path": str(item["path"])}
                    )
                else:
                    raise ValueError(
                        "Invalid variants JSON list entry. Use {name, path}."
                    )
        else:
            raise ValueError("Invalid variants JSON; expected dict or list.")
    deduped = []
    seen = set()
    for v in variants:
        key = v["name"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(v)
    unknown = sorted({variant["name"] for variant in deduped} - ALLOWED_VARIANT_METHODS)
    if unknown:
        raise ValueError(
            "Unexpected anonymization methods requested: {0}".format(", ".join(unknown))
        )
    return deduped


def _write_text_outputs(out_dir, project_name, baseline_result, variant_results):
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.txt")
    runs_path = os.path.join(out_dir, "runs.txt")
    fi_avg_path = os.path.join(out_dir, "feature_importance_avg.txt")
    fi_runs_path = os.path.join(out_dir, "feature_importance_runs.txt")

    summary_cols = [
        "project",
        "method",
        "tim_ipr",
        "my_ipr",
        "auc",
        "f1",
        "pre",
        "rec",
        "acc",
        "n_runs",
        "csv_path",
    ]

    def _fmt(v):
        return "NA" if v is None else str(v)

    baseline_name = _display_method_name(baseline_result["name"])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\t".join(summary_cols) + "\n")
        f.write(
            "\t".join(
                [
                    project_name,
                    baseline_name,
                    "NA",
                    "NA",
                    _fmt(baseline_result["rf_avg"].get("auc")),
                    _fmt(baseline_result["rf_avg"].get("f1")),
                    _fmt(baseline_result["rf_avg"].get("pre")),
                    _fmt(baseline_result["rf_avg"].get("rec")),
                    _fmt(baseline_result["rf_avg"].get("acc")),
                    str(baseline_result["n_runs"]),
                    _fmt(baseline_result.get("baseline_csv")),
                ]
            )
            + "\n"
        )

        for r in variant_results:
            method_name = _display_method_name(r["name"])
            f.write(
                "\t".join(
                    [
                        project_name,
                        method_name,
                        _fmt(r["privacy"].get("tim_ipr")),
                        _fmt(r["privacy"].get("my_ipr_metric")),
                        _fmt(r["rf_avg"].get("auc")),
                        _fmt(r["rf_avg"].get("f1")),
                        _fmt(r["rf_avg"].get("pre")),
                        _fmt(r["rf_avg"].get("rec")),
                        _fmt(r["rf_avg"].get("acc")),
                        str(r["n_runs"]),
                        _fmt(r.get("anon_csv")),
                    ]
                )
                + "\n"
            )

    run_cols = ["project", "method", "run_id", "auc", "f1", "pre", "rec", "acc"]
    with open(runs_path, "w", encoding="utf-8") as f:
        f.write("\t".join(run_cols) + "\n")

        def _write_runs(name, runs):
            for r in runs:
                rid = r.get("run_id", r.get("bootstrap_id"))
                res = r.get("results", {})
                f.write(
                    "\t".join(
                        [
                            project_name,
                            name,
                            str(rid),
                            _fmt(res.get("auc")),
                            _fmt(res.get("f1")),
                            _fmt(res.get("pre")),
                            _fmt(res.get("rec")),
                            _fmt(res.get("acc")),
                        ]
                    )
                    + "\n"
                )

        _write_runs(baseline_name, baseline_result["rf_runs"])
        for r in variant_results:
            _write_runs(_display_method_name(r["name"]), r["rf_runs"])

    fi_avg_cols = ["project", "method", "feature", "importance"]
    with open(fi_avg_path, "w", encoding="utf-8") as f:
        f.write("\t".join(fi_avg_cols) + "\n")

        def _write_fi_avg(name, fi_avg):
            for feat, val in fi_avg.items():
                f.write(
                    "\t".join(
                        [
                            project_name,
                            name,
                            str(feat),
                            _fmt(val),
                        ]
                    )
                    + "\n"
                )

        _write_fi_avg(baseline_name, baseline_result["rf_feat_imp_avg"])
        for r in variant_results:
            _write_fi_avg(_display_method_name(r["name"]), r["rf_feat_imp_avg"])

    fi_run_cols = ["project", "method", "run_id", "feature", "importance"]
    with open(fi_runs_path, "w", encoding="utf-8") as f:
        f.write("\t".join(fi_run_cols) + "\n")

        def _write_fi_runs(name, runs):
            for r in runs:
                rid = r.get("run_id", r.get("bootstrap_id"))
                fi = r.get("feat_importance", {})
                for feat, val in fi.items():
                    f.write(
                        "\t".join(
                            [
                                project_name,
                                name,
                                str(rid),
                                str(feat),
                                _fmt(val),
                            ]
                        )
                        + "\n"
                    )

        _write_fi_runs(baseline_name, baseline_result["rf_runs"])
        for r in variant_results:
            _write_fi_runs(_display_method_name(r["name"]), r["rf_runs"])


def main():
    ap = argparse.ArgumentParser(
        description="Compare privacy + utility: Author Anon vs LLM metrics vs same baseline."
    )
    ap.add_argument("--baseline", required=True, help="Author NON-ANON CSV (baseline).")
    ap.add_argument("--graph-anon", help="Graph-based ANONYMIZED CSV.")
    ap.add_argument("--cluster-lace-anon", help="Cluster + LACE ANONYMIZED CSV.")
    ap.add_argument(
        "--bootstrap-prefix",
        help="Optional: data_split_<dataset> prefix (for *_train_bootstrap.csv, *_test_bootstrap.csv).",
    )
    ap.add_argument(
        "--out-xlsx", default="comparison_results.xlsx", help="Excel report path."
    )
    ap.add_argument(
        "--out-json", default="comparison_results.json", help="JSON summary path."
    )
    ap.add_argument(
        "--rf-runs",
        type=int,
        default=1,
        help="Number of stratified runs (ignored if bootstrap files provided). Use 5 as requested.",
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size for stratified splits when multi-run.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Seed for split generation.")
    ap.add_argument(
        "--metrics-set",
        choices=sorted(METRICS_SETS.keys()),
        default="all",
        help="Which feature set to use: all, la_ld, qids_only.",
    )
    ap.add_argument(
        "--variant",
        action="append",
        help="Multi-variant mode: name=path (repeatable).",
    )
    ap.add_argument(
        "--variants-json",
        help="Multi-variant mode: JSON file with dict or list of {name, path}.",
    )
    ap.add_argument(
        "--out-dir",
        help="Output directory for text summaries (multi-variant mode).",
    )
    ap.add_argument(
        "--project-name",
        default="project",
        help="Project name for text outputs.",
    )
    args = ap.parse_args()

    t0 = time.time()

    # Load baseline once for splits / baseline evaluation
    baseline_df_full = (
        pd.read_csv(args.baseline).dropna(subset=["commit_id"])
        if os.path.isfile(args.baseline)
        else None
    )

    metrics = METRICS_SETS[args.metrics_set]

    use_bootstrap = (
        args.bootstrap_prefix
        and os.path.isfile(f"{args.bootstrap_prefix}_train_bootstrap.csv")
        and os.path.isfile(f"{args.bootstrap_prefix}_test_bootstrap.csv")
    )
    splits = None
    baseline_runs = []
    if use_bootstrap:
        # Reuse bootstrap for baseline by treating df_base as both anon/base
        if baseline_df_full is not None:
            baseline_runs = rf_runs_with_bootstrap(
                baseline_df_full,
                baseline_df_full,
                f"{args.bootstrap_prefix}_train_bootstrap.csv",
                f"{args.bootstrap_prefix}_test_bootstrap.csv",
                metrics,
            )
    else:
        if args.rf_runs > 1 and baseline_df_full is not None:
            splits = generate_stratified_splits(
                baseline_df_full,
                n_runs=args.rf_runs,
                test_size=args.test_size,
                seed=args.seed,
            )
            baseline_runs = evaluate_baseline_runs(baseline_df_full, splits, metrics)
        else:
            # single run fallback
            baseline_runs = rf_single_split(
                baseline_df_full,
                baseline_df_full,
                metrics,
                test_size=args.test_size,
                seed=args.seed,
            )

    baseline_avg, baseline_fi_avg = average_rf_runs(baseline_runs)
    baseline_result = {
        "name": "Original_Baseline",
        "baseline_csv": os.path.abspath(args.baseline),
        "rf_avg": baseline_avg,
        "rf_feat_imp_avg": baseline_fi_avg,
        "rf_runs": baseline_runs,
        "n_runs": len(baseline_runs),
    }

    variants = _load_variants(args.variant, args.variants_json)
    if variants:
        for v in variants:
            if not os.path.isfile(v["path"]):
                raise FileNotFoundError(
                    f"Variant not found: {v['name']} -> {v['path']}"
                )

        privacy_measurer = PrivacyMeasurer(
            ["nf", "nd", "ns", "ent", "ndev", "nuc", "age", "aexp", "asexp", "arexp"],
            ["la", "ld"],
            args.baseline,
            {1: 100, 2: 1000, 4: 1000},
        )

        variant_results = []
        for v in variants:
            res = run_eval_for_variant(
                v["name"],
                v["path"],
                args.baseline,
                metrics,
                args.bootstrap_prefix,
                splits=splits,
                privacy_measurer=privacy_measurer,
            )
            res["median_decrease_vs_original"] = compute_median_decrease(
                baseline_runs, res.get("rf_runs", [])
            )
            variant_results.append(res)

        summary = {
            "baseline": os.path.abspath(args.baseline),
            "metrics_set": args.metrics_set,
            "bootstrap_prefix": args.bootstrap_prefix,
            "time": time.time() - t0,
        }
        baseline_payload = {
            **baseline_result,
            "name": _display_method_name(baseline_result["name"]),
        }
        variant_payload = [
            {
                **result,
                "name": _display_method_name(result["name"]),
            }
            for result in variant_results
        ]
        payload = {
            "summary": summary,
            "baseline_result": baseline_payload,
            "variant_results": variant_payload,
        }
        if args.out_json:
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        out_dir = args.out_dir or os.getcwd()
        _write_text_outputs(
            out_dir, args.project_name, baseline_result, variant_results
        )
        print(json.dumps(payload, indent=2))
        print(f"\nSaved text outputs in: {os.path.abspath(out_dir)}")
        return

    if not args.graph_anon or not args.cluster_lace_anon:
        raise ValueError(
            "--graph-anon and --cluster-lace-anon are required when no --variant is provided."
        )

    # Evaluate Graph anonymized
    author_res = run_eval_for_variant(
        "Graph_Anon",
        args.graph_anon,
        args.baseline,
        metrics,
        args.bootstrap_prefix,
        splits=splits,
    )
    # Evaluate Cluster + LACE metrics
    llm_res = run_eval_for_variant(
        "Cluster_Lace",
        args.cluster_lace_anon,
        args.baseline,
        metrics,
        args.bootstrap_prefix,
        splits=splits,
    )

    # Median decreases vs baseline
    author_res["median_decrease_vs_original"] = compute_median_decrease(
        baseline_runs, author_res.get("rf_runs", [])
    )
    llm_res["median_decrease_vs_original"] = compute_median_decrease(
        baseline_runs, llm_res.get("rf_runs", [])
    )

    summary = {
        "baseline": os.path.abspath(args.baseline),
        "graph_anon": os.path.abspath(args.graph_anon),
        "cluster_lace_anon": os.path.abspath(args.cluster_lace_anon),
        "metrics_set": args.metrics_set,
        "bootstrap_prefix": args.bootstrap_prefix,
        "time": time.time() - t0,
    }

    # Write outputs
    print(
        json.dumps(
            {
                "summary": summary,
                "baseline_result": baseline_result,
                "author_result": author_res,
                "llm_result": llm_res,
            },
            indent=2,
        )
    )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "baseline_result": baseline_result,
                "author_result": author_res,
                "llm_result": llm_res,
            },
            f,
            indent=2,
        )

    to_excel_report(args.out_xlsx, summary, baseline_result, author_res, llm_res)
    print(
        f"\nSaved:\n  JSON  -> {os.path.abspath(args.out_json)}\n  Excel -> {os.path.abspath(args.out_xlsx)}"
    )


if __name__ == "__main__":
    main()
