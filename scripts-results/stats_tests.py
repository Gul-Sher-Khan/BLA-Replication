#!/usr/bin/env python3
import argparse
from itertools import combinations
import logging
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from rename_methods_and_visualize import filter_method_frame, rename_method_series

try:
    from scipy.stats import binomtest, friedmanchisquare, rankdata, wilcoxon
except Exception as exc:
    raise SystemExit(
        "Missing scipy. Install with: pip install scipy\nError: {0}".format(exc)
    )


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("stats-tests")

UTILITY_METRICS = [
    ("auc", "AUC"),
    ("f1", "F1"),
    ("acc", "Accuracy"),
    ("pre", "Precision"),
    ("rec", "Recall"),
]
IPR_METRIC = ("tim_ipr", "IPR")
TEST_COLUMNS = [
    "section",
    "rq",
    "metric",
    "method_a",
    "method_b",
    "n",
    "median_a",
    "median_b",
    "median_diff",
    "cliffs_delta",
    "p_raw",
    "p_holm",
    "reject_holm",
    "test",
    "statistic",
    "df",
    "p_value",
    "n_methods",
    "n_observations",
]

RQ_METHODS = {
    "RQ1": [
        "GMM",
        "LACE",
        "LACE2",
        "Gen",
        "K-DA",
        "Random Add/Delete",
        "Random Switch",
    ],
    "RQ2": ["GMM", "CLA"],
    "RQ3": ["GMM+", "CLA+"],
    "RQ4": [
        "GMM",
        "CLA",
        "GMM+",
        "CLA+",
        "LACE",
        "LACE2",
        "Gen",
        "K-DA",
        "Random Add/Delete",
        "Random Switch",
    ],
}

UTILITY_RQ_PAIRS = {
    "RQ1": [
        ("GMM", "LACE"),
        ("GMM", "LACE2"),
        ("GMM", "Gen"),
        ("GMM", "K-DA"),
        ("GMM", "Random Add/Delete"),
        ("GMM", "Random Switch"),
    ],
    "RQ2": [("GMM", "CLA")],
    "RQ3": [("GMM+", "CLA+")],
    "RQ4": list(combinations(RQ_METHODS["RQ4"], 2)),
}


def find_files(results_root: str, filename: str) -> List[str]:
    matches: List[str] = []
    for root, _, files in os.walk(results_root):
        for entry in files:
            if entry.lower() == filename.lower():
                matches.append(os.path.join(root, entry))
    return sorted(matches)


def _load_tsvs(paths: Sequence[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path, sep="\t") for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_summary(results_root: str) -> pd.DataFrame:
    df = _load_tsvs(find_files(results_root, "summary.txt"))
    if df.empty:
        return df
    df = df.copy()
    df["method"] = rename_method_series(df["method"])
    df = filter_method_frame(df, "method")
    df["tim_ipr"] = pd.to_numeric(df["tim_ipr"], errors="coerce")
    return df


def load_runs(results_root: str) -> pd.DataFrame:
    df = _load_tsvs(find_files(results_root, "runs.txt"))
    if df.empty:
        return df
    df = df.copy()
    df["method"] = rename_method_series(df["method"])
    df = filter_method_frame(df, "method")
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
    for raw_metric, _ in UTILITY_METRICS:
        df[raw_metric] = pd.to_numeric(df[raw_metric], errors="coerce")
    return df


def build_utility_long(runs_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for raw_metric, label in UTILITY_METRICS:
        if raw_metric not in runs_df.columns:
            continue
        subset = runs_df[["project", "method", "run_id", raw_metric]].copy()
        subset = subset.rename(columns={raw_metric: "value"})
        subset["metric"] = label
        subset = subset.dropna(subset=["value", "run_id"])
        records.append(subset)
    if not records:
        return pd.DataFrame(columns=["project", "method", "run_id", "value", "metric"])
    out = pd.concat(records, ignore_index=True)
    out["run_id"] = out["run_id"].astype(int)
    return out


def build_ipr_frame(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=["project", "method", "value", "metric"])
    df = summary_df[["project", "method", "tim_ipr"]].copy()
    df = df.rename(columns={"tim_ipr": "value"})
    df["metric"] = IPR_METRIC[1]
    return df.dropna(subset=["value"])


def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, Dict[str, float | bool]]:
    if not pvals:
        return {}
    ordered = sorted(pvals.items(), key=lambda item: item[1])
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    total = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        scaled = (total - index) * float(p_value)
        running_max = max(running_max, scaled)
        adjusted[name] = min(running_max, 1.0)
    return {
        name: {
            "p_raw": float(pvals[name]),
            "p_holm": float(adjusted[name]),
            "reject": bool(adjusted[name] <= 0.05),
        }
        for name in pvals
    }


def cliffs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.size == 0 or right_arr.size == 0:
        return float("nan")
    diffs = np.sign(left_arr[:, None] - right_arr[None, :])
    return float(diffs.sum() / (left_arr.size * right_arr.size))


def _median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def paired_utility_values(
    utility_df: pd.DataFrame, metric: str, method_a: str, method_b: str
) -> pd.DataFrame:
    left = utility_df[
        (utility_df["metric"] == metric) & (utility_df["method"] == method_a)
    ][["project", "run_id", "value"]].rename(columns={"value": "value_a"})
    right = utility_df[
        (utility_df["metric"] == metric) & (utility_df["method"] == method_b)
    ][["project", "run_id", "value"]].rename(columns={"value": "value_b"})
    merged = left.merge(right, on=["project", "run_id"], how="inner")
    return merged.dropna(subset=["value_a", "value_b"])


def paired_ipr_values(
    ipr_df: pd.DataFrame, method_a: str, method_b: str
) -> pd.DataFrame:
    left = ipr_df[ipr_df["method"] == method_a][["project", "value"]].rename(
        columns={"value": "value_a"}
    )
    right = ipr_df[ipr_df["method"] == method_b][["project", "value"]].rename(
        columns={"value": "value_b"}
    )
    merged = left.merge(right, on=["project"], how="inner")
    return merged.dropna(subset=["value_a", "value_b"])


def wilcoxon_result(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float]:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.size == 0 or right_arr.size == 0:
        return float("nan"), float("nan")
    if np.allclose(left_arr, right_arr):
        return 0.0, 1.0
    result = wilcoxon(left_arr, right_arr, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def wilcoxon_p_value(left: Sequence[float], right: Sequence[float]) -> float:
    _, p_value = wilcoxon_result(left, right)
    return float(p_value)


def observation_matrix(
    df: pd.DataFrame,
    methods: Sequence[str],
    value_col: str,
    index_cols: Sequence[str],
) -> pd.DataFrame:
    matrix = (
        df[df["method"].isin(methods)]
        .pivot_table(
            index=list(index_cols), columns="method", values=value_col, aggfunc="first"
        )
        .reindex(columns=list(methods))
        .dropna()
    )
    return matrix


def mean_rank_table(matrix: pd.DataFrame, rq: str, metric: str) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame(
            columns=["rq", "metric", "method", "mean_rank", "n_observations"]
        )
    ranks = []
    for _, row in matrix.iterrows():
        ranks.append(rankdata(-row.to_numpy(dtype=float), method="average"))
    rank_frame = pd.DataFrame(ranks, columns=matrix.columns)
    return pd.DataFrame(
        {
            "rq": rq,
            "metric": metric,
            "method": matrix.columns.tolist(),
            "mean_rank": rank_frame.mean(axis=0).to_numpy(dtype=float),
            "n_observations": int(matrix.shape[0]),
        }
    )


def omnibus_result(
    matrix: pd.DataFrame,
    rq: str,
    metric: str,
) -> tuple[Dict[str, object], pd.DataFrame]:
    ranks = mean_rank_table(matrix, rq=rq, metric=metric)
    n_methods = int(matrix.shape[1])
    n_observations = int(matrix.shape[0])

    base_row = {
        "section": "omnibus",
        "rq": rq,
        "metric": metric,
        "method_a": "",
        "method_b": "",
        "n": "",
        "median_a": "",
        "median_b": "",
        "median_diff": "",
        "cliffs_delta": "",
        "p_raw": "",
        "p_holm": "",
        "reject_holm": "",
        "n_methods": n_methods,
        "n_observations": n_observations,
    }

    if n_methods < 2 or n_observations == 0:
        row = dict(base_row)
        row.update(
            {
                "test": "Unavailable",
                "statistic": float("nan"),
                "df": float("nan"),
                "p_value": float("nan"),
            }
        )
        return row, ranks

    if n_methods == 2:
        diffs = matrix.iloc[:, 0] - matrix.iloc[:, 1]
        non_zero = diffs[~np.isclose(diffs, 0.0)]
        positive_count = int((non_zero > 0).sum())
        trials = int(non_zero.shape[0])
        p_value = (
            float(
                binomtest(
                    positive_count,
                    max(trials, 1),
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if trials > 0
            else 1.0
        )
        row = dict(base_row)
        row.update(
            {
                "test": "Sign test",
                "statistic": positive_count,
                "df": 1,
                "p_value": p_value,
            }
        )
        return row, ranks

    if n_observations < 2:
        statistic = float("nan")
        p_value = float("nan")
    else:
        statistic, p_value = friedmanchisquare(
            *[matrix[method].to_numpy(dtype=float) for method in matrix.columns]
        )
        statistic = float(statistic)
        p_value = float(p_value)

    row = dict(base_row)
    row.update(
        {
            "test": "Friedman",
            "statistic": statistic,
            "df": max(n_methods - 1, 0),
            "p_value": p_value,
        }
    )
    return row, ranks


def build_utility_pairwise_rows(
    utility_df: pd.DataFrame, rq: str
) -> List[Dict[str, object]]:
    pairs = UTILITY_RQ_PAIRS[rq]
    rows: List[Dict[str, object]] = []
    for _, metric in UTILITY_METRICS:
        pvals: Dict[str, float] = {}
        metric_rows: List[Dict[str, object]] = []
        for method_a, method_b in pairs:
            paired = paired_utility_values(utility_df, metric, method_a, method_b)
            if paired.empty:
                continue
            key = f"{method_a}::{method_b}"
            statistic, p_value = wilcoxon_result(paired["value_a"], paired["value_b"])
            pvals[key] = p_value
            metric_rows.append(
                {
                    "section": "utility",
                    "rq": rq,
                    "metric": metric,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n": int(len(paired)),
                    "median_a": _median(paired["value_a"]),
                    "median_b": _median(paired["value_b"]),
                    "median_diff": _median(paired["value_a"] - paired["value_b"]),
                    "cliffs_delta": cliffs_delta(paired["value_a"], paired["value_b"]),
                    "test": "Wilcoxon signed-rank",
                    "statistic": statistic,
                    "df": "",
                    "p_value": p_value,
                    "n_methods": "",
                    "n_observations": "",
                    "pair_key": key,
                }
            )
        adjusted = holm_bonferroni(pvals)
        for row in metric_rows:
            result = adjusted[row.pop("pair_key")]
            row["p_raw"] = result["p_raw"]
            row["p_holm"] = result["p_holm"]
            row["reject_holm"] = result["reject"]
            rows.append(row)
    return rows


def build_ipr_pairwise_rows(ipr_df: pd.DataFrame, rq: str) -> List[Dict[str, object]]:
    pairs = list(combinations(RQ_METHODS[rq], 2))
    pvals: Dict[str, float] = {}
    pair_rows: List[Dict[str, object]] = []
    for method_a, method_b in pairs:
        paired = paired_ipr_values(ipr_df, method_a, method_b)
        if paired.empty:
            continue
        key = f"{method_a}::{method_b}"
        statistic, p_value = wilcoxon_result(paired["value_a"], paired["value_b"])
        pvals[key] = p_value
        pair_rows.append(
            {
                "section": "ipr",
                "rq": rq,
                "metric": "IPR",
                "method_a": method_a,
                "method_b": method_b,
                "n": int(len(paired)),
                "median_a": _median(paired["value_a"]),
                "median_b": _median(paired["value_b"]),
                "median_diff": _median(paired["value_a"] - paired["value_b"]),
                "cliffs_delta": cliffs_delta(paired["value_a"], paired["value_b"]),
                "test": "Wilcoxon signed-rank",
                "statistic": statistic,
                "df": "",
                "p_value": p_value,
                "n_methods": "",
                "n_observations": "",
                "pair_key": key,
            }
        )

    adjusted = holm_bonferroni(pvals)
    rows: List[Dict[str, object]] = []
    for row in pair_rows:
        result = adjusted[row.pop("pair_key")]
        row["p_raw"] = result["p_raw"]
        row["p_holm"] = result["p_holm"]
        row["reject_holm"] = result["reject"]
        rows.append(row)
    return rows


def build_rq1_to_rq3_tables(
    utility_df: pd.DataFrame, ipr_df: pd.DataFrame, rq: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = build_utility_pairwise_rows(utility_df, rq)
    ipr_matrix = observation_matrix(
        ipr_df, RQ_METHODS[rq], value_col="value", index_cols=["project"]
    )
    ipr_row, ipr_ranks = omnibus_result(ipr_matrix, rq=rq, metric="IPR")
    rows.append(ipr_row)
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=TEST_COLUMNS)
    else:
        out = out.reindex(columns=TEST_COLUMNS)
    return out, ipr_ranks


def build_rq4_outputs(
    utility_df: pd.DataFrame, ipr_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows: List[Dict[str, object]] = build_utility_pairwise_rows(utility_df, "RQ4")
    utility_rank_rows: List[pd.DataFrame] = []
    cliffs_matrices: Dict[str, pd.DataFrame] = {}

    for _, metric in UTILITY_METRICS:
        metric_df = utility_df[utility_df["metric"] == metric]
        matrix = observation_matrix(
            metric_df,
            RQ_METHODS["RQ4"],
            value_col="value",
            index_cols=["project", "run_id"],
        )
        rank_table = mean_rank_table(matrix, rq="RQ4", metric=metric)
        utility_rank_rows.append(rank_table)

        delta_matrix = pd.DataFrame(
            index=RQ_METHODS["RQ4"], columns=RQ_METHODS["RQ4"], dtype=float
        )
        for method_a in RQ_METHODS["RQ4"]:
            for method_b in RQ_METHODS["RQ4"]:
                if method_a == method_b:
                    delta_matrix.loc[method_a, method_b] = 0.0
                    continue
                paired = paired_utility_values(utility_df, metric, method_a, method_b)
                delta_matrix.loc[method_a, method_b] = cliffs_delta(
                    paired["value_a"], paired["value_b"]
                )
        cliffs_matrices[metric] = delta_matrix

    ipr_matrix = observation_matrix(
        ipr_df,
        RQ_METHODS["RQ4"],
        value_col="value",
        index_cols=["project"],
    )
    ipr_row, ipr_ranks = omnibus_result(ipr_matrix, rq="RQ4", metric="IPR")
    rows.append(ipr_row)

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=TEST_COLUMNS)
    else:
        out = out.reindex(columns=TEST_COLUMNS)
    utility_ranks = (
        pd.concat(utility_rank_rows, ignore_index=True)
        if utility_rank_rows
        else pd.DataFrame(
            columns=["rq", "metric", "method", "mean_rank", "n_observations"]
        )
    )
    return out, ipr_ranks, utility_ranks, cliffs_matrices


def write_tsv(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    logger.info("Wrote %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ-focused non-parametric tests for anonymization results."
    )
    parser.add_argument(
        "--results-root", default="results", help="Results root folder."
    )
    args = parser.parse_args()

    summary_df = load_summary(args.results_root)
    runs_df = load_runs(args.results_root)
    if summary_df.empty:
        raise ValueError("No summary.txt files found under the results root")
    if runs_df.empty:
        raise ValueError("No runs.txt files found under the results root")

    utility_df = build_utility_long(runs_df)
    ipr_df = build_ipr_frame(summary_df)
    stats_dir = os.path.join(args.results_root, "stats")

    for rq in ["RQ1", "RQ2", "RQ3"]:
        test_table, ipr_ranks = build_rq1_to_rq3_tables(utility_df, ipr_df, rq)
        write_tsv(test_table, os.path.join(stats_dir, f"{rq.lower()}_tests.tsv"))
        write_tsv(ipr_ranks, os.path.join(stats_dir, f"{rq.lower()}_ipr_ranks.tsv"))

    rq4_tests, rq4_ipr_ranks, rq4_utility_ranks, cliffs_matrices = build_rq4_outputs(
        utility_df, ipr_df
    )
    write_tsv(rq4_tests, os.path.join(stats_dir, "rq4_tests.tsv"))
    write_tsv(rq4_ipr_ranks, os.path.join(stats_dir, "rq4_ipr_ranks.tsv"))
    write_tsv(rq4_utility_ranks, os.path.join(stats_dir, "rq4_utility_ranks.tsv"))
    for metric, matrix in cliffs_matrices.items():
        metric_slug = metric.lower().replace(" ", "_")
        write_tsv(
            matrix,
            os.path.join(stats_dir, f"rq4_cliffs_matrix_{metric_slug}.tsv"),
        )


if __name__ == "__main__":
    main()
