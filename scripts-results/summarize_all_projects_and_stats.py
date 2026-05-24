#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Tuple

import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception as exc:
    raise SystemExit(
        "Missing scipy. Install with: pip install scipy\nError: {0}".format(exc)
    )


CLA_METHODS = [
    "cla_random",
    "cla_gmm_simple",
    "cla_gmm_2",
    "cla_gmm_no_cluster",
    "cla_gmm_2_no_cluster",
    "lace_cluster_lace1",
    "lace_global_lace1",
    "lace_kmeans_lace1",
]
GRAPH_METHODS = ["gen", "k_da_anon", "random_add_delete", "random_switch"]
BASELINE_METHOD = "Original_Baseline"
RUN_METRICS = ["auc", "f1", "acc", "rec"]
SUMMARY_COLUMNS = ["project", "method", "tim_ipr", "auc", "f1", "acc", "rec", "n_runs"]


def _read_tabular(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def load_summary(results_root: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for root, _, files in os.walk(results_root):
        for file_name in files:
            if file_name.lower() != "summary.txt":
                continue
            full_path = os.path.join(root, file_name)
            df = _read_tabular(full_path)
            if "project" not in df.columns:
                df["project"] = os.path.basename(os.path.dirname(full_path))
            rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    for col in ["tim_ipr", "auc", "f1", "acc", "rec", "n_runs"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_runs(results_root: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for root, _, files in os.walk(results_root):
        for file_name in files:
            if file_name.lower() != "runs.txt":
                continue
            full_path = os.path.join(root, file_name)
            df = _read_tabular(full_path)
            if "project" not in df.columns:
                df["project"] = os.path.basename(os.path.dirname(full_path))
            rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    for col in ["run_id"] + RUN_METRICS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def paired_wilcoxon(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    key_cols: List[str],
    left_col: str,
    right_col: str,
    alternative: str = "two-sided",
) -> Tuple[int, float, float, str]:
    merged = pd.merge(
        left_df[key_cols + [left_col]].rename(columns={left_col: "left_val"}),
        right_df[key_cols + [right_col]].rename(columns={right_col: "right_val"}),
        on=key_cols,
        how="inner",
    ).dropna(subset=["left_val", "right_val"])

    n = merged.shape[0]
    if n < 2:
        return n, float("nan"), float("nan"), "insufficient_pairs"

    diffs = merged["left_val"] - merged["right_val"]
    med_diff = float(diffs.median())
    direction = (
        "left>right" if med_diff > 0 else ("left<right" if med_diff < 0 else "equal")
    )

    try:
        _, p_val = wilcoxon(
            merged["left_val"], merged["right_val"], alternative=alternative
        )
    except ValueError:
        p_val = 1.0

    return n, med_diff, float(p_val), direction


def section_combined_summary(summary_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    lines.append("Combined Summary Across Projects")
    if summary_df.empty:
        lines.append("No summary.txt files found.")
        lines.append("")
        return lines

    cols = [c for c in SUMMARY_COLUMNS if c in summary_df.columns]
    table = summary_df[cols].sort_values(["project", "method"]).reset_index(drop=True)

    lines.append("Columns: {0}".format(", ".join(cols)))
    lines.append("")
    lines.append(table.to_csv(sep="\t", index=False, lineterminator="\n").strip())
    lines.append("")
    return lines


def section_run_metric_tests(runs_df: pd.DataFrame, alpha: float) -> List[str]:
    lines: List[str] = []
    lines.append("Run-Level Statistical Tests Per Project")
    lines.append("Test: Wilcoxon signed-rank, paired by run_id, two-sided")
    lines.append("Left=method, Right={0}".format(BASELINE_METHOD))
    lines.append("")

    if runs_df.empty:
        lines.append("No runs.txt files found.")
        lines.append("")
        return lines

    for project in sorted(runs_df["project"].dropna().unique()):
        project_df = runs_df[runs_df["project"] == project].copy()
        base_df = project_df[project_df["method"] == BASELINE_METHOD]
        if base_df.empty:
            lines.append("Project: {0} (missing baseline, skipped)".format(project))
            lines.append("")
            continue

        lines.append("Project: {0}".format(project))
        methods = sorted(
            [m for m in project_df["method"].dropna().unique() if m != BASELINE_METHOD]
        )
        for metric in RUN_METRICS:
            if metric not in project_df.columns:
                lines.append("  Metric {0}: missing".format(metric))
                continue

            lines.append("  Metric {0}".format(metric))
            for method in methods:
                method_df = project_df[project_df["method"] == method]
                n, med_diff, p_val, direction = paired_wilcoxon(
                    method_df,
                    base_df,
                    key_cols=["run_id"],
                    left_col=metric,
                    right_col=metric,
                    alternative="two-sided",
                )
                if n < 2:
                    lines.append("    {0}: n={1}, insufficient pairs".format(method, n))
                else:
                    lines.append(
                        "    {0}: n={1}, med_diff={2:.6f}, {3}, p={4:.6f}, reject={5}".format(
                            method,
                            n,
                            med_diff,
                            direction,
                            p_val,
                            p_val <= alpha,
                        )
                    )
        lines.append("")

    return lines


def section_ipr_tests(summary_df: pd.DataFrame, alpha: float) -> List[str]:
    lines: List[str] = []
    lines.append("IPR Statistical Tests Across Projects")
    lines.append("Test: Wilcoxon signed-rank, paired by project, one-sided Left>Graph")
    lines.append("")

    if summary_df.empty or "tim_ipr" not in summary_df.columns:
        lines.append("tim_ipr is missing in summary data.")
        lines.append("")
        return lines

    ipr_df = summary_df[["project", "method", "tim_ipr"]].copy()
    ipr_df = ipr_df.dropna(subset=["project", "method", "tim_ipr"])
    if ipr_df.empty:
        lines.append("No valid tim_ipr values found.")
        lines.append("")
        return lines

    any_result = False
    for cla in CLA_METHODS:
        for graph in GRAPH_METHODS:
            cla_df = ipr_df[ipr_df["method"] == cla]
            graph_df = ipr_df[ipr_df["method"] == graph]
            n, med_diff, p_val, direction = paired_wilcoxon(
                cla_df,
                graph_df,
                key_cols=["project"],
                left_col="tim_ipr",
                right_col="tim_ipr",
                alternative="greater",
            )
            if n < 2:
                lines.append(
                    "  {0} vs {1}: n={2}, insufficient pairs".format(cla, graph, n)
                )
                continue

            any_result = True
            lines.append(
                "  {0} vs {1}: n={2}, med_diff={3:.6f}, {4}, p={5:.6f}, reject={6}".format(
                    cla,
                    graph,
                    n,
                    med_diff,
                    direction.replace("left", "Left").replace("right", "Graph"),
                    p_val,
                    p_val <= alpha,
                )
            )

    if not any_result:
        lines.append("No comparable CLA-vs-Graph tim_ipr pairs found.")

    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize all projects and run statistical tests in one output file."
    )
    parser.add_argument(
        "--results-root", default="results", help="Path to results directory"
    )
    parser.add_argument(
        "--out-file",
        default="all_projects_summary_and_stats.txt",
        help="Output file name under results root",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    args = parser.parse_args()

    summary_df = load_summary(args.results_root)
    runs_df = load_runs(args.results_root)

    lines: List[str] = []
    lines.append("All Projects Summary + Statistical Tests")
    lines.append("Results root: {0}".format(os.path.abspath(args.results_root)))
    lines.append("Alpha: {0}".format(args.alpha))
    lines.append("")

    lines.extend(section_combined_summary(summary_df))
    lines.extend(section_run_metric_tests(runs_df, args.alpha))
    lines.extend(section_ipr_tests(summary_df, args.alpha))

    out_path = os.path.join(args.results_root, args.out_file)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")

    print("Wrote {0}".format(out_path))


if __name__ == "__main__":
    main()
