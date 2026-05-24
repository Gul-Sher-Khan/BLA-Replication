#!/usr/bin/env python3
import argparse
import os
from typing import List

import pandas as pd

try:
    from scipy.stats import shapiro
except Exception as exc:
    raise SystemExit(
        "Missing scipy. Install with: pip install scipy\nError: {0}".format(exc)
    )


PERF_METRICS = ["auc", "f1", "acc", "rec"]


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
    for col in ["run_id"] + PERF_METRICS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def shapiro_line(values: pd.Series, alpha: float) -> str:
    vals = values.dropna().astype(float)
    n = vals.shape[0]
    if n < 3:
        return "n={0}, insufficient (need >=3)".format(n)
    if n > 5000:
        return "n={0}, too large for Shapiro-Wilk (max 5000)".format(n)

    stat, p_val = shapiro(vals)
    normal = p_val > alpha
    return "n={0}, W={1:.6f}, p={2:.6f}, normal={3}".format(n, stat, p_val, normal)


def section_ipr(summary_df: pd.DataFrame, alpha: float) -> List[str]:
    lines: List[str] = []
    lines.append("IPR Normality (tim_ipr)")
    lines.append("Shapiro-Wilk alpha={0}".format(alpha))
    lines.append("Grouped by method across projects")
    lines.append("")

    if summary_df.empty or "tim_ipr" not in summary_df.columns:
        lines.append("No summary/tim_ipr data found.")
        lines.append("")
        return lines

    ipr_df = summary_df.dropna(subset=["method", "tim_ipr"])[
        ["project", "method", "tim_ipr"]
    ]
    if ipr_df.empty:
        lines.append("No valid tim_ipr values found.")
        lines.append("")
        return lines

    for method in sorted(ipr_df["method"].unique()):
        series = ipr_df[ipr_df["method"] == method]["tim_ipr"]
        lines.append("  {0}: {1}".format(method, shapiro_line(series, alpha)))
    lines.append("")
    return lines


def section_summary_performance(summary_df: pd.DataFrame, alpha: float) -> List[str]:
    lines: List[str] = []
    lines.append("Model Performance Normality (summary-level averages)")
    lines.append("Shapiro-Wilk alpha={0}".format(alpha))
    lines.append("Grouped by metric and method across projects")
    lines.append("")

    if summary_df.empty:
        lines.append("No summary data found.")
        lines.append("")
        return lines

    for metric in PERF_METRICS:
        if metric not in summary_df.columns:
            lines.append("Metric {0}: missing".format(metric))
            continue
        lines.append("Metric {0}".format(metric))
        metric_df = summary_df.dropna(subset=["method", metric])
        for method in sorted(metric_df["method"].unique()):
            series = metric_df[metric_df["method"] == method][metric]
            lines.append("  {0}: {1}".format(method, shapiro_line(series, alpha)))
    lines.append("")
    return lines


def section_run_performance(runs_df: pd.DataFrame, alpha: float) -> List[str]:
    lines: List[str] = []
    lines.append("Model Performance Normality (run-level)")
    lines.append("Shapiro-Wilk alpha={0}".format(alpha))
    lines.append("Grouped by project, metric, and method using run values")
    lines.append("")

    if runs_df.empty:
        lines.append("No runs data found.")
        lines.append("")
        return lines

    for project in sorted(runs_df["project"].dropna().unique()):
        lines.append("Project: {0}".format(project))
        proj_df = runs_df[runs_df["project"] == project]
        methods = sorted(proj_df["method"].dropna().unique())
        for metric in PERF_METRICS:
            if metric not in proj_df.columns:
                lines.append("  Metric {0}: missing".format(metric))
                continue
            lines.append("  Metric {0}".format(metric))
            for method in methods:
                series = proj_df[proj_df["method"] == method][metric]
                lines.append("    {0}: {1}".format(method, shapiro_line(series, alpha)))
        lines.append("")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Shapiro-Wilk normality tests for IPR and model performance."
    )
    parser.add_argument(
        "--results-root", default="results", help="Path to results directory"
    )
    parser.add_argument(
        "--out-file",
        default="shapiro_normality_tests.txt",
        help="Output file name under results root",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    args = parser.parse_args()

    summary_df = load_summary(args.results_root)
    runs_df = load_runs(args.results_root)

    lines: List[str] = []
    lines.append("Shapiro-Wilk Normality Tests")
    lines.append("Results root: {0}".format(os.path.abspath(args.results_root)))
    lines.append("Alpha: {0}".format(args.alpha))
    lines.append("")

    lines.extend(section_ipr(summary_df, args.alpha))
    lines.extend(section_summary_performance(summary_df, args.alpha))
    lines.extend(section_run_performance(runs_df, args.alpha))

    out_path = os.path.join(args.results_root, args.out_file)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")

    print("Wrote {0}".format(out_path))


if __name__ == "__main__":
    main()
