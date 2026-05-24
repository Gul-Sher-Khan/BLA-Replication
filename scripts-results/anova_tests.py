#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Tuple

import pandas as pd

try:
    from scipy.stats import f_oneway
except Exception as exc:
    raise SystemExit(
        "Missing scipy. Install with: pip install scipy\nError: {0}".format(exc)
    )


DEFAULT_METRICS = ["auc", "f1", "acc", "rec"]


def _read_tabular(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


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
    for col in ["run_id", "auc", "f1", "pre", "rec", "acc"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


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
    if "tim_ipr" in out.columns:
        out["tim_ipr"] = pd.to_numeric(out["tim_ipr"], errors="coerce")
    return out


def _anova(groups: Dict[str, pd.Series]) -> Tuple[int, float, float, Dict[str, int]]:
    clean_groups: Dict[str, pd.Series] = {}
    counts: Dict[str, int] = {}
    for name, values in groups.items():
        v = values.dropna().astype(float)
        if v.shape[0] >= 2:
            clean_groups[name] = v
            counts[name] = int(v.shape[0])

    if len(clean_groups) < 2:
        return len(clean_groups), float("nan"), float("nan"), counts

    stat, p_val = f_oneway(*[v.values for v in clean_groups.values()])
    return len(clean_groups), float(stat), float(p_val), counts


def section_run_level_by_project(
    runs_df: pd.DataFrame, metrics: List[str], alpha: float
) -> List[str]:
    lines: List[str] = []
    lines.append("Run-Level One-Way ANOVA Per Project")
    lines.append("Grouping factor: method")
    lines.append("Response variables: {0}".format(", ".join(metrics)))
    lines.append("")

    if runs_df.empty:
        lines.append("No runs.txt files found.")
        lines.append("")
        return lines

    for project in sorted(runs_df["project"].dropna().unique()):
        lines.append("Project: {0}".format(project))
        proj_df = runs_df[runs_df["project"] == project]
        methods = sorted(proj_df["method"].dropna().unique())
        for metric in metrics:
            if metric not in proj_df.columns:
                lines.append("  {0}: missing".format(metric))
                continue

            groups = {
                method: proj_df[proj_df["method"] == method][metric]
                for method in methods
            }
            k, f_stat, p_val, counts = _anova(groups)
            if k < 2:
                lines.append(
                    "  {0}: insufficient groups with >=2 samples".format(metric)
                )
                continue

            counts_text = ", ".join(
                ["{0}:{1}".format(name, counts[name]) for name in sorted(counts.keys())]
            )
            lines.append(
                "  {0}: k={1}, F={2:.6f}, p={3:.6f}, reject={4}, counts=({5})".format(
                    metric,
                    k,
                    f_stat,
                    p_val,
                    p_val <= alpha,
                    counts_text,
                )
            )
        lines.append("")

    return lines


def section_run_level_global(
    runs_df: pd.DataFrame, metrics: List[str], alpha: float
) -> List[str]:
    lines: List[str] = []
    lines.append("Run-Level One-Way ANOVA Across All Projects")
    lines.append("Grouping factor: method")
    lines.append("")

    if runs_df.empty:
        lines.append("No runs data found.")
        lines.append("")
        return lines

    methods = sorted(runs_df["method"].dropna().unique())
    for metric in metrics:
        if metric not in runs_df.columns:
            lines.append("  {0}: missing".format(metric))
            continue

        groups = {
            method: runs_df[runs_df["method"] == method][metric] for method in methods
        }
        k, f_stat, p_val, counts = _anova(groups)
        if k < 2:
            lines.append("  {0}: insufficient groups with >=2 samples".format(metric))
            continue

        counts_text = ", ".join(
            ["{0}:{1}".format(name, counts[name]) for name in sorted(counts.keys())]
        )
        lines.append(
            "  {0}: k={1}, F={2:.6f}, p={3:.6f}, reject={4}, counts=({5})".format(
                metric,
                k,
                f_stat,
                p_val,
                p_val <= alpha,
                counts_text,
            )
        )

    lines.append("")
    return lines


def section_ipr(summary_df: pd.DataFrame, alpha: float) -> List[str]:
    lines: List[str] = []
    lines.append("IPR One-Way ANOVA Across Projects")
    lines.append("Grouping factor: method")
    lines.append("Response variable: tim_ipr")
    lines.append("")

    if summary_df.empty or "tim_ipr" not in summary_df.columns:
        lines.append("No summary/tim_ipr data found.")
        lines.append("")
        return lines

    ipr_df = summary_df.dropna(subset=["method", "tim_ipr"])
    if ipr_df.empty:
        lines.append("No valid tim_ipr values found.")
        lines.append("")
        return lines

    methods = sorted(ipr_df["method"].unique())
    groups = {
        method: ipr_df[ipr_df["method"] == method]["tim_ipr"] for method in methods
    }
    k, f_stat, p_val, counts = _anova(groups)
    if k < 2:
        lines.append("tim_ipr: insufficient groups with >=2 samples")
        lines.append("")
        return lines

    counts_text = ", ".join(
        ["{0}:{1}".format(name, counts[name]) for name in sorted(counts.keys())]
    )
    lines.append(
        "tim_ipr: k={0}, F={1:.6f}, p={2:.6f}, reject={3}, counts=({4})".format(
            k,
            f_stat,
            p_val,
            p_val <= alpha,
            counts_text,
        )
    )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one-way ANOVA tests for IPR and model performance."
    )
    parser.add_argument("--results-root", default="results", help="Results root folder")
    parser.add_argument(
        "--out-file",
        default="anova_tests.txt",
        help="Output file name under results root",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated performance metrics",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    summary_df = load_summary(args.results_root)
    runs_df = load_runs(args.results_root)

    lines: List[str] = []
    lines.append("ANOVA Tests")
    lines.append("Results root: {0}".format(os.path.abspath(args.results_root)))
    lines.append("Metrics: {0}".format(", ".join(metrics)))
    lines.append("Alpha: {0}".format(args.alpha))
    lines.append("")

    lines.extend(section_run_level_by_project(runs_df, metrics, args.alpha))
    lines.extend(section_run_level_global(runs_df, metrics, args.alpha))
    lines.extend(section_ipr(summary_df, args.alpha))

    out_path = os.path.join(args.results_root, args.out_file)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")

    print("Wrote {0}".format(out_path))


if __name__ == "__main__":
    main()
