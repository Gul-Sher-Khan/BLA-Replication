#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from typing import Dict, Iterable, List, Tuple

import pandas as pd

METHOD_RENAMES: Dict[str, str] = {
    "Original_Baseline": "Unanonymized",
    "cla_gmm_2": "CLA+",
    "cla_gmm_2_no_cluster": "GMM+",
    "gmm_joint_2_global": "GMM",
    "gmm_joint_2_cluster": "CLA",
    "gen": "Gen",
    "k_da_anon": "K-DA",
    "random_add_delete": "Random Add/Delete",
    "random_switch": "Random Switch",
    "LACE1": "LACE",
    "LACE2": "LACE2",
    "do_lace1": "LACE",
    "do_lace2": "LACE2",
}

EXCLUDED_METHODS: set[str] = set()

LEGACY_DISPLAY_RENAMES: Dict[str, str] = {
    "Cluster GMM (la/ld)": "CLA+",
    "GMM (la/ld)": "GMM+",
    "Cluster GMM": "CLA",
    "lace": "LACE",
    "lace2": "LACE2",
}

METHOD_ORDER: List[str] = [
    "Unanonymized",
    "CLA+",
    "GMM+",
    "CLA",
    "GMM",
    "LACE",
    "LACE2",
    "Gen",
    "K-DA",
    "Random Add/Delete",
    "Random Switch",
]

METHOD_PALETTE: Dict[str, str] = {
    "Unanonymized": "#6B7280",
    "CLA+": "#D97706",
    "GMM+": "#5EEAD4",
    "CLA": "#115E59",
    "GMM": "#0F766E",
    "LACE": "#F97316",
    "LACE2": "#C2410C",
    "Gen": "#2563EB",
    "K-DA": "#9333EA",
    "Random Add/Delete": "#DC2626",
    "Random Switch": "#0891B2",
}

DISPLAY_METHODS = set(METHOD_RENAMES.values())
RAW_METHODS = set(METHOD_RENAMES.keys())
METHOD_ALIASES: Dict[str, str] = {
    **METHOD_RENAMES,
    **LEGACY_DISPLAY_RENAMES,
    **{method: method for method in DISPLAY_METHODS},
}

TABULAR_METHOD_FILES = {
    "summary.txt",
    "runs.txt",
    "feature_importance_avg.txt",
    "feature_importance_runs.txt",
}


def validate_raw_methods(methods: Iterable[str]) -> None:
    observed = {str(method) for method in methods if pd.notna(method)}
    unexpected = sorted(observed - RAW_METHODS)
    if unexpected:
        raise ValueError(
            "Unexpected raw method codes found in results: {0}".format(
                ", ".join(unexpected)
            )
        )


def validate_display_methods(methods: Iterable[str]) -> None:
    observed = {str(method) for method in methods if pd.notna(method)}
    unexpected = sorted(observed - DISPLAY_METHODS)
    if unexpected:
        raise ValueError(
            "Unexpected display method names found in results: {0}".format(
                ", ".join(unexpected)
            )
        )


def rename_method_series(series: pd.Series) -> pd.Series:
    def _normalize(method):
        if pd.isna(method):
            return method
        text = str(method)
        return METHOD_ALIASES.get(text, text)

    normalized = series.map(_normalize)
    validate_display_methods(normalized.dropna().unique().tolist())
    return normalized


def filter_method_frame(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df.copy()
    for column in columns:
        if column not in filtered.columns:
            continue
        filtered = filtered[~filtered[column].isin(EXCLUDED_METHODS)]
    return filtered.reset_index(drop=True)


def _replace_in_text(content: str, mapping: Dict[str, str]) -> Tuple[str, int]:
    updated = content
    replacements = 0
    for old_name, new_name in sorted(
        mapping.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if old_name == new_name:
            continue
        hits = updated.count(old_name)
        if hits > 0:
            replacements += hits
            updated = updated.replace(old_name, new_name)
    return updated, replacements


def _rewrite_tabular_method_file(
    path: str, mapping: Dict[str, str]
) -> Tuple[bool, int]:
    df = pd.read_csv(path, sep="\t")
    if "method" not in df.columns:
        return False, 0

    before = df["method"].astype(str)
    after = rename_method_series(before)
    changed = int((before != after).sum())
    if changed <= 0:
        return False, 0

    df["method"] = after
    df.to_csv(path, sep="\t", index=False)
    return True, changed


def _rewrite_plain_text_file(path: str, mapping: Dict[str, str]) -> Tuple[bool, int]:
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()

    updated, replacements = _replace_in_text(content, mapping)
    if replacements <= 0:
        return False, 0

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(updated)
    return True, replacements


def rewrite_results_names(results_root: str, mapping: Dict[str, str]) -> Dict[str, int]:
    tabular_files_changed = 0
    tabular_value_changes = 0
    text_files_changed = 0
    text_replacements = 0

    for root, _, files in os.walk(results_root):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            lower_name = file_name.lower()

            if lower_name in TABULAR_METHOD_FILES:
                changed, count = _rewrite_tabular_method_file(full_path, mapping)
                if changed:
                    tabular_files_changed += 1
                    tabular_value_changes += count
                continue

            if not lower_name.endswith(".txt"):
                continue

            changed, count = _rewrite_plain_text_file(full_path, mapping)
            if changed:
                text_files_changed += 1
                text_replacements += count

    return {
        "tabular_files_changed": tabular_files_changed,
        "tabular_value_changes": tabular_value_changes,
        "text_files_changed": text_files_changed,
        "text_replacements": text_replacements,
    }


def rerun_visualization(results_root: str, out_dir: str | None) -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    visualize_script = os.path.join(this_dir, "visualize_results.py")

    cmd: List[str] = [sys.executable, visualize_script, "--results-root", results_root]
    if out_dir:
        cmd.extend(["--out-dir", out_dir])

    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename method names in results files and regenerate visualizations."
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root folder containing experiment results.",
    )
    parser.add_argument(
        "--viz-out-dir",
        default=None,
        help="Optional output directory for regenerated visualizations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = os.path.abspath(args.results_root)

    if not os.path.isdir(results_root):
        raise SystemExit("Results root not found: {0}".format(results_root))

    summary = rewrite_results_names(results_root, METHOD_ALIASES)
    print(
        "Renamed methods: tabular_files_changed={0}, tabular_value_changes={1}, "
        "text_files_changed={2}, text_replacements={3}".format(
            summary["tabular_files_changed"],
            summary["tabular_value_changes"],
            summary["text_files_changed"],
            summary["text_replacements"],
        )
    )

    rerun_visualization(results_root, args.viz_out_dir)


if __name__ == "__main__":
    main()
