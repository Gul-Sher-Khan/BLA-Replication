#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Tuple

import pandas as pd

from rename_methods_and_visualize import EXCLUDED_METHODS

TECHNIQUE_RENAMES: Dict[str, str] = {
    "LACE1": "LACE",
    "LACE2": "LACE2",
    "do_lace1": "LACE",
    "do_lace2": "LACE2",
}

SUMMARY_COLUMNS = [
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

RUN_COLUMNS = ["project", "method", "run_id", "auc", "f1", "pre", "rec", "acc"]


def project_slug(raw_project: str) -> str:
    text = str(raw_project)
    if text.startswith("apache_"):
        return text[len("apache_") :]
    return text


def normalize_technique(raw_technique: str) -> str:
    text = str(raw_technique)
    if text not in TECHNIQUE_RENAMES:
        raise ValueError("Unexpected LACE technique: {0}".format(text))
    return TECHNIQUE_RENAMES[text]


def lace_project_file_map(lace_root: str) -> Dict[str, str]:
    project_files: Dict[str, str] = {}
    for entry in sorted(os.listdir(lace_root)):
        if not entry.endswith("_lace_results_long.csv"):
            continue
        full_path = os.path.join(lace_root, entry)
        if not os.path.isfile(full_path):
            continue
        project_name = entry[: -len("_lace_results_long.csv")]
        project_files[project_slug(project_name)] = full_path
    return project_files


def standardize_long_df(long_df: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "technique_id": "technique",
        "rf_results_acc": "acc",
        "rf_results_auc": "auc",
        "rf_results_f1": "f1",
        "rf_results_pre": "pre",
        "rf_results_rec": "rec",
        "privacy_my_ipr_metric": "my_ipr",
        "privacy_tim_ipr": "tim_ipr",
        "bootstrap_id": "run_id",
    }
    standardized = long_df.rename(columns=column_map).copy()
    required = [
        "project",
        "technique",
        "run_id",
        "tim_ipr",
        "my_ipr",
        "auc",
        "f1",
        "pre",
        "rec",
        "acc",
    ]
    missing = [column for column in required if column not in standardized.columns]
    if missing:
        raise ValueError(
            "Missing required LACE columns: {0}".format(", ".join(sorted(missing)))
        )
    return standardized


def read_existing_tsv(path: str, columns: List[str]) -> pd.DataFrame:
    if os.path.isfile(path):
        return pd.read_csv(path, sep="\t")
    return pd.DataFrame(columns=columns)


def upsert_methods(
    existing: pd.DataFrame, new_rows: pd.DataFrame, methods: List[str]
) -> pd.DataFrame:
    if existing.empty:
        base = pd.DataFrame(columns=new_rows.columns)
    else:
        base = existing[~existing["method"].isin(methods)].copy()
    return pd.concat([base, new_rows], ignore_index=True)


def build_summary_rows(
    long_df: pd.DataFrame, project: str, long_csv_path: str
) -> pd.DataFrame:
    rows = []
    source_long = long_df[long_df["project"].map(project_slug) == project]
    if source_long.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    source_long = source_long.copy()
    source_long["method"] = source_long["technique"].map(normalize_technique)
    grouped = source_long.groupby("method", sort=False)

    for method, method_rows in grouped:
        if method in EXCLUDED_METHODS:
            continue
        n_runs = int(method_rows["run_id"].nunique())
        rows.append(
            {
                "project": project,
                "method": method,
                "tim_ipr": method_rows["tim_ipr"].mean(),
                "my_ipr": method_rows["my_ipr"].mean(),
                "auc": method_rows["auc"].mean(),
                "f1": method_rows["f1"].mean(),
                "pre": method_rows["pre"].mean(),
                "rec": method_rows["rec"].mean(),
                "acc": method_rows["acc"].mean(),
                "n_runs": n_runs,
                "csv_path": long_csv_path,
            }
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_run_rows(long_df: pd.DataFrame, project: str) -> pd.DataFrame:
    source = long_df[long_df["project"].map(project_slug) == project].copy()
    if source.empty:
        return pd.DataFrame(columns=RUN_COLUMNS)

    source["method"] = source["technique"].map(normalize_technique)
    source = source[~source["method"].isin(EXCLUDED_METHODS)].copy()
    if source.empty:
        return pd.DataFrame(columns=RUN_COLUMNS)

    out = pd.DataFrame(
        {
            "project": project,
            "method": source["method"],
            "run_id": pd.to_numeric(source["run_id"], errors="raise").astype(int),
            "auc": source["auc"],
            "f1": source["f1"],
            "pre": source["pre"],
            "rec": source["rec"],
            "acc": source["acc"],
        }
    )
    return out.sort_values(["method", "run_id"]).reset_index(drop=True)


def import_lace_results(
    lace_root: str, results_root: str, dry_run: bool = False
) -> None:
    methods = list(TECHNIQUE_RENAMES.values())
    if not os.path.isdir(lace_root):
        raise FileNotFoundError("Missing LACE results folder: {0}".format(lace_root))

    project_files = lace_project_file_map(lace_root)
    if not project_files:
        raise FileNotFoundError(
            "No per-project LACE CSVs found under: {0}".format(lace_root)
        )

    for project, long_path in sorted(project_files.items()):
        project_dir = os.path.join(results_root, project)
        summary_out_path = os.path.join(project_dir, "summary.txt")
        runs_out_path = os.path.join(project_dir, "runs.txt")

        long_df = standardize_long_df(pd.read_csv(long_path))
        long_csv_path = os.path.abspath(long_path)

        summary_rows = build_summary_rows(long_df, project, long_csv_path)
        run_rows = build_run_rows(long_df, project)
        if summary_rows.empty or run_rows.empty:
            print("Skipping {0}: no LACE rows found".format(project))
            continue

        existing_summary = read_existing_tsv(summary_out_path, SUMMARY_COLUMNS)
        existing_runs = read_existing_tsv(runs_out_path, RUN_COLUMNS)
        updated_summary = upsert_methods(existing_summary, summary_rows, methods)
        updated_runs = upsert_methods(existing_runs, run_rows, methods)

        if not dry_run:
            os.makedirs(project_dir, exist_ok=True)
            updated_summary.to_csv(summary_out_path, sep="\t", index=False)
            updated_runs.to_csv(runs_out_path, sep="\t", index=False)

        print(
            "{0}: upserted {1} summary rows and {2} run rows".format(
                project, len(summary_rows), len(run_rows)
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import LACE/LACE2 CSV outputs into per-project result folders."
    )
    parser.add_argument(
        "--lace-root",
        default="lace-results",
        help="Folder containing per-project *_lace_results_long.csv files.",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root folder containing per-project result folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report rows that would be upserted without writing result files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import_lace_results(
        lace_root=os.path.abspath(args.lace_root),
        results_root=os.path.abspath(args.results_root),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
