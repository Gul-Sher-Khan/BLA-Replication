#!/usr/bin/env python3
"""Run the RQ5 sensitivity analysis without rerunning the full study.

This script sweeps fixed GMM component counts k for all GMM-based methods and
quantile bins Q for the clustered methods only.
"""

import argparse
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import numpy as np
    from scipy.stats import friedmanchisquare, rankdata, wilcoxon
except Exception as exc:
    raise SystemExit(
        "Missing statistical dependencies. Install with: pip install numpy scipy\n"
        "Error: {0}".format(exc)
    )

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit(
        "Missing matplotlib. Install with: pip install matplotlib\nError: {0}".format(
            exc
        )
    )

from compare_privacy_and_utility import (
    METRICS_SETS,
    PrivacyMeasurer,
    generate_stratified_splits,
    run_eval_for_variant,
)

DEFAULT_PROJECTS = ["cassandra", "flink", "groovy", "ignite", "openstack", "qt"]
DEFAULT_Q_VALUES = [5, 10, 20, 50]
DEFAULT_GMM_COMPONENTS = [1, 2, 3]
DEFAULT_PRIVACY_CONFIGURATION = {1: 100, 2: 1000, 4: 1000}
PROJECT_LABELS = {
    "cassandra": "Cassandra",
    "flink": "Flink",
    "groovy": "Groovy",
    "ignite": "Ignite",
    "openstack": "OpenStack",
    "qt": "Qt",
}

for _thread_env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_var, "1")


@dataclass(frozen=True)
class MethodSpec:
    raw_name: str
    display_name: str
    script_key: str
    method_arg: str
    clustered: bool


METHOD_SPECS: Dict[str, MethodSpec] = {
    "gmm_joint_2_global": MethodSpec(
        raw_name="gmm_joint_2_global",
        display_name="GMM",
        script_key="joint",
        method_arg="gmm_joint_2_global",
        clustered=False,
    ),
    "cla_gmm_2_no_cluster": MethodSpec(
        raw_name="cla_gmm_2_no_cluster",
        display_name="GMM+",
        script_key="stats",
        method_arg="gmm_2_no_cluster",
        clustered=False,
    ),
    "gmm_joint_2_cluster": MethodSpec(
        raw_name="gmm_joint_2_cluster",
        display_name="CLA",
        script_key="joint",
        method_arg="gmm_joint_2_cluster",
        clustered=True,
    ),
    "cla_gmm_2": MethodSpec(
        raw_name="cla_gmm_2",
        display_name="CLA+",
        script_key="stats",
        method_arg="gmm_2",
        clustered=True,
    ),
}
DEFAULT_METHODS = list(METHOD_SPECS.keys())
RQ5_TEST_METRICS = [
    ("tim_ipr", "IPR"),
    ("auc", "AUC"),
    ("f1", "F1"),
    ("acc", "Accuracy"),
    ("pre", "Precision"),
    ("rec", "Recall"),
]


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path_value: Optional[str], fallback_relative: str) -> str:
    root = _repo_root()
    if not path_value:
        return os.path.abspath(os.path.join(root, fallback_relative))
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)
    cwd_candidate = os.path.abspath(path_value)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(root, path_value))


def _resolve_output_path(path_value: Optional[str], default_path: str) -> str:
    if not path_value:
        return os.path.abspath(default_path)
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)
    cwd_candidate = os.path.abspath(path_value)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(_repo_root(), path_value))


def _parse_int_list(raw_values: Sequence[str], label: str) -> List[int]:
    values = [int(value) for value in raw_values]
    if not values:
        raise SystemExit(f"{label} must not be empty")
    return values


def _resolve_max_workers(raw_value: int) -> int:
    if raw_value and raw_value > 0:
        return raw_value
    return max(1, min(os.cpu_count() or 1, 4))


def _single_thread_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RQ5 multi-method sensitivity sweep for selected projects."
    )
    parser.add_argument(
        "--projects-root",
        default=None,
        help="Root folder containing per-project input.csv files (default: projects).",
    )
    parser.add_argument(
        "--anon-root",
        default=None,
        help="Root folder for anonymized CSV outputs (default: anon_results).",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Root folder containing the existing main experiment results (default: results).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Legacy single-folder output directory for all RQ5 artifacts.",
    )
    parser.add_argument(
        "--stats-dir",
        default=None,
        help="Directory for RQ5 tables/text outputs (default: results/stats).",
    )
    parser.add_argument(
        "--viz-dir",
        default=None,
        help="Directory for RQ5 figures (default: results/visualizations/rq5).",
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        default=DEFAULT_PROJECTS,
        help="Projects to evaluate (default: all study projects).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=DEFAULT_METHODS,
        help="Methods to evaluate (default: GMM GMM+ CLA CLA+ raw codes).",
    )
    parser.add_argument(
        "--q-values",
        nargs="+",
        default=[str(value) for value in DEFAULT_Q_VALUES],
        help="Quantile bin counts for clustered methods (default: 5 10 20 50).",
    )
    parser.add_argument(
        "--gmm-components",
        nargs="+",
        default=[str(value) for value in DEFAULT_GMM_COMPONENTS],
        help="Fixed GMM component counts to sweep (default: 1 2 3).",
    )
    parser.add_argument(
        "--rf-runs",
        type=int,
        default=30,
        help="Number of stratified RF runs per configuration (default: 30).",
    )
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=1,
        help="Parallel jobs inside each RandomForest fit (default: 1).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Parallel worker count for RQ5 configurations (default: auto, capped at 4).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction for the RF evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for RF split generation (default: 42).",
    )
    parser.add_argument(
        "--metrics-set",
        choices=sorted(METRICS_SETS.keys()),
        default="all",
        help="Feature set for the downstream RF evaluation (default: all).",
    )
    parser.add_argument(
        "--force-anon",
        action="store_true",
        help="Regenerate parameterized anonymized CSVs even if they already exist.",
    )
    parser.add_argument(
        "--skip-existing-eval",
        action="store_true",
        help="Skip a configuration if it is already present in the saved RQ5 summary.",
    )
    parser.add_argument(
        "--no-default-reference",
        action="store_true",
        help="Do not load the existing main-pipeline method result as a BIC reference row.",
    )
    return parser.parse_args()


def _project_input_csv(projects_root: str, project: str) -> str:
    return os.path.join(projects_root, project, "input.csv")


def _q_label(spec: MethodSpec, q_value: Optional[int]) -> str:
    if spec.clustered and q_value is not None:
        return f"Q={q_value}"
    return "global"


def _variant_name(spec: MethodSpec, q_value: Optional[int], gmm_components: int) -> str:
    if spec.clustered and q_value is not None:
        return f"{spec.raw_name}_q{q_value}_k{gmm_components}"
    return f"{spec.raw_name}_k{gmm_components}"


def _variant_output_csv(
    anon_root: str,
    project: str,
    spec: MethodSpec,
    q_value: Optional[int],
    gmm_components: int,
) -> str:
    file_name = f"{_variant_name(spec, q_value, gmm_components)}.csv"
    return os.path.join(anon_root, "rq5_sensitivity", project, spec.raw_name, file_name)


def _rq5_summary_path(stats_dir: str) -> str:
    return os.path.join(stats_dir, "rq5_grid_summary.tsv")


def _load_existing_table(path: str, required_columns: Sequence[str]) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    if any(column not in df.columns for column in required_columns):
        return pd.DataFrame()
    return df


def _load_existing_rq5_summary(stats_dir: str) -> pd.DataFrame:
    return _load_existing_table(
        _rq5_summary_path(stats_dir),
        ["project", "method", "selection", "q_label", "gmm_components"],
    )


def _load_existing_rq5_runs(stats_dir: str) -> pd.DataFrame:
    return _load_existing_table(
        os.path.join(stats_dir, "rq5_grid_runs.tsv"),
        ["project", "method", "selection", "q_label", "gmm_components", "run_id"],
    )


def _load_existing_default_reference(
    results_root: str, project: str, spec: MethodSpec
) -> Optional[Dict[str, object]]:
    summary_path = os.path.join(results_root, project, "summary.txt")
    if not os.path.isfile(summary_path):
        return None

    df = pd.read_csv(summary_path, sep="\t")
    if "method" not in df.columns:
        return None

    match = df[df["method"] == spec.display_name]
    if match.empty:
        return None

    row = match.iloc[0].to_dict()
    q_value = 20 if spec.clustered else None
    return {
        "project": project,
        "method": spec.raw_name,
        "method_display": spec.display_name,
        "selection": "bic",
        "q_label": _q_label(spec, q_value),
        "bins": q_value,
        "gmm_components": "bic",
        "tim_ipr": row.get("tim_ipr"),
        "my_ipr": row.get("my_ipr"),
        "auc": row.get("auc"),
        "f1": row.get("f1"),
        "pre": row.get("pre"),
        "rec": row.get("rec"),
        "acc": row.get("acc"),
        "n_runs": row.get("n_runs"),
        "csv_path": row.get("csv_path"),
        "source": "existing_results",
    }


def _iter_configurations(
    spec: MethodSpec, q_values: Sequence[int], gmm_components: Sequence[int]
) -> List[Tuple[Optional[int], int]]:
    if spec.clustered:
        return [
            (q_value, component) for q_value in q_values for component in gmm_components
        ]
    return [(None, component) for component in gmm_components]


def _run_anonymizer(
    script_paths: Dict[str, str],
    spec: MethodSpec,
    input_csv: str,
    output_csv: str,
    q_value: Optional[int],
    gmm_components: int,
    force_anon: bool,
) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if os.path.isfile(output_csv) and not force_anon:
        print("      Reuse anonymized CSV: {0}".format(output_csv), flush=True)
        return

    start = time.time()
    cmd = [
        sys.executable,
        script_paths[spec.script_key],
        "--input-csv",
        input_csv,
        "--output-csv",
        output_csv,
        "--method",
        spec.method_arg,
        "--gmm-components",
        str(gmm_components),
    ]
    if spec.clustered and q_value is not None:
        cmd.extend(["--bins", str(q_value)])
    subprocess.run(cmd, check=True, env=_single_thread_env())
    print(
        "      Anonymized {0} in {1:.1f}s".format(
            os.path.basename(output_csv), time.time() - start
        ),
        flush=True,
    )


def _evaluate_configuration(
    script_paths: Dict[str, str],
    spec: MethodSpec,
    project: str,
    input_csv: str,
    anon_root: str,
    q_value: Optional[int],
    component_count: int,
    force_anon: bool,
    metrics: Sequence[str],
    splits,
    privacy_measurer: PrivacyMeasurer,
    rf_n_jobs: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    output_csv = _variant_output_csv(anon_root, project, spec, q_value, component_count)
    config_start = time.time()
    _run_anonymizer(
        script_paths,
        spec,
        input_csv,
        output_csv,
        q_value,
        component_count,
        force_anon,
    )
    eval_start = time.time()
    result = run_eval_for_variant(
        _variant_name(spec, q_value, component_count),
        output_csv,
        input_csv,
        metrics,
        splits=splits,
        privacy_measurer=privacy_measurer,
        rf_n_jobs=rf_n_jobs,
    )
    print(
        "      Evaluated {0} in {1:.1f}s; config total {2:.1f}s".format(
            os.path.basename(output_csv),
            time.time() - eval_start,
            time.time() - config_start,
        ),
        flush=True,
    )
    return (
        _build_summary_row(project, spec, q_value, component_count, result),
        _build_runs_rows(project, spec, q_value, component_count, result),
    )


def _should_skip_configuration(
    existing_summary: pd.DataFrame,
    project: str,
    method: str,
    q_label: str,
    gmm_components: int,
    skip_existing_eval: bool,
) -> bool:
    if not skip_existing_eval or existing_summary.empty:
        return False
    mask = (
        (existing_summary["project"] == project)
        & (existing_summary["method"] == method)
        & (existing_summary["selection"] == "fixed")
        & (existing_summary["q_label"] == q_label)
        & (existing_summary["gmm_components"] == gmm_components)
    )
    return bool(mask.any())


def _build_summary_row(
    project: str,
    spec: MethodSpec,
    q_value: Optional[int],
    gmm_components: int,
    result: Dict[str, object],
) -> Dict[str, object]:
    privacy = result["privacy"]
    rf_avg = result["rf_avg"]
    return {
        "project": project,
        "method": spec.raw_name,
        "method_display": spec.display_name,
        "selection": "fixed",
        "q_label": _q_label(spec, q_value),
        "bins": q_value,
        "gmm_components": gmm_components,
        "tim_ipr": privacy.get("tim_ipr"),
        "my_ipr": privacy.get("my_ipr_metric"),
        "auc": rf_avg.get("auc"),
        "f1": rf_avg.get("f1"),
        "pre": rf_avg.get("pre"),
        "rec": rf_avg.get("rec"),
        "acc": rf_avg.get("acc"),
        "n_runs": result.get("n_runs"),
        "csv_path": result.get("anon_csv"),
    }


def _build_runs_rows(
    project: str,
    spec: MethodSpec,
    q_value: Optional[int],
    gmm_components: int,
    result: Dict[str, object],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    q_label = _q_label(spec, q_value)
    for run in result["rf_runs"]:
        metrics = run["results"]
        rows.append(
            {
                "project": project,
                "method": spec.raw_name,
                "method_display": spec.display_name,
                "selection": "fixed",
                "q_label": q_label,
                "bins": q_value,
                "gmm_components": gmm_components,
                "run_id": run.get("run_id", run.get("bootstrap_id")),
                "auc": metrics.get("auc"),
                "f1": metrics.get("f1"),
                "pre": metrics.get("pre"),
                "rec": metrics.get("rec"),
                "acc": metrics.get("acc"),
            }
        )
    return rows


def _write_tsv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _merge_summary_rows(
    existing_df: pd.DataFrame, new_rows: Sequence[Dict[str, object]]
) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    if existing_df.empty:
        return new_df
    if new_df.empty:
        return existing_df.copy()
    merged = pd.concat([existing_df, new_df], ignore_index=True)
    return merged.drop_duplicates(
        subset=["project", "method", "selection", "q_label", "gmm_components"],
        keep="last",
    ).reset_index(drop=True)


def _merge_runs_rows(
    existing_df: pd.DataFrame, new_rows: Sequence[Dict[str, object]]
) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    if existing_df.empty:
        return new_df
    if new_df.empty:
        return existing_df.copy()
    merged = pd.concat([existing_df, new_df], ignore_index=True)
    return merged.drop_duplicates(
        subset=[
            "project",
            "method",
            "selection",
            "q_label",
            "gmm_components",
            "run_id",
        ],
        keep="last",
    ).reset_index(drop=True)


def _metric_bounds(df: pd.DataFrame, metric: str) -> Tuple[float, float]:
    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    if values.empty:
        return (0.0, 1.0)
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-9:
        pad = 0.01 if vmax == 0 else abs(vmax) * 0.01
        return (vmin - pad, vmax + pad)
    return (vmin, vmax)


def _rq5_fixed_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df[summary_df["selection"] == "fixed"].copy()
    if df.empty:
        return df
    for column in [
        "bins",
        "gmm_components",
        "tim_ipr",
        "my_ipr",
        "auc",
        "f1",
        "pre",
        "rec",
        "acc",
        "n_runs",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _combined_heatmap_rows(
    methods: Sequence[str], q_values: Sequence[int]
) -> List[Tuple[str, Optional[int], str]]:
    rows: List[Tuple[str, Optional[int], str]] = []
    for method in methods:
        spec = METHOD_SPECS[method]
        if spec.clustered:
            for q_value in q_values:
                rows.append((method, q_value, f"{spec.display_name} Q={q_value}"))
        else:
            rows.append((method, None, f"{spec.display_name} global"))
    return rows


def _combined_heatmap_values(
    summary_df: pd.DataFrame,
    methods: Sequence[str],
    q_values: Sequence[int],
    gmm_components: Sequence[int],
) -> pd.DataFrame:
    fixed_df = _rq5_fixed_summary(summary_df)
    rows: List[Dict[str, object]] = []
    for method, q_value, label in _combined_heatmap_rows(methods, q_values):
        spec = METHOD_SPECS[method]
        subset = fixed_df[fixed_df["method"] == spec.raw_name].copy()
        if spec.clustered:
            subset = subset[subset["bins"] == q_value]
        for metric, metric_label in [("tim_ipr", "IPR"), ("f1", "F1")]:
            for component in gmm_components:
                values = pd.to_numeric(
                    subset[subset["gmm_components"] == component][metric],
                    errors="coerce",
                ).dropna()
                rows.append(
                    {
                        "metric": metric,
                        "metric_display": metric_label,
                        "method": spec.raw_name,
                        "method_display": spec.display_name,
                        "configuration": label,
                        "bins": q_value,
                        "gmm_components": component,
                        "mean": float(values.mean()) if not values.empty else np.nan,
                        "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                        "n_projects": int(values.shape[0]),
                    }
                )
    return pd.DataFrame(rows)


def _remove_stale_method_heatmaps(viz_dir: str, methods: Sequence[str]) -> None:
    for method in methods:
        for extension in (".png", ".pdf"):
            path = os.path.join(viz_dir, f"rq5_{method}_heatmaps{extension}")
            if os.path.isfile(path):
                os.remove(path)


def _plot_combined_heatmap(
    summary_df: pd.DataFrame,
    methods: Sequence[str],
    q_values: Sequence[int],
    gmm_components: Sequence[int],
    viz_dir: str,
) -> None:
    heatmap_df = _combined_heatmap_values(summary_df, methods, q_values, gmm_components)
    if heatmap_df.empty:
        return

    row_labels = [
        row_label for _, _, row_label in _combined_heatmap_rows(methods, q_values)
    ]
    metrics = [("tim_ipr", "IPR", "Greys"), ("f1", "F1", "Greys")]
    fig, axes = plt.subplots(1, len(metrics), squeeze=False)
    fig.set_size_inches(8.2, max(3.15, 0.24 * len(row_labels) + 0.82))
    images = []

    for metric_index, (metric, metric_label, cmap_name) in enumerate(metrics):
        ax = axes[0][metric_index]
        metric_df = heatmap_df[heatmap_df["metric"] == metric]
        pivot = (
            metric_df.pivot(
                index="configuration", columns="gmm_components", values="mean"
            )
            .reindex(index=row_labels, columns=gmm_components)
            .astype(float)
        )
        image = ax.imshow(
            pivot.to_numpy(dtype=float),
            aspect="auto",
            cmap=cmap_name,
            vmin=_metric_bounds(metric_df.rename(columns={"mean": metric}), metric)[0],
            vmax=_metric_bounds(metric_df.rename(columns={"mean": metric}), metric)[1],
        )
        images.append(image)
        ax.set_xticks(range(len(gmm_components)))
        ax.set_xticklabels([str(value) for value in gmm_components])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels if metric_index == 0 else [])
        ax.set_xlabel("Fixed GMM components (k)")
        ax.set_title(f"Mean {metric_label} across projects", fontsize=12)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10, left=(metric_index == 0))

        for row_index, row_label in enumerate(row_labels):
            for col_index, component_value in enumerate(gmm_components):
                cell_value = pivot.loc[row_label, component_value]
                label = "n/a" if pd.isna(cell_value) else f"{float(cell_value):.3f}"
                if pd.isna(cell_value):
                    text_color = "black"
                else:
                    text_color = (
                        "white" if image.norm(float(cell_value)) >= 0.55 else "black"
                    )
                ax.text(
                    col_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    if images:
        colorbar_ax = fig.add_axes([0.942, 0.20, 0.016, 0.54])
        colorbar = fig.colorbar(images[-1], cax=colorbar_ax)
        colorbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "RQ5 fixed-parameter sensitivity, aggregated across projects",
        y=0.98,
        fontsize=14,
    )
    fig.subplots_adjust(left=0.15, right=0.91, top=0.84, bottom=0.14, wspace=0.16)
    os.makedirs(viz_dir, exist_ok=True)
    out_base = os.path.join(viz_dir, "rq5_combined_heatmap")
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmaps(
    summary_df: pd.DataFrame,
    projects: Sequence[str],
    methods: Sequence[str],
    q_values: Sequence[int],
    gmm_components: Sequence[int],
    viz_dir: str,
) -> None:
    del projects
    _remove_stale_method_heatmaps(viz_dir, methods)
    _plot_combined_heatmap(summary_df, methods, q_values, gmm_components, viz_dir)


def _holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, Dict[str, object]]:
    if not pvals:
        return {}
    ordered = sorted(pvals.items(), key=lambda item: item[1])
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    total = len(ordered)
    for index, (key, p_value) in enumerate(ordered):
        scaled = (total - index) * float(p_value)
        running_max = max(running_max, scaled)
        adjusted[key] = min(running_max, 1.0)
    return {
        key: {
            "p_raw": float(pvals[key]),
            "p_holm": float(adjusted[key]),
            "reject_holm": bool(adjusted[key] <= 0.05),
        }
        for key in pvals
    }


def _cliffs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.size == 0 or right_arr.size == 0:
        return float("nan")
    diffs = np.sign(left_arr[:, None] - right_arr[None, :])
    return float(diffs.sum() / (left_arr.size * right_arr.size))


def _wilcoxon_result(
    left: Sequence[float], right: Sequence[float]
) -> Tuple[float, float]:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.size == 0 or right_arr.size == 0:
        return float("nan"), float("nan")
    if np.allclose(left_arr, right_arr):
        return 0.0, 1.0
    try:
        result = wilcoxon(left_arr, right_arr, alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return 0.0, 1.0


def _condition_matrix(
    df: pd.DataFrame,
    condition_col: str,
    conditions: Sequence[int],
    value_col: str,
    index_cols: Sequence[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(conditions))
    matrix = (
        df.pivot_table(
            index=list(index_cols),
            columns=condition_col,
            values=value_col,
            aggfunc="mean",
        )
        .reindex(columns=list(conditions))
        .dropna()
    )
    return matrix


def _rank_rows(
    matrix: pd.DataFrame,
    factor: str,
    method_label: str,
    metric_label: str,
    condition_prefix: str,
) -> List[Dict[str, object]]:
    if matrix.empty:
        return []
    ranks = []
    for _, row in matrix.iterrows():
        ranks.append(rankdata(-row.to_numpy(dtype=float), method="average"))
    rank_frame = pd.DataFrame(ranks, columns=matrix.columns)
    rows: List[Dict[str, object]] = []
    for condition in matrix.columns:
        rows.append(
            {
                "factor": factor,
                "method": method_label,
                "metric": metric_label,
                "condition": f"{condition_prefix}={int(condition)}",
                "mean_rank": float(rank_frame[condition].mean()),
                "n_blocks": int(matrix.shape[0]),
            }
        )
    return rows


def _omnibus_row(
    matrix: pd.DataFrame,
    factor: str,
    method_label: str,
    metric_label: str,
) -> Dict[str, object]:
    row = {
        "factor": factor,
        "method": method_label,
        "metric": metric_label,
        "test": "Friedman",
        "n_conditions": int(matrix.shape[1]) if not matrix.empty else 0,
        "n_blocks": int(matrix.shape[0]) if not matrix.empty else 0,
        "statistic": float("nan"),
        "df": int(matrix.shape[1] - 1) if not matrix.empty else 0,
        "p_value": float("nan"),
    }
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        row["test"] = "Unavailable"
        return row
    try:
        statistic, p_value = friedmanchisquare(
            *[matrix[column].to_numpy(dtype=float) for column in matrix.columns]
        )
        row["statistic"] = float(statistic)
        row["p_value"] = float(p_value)
    except ValueError:
        row["statistic"] = 0.0
        row["p_value"] = 1.0
    return row


def _pairwise_rows(
    matrix: pd.DataFrame,
    factor: str,
    method_label: str,
    metric_label: str,
    condition_prefix: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    pvals: Dict[str, float] = {}
    for left_condition, right_condition in combinations(matrix.columns, 2):
        left = matrix[left_condition].to_numpy(dtype=float)
        right = matrix[right_condition].to_numpy(dtype=float)
        statistic, p_value = _wilcoxon_result(left, right)
        key = f"{left_condition}::{right_condition}"
        pvals[key] = p_value
        rows.append(
            {
                "factor": factor,
                "method": method_label,
                "metric": metric_label,
                "condition_a": f"{condition_prefix}={int(left_condition)}",
                "condition_b": f"{condition_prefix}={int(right_condition)}",
                "n": int(matrix.shape[0]),
                "median_a": float(np.median(left)),
                "median_b": float(np.median(right)),
                "median_diff": float(np.median(left - right)),
                "cliffs_delta": _cliffs_delta(left, right),
                "test": "Wilcoxon signed-rank",
                "statistic": statistic,
                "pair_key": key,
            }
        )

    adjusted = _holm_bonferroni(pvals)
    for row in rows:
        correction = adjusted[row.pop("pair_key")]
        row["p_raw"] = correction["p_raw"]
        row["p_holm"] = correction["p_holm"]
        row["reject_holm"] = correction["reject_holm"]
    return rows


def _write_rq5_statistical_tests(
    summary_df: pd.DataFrame,
    methods: Sequence[str],
    q_values: Sequence[int],
    gmm_components: Sequence[int],
    stats_dir: str,
) -> None:
    fixed_df = _rq5_fixed_summary(summary_df)
    omnibus_rows: List[Dict[str, object]] = []
    pairwise_rows: List[Dict[str, object]] = []
    rank_rows: List[Dict[str, object]] = []

    for method in methods:
        spec = METHOD_SPECS[method]
        method_df = fixed_df[fixed_df["method"] == spec.raw_name].copy()
        if method_df.empty:
            continue
        k_index_cols = ["project", "q_label"] if spec.clustered else ["project"]
        for metric, metric_label in RQ5_TEST_METRICS:
            matrix = _condition_matrix(
                method_df,
                "gmm_components",
                gmm_components,
                metric,
                k_index_cols,
            )
            omnibus_rows.append(
                _omnibus_row(matrix, "gmm_components", spec.display_name, metric_label)
            )
            pairwise_rows.extend(
                _pairwise_rows(
                    matrix,
                    "gmm_components",
                    spec.display_name,
                    metric_label,
                    "k",
                )
            )
            rank_rows.extend(
                _rank_rows(
                    matrix,
                    "gmm_components",
                    spec.display_name,
                    metric_label,
                    "k",
                )
            )

        if not spec.clustered:
            continue

        for metric, metric_label in RQ5_TEST_METRICS:
            matrix = _condition_matrix(
                method_df,
                "bins",
                q_values,
                metric,
                ["project", "gmm_components"],
            )
            omnibus_rows.append(
                _omnibus_row(matrix, "quantile_bins", spec.display_name, metric_label)
            )
            pairwise_rows.extend(
                _pairwise_rows(
                    matrix,
                    "quantile_bins",
                    spec.display_name,
                    metric_label,
                    "Q",
                )
            )
            rank_rows.extend(
                _rank_rows(
                    matrix,
                    "quantile_bins",
                    spec.display_name,
                    metric_label,
                    "Q",
                )
            )

    _write_tsv(
        pd.DataFrame(omnibus_rows),
        os.path.join(stats_dir, "rq5_tests.tsv"),
    )
    _write_tsv(
        pd.DataFrame(pairwise_rows),
        os.path.join(stats_dir, "rq5_pairwise_tests.tsv"),
    )
    _write_tsv(
        pd.DataFrame(rank_rows),
        os.path.join(stats_dir, "rq5_ranks.tsv"),
    )


def _project_method_summary_lines(
    project: str,
    spec: MethodSpec,
    summary_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> List[str]:
    lines: List[str] = []
    subset = summary_df[
        (summary_df["project"] == project)
        & (summary_df["method"] == spec.raw_name)
        & (summary_df["selection"] == "fixed")
    ].copy()
    if subset.empty:
        lines.append(
            "{0} {1}: no fixed-k results found.".format(
                PROJECT_LABELS.get(project, project.title()), spec.display_name
            )
        )
        return lines

    lines.append(
        "{0} {1}: fixed-k F1 spans {2:.3f}-{3:.3f}; IPR spans {4:.3f}-{5:.3f}.".format(
            PROJECT_LABELS.get(project, project.title()),
            spec.display_name,
            float(subset["f1"].min()),
            float(subset["f1"].max()),
            float(subset["tim_ipr"].min()),
            float(subset["tim_ipr"].max()),
        )
    )

    best_row = subset.sort_values(["f1", "tim_ipr"], ascending=[False, False]).iloc[0]
    if spec.clustered:
        lines.append(
            "{0} {1}: best fixed setting is Q={2}, k={3}, F1={4:.3f}, IPR={5:.3f}.".format(
                PROJECT_LABELS.get(project, project.title()),
                spec.display_name,
                int(best_row["bins"]),
                int(best_row["gmm_components"]),
                float(best_row["f1"]),
                float(best_row["tim_ipr"]),
            )
        )
    else:
        lines.append(
            "{0} {1}: best fixed setting is global, k={2}, F1={3:.3f}, IPR={4:.3f}.".format(
                PROJECT_LABELS.get(project, project.title()),
                spec.display_name,
                int(best_row["gmm_components"]),
                float(best_row["f1"]),
                float(best_row["tim_ipr"]),
            )
        )

    if reference_df.empty:
        ref_subset = pd.DataFrame()
    else:
        ref_subset = reference_df[
            (reference_df["project"] == project)
            & (reference_df["method"] == spec.raw_name)
        ].copy()
    if not ref_subset.empty:
        ref_row = ref_subset.iloc[0]
        lines.append(
            "{0} {1}: existing BIC reference reports F1={2:.3f}, IPR={3:.3f}.".format(
                PROJECT_LABELS.get(project, project.title()),
                spec.display_name,
                float(ref_row["f1"]),
                float(ref_row["tim_ipr"]),
            )
        )

    if spec.clustered:
        q20_subset = subset[subset["bins"] == 20].copy()
        if not q20_subset.empty:
            q20_best = q20_subset.sort_values(
                ["f1", "tim_ipr"], ascending=[False, False]
            ).iloc[0]
            lines.append(
                "{0} {1}: at Q=20, the best fixed-k setting is k={2} with F1={3:.3f} and IPR={4:.3f}.".format(
                    PROJECT_LABELS.get(project, project.title()),
                    spec.display_name,
                    int(q20_best["gmm_components"]),
                    float(q20_best["f1"]),
                    float(q20_best["tim_ipr"]),
                )
            )
    return lines


def _write_text_summary(
    summary_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    projects: Sequence[str],
    methods: Sequence[str],
    q_values: Sequence[int],
    gmm_components: Sequence[int],
    stats_dir: str,
    rf_runs: int,
    metrics_set: str,
) -> None:
    method_labels = [METHOD_SPECS[method].display_name for method in methods]
    lines = [
        "RQ5 multi-method sensitivity analysis",
        "",
        "Projects: {0}".format(
            ", ".join(
                PROJECT_LABELS.get(project, project.title()) for project in projects
            )
        ),
        "Methods: {0}".format(", ".join(method_labels)),
        "Q sweep for clustered methods: {0}".format(
            ", ".join(str(value) for value in q_values)
        ),
        "k sweep for all GMM-based methods: {0}".format(
            ", ".join(str(value) for value in gmm_components)
        ),
        f"RF runs per configuration: {rf_runs}",
        f"Metrics set: {metrics_set}",
        "",
    ]
    for project in projects:
        for method in methods:
            lines.extend(
                _project_method_summary_lines(
                    project,
                    METHOD_SPECS[method],
                    summary_df,
                    reference_df,
                )
            )
        lines.append("")

    os.makedirs(stats_dir, exist_ok=True)
    path = os.path.join(stats_dir, "rq5_summary.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def _build_configuration_space(
    projects: Sequence[str],
    methods: Sequence[str],
    q_values: Sequence[int],
    gmm_components: Sequence[int],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for project in projects:
        for method in methods:
            spec = METHOD_SPECS[method]
            for q_value, component in _iter_configurations(
                spec, q_values, gmm_components
            ):
                rows.append(
                    {
                        "project": project,
                        "method": spec.raw_name,
                        "method_display": spec.display_name,
                        "q_label": _q_label(spec, q_value),
                        "bins": q_value,
                        "gmm_components": component,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    q_values = _parse_int_list(args.q_values, "q-values")
    gmm_components = _parse_int_list(args.gmm_components, "gmm-components")
    max_workers = _resolve_max_workers(args.max_workers)

    projects_root = _resolve_path(args.projects_root, "projects")
    anon_root = _resolve_path(args.anon_root, "anon_results")
    results_root = _resolve_path(args.results_root, "results")
    script_paths = {
        "stats": os.path.join(_repo_root(), "scripts-anon", "anonymizer-stats.py"),
        "joint": os.path.join(_repo_root(), "scripts-anon", "anonymizer_gmm_joint.py"),
    }

    if args.out_dir:
        artifact_root = _resolve_output_path(
            args.out_dir, os.path.join(results_root, "rq5_sensitivity")
        )
        stats_dir = artifact_root
        viz_dir = artifact_root
    else:
        stats_dir = _resolve_output_path(
            args.stats_dir, os.path.join(results_root, "stats")
        )
        viz_dir = _resolve_output_path(
            args.viz_dir, os.path.join(results_root, "visualizations", "rq5")
        )

    for script_path in script_paths.values():
        if not os.path.isfile(script_path):
            raise SystemExit(f"Anonymizer not found: {script_path}")
    if not os.path.isdir(projects_root):
        raise SystemExit(f"Projects root not found: {projects_root}")

    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    existing_summary = _load_existing_rq5_summary(stats_dir)
    existing_runs = _load_existing_rq5_runs(stats_dir)
    summary_df = existing_summary.copy()
    runs_df = existing_runs.copy()
    summary_rows: List[Dict[str, object]] = []
    runs_rows: List[Dict[str, object]] = []
    reference_rows: List[Dict[str, object]] = []
    metrics = METRICS_SETS[args.metrics_set]
    scheduled_tasks: List[Dict[str, object]] = []

    for project in args.projects:
        input_csv = _project_input_csv(projects_root, project)
        if not os.path.isfile(input_csv):
            raise SystemExit(f"Missing input.csv for project '{project}': {input_csv}")

        print("==> Project {0}".format(PROJECT_LABELS.get(project, project.title())))
        baseline_df = pd.read_csv(input_csv).dropna(subset=["commit_id"])
        splits = generate_stratified_splits(
            baseline_df,
            n_runs=args.rf_runs,
            test_size=args.test_size,
            seed=args.seed,
        )
        privacy_measurer = PrivacyMeasurer(
            ["nf", "nd", "ns", "ent", "ndev", "nuc", "age", "aexp", "asexp", "arexp"],
            ["la", "ld"],
            input_csv,
            DEFAULT_PRIVACY_CONFIGURATION,
        )

        if not args.no_default_reference:
            for method in args.methods:
                reference = _load_existing_default_reference(
                    results_root, project, METHOD_SPECS[method]
                )
                if reference is not None:
                    reference_rows.append(reference)

        for method in args.methods:
            spec = METHOD_SPECS[method]
            print("  Method {0}".format(spec.display_name))
            for q_value, component_count in _iter_configurations(
                spec, q_values, gmm_components
            ):
                q_label = _q_label(spec, q_value)
                if _should_skip_configuration(
                    existing_summary,
                    project,
                    spec.raw_name,
                    q_label,
                    component_count,
                    args.skip_existing_eval,
                ):
                    print(
                        "    Skip {0}, k={1}: already present in saved RQ5 summary".format(
                            q_label,
                            component_count,
                        )
                    )
                    continue

                print("    Queue {0}, k={1}".format(q_label, component_count))
                scheduled_tasks.append(
                    {
                        "project": project,
                        "spec": spec,
                        "input_csv": input_csv,
                        "q_value": q_value,
                        "component_count": component_count,
                        "splits": splits,
                        "privacy_measurer": privacy_measurer,
                    }
                )

    if scheduled_tasks:
        print(
            "==> Running {0} RQ5 configurations with up to {1} parallel workers and rf_n_jobs={2}".format(
                len(scheduled_tasks),
                max_workers,
                args.rf_n_jobs,
            )
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for task in scheduled_tasks:
                future = executor.submit(
                    _evaluate_configuration,
                    script_paths,
                    task["spec"],
                    task["project"],
                    task["input_csv"],
                    anon_root,
                    task["q_value"],
                    task["component_count"],
                    args.force_anon,
                    metrics,
                    task["splits"],
                    task["privacy_measurer"],
                    args.rf_n_jobs,
                )
                future_map[future] = task

            completed = 0
            total = len(future_map)
            for future in as_completed(future_map):
                task = future_map[future]
                spec = task["spec"]
                q_label = _q_label(spec, task["q_value"])
                try:
                    summary_row, task_runs = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        "RQ5 failed for project={0}, method={1}, {2}, k={3}: {4}".format(
                            task["project"],
                            spec.display_name,
                            q_label,
                            task["component_count"],
                            exc,
                        )
                    ) from exc
                summary_rows.append(summary_row)
                runs_rows.extend(task_runs)
                summary_df = _merge_summary_rows(summary_df, [summary_row])
                runs_df = _merge_runs_rows(runs_df, task_runs)
                _write_tsv(summary_df, _rq5_summary_path(stats_dir))
                _write_tsv(runs_df, os.path.join(stats_dir, "rq5_grid_runs.tsv"))
                completed += 1
                print(
                    "Completed {0}/{1}: {2} {3} {4}, k={5} (checkpoint saved)".format(
                        completed,
                        total,
                        PROJECT_LABELS.get(task["project"], task["project"].title()),
                        spec.display_name,
                        q_label,
                        task["component_count"],
                    )
                )

    reference_df = pd.DataFrame(reference_rows)
    if not reference_df.empty:
        reference_df = reference_df.drop_duplicates(
            subset=["project", "method", "selection"], keep="last"
        )

    _write_tsv(summary_df, _rq5_summary_path(stats_dir))
    _write_tsv(runs_df, os.path.join(stats_dir, "rq5_grid_runs.tsv"))
    if not reference_df.empty:
        _write_tsv(reference_df, os.path.join(stats_dir, "rq5_default_reference.tsv"))
    _write_tsv(
        _build_configuration_space(
            args.projects, args.methods, q_values, gmm_components
        ),
        os.path.join(stats_dir, "rq5_configuration_space.tsv"),
    )

    if not summary_df.empty:
        _write_tsv(
            _combined_heatmap_values(
                summary_df, args.methods, q_values, gmm_components
            ),
            os.path.join(stats_dir, "rq5_combined_heatmap_values.tsv"),
        )
        _write_rq5_statistical_tests(
            summary_df, args.methods, q_values, gmm_components, stats_dir
        )
        _plot_heatmaps(
            summary_df, args.projects, args.methods, q_values, gmm_components, viz_dir
        )
        _write_text_summary(
            summary_df,
            reference_df,
            args.projects,
            args.methods,
            q_values,
            gmm_components,
            stats_dir,
            args.rf_runs,
            args.metrics_set,
        )

    print(
        "RQ5 artifacts written to stats={0} and visualizations={1}".format(
            stats_dir,
            viz_dir,
        )
    )


if __name__ == "__main__":
    main()
