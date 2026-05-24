#!/usr/bin/env python3
import argparse
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from rename_methods_and_visualize import (
    METHOD_ORDER,
    METHOD_PALETTE,
    filter_method_frame,
    rename_method_series,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter
except Exception as exc:
    raise SystemExit(
        "Missing matplotlib. Install with: pip install matplotlib\nError: {0}".format(
            exc
        )
    )

try:
    from adjustText import adjust_text
except Exception:
    adjust_text = None

try:
    from scipy.interpolate import PchipInterpolator
except Exception:
    PchipInterpolator = None


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"

UTILITY_METRICS = [
    ("auc", "AUC"),
    ("f1", "F1"),
    ("acc", "Accuracy"),
    ("pre", "Precision"),
    ("rec", "Recall"),
]
IPR_METRIC = ("tim_ipr", "IPR")
FEATURE_ORDER = [
    "la",
    "ld",
    "nf",
    "nd",
    "ns",
    "ent",
    "ndev",
    "nuc",
    "age",
    "aexp",
    "asexp",
    "arexp",
]
PARETO_AXIS_LIMITS = (0.0, 1.0)
PARETO_SCALE_EPSILON = 0.10
IPR_LEVELS = [
    (0.65, "IPR level 1"),
    (0.80, "IPR level 2"),
]
RANKING_METHODS = [
    "CLA+",
    "GMM+",
    "CLA",
    "GMM",
    "LACE",
    "LACE2",
    "K-DA",
    "Gen",
    "Random Add/Delete",
    "Random Switch",
]
RANKING_METHOD_LABELS = {
    "CLA+": "CLA+",
    "GMM+": "GMM+",
    "CLA": "CLA",
    "GMM": "GMM",
    "LACE": "LACE",
    "LACE2": "LACE2",
    "K-DA": "k-DA",
    "Gen": "Gen",
    "Random Add/Delete": "Random Add/Delete",
    "Random Switch": "Random Switch",
}
RANKING_POINT_LABELS = {
    "CLA+": "CLA+",
    "GMM+": "GMM+",
    "CLA": "CLA",
    "GMM": "GMM",
    "LACE": "LACE",
    "LACE2": "LACE2",
    "K-DA": "k-DA",
    "Gen": "Gen",
    "Random Add/Delete": "RAD",
    "Random Switch": "RS",
}
RANKING_PROJECT_ORDER = [
    ("cassandra", "Cassandra"),
    ("flink", "Flink"),
    ("groovy", "Groovy"),
    ("ignite", "Ignite"),
    ("openstack", "OpenStack"),
    ("qt", "Qt"),
]
RANKING_METHOD_COLORS = {
    "CLA+": "#B91C1C",
    "GMM+": "#1D4ED8",
    "CLA": "#047857",
    "GMM": "#7C3AED",
    "LACE": "#A16207",
    "LACE2": "#78716C",
    "K-DA": "#6B7280",
    "Gen": "#94A3B8",
    "Random Add/Delete": "#A8A29E",
    "Random Switch": "#7C8EA3",
}
RANKING_LINESTYLES = {
    "CLA+": "-",
    "GMM+": "-",
    "CLA": "-",
    "GMM": "-",
    "LACE": (0, (6, 2)),
    "LACE2": (0, (2, 2)),
    "K-DA": (0, (4, 1, 1, 1)),
    "Gen": (0, (8, 2)),
    "Random Add/Delete": (0, (5, 1, 1, 1)),
    "Random Switch": (0, (3, 2)),
}
RANKING_LINEWIDTHS = {
    "CLA+": 2.8,
    "GMM+": 2.8,
    "CLA": 2.8,
    "GMM": 2.8,
    "LACE": 2.1,
    "LACE2": 2.1,
    "K-DA": 2.1,
    "Gen": 2.1,
    "Random Add/Delete": 2.1,
    "Random Switch": 2.1,
}
RQ_PLOT_CONFIGS = {
    "RQ1": {
        "methods": [
            "Unanonymized",
            "GMM",
            "LACE",
            "LACE2",
            "Gen",
            "K-DA",
            "Random Add/Delete",
            "Random Switch",
        ],
        "title": "All-feature global GMM vs anonymization methods",
        "stats_file": "rq1_tests.tsv",
    },
    "RQ2": {
        "methods": ["Unanonymized", "GMM", "CLA"],
        "title": "All-feature global vs clustered",
        "stats_file": "rq2_tests.tsv",
    },
    "RQ3": {
        "methods": ["Unanonymized", "GMM+", "CLA+"],
        "title": "la/ld-only global vs clustered",
        "stats_file": "rq3_tests.tsv",
    },
    "RQ4": {
        "methods": [
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
        ],
        "title": "All methods overview",
        "stats_file": "rq4_tests.tsv",
    },
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


def load_feature_runs(results_root: str) -> pd.DataFrame:
    df = _load_tsvs(find_files(results_root, "feature_importance_runs.txt"))
    if df.empty:
        return df
    df = df.copy()
    df["method"] = rename_method_series(df["method"])
    df = filter_method_frame(df, "method")
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
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


def load_stats_table(results_root: str, rq: str) -> pd.DataFrame:
    path = os.path.join(results_root, "stats", RQ_PLOT_CONFIGS[rq]["stats_file"])
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    for column in ["method", "method_a", "method_b"]:
        if column in df.columns:
            df[column] = rename_method_series(df[column])
    df = filter_method_frame(df, "method", "method_a", "method_b")
    return df


def ordered_methods(methods: Iterable[str]) -> List[str]:
    present = set(methods)
    return [method for method in METHOD_ORDER if method in present]


def p_label(value: float) -> str:
    if pd.isna(value):
        return "p=n/a"
    if value < 0.001:
        return "p<0.001"
    if value < 0.01:
        return "p<0.01"
    if value < 0.05:
        return "p<0.05"
    return "p≥0.05"


def save_figure(fig: plt.Figure, out_base: str, size: Tuple[float, float]) -> None:
    fig.set_size_inches(*size)
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pdf_figure(fig: plt.Figure, out_path: str, size: Tuple[float, float]) -> None:
    fig.set_size_inches(*size)
    fig.tight_layout()
    out_base, _ = os.path.splitext(out_path)
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_bars(df: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    grouped = (
        df.groupby("method")["value"]
        .agg(
            median="median",
            q1=lambda values: float(np.quantile(values, 0.25)),
            q3=lambda values: float(np.quantile(values, 0.75)),
        )
        .reindex(methods)
    )
    return grouped


def build_project_rank_frame(summary_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    method_priority = {method: index for index, method in enumerate(RANKING_METHODS)}
    rows = []
    for project_slug, project_label in RANKING_PROJECT_ORDER:
        project_df = summary_df[
            (summary_df["project"] == project_slug)
            & (summary_df["method"].isin(RANKING_METHODS))
        ][["method", metric_col]].copy()
        if len(project_df) != len(RANKING_METHODS):
            raise ValueError(
                "Expected {0} ranking methods for {1} in {2}, found {3}".format(
                    len(RANKING_METHODS), project_label, metric_col, len(project_df)
                )
            )
        project_df[metric_col] = pd.to_numeric(project_df[metric_col], errors="coerce")
        if project_df[metric_col].isna().any():
            raise ValueError(
                "Missing ranking values for {0} in {1}".format(
                    project_label, metric_col
                )
            )
        project_df["method_order"] = project_df["method"].map(method_priority)
        project_df = project_df.sort_values(
            by=[metric_col, "method_order"], ascending=[False, True]
        ).reset_index(drop=True)
        project_df["rank"] = np.arange(1, len(project_df) + 1)
        project_df["project"] = project_slug
        project_df["project_label"] = project_label
        rows.append(project_df[["project", "project_label", "method", "rank"]])
    return pd.concat(rows, ignore_index=True)


def plot_rq4_rankings(summary_df: pd.DataFrame, results_root: str) -> None:
    workspace_root = os.path.dirname(os.path.abspath(results_root))
    figs_dir = os.path.join(workspace_root, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    jitter_values = np.linspace(-0.18, 0.18, len(RANKING_METHODS))
    method_jitter = {
        method: float(jitter_values[index])
        for index, method in enumerate(RANKING_METHODS)
    }
    x_positions = np.arange(len(RANKING_PROJECT_ORDER), dtype=float)
    project_index = {
        slug: index for index, (slug, _) in enumerate(RANKING_PROJECT_ORDER)
    }

    metric_specs = [
        ("auc", "AUC", "rq4_auc.pdf"),
        ("f1", "F1", "rq4_f1.pdf"),
        ("tim_ipr", "IPR", "rq4_ipr.pdf"),
    ]

    for metric_col, metric_label, filename in metric_specs:
        rank_df = build_project_rank_frame(summary_df, metric_col)
        fig, ax = plt.subplots()

        legend_handles = []
        for method in RANKING_METHODS:
            method_df = rank_df[rank_df["method"] == method].copy()
            method_df["x"] = method_df["project"].map(project_index).astype(float)
            method_df["rank_jittered"] = (
                method_df["rank"].astype(float) + method_jitter[method]
            )

            ax.plot(
                method_df["x"],
                method_df["rank_jittered"],
                color=RANKING_METHOD_COLORS[method],
                linestyle=RANKING_LINESTYLES[method],
                linewidth=RANKING_LINEWIDTHS[method],
                alpha=0.98,
                zorder=2,
            )

            for _, row in method_df.iterrows():
                ax.text(
                    float(row["x"]),
                    float(row["rank_jittered"]),
                    RANKING_POINT_LABELS[method],
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=RANKING_METHOD_COLORS[method],
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.72,
                        "edgecolor": "none",
                        "pad": 0.15,
                    },
                    zorder=3,
                )

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=RANKING_METHOD_COLORS[method],
                    linestyle=RANKING_LINESTYLES[method],
                    linewidth=RANKING_LINEWIDTHS[method],
                    label=RANKING_METHOD_LABELS[method],
                )
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [label for _, label in RANKING_PROJECT_ORDER],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel("Rank")
        ax.set_title("{0} Method Rankings by Project".format(metric_label))
        ax.set_ylim(10.5, 0.5)
        ax.set_yticks(np.arange(1, 11, 1))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.grid(True, axis="both", color="#D1D5DB", linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            borderaxespad=0.0,
        )

        save_pdf_figure(fig, os.path.join(figs_dir, filename), (9.6, 5.8))


def sort_methods_by_value(
    summary: pd.DataFrame, methods: Sequence[str], value_col: str
) -> List[str]:
    order_rank = {method: index for index, method in enumerate(methods)}
    sortable = summary.loc[[method for method in methods if method in summary.index]]
    sortable = sortable.copy()
    sortable["_order_rank"] = [order_rank[method] for method in sortable.index]
    sortable["_sort_value"] = pd.to_numeric(sortable[value_col], errors="coerce")
    sortable = sortable.sort_values(
        by=["_sort_value", "_order_rank"],
        ascending=[False, True],
        na_position="last",
    )
    return sortable.index.tolist()


def set_value_axis(
    ax, lows: Sequence[float], highs: Sequence[float], extra_top: float = 0.0
) -> None:
    finite_lows = np.asarray([value for value in lows if pd.notna(value)], dtype=float)
    finite_highs = np.asarray(
        [value for value in highs if pd.notna(value)], dtype=float
    )
    if finite_lows.size == 0 or finite_highs.size == 0:
        return
    lo = float(np.min(finite_lows))
    hi = float(np.max(finite_highs))
    span = hi - lo
    margin = max(span * 0.05, 0.01)
    ax.set_ylim(lo - margin, hi + margin + extra_top)


def format_value_label(value: float) -> str:
    return f"{value:.3f}"


def add_bar_value_labels(
    ax,
    x_positions: Sequence[float],
    values: Sequence[float],
    anchors: Sequence[float],
    fontsize: int = 8,
) -> None:
    ymin, ymax = ax.get_ylim()
    offset = max((ymax - ymin) * 0.012, 0.005)
    for xpos, value, anchor in zip(x_positions, values, anchors):
        if pd.isna(value) or pd.isna(anchor):
            continue
        ax.text(
            xpos,
            float(anchor) + offset,
            format_value_label(float(value)),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=False,
        )


def add_ipr_level_lines(ax, label_x: float = 0.985, fontsize: int = 8) -> None:
    for level, label in IPR_LEVELS:
        ax.axhline(
            level,
            color="#4B5563",
            linestyle=":",
            linewidth=1.0,
            alpha=0.75,
            zorder=1,
        )
        ax.text(
            label_x,
            level,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=fontsize,
            color="#374151",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.2},
        )


def include_ipr_levels_in_axis(ax, lower_floor: float | None = None) -> None:
    ymin, ymax = ax.get_ylim()
    level_values = [level for level, _ in IPR_LEVELS]
    if lower_floor is not None:
        ymin = min(ymin, lower_floor)
    ymin = min(ymin, min(level_values) - 0.03)
    ymax = max(ymax, max(level_values) + 0.03)
    ax.set_ylim(ymin, ymax)


def set_percent_axis(ax) -> None:
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def add_inside_bar_label(
    ax,
    x_position: float,
    value: float,
    fontsize: int = 7,
) -> None:
    if pd.isna(value):
        return
    y_position = max(float(value) - 0.035, 0.025)
    ax.text(
        x_position,
        y_position,
        f"{float(value):.2f}",
        ha="center",
        va="top",
        fontsize=fontsize,
        color="white",
        fontweight="bold",
        bbox={
            "facecolor": "black",
            "alpha": 0.24,
            "edgecolor": "none",
            "pad": 0.15,
        },
    )


def method_color(method: str, rq: str | None = None) -> str:
    if rq in {"RQ2", "RQ3"}:
        rq3_colors = {
            "GMM+": METHOD_PALETTE["GMM"],
            "CLA+": METHOD_PALETTE["Random Switch"],
            "GMM": METHOD_PALETTE["GMM"],
            "CLA": METHOD_PALETTE["Random Switch"],
        }
        return rq3_colors.get(method, METHOD_PALETTE[method])
    return METHOD_PALETTE[method]


def add_grid_significance_labels(
    ax, stats_df: pd.DataFrame, rq: str, metric: str
) -> None:
    if stats_df.empty:
        return
    if metric == "IPR":
        rows = stats_df[
            (stats_df["section"] == "omnibus") & (stats_df["metric"] == "IPR")
        ].copy()
        if rows.empty:
            return
        p_value = pd.to_numeric(rows.iloc[0].get("p_value"), errors="coerce")
        if pd.isna(p_value) or float(p_value) >= 0.05:
            return
        text = f"Sign test: {p_label(float(p_value))}"
    else:
        rows = stats_df[
            (stats_df["section"] == "utility") & (stats_df["metric"] == metric)
        ].copy()
        if rows.empty:
            return
        rows["p_holm"] = pd.to_numeric(rows["p_holm"], errors="coerce")
        rows = rows[rows["p_holm"] < 0.05]
        if rows.empty:
            return
        if rq == "RQ1":
            parts = []
            for _, row in rows.iterrows():
                other = row["method_b"]
                if row["method_a"] != "GMM":
                    other = row["method_a"]
                parts.append(f"{other}: {p_label(float(row['p_holm']))}")
            text = "Significant vs GMM\n" + "\n".join(parts)
        else:
            row = rows.iloc[0]
            text = (
                f"{row['method_a']} vs {row['method_b']}\n"
                f"Holm: {p_label(float(row['p_holm']))}"
            )
    ax.text(
        0.02,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        color="#111827",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 0.25},
    )


def add_point_value_label(
    ax,
    x_value: float,
    y_value: float,
    x_metric: str,
    y_metric: str,
    method: str,
) -> None:
    ax.annotate(
        f"{method} ({format_value_label(x_value)}, {format_value_label(y_value)})",
        (x_value, y_value),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.2},
    )


def build_pareto_summary(
    utility_subset: pd.DataFrame, ipr_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    present_methods = set(utility_subset["method"]).intersection(set(ipr_df["method"]))
    for method in ordered_methods(present_methods):
        utility_values = utility_subset[utility_subset["method"] == method][
            "value"
        ].to_numpy(dtype=float)
        ipr_values = ipr_df[ipr_df["method"] == method]["value"].to_numpy(dtype=float)
        if utility_values.size == 0 or ipr_values.size == 0:
            continue
        rows.append(
            {
                "method": method,
                "utility_median": float(np.median(utility_values)),
                "utility_q1": float(np.quantile(utility_values, 0.25)),
                "utility_q3": float(np.quantile(utility_values, 0.75)),
                "ipr_median": float(np.median(ipr_values)),
                "ipr_min": float(np.min(ipr_values)),
                "ipr_max": float(np.max(ipr_values)),
            }
        )
    points = pd.DataFrame(rows)

    baseline_rows = []
    baseline_values = utility_subset[utility_subset["method"] == "Unanonymized"][
        "value"
    ].to_numpy(dtype=float)
    if baseline_values.size:
        baseline_rows.append(
            {
                "method": "Unanonymized",
                "utility_median": float(np.median(baseline_values)),
                "utility_q1": float(np.quantile(baseline_values, 0.25)),
                "utility_q3": float(np.quantile(baseline_values, 0.75)),
                "ipr_median": 1.0,
                "ipr_min": 1.0,
                "ipr_max": 1.0,
            }
        )
    baseline = pd.DataFrame(baseline_rows)
    return points, baseline


PARETO_LABEL_OFFSETS: Dict[str, tuple[int, int]] = {
    "CLA+": (10, 28),
    "GMM+": (-12, 20),
    "CLA": (34, -20),
    "GMM": (-48, 10),
    "LACE": (10, -14),
    "LACE2": (12, -28),
    "Unanonymized": (-54, 12),
    "Gen": (-16, -20),
    "K-DA": (5, -16),
    "Random Add/Delete": (-40, -20),
    "Random Switch": (-54, -18),
}


def add_manual_pareto_labels(ax, label_df: pd.DataFrame) -> None:
    fallback_offsets = [(8, 12), (8, -12), (-12, 10), (-12, -10)]
    ordered = label_df.sort_values(
        by=["ipr_median", "utility_median"], ascending=[False, True]
    ).reset_index(drop=True)
    for index, row in ordered.iterrows():
        method = str(row["method"])
        dx, dy = PARETO_LABEL_OFFSETS.get(
            method, fallback_offsets[index % len(fallback_offsets)]
        )
        ax.annotate(
            method,
            (row["utility_median"], row["ipr_median"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5},
            bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none", "pad": 0.2},
        )


def add_pareto_labels(ax, label_df: pd.DataFrame) -> None:
    if label_df.empty:
        return
    add_manual_pareto_labels(ax, label_df)


def add_pareto_table(
    ax_table, utility_metric: str, points: pd.DataFrame, baseline: pd.DataFrame
) -> None:
    ax_table.axis("off")
    table_df = pd.concat([baseline, points], ignore_index=True)
    if table_df.empty:
        return
    table_df = table_df.copy()
    table_df[utility_metric] = table_df["utility_median"].map(format_value_label)
    table_df["IPR"] = table_df["ipr_median"].map(format_value_label)
    table_df = table_df[["method", utility_metric, "IPR"]]

    table = ax_table.table(
        cellText=table_df.values.tolist(),
        colLabels=["Method", utility_metric, "IPR"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.62, 0.19, 0.19],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    for column_index in range(3):
        table[(0, column_index)].set_text_props(weight="bold")
        table[(0, column_index)].set_facecolor("#E5E7EB")

    for row_index, method in enumerate(table_df["method"], start=1):
        if method == "Unanonymized":
            table[(row_index, 0)].set_facecolor("#F3F4F6")
        else:
            table[(row_index, 0)].set_facecolor(METHOD_PALETTE[method])
            table[(row_index, 0)].set_text_props(color="white")


def shade_dominated_region(ax, frontier: pd.DataFrame) -> None:
    if frontier.empty:
        return
    x_min, x_max = ax.get_xlim()
    y_min, _ = ax.get_ylim()
    ordered = frontier.sort_values(by="utility_median")
    x_vals = ordered["utility_median"].to_numpy(dtype=float)
    y_vals = ordered["ipr_median"].to_numpy(dtype=float)
    envelope = np.maximum.accumulate(y_vals[::-1])[::-1]

    step_x = np.concatenate(([x_min], x_vals))
    step_y = np.concatenate(([envelope[0]], envelope))
    ax.fill_between(step_x, step_y, y_min, step="post", color="#CBD5E1", alpha=0.18)
    ax.fill_between(
        [x_vals[-1], x_max],
        [envelope[-1], envelope[-1]],
        y_min,
        color="#CBD5E1",
        alpha=0.18,
    )


def bounded_high_end_log_forward(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    usable_span = 1.0 - PARETO_SCALE_EPSILON
    denominator = -np.log(PARETO_SCALE_EPSILON)
    return -np.log1p(-usable_span * clipped) / denominator


def bounded_high_end_log_inverse(values: np.ndarray) -> np.ndarray:
    scaled = np.asarray(values, dtype=float)
    usable_span = 1.0 - PARETO_SCALE_EPSILON
    denominator = -np.log(PARETO_SCALE_EPSILON)
    restored = (1.0 - np.exp(-scaled * denominator)) / usable_span
    return np.clip(restored, 0.0, 1.0)


def style_pareto_axes(ax, utility_metric: str) -> None:
    ax.set_xlim(*PARETO_AXIS_LIMITS)
    ax.set_ylim(*PARETO_AXIS_LIMITS)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlabel(utility_metric)
    ax.set_ylabel("IPR")
    ax.set_title(f"{utility_metric} vs IPR")
    ax.grid(alpha=0.24)


def annotate_pairwise_bars(
    ax, methods: Sequence[str], stats_df: pd.DataFrame, bar_summary: pd.DataFrame
) -> None:
    if stats_df.empty:
        return
    positions = {method: index for index, method in enumerate(methods)}
    highs = []
    for method in methods:
        if method in bar_summary.index and pd.notna(bar_summary.loc[method, "q3"]):
            highs.append(float(bar_summary.loc[method, "q3"]))
        elif method in bar_summary.index and pd.notna(
            bar_summary.loc[method, "median"]
        ):
            highs.append(float(bar_summary.loc[method, "median"]))
    if not highs:
        return
    ymax = max(highs)
    ymin = min(highs)
    step = max((ymax - ymin) * 0.12, 0.02)
    current = ymax + step
    for _, row in stats_df.iterrows():
        method_a = row["method_a"]
        method_b = row["method_b"]
        if method_a not in positions or method_b not in positions:
            continue
        x1 = positions[method_a]
        x2 = positions[method_b]
        ax.plot(
            [x1, x1, x2, x2],
            [current, current + step, current + step, current],
            color="#374151",
            linewidth=1.0,
        )
        ax.text(
            (x1 + x2) / 2.0,
            current + step * 1.05,
            p_label(float(row["p_holm"])),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        current += step * 2.0
    set_value_axis(
        ax,
        bar_summary["q1"].tolist(),
        bar_summary["q3"].tolist(),
        extra_top=current - ymax,
    )


def annotate_pairwise_summary(
    ax, stats_df: pd.DataFrame, metric: str, section: str
) -> None:
    if stats_df.empty:
        return
    comparisons = stats_df[
        (stats_df["section"] == section) & (stats_df["metric"] == metric)
    ].copy()
    if comparisons.empty:
        return
    rejected = (
        pd.to_numeric(comparisons["reject_holm"], errors="coerce")
        .fillna(False)
        .astype(bool)
    )
    ax.text(
        0.98,
        0.98,
        f"Wilcoxon + Holm: {int(rejected.sum())}/{len(comparisons)} sig.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )


def plot_rq_bar_charts(
    utility_df: pd.DataFrame,
    ipr_df: pd.DataFrame,
    results_root: str,
    out_dir: str,
) -> None:
    target_dir = os.path.join(out_dir, "rq_plots")
    os.makedirs(target_dir, exist_ok=True)

    for rq, config in RQ_PLOT_CONFIGS.items():
        stats_df = load_stats_table(results_root, rq)

        for _, metric in [*UTILITY_METRICS, IPR_METRIC]:
            is_ipr = metric == "IPR"
            data_source = ipr_df if is_ipr else utility_df
            subset = data_source[
                (data_source["metric"] == metric)
                & (data_source["method"].isin(config["methods"]))
            ].copy()
            if subset.empty:
                continue

            methods = config["methods"]
            summary = summarize_bars(subset, methods)
            methods = sort_methods_by_value(summary, methods, "median")
            summary = summary.reindex(methods)
            fig, ax = plt.subplots()
            x_positions = np.arange(len(methods))
            draw_lows: List[float] = []
            draw_highs: List[float] = []
            label_values: List[float] = []
            label_anchors: List[float] = []
            label_positions: List[float] = []
            zero_iqr = True

            for index, method in enumerate(methods):
                if method not in summary.index or pd.isna(
                    summary.loc[method, "median"]
                ):
                    continue
                median = float(summary.loc[method, "median"])
                q1 = float(summary.loc[method, "q1"])
                q3 = float(summary.loc[method, "q3"])
                yerr = None
                if not np.isclose(q1, q3):
                    zero_iqr = False
                    yerr = np.array([[median - q1], [q3 - median]])
                ax.bar(
                    x_positions[index],
                    median,
                    color=method_color(method, rq),
                    alpha=0.85,
                    yerr=yerr,
                    capsize=3 if yerr is not None else 0,
                )
                draw_lows.append(q1)
                draw_highs.append(q3)
                label_positions.append(float(x_positions[index]))
                label_values.append(median)
                label_anchors.append(max(median, q3))

            ax.set_xticks(x_positions)
            ax.set_xticklabels(methods, rotation=25, ha="right")
            ax.set_ylabel(metric)
            ax.set_title(f"{rq} - {config['title']} ({metric})")
            ax.grid(axis="y", alpha=0.2)

            if rq in {"RQ1", "RQ2", "RQ3"} and not is_ipr and not stats_df.empty:
                comparisons = stats_df[
                    (stats_df["section"] == "utility") & (stats_df["metric"] == metric)
                ]
                annotate_pairwise_bars(ax, methods, comparisons, summary)
            else:
                span = (
                    (max(draw_highs) - min(draw_lows))
                    if draw_lows and draw_highs
                    else 0.0
                )
                set_value_axis(
                    ax, draw_lows, draw_highs, extra_top=max(span * 0.05, 0.02)
                )
                if not stats_df.empty:
                    omnibus = stats_df[
                        (stats_df["section"] == "omnibus")
                        & (stats_df["metric"] == metric)
                    ]
                    if not omnibus.empty:
                        row = omnibus.iloc[0]
                        ax.text(
                            0.98,
                            0.98,
                            f"{row['test']}: {p_label(float(row['p_value']))}",
                            transform=ax.transAxes,
                            ha="right",
                            va="top",
                            fontsize=9,
                        )
                    elif rq == "RQ4":
                        annotate_pairwise_summary(
                            ax,
                            stats_df,
                            metric,
                            "ipr" if is_ipr else "utility",
                        )
            if is_ipr:
                include_ipr_levels_in_axis(ax)
                add_ipr_level_lines(ax)
            if label_positions:
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax + max((ymax - ymin) * 0.04, 0.02))
                add_bar_value_labels(ax, label_positions, label_values, label_anchors)
            if is_ipr and zero_iqr:
                ax.text(
                    0.02,
                    0.98,
                    "IPR IQR = 0 per project",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                )
            save_figure(
                fig,
                os.path.join(target_dir, f"{rq.lower()}_{metric.lower()}"),
                (6.0, 4.0),
            )


def plot_rq_metric_grid(
    utility_df: pd.DataFrame,
    ipr_df: pd.DataFrame,
    results_root: str,
    out_dir: str,
) -> None:
    target_dir = os.path.join(out_dir, "rq_plots")
    os.makedirs(target_dir, exist_ok=True)
    rqs = ["RQ1", "RQ2", "RQ3"]
    metrics = ["AUC", "F1", "IPR"]
    fig, axes = plt.subplots(len(rqs), len(metrics), squeeze=False, sharey=True)

    for row_index, rq in enumerate(rqs):
        methods = RQ_PLOT_CONFIGS[rq]["methods"]
        stats_df = load_stats_table(results_root, rq)
        for col_index, metric in enumerate(metrics):
            ax = axes[row_index][col_index]
            data_source = ipr_df if metric == "IPR" else utility_df
            subset = data_source[
                (data_source["metric"] == metric)
                & (data_source["method"].isin(methods))
            ].copy()
            summary = summarize_bars(subset, methods)
            plot_methods = [method for method in methods if method in summary.index]
            summary = summary.reindex(plot_methods)
            x_positions = np.arange(len(plot_methods))

            for index, method in enumerate(plot_methods):
                if pd.isna(summary.loc[method, "median"]):
                    continue
                median = float(summary.loc[method, "median"])
                q1 = float(summary.loc[method, "q1"])
                q3 = float(summary.loc[method, "q3"])
                yerr = None
                if not np.isclose(q1, q3):
                    yerr = np.array([[median - q1], [q3 - median]])
                ax.bar(
                    index,
                    median,
                    yerr=yerr,
                    color=method_color(method, rq),
                    alpha=0.88,
                    capsize=2.5 if yerr is not None else 0,
                )
                add_inside_bar_label(ax, float(index), median)

            set_percent_axis(ax)
            if metric == "IPR":
                add_ipr_level_lines(ax, fontsize=7)
            add_grid_significance_labels(ax, stats_df, rq, metric)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(plot_methods, rotation=32, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.18)
            if row_index == 0:
                ax.set_title(metric)
            if col_index == 0:
                ax.set_ylabel(f"{rq}\nScore")
            else:
                ax.tick_params(axis="y", labelleft=False)

    fig.suptitle("RQ1-RQ3 Utility and Privacy Scores", y=0.995)
    save_figure(
        fig,
        os.path.join(target_dir, "rq1_rq2_rq3_auc_f1_ipr_grid"),
        (11.5, 7.6),
    )


def non_dominated_points(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, candidate in df.iterrows():
        dominated = False
        for _, challenger in df.iterrows():
            if challenger["method"] == candidate["method"]:
                continue
            if (
                challenger["utility_median"] >= candidate["utility_median"]
                and challenger["ipr_median"] >= candidate["ipr_median"]
                and (
                    challenger["utility_median"] > candidate["utility_median"]
                    or challenger["ipr_median"] > candidate["ipr_median"]
                )
            ):
                dominated = True
                break
        if not dominated:
            rows.append(candidate)
    out = pd.DataFrame(rows)
    return (
        out.sort_values(by=["utility_median", "ipr_median"]) if not out.empty else out
    )


def pareto_curve_coordinates(frontier: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = frontier.sort_values(by=["utility_median", "ipr_median"]).reset_index(
        drop=True
    )
    if ordered.empty:
        return np.asarray([]), np.asarray([])

    ordered = (
        ordered.groupby("utility_median", as_index=False)["ipr_median"]
        .max()
        .sort_values(by="utility_median")
    )
    x_vals = ordered["utility_median"].to_numpy(dtype=float)
    y_vals = ordered["ipr_median"].to_numpy(dtype=float)
    if len(x_vals) < 2:
        return x_vals, y_vals

    if len(x_vals) == 2:
        t = np.linspace(0.0, 1.0, 80)
        control_x = (x_vals[0] + x_vals[1]) / 2.0
        control_y = (y_vals[0] + y_vals[1]) / 2.0 + 0.03
        curve_x = (
            (1 - t) ** 2 * x_vals[0] + 2 * (1 - t) * t * control_x + t**2 * x_vals[1]
        )
        curve_y = (
            (1 - t) ** 2 * y_vals[0] + 2 * (1 - t) * t * control_y + t**2 * y_vals[1]
        )
        return curve_x, np.clip(curve_y, 0.0, 1.0)

    curve_x = np.linspace(float(x_vals[0]), float(x_vals[-1]), 160)
    if PchipInterpolator is None:
        curve_y = np.interp(curve_x, x_vals, y_vals)
    else:
        curve_y = PchipInterpolator(x_vals, y_vals)(curve_x)
    return curve_x, np.clip(curve_y, 0.0, 1.0)


def plot_pareto_scatter(
    utility_df: pd.DataFrame, ipr_df: pd.DataFrame, out_dir: str
) -> None:
    target_dir = os.path.join(out_dir, "pareto")
    os.makedirs(target_dir, exist_ok=True)
    for _, utility_metric in [("f1", "F1"), ("auc", "AUC")]:
        utility_subset = utility_df[utility_df["metric"] == utility_metric]
        points, baseline = build_pareto_summary(utility_subset, ipr_df)
        if points.empty and baseline.empty:
            continue

        variants: List[tuple[pd.DataFrame, str, str]] = [
            (
                pd.concat([points, baseline], ignore_index=True),
                f"{utility_metric.lower()}_ipr_scatter",
                f"{utility_metric} vs IPR",
            ),
        ]
        if not baseline.empty:
            baseline_at_zero = baseline.copy()
            baseline_at_zero[["ipr_median", "ipr_min", "ipr_max"]] = 0.0
            variants.append(
                (
                    pd.concat([points, baseline_at_zero], ignore_index=True),
                    f"{utility_metric.lower()}_ipr_scatter_with_unanonymized_ipr0",
                    f"{utility_metric} vs IPR",
                )
            )

        for plot_points, file_stem, plot_title in variants:
            if plot_points.empty:
                continue
            fig, ax = plt.subplots()
            style_pareto_axes(ax, utility_metric)
            ax.set_title(plot_title)

            legend_handles = []
            for _, row in plot_points.iterrows():
                method = row["method"]
                ax.scatter(
                    row["utility_median"],
                    row["ipr_median"],
                    s=86,
                    color=METHOD_PALETTE[method],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        label=method,
                        markerfacecolor=METHOD_PALETTE[method],
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                        markersize=7,
                    )
                )

            frontier = non_dominated_points(plot_points)
            if len(frontier) >= 2:
                curve_x, curve_y = pareto_curve_coordinates(frontier)
                ax.plot(
                    curve_x,
                    curve_y,
                    linestyle="-",
                    color="#374151",
                    linewidth=1.8,
                    zorder=2,
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="-",
                        color="#374151",
                        linewidth=1.8,
                        label="Pareto frontier",
                    )
                )

            x_values = plot_points["utility_median"].to_numpy(dtype=float)
            y_values = plot_points["ipr_median"].to_numpy(dtype=float)
            x_margin = max(
                (float(np.max(x_values)) - float(np.min(x_values))) * 0.18, 0.025
            )
            y_margin = max(
                (float(np.max(y_values)) - float(np.min(y_values))) * 0.14, 0.035
            )
            ax.set_xlim(
                max(0.0, float(np.min(x_values)) - x_margin),
                min(1.0, float(np.max(x_values)) + x_margin),
            )
            ax.set_ylim(
                max(0.0, min(float(np.min(y_values)) - y_margin, 0.60)),
                min(1.0, max(float(np.max(y_values)) + y_margin, 0.84)),
            )
            add_ipr_level_lines(ax)
            add_pareto_labels(ax, plot_points)
            ax.legend(
                handles=legend_handles,
                loc="lower left",
                bbox_to_anchor=(0.01, 0.34),
                ncol=2,
                fontsize=7,
                frameon=True,
                framealpha=0.82,
                borderpad=0.5,
                handlelength=1.6,
                columnspacing=0.9,
            )
            save_figure(
                fig,
                os.path.join(target_dir, file_stem),
                (6.6, 4.8),
            )


def plot_rq4_ablation(
    utility_df: pd.DataFrame, ipr_df: pd.DataFrame, out_dir: str
) -> None:
    target_dir = os.path.join(out_dir, "rq4")
    os.makedirs(target_dir, exist_ok=True)
    metrics = ["AUC", "F1", "IPR"]
    projects = sorted(
        set(utility_df["project"].dropna().astype(str)).union(
            set(ipr_df["project"].dropna().astype(str))
        )
    )
    if not projects:
        return
    methods = RQ_PLOT_CONFIGS["RQ4"]["methods"]
    fig, axes = plt.subplots(len(projects), len(metrics), squeeze=False)
    for row_index, project in enumerate(projects):
        for col_index, metric in enumerate(metrics):
            ax = axes[row_index][col_index]
            if metric == "IPR":
                subset = ipr_df[
                    (ipr_df["project"] == project) & (ipr_df["method"].isin(methods))
                ].copy()
                agg = (
                    subset.groupby("method")["value"]
                    .agg(mean="mean", std="std")
                    .reindex(methods)
                )
            else:
                subset = utility_df[
                    (utility_df["project"] == project)
                    & (utility_df["metric"] == metric)
                    & (utility_df["method"].isin(methods))
                ].copy()
                agg = (
                    subset.groupby("method")["value"]
                    .agg(mean="mean", std="std")
                    .reindex(methods)
                )
            plot_methods = sort_methods_by_value(agg, methods, "mean")
            agg = agg.reindex(plot_methods)
            positions = np.arange(len(plot_methods))
            lows = []
            highs = []
            label_values: List[float] = []
            label_anchors: List[float] = []
            label_positions: List[float] = []
            for index, method in enumerate(plot_methods):
                if method not in agg.index or pd.isna(agg.loc[method, "mean"]):
                    continue
                mean = float(agg.loc[method, "mean"])
                std = (
                    float(agg.loc[method, "std"])
                    if pd.notna(agg.loc[method, "std"])
                    else 0.0
                )
                ax.bar(
                    index,
                    mean,
                    yerr=std if std > 0 else None,
                    color=METHOD_PALETTE[method],
                    alpha=0.85,
                    capsize=3,
                )
                lows.append(mean - std)
                highs.append(mean + std)
                label_positions.append(float(index))
                label_values.append(mean)
                label_anchors.append(mean + std)
            ax.set_xticks(positions)
            ax.set_xticklabels(plot_methods, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{project} - {metric}")
            ax.grid(axis="y", alpha=0.2)
            span = (max(highs) - min(lows)) if lows and highs else 0.0
            set_value_axis(ax, lows, highs, extra_top=max(span * 0.05, 0.02))
            if metric == "IPR":
                include_ipr_levels_in_axis(ax)
                add_ipr_level_lines(ax, fontsize=7)
            if label_positions:
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax + max((ymax - ymin) * 0.04, 0.02))
                add_bar_value_labels(
                    ax, label_positions, label_values, label_anchors, fontsize=7
                )
    save_figure(
        fig,
        os.path.join(target_dir, "project_ablation_grid"),
        (14.0, max(8.0, 2.6 * len(projects))),
    )


def plot_feature_importance_heatmap(feature_runs: pd.DataFrame, out_dir: str) -> None:
    if feature_runs.empty:
        return
    target_dir = os.path.join(out_dir, "discussion")
    os.makedirs(target_dir, exist_ok=True)
    summary = (
        feature_runs.groupby(["feature", "method"])["importance"]
        .median()
        .unstack("method")
        .reindex(index=FEATURE_ORDER)
        .reindex(columns=ordered_methods(feature_runs["method"].unique()))
    )
    if summary.empty:
        return
    baseline = (
        summary["Unanonymized"]
        if "Unanonymized" in summary.columns
        else pd.Series(0.0, index=summary.index)
    )
    delta = summary.sub(baseline, axis=0)
    delta_values = delta.to_numpy(dtype=float)
    finite = delta_values[np.isfinite(delta_values)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots()
    image = ax.imshow(
        delta_values,
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
    )
    ax.set_xticks(range(len(summary.columns)))
    ax.set_xticklabels(summary.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(summary.index)))
    ax.set_yticklabels(summary.index)
    ax.set_title("Feature Importance Relative to Unanonymized")
    for row_index in range(summary.shape[0]):
        for col_index in range(summary.shape[1]):
            value = summary.iat[row_index, col_index]
            if pd.isna(value):
                continue
            ax.text(
                col_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Delta vs Unanonymized")
    save_figure(fig, os.path.join(target_dir, "feature_importance_heatmap"), (7.0, 4.5))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RQ-specific visualizations for anonymization experiments."
    )
    parser.add_argument(
        "--results-root", default="results", help="Results root folder."
    )
    parser.add_argument("--out-dir", default=None, help="Output directory for figures.")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_root, "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    summary_df = load_summary(args.results_root)
    runs_df = load_runs(args.results_root)
    feature_runs = load_feature_runs(args.results_root)
    if summary_df.empty or runs_df.empty:
        raise ValueError(
            "Expected both summary.txt and runs.txt under the results root"
        )

    utility_df = build_utility_long(runs_df)
    ipr_df = build_ipr_frame(summary_df)

    plot_rq_bar_charts(utility_df, ipr_df, args.results_root, out_dir)
    plot_rq_metric_grid(utility_df, ipr_df, args.results_root, out_dir)
    plot_pareto_scatter(utility_df, ipr_df, out_dir)
    plot_rq4_ablation(utility_df, ipr_df, out_dir)
    plot_feature_importance_heatmap(feature_runs, out_dir)
    plot_rq4_rankings(summary_df, args.results_root)

    print("Wrote visualizations to {0}".format(os.path.abspath(out_dir)))


if __name__ == "__main__":
    main()
