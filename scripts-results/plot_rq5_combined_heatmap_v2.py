#!/usr/bin/env python3
"""Plot the RQ5 combined heatmap from existing summary tables only."""

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMBINED_VALUES = os.path.join(
    ROOT, "results", "stats", "rq5_combined_heatmap_values.tsv"
)
GRID_SUMMARY = os.path.join(ROOT, "results", "stats", "rq5_grid_summary.tsv")
OUT_DIR = os.path.join(ROOT, "results", "visualizations", "rq5")

LABEL_FIXES: Dict[str, str] = {
    "GMM global": "GMM",
    "GMM+ global": "GMM+",
    "CLA Q=5": "CLA (Q=5)",
    "CLA Q=10": "CLA (Q=10)",
    "CLA Q=20": "CLA (Q=20)",
    "CLA Q=50": "CLA (Q=50)",
    "CLA+ Q=5": "CLA+ (Q=5)",
    "CLA+ Q=10": "CLA+ (Q=10)",
    "CLA+ Q=20": "CLA+ (Q=20)",
    "CLA+ Q=50": "CLA+ (Q=50)",
}

ROW_ORDER: List[str] = [
    "GMM",
    "GMM+",
    "CLA (Q=5)",
    "CLA (Q=10)",
    "CLA (Q=20)",
    "CLA (Q=50)",
    "CLA+ (Q=5)",
    "CLA+ (Q=10)",
    "CLA+ (Q=20)",
    "CLA+ (Q=50)",
]
K_ORDER = [1, 2, 3]


def load_combined_values() -> pd.DataFrame:
    if not os.path.isfile(COMBINED_VALUES):
        raise FileNotFoundError(COMBINED_VALUES)
    df = pd.read_csv(COMBINED_VALUES, sep="\t")
    df["gmm_components"] = pd.to_numeric(df["gmm_components"], errors="coerce").astype(
        "Int64"
    )
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["plot_label"] = df["configuration"].replace(LABEL_FIXES)
    return df


def auc_from_grid_summary(existing_values: pd.DataFrame) -> pd.DataFrame:
    if not os.path.isfile(GRID_SUMMARY):
        raise FileNotFoundError(GRID_SUMMARY)

    grid = pd.read_csv(GRID_SUMMARY, sep="\t")
    grid = grid[grid["selection"] == "fixed"].copy()
    grid["gmm_components"] = pd.to_numeric(
        grid["gmm_components"], errors="coerce"
    ).astype("Int64")
    grid["bins"] = pd.to_numeric(grid["bins"], errors="coerce")
    grid["auc"] = pd.to_numeric(grid["auc"], errors="coerce")

    config_meta = (
        existing_values[
            [
                "method",
                "method_display",
                "configuration",
                "plot_label",
                "bins",
                "gmm_components",
            ]
        ]
        .drop_duplicates()
        .copy()
    )
    config_meta["bins"] = pd.to_numeric(config_meta["bins"], errors="coerce")
    config_meta["gmm_components"] = pd.to_numeric(
        config_meta["gmm_components"], errors="coerce"
    ).astype("Int64")

    rows = []
    for _, meta in config_meta.iterrows():
        subset = grid[
            (grid["method"] == meta["method"])
            & (grid["gmm_components"] == meta["gmm_components"])
        ].copy()
        if pd.notna(meta["bins"]):
            subset = subset[subset["bins"] == meta["bins"]]

        values = subset["auc"].dropna()
        rows.append(
            {
                "metric": "auc",
                "metric_display": "AUC",
                "method": meta["method"],
                "method_display": meta["method_display"],
                "configuration": meta["configuration"],
                "plot_label": meta["plot_label"],
                "bins": meta["bins"],
                "gmm_components": int(meta["gmm_components"]),
                "mean": float(values.mean()) if not values.empty else float("nan"),
                "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "n_projects": int(values.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def build_plot_data() -> pd.DataFrame:
    combined = load_combined_values()
    if "auc" in set(combined["metric"]):
        auc_values = combined[combined["metric"] == "auc"].copy()
    else:
        auc_values = auc_from_grid_summary(combined)

    plot_df = pd.concat(
        [
            combined[combined["metric"].isin(["tim_ipr", "f1"])],
            auc_values,
        ],
        ignore_index=True,
    )
    plot_df["plot_label"] = plot_df["configuration"].replace(LABEL_FIXES)
    return plot_df


def plot_heatmap(plot_df: pd.DataFrame) -> None:
    cmap = "Greys"
    panels = [
        ("tim_ipr", "Mean IPR"),
        ("f1", "Mean F1"),
        ("auc", "Mean AUC"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.15), constrained_layout=False)
    images = []
    for panel_index, (metric, title) in enumerate(panels):
        ax = axes[panel_index]
        metric_df = plot_df[plot_df["metric"] == metric].copy()
        pivot = (
            metric_df.pivot_table(
                index="plot_label",
                columns="gmm_components",
                values="mean",
                aggfunc="first",
            )
            .reindex(index=ROW_ORDER, columns=K_ORDER)
            .astype(float)
        )

        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap)
        images.append(image)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xticks(range(len(K_ORDER)))
        ax.set_xticklabels([f"k={value}" for value in K_ORDER], fontsize=9)
        ax.set_yticks(range(len(ROW_ORDER)))
        ax.set_yticklabels(ROW_ORDER if panel_index == 0 else [], fontsize=9)
        ax.tick_params(axis="y", length=0)

        for separator in [1.5, 5.5]:
            ax.axhline(separator, color="#333333", linewidth=0.8)

        for row_index, row_label in enumerate(ROW_ORDER):
            for col_index, component in enumerate(K_ORDER):
                value = pivot.loc[row_label, component]
                label = "" if pd.isna(value) else f"{value:.3f}"
                if pd.isna(value):
                    text_color = "black"
                else:
                    text_color = (
                        "white" if image.norm(float(value)) >= 0.55 else "black"
                    )
                ax.text(
                    col_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=text_color,
                )

    if images:
        colorbar_ax = fig.add_axes([0.945, 0.19, 0.016, 0.58])
        colorbar = fig.colorbar(images[-1], cax=colorbar_ax)
        colorbar.ax.tick_params(labelsize=8)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.subplots_adjust(left=0.13, right=0.91, top=0.86, bottom=0.13, wspace=0.14)
    fig.savefig(
        os.path.join(OUT_DIR, "rq5_combined_heatmap_v2.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(OUT_DIR, "rq5_combined_heatmap_v2.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    plot_heatmap(build_plot_data())


if __name__ == "__main__":
    main()
