#!/usr/bin/env python3
"""
DPP vs FOG Scatter Plot: Diversity vs Quality Trade-off

Key question: Do diverse designs maintain quality?
- DPP (higher = more diverse)
- FOG (lower = better quality)
"""

import matplotlib.pyplot as plt
from scipy import stats

from utils import (
    MIN_CORRELATION_SAMPLES,
    PLOT_STYLE,
    get_combined_global_df,
    load_data,
    save_figure,
    setup_style,
)


def plot_dpp_vs_fog(combined_df, output_path=None):
    """Create DPP vs FOG scatter plot (NeurIPS format)."""
    setup_style()
    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col"], constrained_layout=True
    )

    markers = PLOT_STYLE["markers"]
    colors = PLOT_STYLE["colors"]
    font_sizes = PLOT_STYLE["font_sizes"]

    for model in combined_df["model"].unique():
        for problem in combined_df["problem"].unique():
            mask = (combined_df["model"] == model) & (combined_df["problem"] == problem)
            subset = combined_df[mask]
            if len(subset) > 0:
                ax.scatter(
                    subset["dpp"],
                    subset["fog"],
                    label=f"{model} ({problem})",
                    marker=markers.get(model, "o"),
                    c=colors.get(problem, "gray"),
                    alpha=PLOT_STYLE["alpha"],
                    s=PLOT_STYLE["marker_size"],
                    edgecolors="white",
                    linewidth=0.3,
                )

    ax.set_xlabel("DPP Diversity")
    ax.set_ylabel("Final Optimality Gap")

    # Correlation annotation
    valid_data = combined_df.dropna(subset=["dpp", "fog"])
    if len(valid_data) > MIN_CORRELATION_SAMPLES:
        r, _ = stats.pearsonr(valid_data["dpp"], valid_data["fog"])
        ax.annotate(
            f"$r$ = {r:.2f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=font_sizes["annotation"],
            verticalalignment="top",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "0.7",
            },
        )

    ax.legend(loc="upper right", fontsize=font_sizes["legend"])
    ax.grid(True, alpha=0.3)

    if output_path:
        save_figure(fig, output_path)

    return fig


def main():
    """Generate DPP vs FOG plot."""
    print("Loading data...")
    data = load_data()
    combined_df = get_combined_global_df(data)

    if combined_df is not None:
        plot_dpp_vs_fog(combined_df, "dpp_vs_fog.png")
    else:
        print("No data available")


if __name__ == "__main__":
    main()
