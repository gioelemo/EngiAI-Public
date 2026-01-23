#!/usr/bin/env python3
"""
Design Quality Distribution Plot

Violin/box plot showing per-design overall_score distribution.
Shows quality consistency across designs.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    PLOT_STYLE,
    get_combined_design_df,
    load_data,
    save_figure,
    setup_style,
)


def plot_design_quality(combined_design_df, output_path=None):
    """Create violin plot of design quality distribution (NeurIPS format)."""
    setup_style()
    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col_tall"], constrained_layout=True
    )

    # Filter to valid designs
    valid_designs = combined_design_df[combined_design_df["design_found"]]
    font_sizes = PLOT_STYLE["font_sizes"]

    sns.violinplot(
        data=valid_designs,
        x="source",
        y="overall_score",
        hue="source",
        palette=[PLOT_STYLE["colors"]["beams2d"], PLOT_STYLE["colors"]["photonics2d"]],
        inner="box",
        legend=False,
        ax=ax,
        linewidth=0.5,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Overall Design Score")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    # Rotate x-tick labels for readability
    ax.tick_params(axis="x", rotation=15)

    # Statistics annotations (smaller, cleaner)
    for i, source in enumerate(valid_designs["source"].unique()):
        subset = valid_designs[valid_designs["source"] == source]["overall_score"]
        stats_text = f"$\\mu$={subset.mean():.2f}"
        ax.annotate(
            stats_text,
            xy=(i, 0.02),
            ha="center",
            fontsize=font_sizes["annotation"],
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "0.8",
            },
        )

    if output_path:
        save_figure(fig, output_path)

    return fig


def main():
    """Generate design quality distribution plot."""
    print("Loading data...")
    data = load_data()
    combined_design_df = get_combined_design_df(data)

    if combined_design_df is not None:
        plot_design_quality(combined_design_df, "design_quality_distribution.png")
    else:
        print("No design-level data available")


if __name__ == "__main__":
    main()
