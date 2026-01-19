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
    """Create violin plot of design quality distribution."""
    setup_style()
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_violin"])

    # Filter to valid designs
    valid_designs = combined_design_df[combined_design_df["design_found"]]

    sns.violinplot(
        data=valid_designs,
        x="source",
        y="overall_score",
        hue="source",
        palette=[PLOT_STYLE["colors"]["beams2d"], PLOT_STYLE["colors"]["photonics2d"]],
        inner="box",
        legend=False,
        ax=ax,
    )

    ax.set_xlabel("Model and Problem", fontsize=12)
    ax.set_ylabel("Overall Design Score (0-1)", fontsize=12)
    ax.set_title(
        "Distribution of Design Quality Scores", fontsize=14, fontweight="bold"
    )
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    # Statistics annotations
    for i, source in enumerate(valid_designs["source"].unique()):
        subset = valid_designs[valid_designs["source"] == source]["overall_score"]
        stats_text = (
            f"mean={subset.mean():.3f}\nstd={subset.std():.3f}\nn={len(subset)}"
        )
        ax.annotate(
            stats_text,
            xy=(i, 0.05),
            ha="center",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    plt.tight_layout()

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
