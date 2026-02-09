#!/usr/bin/env python3
"""
Design Quality Distribution Plot

Violin/box plot showing per-design output_quality_score distribution.
Note: Uses 'overall_score' field which is the output_quality scorer result
(partial score with design_quality + printability only).
For the full weighted score, use combined_overall_score instead.
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


def plot_design_quality(combined_design_df, output_path=None, output_dir=None):
    """Create violin plot of design quality distribution (NeurIPS format).

    Args:
        combined_design_df: DataFrame with design-level metrics
        output_path: Output filename (e.g., "design_quality_distribution.png")
        output_dir: Optional output directory (default: figures/)
    """
    setup_style()
    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col_tall"], constrained_layout=True
    )

    # Filter to valid designs
    valid_designs = combined_design_df[combined_design_df["design_found"]].copy()
    font_sizes = PLOT_STYLE["font_sizes"]

    # When all entries share the same problem, use just model names (no redundant problem suffix)
    single_problem = (
        valid_designs["problem"].nunique() == 1
        if "problem" in valid_designs.columns
        else False
    )
    if single_problem and "model_display" in valid_designs.columns:
        valid_designs["_label"] = valid_designs["model_display"]
    else:
        valid_designs["_label"] = valid_designs["source"]

    sns.violinplot(
        data=valid_designs,
        x="_label",
        y="overall_score",
        hue="_label",
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
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    # Statistics annotations (smaller, cleaner)
    for i, source in enumerate(valid_designs["_label"].unique()):
        subset = valid_designs[valid_designs["_label"] == source]["overall_score"]
        stats_text = f"$\\mu$={subset.mean():.2f}, $\\sigma$={subset.std():.2f}"
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
        save_figure(fig, output_path, output_dir=output_dir)

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
