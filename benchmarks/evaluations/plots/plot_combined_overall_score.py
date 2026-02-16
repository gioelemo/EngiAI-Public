#!/usr/bin/env python3
"""
Combined Overall Score Distribution Plot

Violin/box plot showing per-design combined_overall_score distribution.
This shows the full weighted overall score across all three categories:
- Design Quality (65%): IoU, pixel accuracy, constraints, objectives, printability
- Tool Efficiency (20%): Efficiency ratio (optimal/actual calls)
- Task Completion (15%): Success rate

For individual category scores, see plot_design_quality.py.
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


def plot_combined_overall_score(combined_design_df, output_path=None, output_dir=None):
    """Create violin plot of combined overall score distribution (publication format).

    Args:
        combined_design_df: DataFrame with design-level metrics
        output_path: Output filename (e.g., "combined_overall_score_distribution.png")
        output_dir: Optional output directory (default: figures/)
    """
    setup_style()
    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col_tall"], constrained_layout=True
    )

    # Filter to rows with a valid combined_overall_score (includes designs not found
    # but scored, e.g. natural prompts where asking for clarification is correct)
    valid_designs = combined_design_df[
        combined_design_df["combined_overall_score"].notna()
    ].copy()
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

    # Assign colors dynamically based on number of unique labels
    n_labels = valid_designs["_label"].nunique()
    palette = PLOT_STYLE["color_palette"][:n_labels]

    sns.violinplot(
        data=valid_designs,
        x="_label",
        y="combined_overall_score",
        hue="_label",
        palette=palette,
        inner="box",
        legend=False,
        ax=ax,
        linewidth=0.5,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Combined Overall Score")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    # Rotate x-tick labels for readability
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    # Mean annotations
    for i, source in enumerate(valid_designs["_label"].unique()):
        subset = valid_designs[valid_designs["_label"] == source][
            "combined_overall_score"
        ]
        ax.annotate(
            f"$\\mu$={subset.mean():.2f}",
            xy=(i, 0.02),
            ha="center",
            va="bottom",
            fontsize=font_sizes["annotation"],
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "0.8",
            },
        )

    if output_path:
        save_figure(fig, output_path, output_dir=output_dir)

    return fig


def main():
    """Generate combined overall score distribution plot."""
    print("Loading data...")
    data = load_data()
    combined_design_df = get_combined_design_df(data)

    if combined_design_df is not None:
        plot_combined_overall_score(
            combined_design_df, "combined_overall_score_distribution.png"
        )
    else:
        print("No design-level data available")


if __name__ == "__main__":
    main()
