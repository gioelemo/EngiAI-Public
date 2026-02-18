#!/usr/bin/env python3
"""
IoU vs Objective Score Scatter Plot

Shows whether LLM designs capture both structure and function.
- IoU: Topology match (does the shape match?)
- Objective Score: Performance match (does it perform similarly?)
"""

import matplotlib.pyplot as plt

from utils import (
    PLOT_STYLE,
    get_combined_design_df,
    load_data,
    save_figure,
    setup_style,
)


def plot_iou_vs_objective(combined_design_df, output_path=None, output_dir=None):
    """Create IoU vs Objective Score scatter plot (publication format).

    Args:
        combined_design_df: DataFrame with design-level metrics
        output_path: Output filename (e.g., "iou_vs_objective.png")
        output_dir: Optional output directory (default: figures/)
    """
    setup_style()
    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col"], constrained_layout=True
    )

    valid = combined_design_df[combined_design_df["design_found"]]
    colors = PLOT_STYLE["colors"]
    font_sizes = PLOT_STYLE["font_sizes"]

    for problem in valid["problem"].unique():
        subset = valid[valid["problem"] == problem]
        ax.scatter(
            subset["iou"],
            subset["objective_score"],
            c=colors.get(problem, "gray"),
            alpha=PLOT_STYLE["alpha"],
            s=PLOT_STYLE["marker_size"],
            label=problem,
            edgecolors="white",
            linewidth=0.3,
        )

    # Reference lines (80% threshold)
    ax.axhline(y=0.8, color="0.6", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.axvline(x=0.8, color="0.6", linestyle="--", linewidth=0.5, alpha=0.7)

    ax.set_xlabel("IoU")
    ax.set_ylabel("Objective Score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=font_sizes["legend"])
    ax.grid(True, alpha=0.3)

    if output_path:
        save_figure(fig, output_path, output_dir=output_dir)

    return fig


def main():
    """Generate IoU vs Objective plot."""
    print("Loading data...")
    data = load_data()
    combined_design_df = get_combined_design_df(data)

    if combined_design_df is not None:
        plot_iou_vs_objective(combined_design_df, "iou_vs_objective.png")
    else:
        print("No design-level data available")


if __name__ == "__main__":
    main()
