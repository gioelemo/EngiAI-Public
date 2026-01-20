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


def plot_iou_vs_objective(combined_design_df, output_path=None):
    """Create IoU vs Objective Score scatter plot."""
    setup_style()
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_scatter"])

    valid = combined_design_df[combined_design_df["design_found"]]
    colors = PLOT_STYLE["colors"]

    for problem in valid["problem"].unique():
        subset = valid[valid["problem"] == problem]
        ax.scatter(
            subset["iou"],
            subset["objective_score"],
            c=colors.get(problem, "gray"),
            alpha=0.6,
            s=50,
            label=f"GPT-4.1 ({problem})",
            edgecolors="white",
            linewidth=0.3,
        )

    # Reference lines (80% threshold)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="80% threshold")
    ax.axvline(x=0.8, color="gray", linestyle="--", alpha=0.5)

    # Quadrant annotations
    ax.annotate(
        "High IoU\nHigh Quality", xy=(0.9, 0.9), ha="center", fontsize=9, alpha=0.7
    )
    ax.annotate(
        "Low IoU\nHigh Quality", xy=(0.1, 0.9), ha="center", fontsize=9, alpha=0.7
    )

    ax.set_xlabel("IoU (Topology Match)", fontsize=12)
    ax.set_ylabel("Objective Score (Performance Match)", fontsize=12)
    ax.set_title(
        "Structure vs Function: Does Shape Match Performance?",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

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
