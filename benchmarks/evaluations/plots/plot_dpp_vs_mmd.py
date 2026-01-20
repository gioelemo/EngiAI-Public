#!/usr/bin/env python3
"""
DPP vs MMD Scatter Plot: Diversity vs Distribution Match

Key question: Does diversity come at the cost of distribution match?
- DPP (higher = more diverse)
- MMD (lower = better distribution match)
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


def plot_dpp_vs_mmd(combined_df, output_path=None):
    """Create DPP vs MMD scatter plot."""
    setup_style()
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_scatter"])

    markers = PLOT_STYLE["markers"]
    colors = PLOT_STYLE["colors"]

    for model in combined_df["model"].unique():
        for problem in combined_df["problem"].unique():
            mask = (combined_df["model"] == model) & (combined_df["problem"] == problem)
            subset = combined_df[mask]
            if len(subset) > 0:
                ax.scatter(
                    subset["dpp"],
                    subset["mmd"],
                    label=f"{model} ({problem})",
                    marker=markers.get(model, "o"),
                    c=colors.get(problem, "gray"),
                    alpha=PLOT_STYLE["alpha"],
                    s=PLOT_STYLE["marker_size"],
                    edgecolors="white",
                    linewidth=0.5,
                )

    ax.set_xlabel("DPP Diversity Score (higher = more diverse)", fontsize=12)
    ax.set_ylabel("MMD (lower = better distribution match)", fontsize=12)
    ax.set_title("Diversity vs Distribution Match", fontsize=14, fontweight="bold")

    # Correlation annotation
    valid_data = combined_df.dropna(subset=["dpp", "mmd"])
    if len(valid_data) > MIN_CORRELATION_SAMPLES:
        r, p = stats.pearsonr(valid_data["dpp"], valid_data["mmd"])
        ax.annotate(
            f"r = {r:.3f} (p = {p:.3f})",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def main():
    """Generate DPP vs MMD plot."""
    print("Loading data...")
    data = load_data()
    combined_df = get_combined_global_df(data)

    if combined_df is not None:
        plot_dpp_vs_mmd(combined_df, "dpp_vs_mmd.png")
    else:
        print("No data available")


if __name__ == "__main__":
    main()
