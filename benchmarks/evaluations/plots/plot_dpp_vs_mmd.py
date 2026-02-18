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
    get_model_style,
    load_data,
    make_label,
    save_figure,
    setup_style,
)


def plot_dpp_vs_mmd(combined_df, output_path=None, output_dir=None):
    """Create DPP vs MMD scatter plot (publication format).

    Args:
        combined_df: DataFrame with global metrics
        output_path: Output filename (e.g., "dpp_vs_mmd.png")
        output_dir: Optional output directory (default: figures/)
    """
    setup_style()
    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col"], constrained_layout=True
    )

    # Get dynamic styles for models
    models = list(combined_df["model"].unique())
    model_styles = get_model_style(models)
    colors = PLOT_STYLE["colors"]
    font_sizes = PLOT_STYLE["font_sizes"]
    single_problem = combined_df["problem"].nunique() == 1

    for model in models:
        for problem in combined_df["problem"].unique():
            mask = (combined_df["model"] == model) & (combined_df["problem"] == problem)
            subset = combined_df[mask]
            if len(subset) > 0:
                style = model_styles[model]
                ax.scatter(
                    subset["dpp"],
                    subset["mmd"],
                    label=make_label(model, problem, single_problem),
                    marker=style["marker"],
                    c=colors.get(problem, style["color"]),
                    alpha=PLOT_STYLE["alpha"],
                    s=PLOT_STYLE["marker_size"],
                    edgecolors="white",
                    linewidth=0.3,
                )

    ax.set_xlabel("DPP Diversity")
    ax.set_ylabel("MMD")

    # Correlation annotation
    valid_data = combined_df.dropna(subset=["dpp", "mmd"])
    if len(valid_data) > MIN_CORRELATION_SAMPLES:
        r, _ = stats.pearsonr(valid_data["dpp"], valid_data["mmd"])
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
                "edgecolor": "0.8",
            },
        )

    ax.legend(loc="upper right", fontsize=font_sizes["legend"])
    ax.grid(True, alpha=0.3)

    if output_path:
        save_figure(fig, output_path, output_dir=output_dir)

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
