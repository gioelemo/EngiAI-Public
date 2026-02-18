#!/usr/bin/env python3
"""
DPP vs FOG Scatter Plot: Diversity vs Quality Trade-off

Key question: Do diverse designs maintain quality?
- DPP (higher = more diverse)
- FOG (lower = better quality)
"""

import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import ConvexHull

from utils import (
    MIN_CORRELATION_SAMPLES,
    PLOT_STYLE,
    get_combined_global_df,
    get_model_style,
    identify_pareto_front,
    load_data,
    make_label,
    save_figure,
    setup_style,
)

# Minimum points required to compute convex hull
MIN_HULL_POINTS = 3


def plot_dpp_vs_fog(combined_df, output_path=None, output_dir=None):
    """Create DPP vs FOG scatter plot (publication format).

    Args:
        combined_df: DataFrame with global metrics
        output_path: Output filename (e.g., "dpp_vs_fog.png")
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
                    subset["fog"],
                    label=make_label(model, problem, single_problem),
                    marker=style["marker"],
                    c=colors.get(problem, style["color"]),
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
                "edgecolor": "0.8",
            },
        )

    ax.legend(loc="upper right", fontsize=font_sizes["legend"])
    ax.grid(True, alpha=0.3)

    if output_path:
        save_figure(fig, output_path, output_dir=output_dir)

    return fig


def plot_tradeoff_with_pareto(
    combined_df, x_col="dpp", y_col="fog", output_name="pareto_tradeoff.png"
):
    setup_style()
    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col"], constrained_layout=True
    )

    models = combined_df["model"].unique()
    styles = get_model_style(models)

    # 1. Plot Model Territories (Hulls) and Points
    for model in models:
        subset = combined_df[combined_df["model"] == model].dropna(
            subset=[x_col, y_col]
        )
        if len(subset) < MIN_HULL_POINTS:
            continue

        style = styles[model]
        pts = subset[[x_col, y_col]].values

        # Shade the model territory (The "Hull")
        try:
            hull = ConvexHull(pts)
            poly = plt.Polygon(
                pts[hull.vertices], alpha=0.1, color=style["color"], lw=0
            )
            ax.add_patch(poly)
        except (ValueError, Exception):
            # ConvexHull may fail if points are collinear or other geometric issues
            pass

        # FIX: Ensure label is passed here for the legend
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            label=model.split("-")[0],  # Simplify name for legend
            marker=style["marker"],
            color=style["color"],
            s=PLOT_STYLE["marker_size"],
            alpha=PLOT_STYLE["alpha"],
            edgecolors="white",
            lw=0.3,
        )

    # 1. ADD PADDING TO THE Y-AXIS
    # This lifts the plot so the Pareto line at y=0 is visible
    y_max = combined_df[y_col].max()
    ax.set_ylim(-50, y_max * 1.1)  # -50 provides the "breathing room"

    # 2. DRAW THE PARETO LINE ON THE HIGHEST LAYER
    raw_pts = combined_df[[x_col, y_col]].dropna().values
    if len(raw_pts) > 0:
        mask = identify_pareto_front(raw_pts, minimize_x=False, minimize_y=True)
        pareto_pts = raw_pts[mask]
        pareto_pts = pareto_pts[pareto_pts[:, 0].argsort()]

        # Increase linewidth and ensure zorder is very high
        ax.plot(
            pareto_pts[:, 0],
            pareto_pts[:, 1],
            color="black",
            linewidth=3.0,  # Thicker line for visibility
            linestyle="--",
            label="Pareto Frontier",
            zorder=100,
        )  # Ensures it stays above the orange hull

    # Labeling
    ax.set_xlabel("Diversity (DPP) $\\rightarrow$")
    ax.set_ylabel("$\\leftarrow$ Optimality Gap (FOG)")

    # IMPROVED LEGEND: Place it outside to the right or top
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=PLOT_STYLE["font_sizes"]["legend"],
        frameon=True,
        title="Models \& Frontier",
    )

    save_figure(fig, output_name)


def main():
    """Generate DPP vs FOG plot."""
    print("Loading data...")
    data = load_data()
    combined_df = get_combined_global_df(data)

    if combined_df is not None:
        plot_dpp_vs_fog(combined_df, "dpp_vs_fog.png")
        plot_tradeoff_with_pareto(
            combined_df, "dpp", "fog", "tradeoff_diversity_quality.png"
        )
    else:
        print("No data available")


if __name__ == "__main__":
    main()
