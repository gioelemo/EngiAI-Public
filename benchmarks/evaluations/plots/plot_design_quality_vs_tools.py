"""
Plot design quality score vs total tool calls for each prompt style.

Creates a scatter plot per prompt style with per-model markers and a
mean trend line, showing whether extra tool calls improve the actual
engineering output (design quality) or not.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.evaluations.plots.utils import (
    PLOT_STYLE,
    get_combined_design_df,
    get_model_style,
    load_data,
    save_figure,
    setup_style,
)


def plot_design_quality_vs_tools(df, output_dir=None):
    """Scatter plot of design quality score vs total tool calls, per model.

    Args:
        df: Combined design DataFrame (must contain total_tools,
            design_quality_score, model columns).
        output_dir: Optional output directory for saving.
    """
    setup_style()

    required = {"total_tools", "design_quality_score", "model"}
    if not required.issubset(df.columns):
        print(f"Missing columns: {required - set(df.columns)}")
        return

    # Drop rows where design was not found (NaN quality)
    plot_df = df.dropna(subset=["total_tools", "design_quality_score"]).copy()
    if len(plot_df) == 0:
        print("No valid data after dropping NaNs")
        return

    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col_tall"], constrained_layout=True
    )

    models = sorted(plot_df["model"].unique())
    styles = get_model_style(models)
    font_sizes = PLOT_STYLE["font_sizes"]

    # Scatter per model
    for model in models:
        subset = plot_df[plot_df["model"] == model]
        style = styles[model]
        # Jitter x slightly so points don't overlap
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(subset))
        ax.scatter(
            subset["total_tools"].values + jitter,
            subset["design_quality_score"].values,
            color=style["color"],
            marker=style["marker"],
            s=PLOT_STYLE["marker_size"],
            alpha=0.65,
            label=model,
            edgecolors="white",
            linewidths=0.3,
        )

    # Mean trend line (across all models)
    tool_counts = sorted(plot_df["total_tools"].unique())
    means = [
        plot_df.loc[plot_df["total_tools"] == tc, "design_quality_score"].mean()
        for tc in tool_counts
    ]
    ax.plot(
        tool_counts,
        means,
        color="black",
        linewidth=1.2,
        linestyle="--",
        marker="D",
        markersize=4,
        zorder=10,
        label=r"Mean ($\mu$)",
    )

    # Annotate means
    for tc, mu in zip(tool_counts, means, strict=True):
        ax.annotate(
            f"{mu:.2f}",
            xy=(tc, mu),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=font_sizes["annotation"],
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "0.7",
            },
        )

    ax.set_xlabel("Total Tools Used")
    ax.set_ylabel("Design Quality Score")
    ax.set_xticks(tool_counts)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        fontsize=font_sizes["legend"],
        loc="lower left",
        framealpha=0.9,
    )

    save_figure(fig, "design_quality_vs_tools.png", output_dir)
    return fig


def plot_co_vs_dq_by_tools(df, output_dir=None):
    """Single scatter: CO and DQ scores vs total tools, per model.

    Both metrics share one panel with distinct marker shapes (circles for CO,
    triangles for DQ) and model-specific colours.  Two mean trend lines show
    that CO declines with more tools while DQ stays flat.

    Args:
        df: Combined design DataFrame for a single prompt style.
        output_dir: Optional output directory for saving.
    """
    setup_style()

    required = {
        "total_tools",
        "combined_overall_score",
        "design_quality_score",
        "model",
    }
    if not required.issubset(df.columns):
        print(f"Missing columns: {required - set(df.columns)}")
        return

    co_df = df.dropna(subset=["total_tools", "combined_overall_score"]).copy()
    dq_df = df.dropna(subset=["total_tools", "design_quality_score"]).copy()
    if len(co_df) == 0:
        print("No valid data")
        return

    font_sizes = PLOT_STYLE["font_sizes"]
    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col_tall"], constrained_layout=True
    )

    models = sorted(co_df["model"].unique())
    styles = get_model_style(models)

    # CO markers: circles (o), DQ markers: triangles (^)
    co_marker = "o"
    dq_marker = "^"

    # --- Scatter: CO per model ---
    for model in models:
        subset = co_df[co_df["model"] == model]
        style = styles[model]
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(subset))
        ax.scatter(
            subset["total_tools"].values + jitter,
            subset["combined_overall_score"].values,
            color=style["color"],
            marker=co_marker,
            s=PLOT_STYLE["marker_size"],
            alpha=0.55,
            edgecolors="white",
            linewidths=0.3,
        )

    # --- Scatter: DQ per model ---
    for model in models:
        subset = dq_df[dq_df["model"] == model]
        style = styles[model]
        jitter = np.random.default_rng(99).uniform(-0.12, 0.12, len(subset))
        ax.scatter(
            subset["total_tools"].values + jitter,
            subset["design_quality_score"].values,
            color=style["color"],
            marker=dq_marker,
            s=PLOT_STYLE["marker_size"],
            alpha=0.55,
            edgecolors="white",
            linewidths=0.3,
        )

    # --- CO mean trend line ---
    tool_counts = sorted(co_df["total_tools"].unique())
    co_means = [
        co_df.loc[co_df["total_tools"] == tc, "combined_overall_score"].mean()
        for tc in tool_counts
    ]
    ax.plot(
        tool_counts,
        co_means,
        color="black",
        linewidth=1.2,
        linestyle="-",
        marker=co_marker,
        markersize=4,
        zorder=10,
        label="CO mean",
    )

    # --- DQ mean trend line ---
    dq_tool_counts = sorted(dq_df["total_tools"].unique())
    dq_means = [
        dq_df.loc[dq_df["total_tools"] == tc, "design_quality_score"].mean()
        for tc in dq_tool_counts
    ]
    ax.plot(
        dq_tool_counts,
        dq_means,
        color="black",
        linewidth=1.2,
        linestyle="--",
        marker=dq_marker,
        markersize=4,
        zorder=10,
        label="DQ mean",
    )

    # Annotate CO means (above, since CO > DQ)
    for tc, mu in zip(tool_counts, co_means, strict=True):
        ax.annotate(
            f"{mu:.2f}",
            xy=(tc, mu),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=font_sizes["annotation"],
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "0.7",
            },
        )

    # Annotate DQ means (below, since DQ < CO)
    for tc, mu in zip(dq_tool_counts, dq_means, strict=True):
        ax.annotate(
            f"{mu:.2f}",
            xy=(tc, mu),
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            fontsize=font_sizes["annotation"],
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "0.7",
            },
        )

    ax.set_xlabel("Total Tools Used")
    ax.set_ylabel("Score")
    ax.set_xticks(tool_counts)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    # Build legend: model colours + metric shapes
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=styles[model]["color"],
            markersize=5,
            label=model,
        )
        for model in models
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker=co_marker,
            color="black",
            linestyle="-",
            markersize=4,
            label="CO mean",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            marker=dq_marker,
            color="black",
            linestyle="--",
            markersize=4,
            label="DQ mean",
        )
    )
    ax.legend(
        handles=handles,
        fontsize=font_sizes["legend"],
        loc="lower left",
        framealpha=0.9,
    )

    save_figure(fig, "co_vs_dq_by_tools.png", output_dir)
    return fig


def main():
    """Main execution — generates plots for all prompt styles."""
    setup_style()

    print("Loading data...")
    data = load_data()
    combined = get_combined_design_df(data)

    if combined is None or len(combined) == 0:
        print("No design data found.")
        return

    if "total_tools" not in combined.columns:
        print("No tool usage data. Re-run extract_data.py.")
        return

    # Generate per prompt-style
    for style in sorted(combined["prompt_style"].dropna().unique()):
        subset = combined[combined["prompt_style"] == style]
        if len(subset) == 0:
            continue

        # Determine output dir (same structure as other plots)
        rag_statuses = subset["rag_status"].dropna().unique()
        for rag in rag_statuses:
            rag_subset = subset[subset["rag_status"] == rag]
            if len(rag_subset) == 0:
                continue

            problems = rag_subset["problem"].dropna().unique()
            for problem in problems:
                prob_subset = rag_subset[rag_subset["problem"] == problem]
                out_dir = Path(__file__).parent / "figures" / problem / style / rag
                out_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"  Plotting {problem}/{style}/{rag} ({len(prob_subset)} samples)"
                )
                plot_design_quality_vs_tools(prob_subset, output_dir=out_dir)
                plot_co_vs_dq_by_tools(prob_subset, output_dir=out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
