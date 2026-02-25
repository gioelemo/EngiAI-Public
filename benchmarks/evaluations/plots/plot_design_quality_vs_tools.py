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
    for tc, mu in zip(tool_counts, means):
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
    """Side-by-side: CO score vs tools (violin) and DQ score vs tools (scatter).

    Shows that CO declines with more tools while design quality stays flat,
    demonstrating the decline is driven by efficiency penalties, not quality.

    Args:
        df: Combined design DataFrame for a single prompt style.
        output_dir: Optional output directory for saving.
    """
    setup_style()

    required = {"total_tools", "combined_overall_score", "design_quality_score", "model"}
    if not required.issubset(df.columns):
        print(f"Missing columns: {required - set(df.columns)}")
        return

    import seaborn as sns

    plot_df = df.dropna(subset=["total_tools", "combined_overall_score"]).copy()
    dq_df = df.dropna(subset=["total_tools", "design_quality_score"]).copy()
    if len(plot_df) == 0:
        print("No valid data")
        return

    font_sizes = PLOT_STYLE["font_sizes"]
    fig, (ax_co, ax_dq) = plt.subplots(
        1, 2, figsize=PLOT_STYLE["figsize_full_width"], constrained_layout=True
    )

    # --- Left panel: CO violin (same as performance_by_tools_overall) ---
    tool_counts_co = sorted(plot_df["total_tools"].unique())
    n_groups = len(tool_counts_co)
    palette = PLOT_STYLE["color_palette"][:n_groups]

    sns.violinplot(
        data=plot_df,
        x="total_tools",
        y="combined_overall_score",
        hue="total_tools",
        legend=False,
        inner="box",
        palette=palette,
        linewidth=0.5,
        ax=ax_co,
    )

    # Mean annotations for CO
    y_bot_co = plot_df["combined_overall_score"].min()
    for j, tc in enumerate(tool_counts_co):
        subset = plot_df[plot_df["total_tools"] == tc]["combined_overall_score"]
        if len(subset) > 0:
            ax_co.annotate(
                f"$\\mu$={subset.mean():.2f}",
                xy=(j, y_bot_co),
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

    ax_co.set_ylabel("Combined Overall Score")
    ax_co.set_xlabel("Total Tools Used")
    ax_co.grid(True, axis="y", alpha=0.3)
    ax_co.set_title("(a)", fontsize=font_sizes["axes_label"], loc="left")

    # --- Right panel: DQ scatter (same as design_quality_vs_tools) ---
    models = sorted(dq_df["model"].unique())
    styles = get_model_style(models)

    for model in models:
        subset = dq_df[dq_df["model"] == model]
        style = styles[model]
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(subset))
        ax_dq.scatter(
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

    # Mean trend line
    tool_counts_dq = sorted(dq_df["total_tools"].unique())
    means = [
        dq_df.loc[dq_df["total_tools"] == tc, "design_quality_score"].mean()
        for tc in tool_counts_dq
    ]
    ax_dq.plot(
        tool_counts_dq,
        means,
        color="black",
        linewidth=1.2,
        linestyle="--",
        marker="D",
        markersize=4,
        zorder=10,
        label=r"Mean ($\mu$)",
    )

    for tc, mu in zip(tool_counts_dq, means):
        ax_dq.annotate(
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

    ax_dq.set_xlabel("Total Tools Used")
    ax_dq.set_ylabel("Design Quality Score")
    ax_dq.set_xticks(tool_counts_dq)
    ax_dq.set_ylim(-0.05, 1.05)
    ax_dq.grid(True, axis="y", alpha=0.3)
    ax_dq.legend(fontsize=font_sizes["legend"], loc="lower left", framealpha=0.9)
    ax_dq.set_title("(b)", fontsize=font_sizes["axes_label"], loc="left")

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
                out_dir = (
                    Path(__file__).parent
                    / "figures"
                    / problem
                    / style
                    / rag
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Plotting {problem}/{style}/{rag} ({len(prob_subset)} samples)")
                plot_design_quality_vs_tools(prob_subset, output_dir=out_dir)
                plot_co_vs_dq_by_tools(prob_subset, output_dir=out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
