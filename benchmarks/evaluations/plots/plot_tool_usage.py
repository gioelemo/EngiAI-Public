"""
Generate tool usage visualizations for agent evaluations.

This script creates plots showing:
- Tool usage frequency across models
- Tool usage distribution per problem
- Correlation between tool usage and performance
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import (
    PLOT_STYLE,
    get_combined_design_df,
    get_model_style,
    load_data,
    save_figure,
    setup_style,
)


def plot_tool_usage_frequency(tool_data, output_dir=None):
    """Plot frequency of each tool across all models (publication format).

    Args:
        tool_data: Combined tool usage DataFrame
        output_dir: Optional output directory for saving
    """
    if tool_data is None or len(tool_data) == 0:
        print("No tool usage data available")
        return

    setup_style()

    # Get all tool columns (exclude metric fields like tool_efficiency_score)
    exclude_metrics = {"tool_efficiency_score", "tool_completion_score"}
    tool_columns = [
        col
        for col in tool_data.columns
        if col.startswith("tool_") and col not in exclude_metrics
    ]

    if not tool_columns:
        print("No tool columns found in data")
        return

    # Sum tool usage across all examples and models
    tool_totals = {}
    for col in tool_columns:
        tool_name = col.replace("tool_", "")
        tool_totals[tool_name] = tool_data[col].fillna(0).sum()

    # Sort by frequency
    sorted_tools = sorted(tool_totals.items(), key=lambda x: x[1], reverse=True)
    tool_names = [t[0] for t in sorted_tools]
    tool_counts = [t[1] for t in sorted_tools]

    font_sizes = PLOT_STYLE["font_sizes"]

    # Create plot
    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col_tall"], constrained_layout=True
    )

    bars = ax.barh(
        tool_names,
        tool_counts,
        color=PLOT_STYLE["color_palette"][0],
        height=0.7,
        alpha=PLOT_STYLE["alpha"],
        edgecolor="white",
        linewidth=0.5,
    )

    # Add count labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=font_sizes["annotation"],
        )

    ax.set_xlabel("Usage Count")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)

    return save_figure(fig, "tool_usage_frequency.png", output_dir)


def plot_tool_usage_by_model(tool_data, output_dir=None):
    """Plot tool usage comparison across models (publication format).

    Args:
        tool_data: Combined tool usage DataFrame
        output_dir: Optional output directory for saving
    """
    if tool_data is None or len(tool_data) == 0:
        print("No tool usage data available")
        return

    setup_style()
    font_sizes = PLOT_STYLE["font_sizes"]

    # Calculate average tools per example for each model
    model_stats = (
        tool_data.groupby(["model", "problem"])
        .agg(
            {
                "total_tools": ["mean", "std", "count"],
                "unique_tools": ["mean", "std"],
            }
        )
        .reset_index()
    )

    model_stats.columns = [
        "model",
        "problem",
        "avg_total",
        "std_total",
        "n_samples",
        "avg_unique",
        "std_unique",
    ]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=PLOT_STYLE["figsize_full_width"], constrained_layout=True
    )

    # Plot 1: Average total tools
    x = np.arange(len(model_stats))
    width = 0.6

    # Get dynamic styles for models
    models = list(model_stats["model"].unique())
    model_styles = get_model_style(models)
    colors = [model_styles[model]["color"] for model in model_stats["model"]]

    # Check if only one problem (don't show problem name if so)
    single_problem = len(model_stats["problem"].unique()) == 1

    bars1 = ax1.bar(
        x,
        model_stats["avg_total"],
        width,
        yerr=model_stats["std_total"],
        color=colors,
        alpha=PLOT_STYLE["alpha"],
        capsize=2,
        edgecolor="white",
        linewidth=0.5,
    )

    ax1.set_xlabel("")
    ax1.set_ylabel("Avg. Total Tool Calls")
    ax1.set_xticks(x)
    if single_problem:
        ax1.set_xticklabels(
            [row["model"] for _, row in model_stats.iterrows()],
            rotation=0,
            ha="center",
        )
    else:
        ax1.set_xticklabels(
            [
                f"{row['model']}\n({row['problem']})"
                for _, row in model_stats.iterrows()
            ],
            rotation=0,
            ha="center",
        )
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=font_sizes["annotation"],
        )

    # Plot 2: Average unique tools
    bars2 = ax2.bar(
        x,
        model_stats["avg_unique"],
        width,
        yerr=model_stats["std_unique"],
        color=colors,
        alpha=PLOT_STYLE["alpha"],
        capsize=2,
        edgecolor="white",
        linewidth=0.5,
    )

    ax2.set_xlabel("")
    ax2.set_ylabel("Avg. Unique Tools")
    ax2.set_xticks(x)
    single_problem = len(model_stats["problem"].unique()) == 1
    if single_problem:
        ax2.set_xticklabels(
            [row["model"] for _, row in model_stats.iterrows()],
            rotation=0,
            ha="center",
        )
    else:
        ax2.set_xticklabels(
            [
                f"{row['model']}\n({row['problem']})"
                for _, row in model_stats.iterrows()
            ],
            rotation=0,
            ha="center",
        )
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=font_sizes["annotation"],
        )

    return save_figure(fig, "tool_usage_by_model.png", output_dir)


def plot_tool_usage_vs_performance(tool_data, design_data, output_dir=None):  # noqa: ARG001, PLR0911
    """Plot correlation between tool usage and performance (publication format).

    Args:
        tool_data: Combined design DataFrame with tool usage metrics
        design_data: Combined design DataFrame (unused, kept for compatibility)
        output_dir: Optional output directory for saving
    """
    if tool_data is None:
        print("Missing data for correlation plot")
        return

    setup_style()
    font_sizes = PLOT_STYLE["font_sizes"]

    # In the JSON pipeline, tool_data and design_data are the same DataFrame
    merged = tool_data

    if len(merged) == 0:
        print("No data available for correlation plot")
        return

    # Check for required columns
    required_cols = ["total_tools", "unique_tools", "combined_overall_score", "iou"]
    missing_cols = [col for col in required_cols if col not in merged.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return

    # Check if tool usage data is actually populated (not all null)
    if merged["total_tools"].isna().all() or merged["unique_tools"].isna().all():
        print("⚠️  Tool usage metrics (total_tools, unique_tools) are not available")
        print("   Skipping tool usage vs performance correlation plot")
        return

    # Drop rows with missing tool data for correlation analysis
    merged = merged.dropna(subset=["total_tools", "unique_tools"])
    if len(merged) == 0:
        print("No valid tool usage data for correlation plot")
        return

    # Create scatter plots
    fig, axes = plt.subplots(
        2, 2, figsize=PLOT_STYLE["figsize_full_width_tall"], constrained_layout=True
    )

    metrics = [
        ("total_tools", "combined_overall_score", "Total Tools", "Score"),
        ("unique_tools", "combined_overall_score", "Unique Tools", "Score"),
        ("total_tools", "iou", "Total Tools", "IoU"),
        ("unique_tools", "iou", "Unique Tools", "IoU"),
    ]

    # Get dynamic styles for models
    models = list(merged["model"].unique()) if "model" in merged.columns else []
    if not models:
        print("No model information available in data")
        return

    model_styles = get_model_style(models)

    for ax, (x_col, y_col, x_label, y_label) in zip(axes.flat, metrics, strict=False):
        # Plot by model
        for model in models:
            model_data = merged[merged["model"] == model]
            style = model_styles[model]
            ax.scatter(
                model_data[x_col],
                model_data[y_col],
                label=model,
                marker=style["marker"],
                color=style["color"],
                alpha=PLOT_STYLE["alpha"],
                s=PLOT_STYLE["marker_size"],
                edgecolors="white",
                linewidth=0.3,
            )

        # Add trend line (drop rows with missing values first)
        trend_data = merged[[x_col, y_col]].dropna()
        min_points_for_trend = 2
        if len(trend_data) > min_points_for_trend:
            z = np.polyfit(trend_data[x_col], trend_data[y_col], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(trend_data[x_col].min(), trend_data[x_col].max(), 100)
            ax.plot(x_trend, p(x_trend), "k--", alpha=0.3, linewidth=0.5)

            # Calculate correlation
            corr = trend_data[x_col].corr(trend_data[y_col])
            ax.text(
                0.05,
                0.95,
                f"$r$ = {corr:.2f}",
                transform=ax.transAxes,
                fontsize=font_sizes["annotation"],
                verticalalignment="top",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "0.8",
                },
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(fontsize=font_sizes["legend"], loc="lower right")
        ax.grid(alpha=0.3)

    return save_figure(fig, "tool_usage_vs_performance.png", output_dir)


def plot_tool_heatmap_by_model(tool_data, output_dir=None):
    """Create heatmap showing tool usage percentages by model."""
    if tool_data is None or len(tool_data) == 0:
        print("No tool usage data available")
        return

    setup_style()
    font_sizes = PLOT_STYLE["font_sizes"]

    # 1. Get tool columns (exclude metric fields like tool_efficiency_score)
    # Only include fields from actual tool calls, not computed scores
    exclude_metrics = {"tool_efficiency_score", "tool_completion_score"}
    tool_columns = [
        col
        for col in tool_data.columns
        if col.startswith("tool_") and col not in exclude_metrics
    ]

    # 1. Calculate average tool calls per sample
    # Mean of actual counts (expressed as %, where 100% = 1 call per sample)
    # fillna(0) ensures NaN/missing values are treated as 0 calls
    model_tool_usage = (
        tool_data.groupby("model")[tool_columns].apply(lambda x: x.fillna(0).mean()).T
        * 100
    )
    # 2. ABBREVIATE NAMES HERE
    model_tool_usage.columns = [
        c.replace("-instruct-2507-q8-0", "-Inst") for c in model_tool_usage.columns
    ]
    # 2. Clean up tool names
    model_tool_usage.index = [
        idx.replace("tool_", "") for idx in model_tool_usage.index
    ]

    # 3. Filter out tools with zero usage and sort
    model_tool_usage = model_tool_usage[(model_tool_usage.sum(axis=1) > 0)].sort_values(
        by=model_tool_usage.columns.tolist(), ascending=False
    )

    n_tools = len(model_tool_usage)
    fig_height = min(max(2.4, n_tools * 0.2), 5.0)
    fig, ax = plt.subplots(
        figsize=(PLOT_STYLE["figsize_single_col"][0], fig_height),
        constrained_layout=True,
    )

    sns.heatmap(
        model_tool_usage,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        # Label: average calls per sample, expressed as % (100% = 1 call)
        cbar_kws={"label": r"Avg. Calls per Sample (\%)", "shrink": 0.8},
        linewidths=0.3,
        ax=ax,
        annot_kws={"size": font_sizes["annotation"]},
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    return save_figure(fig, "tool_usage_heatmap_pct.png", output_dir)


def plot_tool_heatmap_with_std(tool_data, output_dir=None):
    """Create heatmap showing tool usage mean and std by model."""
    if tool_data is None or len(tool_data) == 0:
        print("No tool usage data available")
        return

    setup_style()
    font_sizes = PLOT_STYLE["font_sizes"]

    # 1. Get tool columns (exclude metric fields like tool_efficiency_score)
    # Only include fields from actual tool calls, not computed scores
    exclude_metrics = {"tool_efficiency_score", "tool_completion_score"}
    tool_columns = [
        col
        for col in tool_data.columns
        if col.startswith("tool_") and col not in exclude_metrics
    ]

    # Calculate mean and std for each model-tool combination
    # Use actual counts (expressed as %, where 100% = 1 call per sample)
    # fillna(0) ensures NaN/missing values are treated as 0 calls
    model_tool_mean = (
        tool_data.groupby("model")[tool_columns].apply(lambda x: x.fillna(0).mean()).T
        * 100
    )
    model_tool_std = (
        tool_data.groupby("model")[tool_columns].apply(lambda x: x.fillna(0).std()).T
        * 100
    )

    # ABBREVIATE NAMES
    model_tool_mean.columns = [
        c.replace("-instruct-2507-q8-0", "-Inst") for c in model_tool_mean.columns
    ]
    model_tool_std.columns = [
        c.replace("-instruct-2507-q8-0", "-Inst") for c in model_tool_std.columns
    ]

    # Clean up tool names
    model_tool_mean.index = [idx.replace("tool_", "") for idx in model_tool_mean.index]
    model_tool_std.index = [idx.replace("tool_", "") for idx in model_tool_std.index]

    # Filter out tools with zero usage and sort
    model_tool_mean = model_tool_mean[(model_tool_mean.sum(axis=1) > 0)].sort_values(
        by=model_tool_mean.columns.tolist(), ascending=False
    )
    model_tool_std = model_tool_std.loc[model_tool_mean.index]

    # Create custom annotations with mean±std format
    annot_labels = np.empty_like(model_tool_mean, dtype=object)
    for i in range(model_tool_mean.shape[0]):
        for j in range(model_tool_mean.shape[1]):
            mean_val = model_tool_mean.iloc[i, j]
            std_val = model_tool_std.iloc[i, j]
            annot_labels[i, j] = f"{mean_val:.1f}\n±{std_val:.1f}"

    n_tools = len(model_tool_mean)
    fig_height = min(max(2.4, n_tools * 0.2), 5.0)
    fig, ax = plt.subplots(
        figsize=(PLOT_STYLE["figsize_single_col"][0], fig_height),
        constrained_layout=True,
    )

    sns.heatmap(
        model_tool_mean,
        annot=annot_labels,
        fmt="",  # Empty format since we're using custom strings
        cmap="YlOrRd",
        cbar_kws={"label": r"Avg. Calls per Sample (\%)", "shrink": 0.8},
        linewidths=0.3,
        ax=ax,
        annot_kws={
            "size": font_sizes["annotation"] - 1
        },  # Slightly smaller for two lines
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    return save_figure(fig, "tool_usage_heatmap_with_std.png", output_dir)


def plot_tool_usage_delta_heatmap(tool_data, output_dir=None):
    """Plot how much each model deviates from the average tool usage."""
    setup_style()

    # 1. Calculate usage rate per model (exclude metric fields)
    exclude_metrics = {"tool_efficiency_score", "tool_completion_score"}
    tool_columns = [
        col
        for col in tool_data.columns
        if col.startswith("tool_") and col not in exclude_metrics
    ]

    # Calculate mean and std for each model-tool combination
    # First fill NaN with 0 to treat missing values as 0 calls
    model_tool_stats = (
        tool_data.groupby("model")[tool_columns]
        .apply(lambda x: x.fillna(0))
        .groupby("model")
        .agg(["mean", "std"])
        * 100
    )

    # Get just the means for the main heatmap
    model_usage = model_tool_stats.xs("mean", level=1, axis=1)

    # 2. Calculate the average usage across ALL models
    avg_usage = model_usage.mean()

    # 3. Calculate Delta (Difference from Mean)
    delta_usage = (model_usage - avg_usage).T

    # Get standard deviations (transposed to match delta_usage shape)
    std_usage = model_tool_stats.xs("std", level=1, axis=1).T

    # Clean up names
    delta_usage.index = [idx.replace("tool_", "") for idx in delta_usage.index]
    std_usage.index = [idx.replace("tool_", "") for idx in std_usage.index]

    # Abbreviate model names as discussed
    delta_usage.columns = [
        c.replace("-instruct-2507-q8-0", "-Inst") for c in delta_usage.columns
    ]
    std_usage.columns = [
        c.replace("-instruct-2507-q8-0", "-Inst") for c in std_usage.columns
    ]

    # Create custom annotations with mean±std format
    annot_labels = np.empty_like(delta_usage, dtype=object)
    for i in range(delta_usage.shape[0]):
        for j in range(delta_usage.shape[1]):
            mean_val = delta_usage.iloc[i, j]
            std_val = std_usage.iloc[i, j]
            annot_labels[i, j] = f"{mean_val:.1f}\n±{std_val:.1f}"

    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col_tall"],  # Use standard tall format
        constrained_layout=True,
    )

    # Use a diverging colormap (RdBu_r: Red is more, Blue is less)
    sns.heatmap(
        delta_usage,
        annot=annot_labels,
        fmt="",  # Empty format since we're using custom strings
        cmap="RdBu_r",
        center=0,
        linewidths=0.3,
        ax=ax,
        cbar_kws={
            "label": r"$\Delta$ Avg. Calls per Sample (\%)",
            "shrink": 0.8,
        },  # LaTeX math
        annot_kws={
            "size": PLOT_STYLE["font_sizes"]["annotation"] - 1
        },  # Slightly smaller for two lines
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    save_figure(
        fig, "tool_usage_delta_heatmap.png", output_dir
    )  # Uses bbox_inches='tight'


def _plot_single_performance_by_tool_count(df, metric, ylabel, output_name, output_dir):
    """Plot a single violin of a metric grouped by total tool count.

    Args:
        df: DataFrame with tool usage and performance data
        metric: Column name to plot on y-axis
        ylabel: Display label for y-axis
        output_name: Output filename
        output_dir: Optional output directory
    """
    setup_style()
    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col_tall"], constrained_layout=True
    )

    n_groups = df["total_tools"].nunique()
    palette = PLOT_STYLE["color_palette"][:n_groups]

    sns.violinplot(
        data=df,
        x="total_tools",
        y=metric,
        hue="total_tools",
        legend=False,
        inner="box",
        palette=palette,
        linewidth=0.5,
        ax=ax,
    )

    # Mean annotations
    font_sizes = PLOT_STYLE["font_sizes"]
    y_bot = df[metric].min()
    for j, tool_count in enumerate(sorted(df["total_tools"].unique())):
        subset = df[df["total_tools"] == tool_count][metric]
        if len(subset) > 0:
            ax.annotate(
                f"$\\mu$={subset.mean():.2f}",
                xy=(j, y_bot),
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

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Total Tools Used")
    ax.grid(True, axis="y", alpha=0.3)

    save_figure(fig, output_name, output_dir)
    return fig


def plot_performance_distribution_by_tool_count(
    combined_tools,
    combined_design,  # noqa: ARG001
    output_dir=None,
):
    """Show the distribution of performance metrics relative to tools used.

    Generates two separate figures: one for overall score, one for IoU.

    Args:
        combined_tools: Combined design DataFrame with tool usage metrics
        combined_design: Unused, kept for compatibility
        output_dir: Optional output directory for saving
    """
    df = combined_tools

    if "total_tools" not in df.columns or "combined_overall_score" not in df.columns:
        print("Missing required columns (total_tools or combined_overall_score)")
        return

    _plot_single_performance_by_tool_count(
        df,
        "combined_overall_score",
        "Overall Score",
        "performance_by_tools_overall.png",
        output_dir,
    )
    _plot_single_performance_by_tool_count(
        df,
        "iou",
        "IoU",
        "performance_by_tools_iou.png",
        output_dir,
    )


def main():
    """Main execution."""
    setup_style()

    print("Loading design data (includes tool usage metrics)...")
    data = load_data()
    combined_design = get_combined_design_df(data)

    if combined_design is None or len(combined_design) == 0:
        print("No design data found. Please run extract_data.py first.")
        print("\nExample:")
        print("  python benchmarks/evaluations/extract_data.py \\")
        print("    --project YOUR_PROJECT \\")
        print("    --model openai:gpt-5.1 \\")
        print("    --problem beams2d")
        return

    # Check if tool usage data is available
    if "total_tools" not in combined_design.columns:
        print("\n⚠️  Tool usage data not found in design data.")
        print("Tool usage metrics (total_tools, unique_tools, tool_*) are not present.")
        print("This might mean:")
        print("  1. The tool_use scorer didn't output these fields")
        print("  2. The data was extracted before tool usage support was added")
        print("\nRe-run extract_data.py to get the latest data with tool usage.")
        return

    print(f"Loaded {len(combined_design)} design records with tool usage data")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("TOOL USAGE STATISTICS")
    print("=" * 60)
    print(f"Total examples: {len(combined_design)}")
    if "total_tools" in combined_design.columns:
        print(
            f"Average tools per example: {combined_design['total_tools'].mean():.2f} +/- {combined_design['total_tools'].std():.2f}"
        )
    if "unique_tools" in combined_design.columns:
        print(
            f"Average unique tools: {combined_design['unique_tools'].mean():.2f} +/- {combined_design['unique_tools'].std():.2f}"
        )
    print("=" * 60)

    # Generate plots
    print("\nGenerating tool usage visualizations...")

    plot_tool_usage_frequency(combined_design)
    plot_tool_usage_by_model(combined_design)
    plot_tool_heatmap_by_model(combined_design)
    plot_tool_heatmap_with_std(combined_design)

    # Correlation plots (design data already has performance metrics)
    print("\nGenerating correlation plots...")
    plot_tool_usage_vs_performance(combined_design, combined_design)
    plot_performance_distribution_by_tool_count(combined_design, combined_design)
    plot_tool_usage_delta_heatmap(combined_design)

    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()
