"""
Generate tool usage visualizations for agent evaluations.

This script creates plots showing:
- Tool usage frequency across models
- Tool usage distribution per problem
- Correlation between tool usage and performance
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import (
    PLOT_STYLE,
    get_combined_tool_usage_df,
    get_model_style,
    load_data,
    load_tool_usage_data,
    save_figure,
    setup_style,
)


def plot_tool_usage_frequency(tool_data, output_dir=None):
    """Plot frequency of each tool across all models (NeurIPS format).

    Args:
        tool_data: Combined tool usage DataFrame
        output_dir: Optional output directory for saving
    """
    if tool_data is None or len(tool_data) == 0:
        print("No tool usage data available")
        return

    setup_style()

    # Get all tool columns (columns starting with 'tool_')
    tool_columns = [col for col in tool_data.columns if col.startswith("tool_")]

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

    # Use a neutral color from the colorblind-friendly palette
    default_color = "#0072B2"  # Blue
    bars = ax.barh(tool_names, tool_counts, color=default_color, height=0.7)

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
    """Plot tool usage comparison across models (NeurIPS format).

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


def plot_tool_usage_vs_performance(tool_data, design_data, output_dir=None):
    """Plot correlation between tool usage and performance (NeurIPS format).

    Args:
        tool_data: Combined tool usage DataFrame
        design_data: Combined design metrics DataFrame
        output_dir: Optional output directory for saving
    """
    if tool_data is None or design_data is None:
        print("Missing data for correlation plot")
        return

    setup_style()
    font_sizes = PLOT_STYLE["font_sizes"]

    # Merge tool usage with design metrics
    merged = tool_data.merge(
        design_data,
        on=["example_id", "model_id", "problem_id"],
        how="inner",
        suffixes=("_tool", "_design"),
    )

    if len(merged) == 0:
        print("No matching examples found between tool usage and design metrics")
        return

    # Create scatter plots
    fig, axes = plt.subplots(
        2, 2, figsize=PLOT_STYLE["figsize_full_width_tall"], constrained_layout=True
    )

    metrics = [
        ("total_tools", "overall_score", "Total Tools", "Score"),
        ("unique_tools", "overall_score", "Unique Tools", "Score"),
        ("total_tools", "iou", "Total Tools", "IoU"),
        ("unique_tools", "iou", "Unique Tools", "IoU"),
    ]

    # Get dynamic styles for models
    models = list(merged["model_tool"].unique())
    model_styles = get_model_style(models)

    for ax, (x_col, y_col, x_label, y_label) in zip(axes.flat, metrics, strict=False):
        # Plot by model
        for model in models:
            model_data = merged[merged["model_tool"] == model]
            style = model_styles[model]
            ax.scatter(
                model_data[x_col],
                model_data[y_col],
                label=model,
                marker=style["marker"],
                color=style["color"],
                alpha=PLOT_STYLE["alpha"],
                s=PLOT_STYLE["marker_size"],
            )

        # Add trend line
        min_points_for_trend = 2
        if len(merged) > min_points_for_trend:
            z = np.polyfit(merged[x_col], merged[y_col], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(merged[x_col].min(), merged[x_col].max(), 100)
            ax.plot(x_trend, p(x_trend), "k--", alpha=0.3, linewidth=0.5)

            # Calculate correlation
            corr = merged[x_col].corr(merged[y_col])
            ax.text(
                0.05,
                0.95,
                f"$r$ = {corr:.2f}",
                transform=ax.transAxes,
                fontsize=font_sizes["annotation"],
                verticalalignment="top",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "0.7",
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

    # 1. Get tool columns
    tool_columns = [col for col in tool_data.columns if col.startswith("tool_")]

    # 1. Calculate average usage per example for each model
    # Dividing by the count of rows (examples) for each model
    model_tool_usage = tool_data.groupby("model")[tool_columns].mean().T * 100
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
        # Updated label to clarify this is the usage rate
        cbar_kws={"label": r"Usage Rate (\%)", "shrink": 0.8},
        linewidths=0.3,
        ax=ax,
        annot_kws={"size": font_sizes["annotation"]},
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    return save_figure(fig, "tool_usage_heatmap_pct.png", output_dir)


def plot_tool_usage_delta_heatmap(tool_data, output_dir=None):
    """Plot how much each model deviates from the average tool usage."""
    # 1. Calculate usage rate per model
    tool_columns = [col for col in tool_data.columns if col.startswith("tool_")]
    model_usage = tool_data.groupby("model")[tool_columns].mean() * 100

    # 2. Calculate the average usage across ALL models
    avg_usage = model_usage.mean()

    # 3. Calculate Delta (Difference from Mean)
    delta_usage = (model_usage - avg_usage).T
    delta_usage.index = [idx.replace("tool_", "") for idx in delta_usage.index]

    # Abbreviate model names as discussed
    delta_usage.columns = [
        c.replace("-instruct-2507-q8-0", "-Inst") for c in delta_usage.columns
    ]

    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col_tall"],  # Use standard tall format
        constrained_layout=True,
    )

    # Use a diverging colormap (RdBu_r: Red is more, Blue is less)
    sns.heatmap(
        delta_usage,
        annot=True,
        fmt=".1f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": r"$\Delta$ Usage Rate (\%)", "shrink": 0.8},  # LaTeX math
        annot_kws={"size": PLOT_STYLE["font_sizes"]["annotation"]},
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    save_figure(
        fig, "tool_usage_delta_heatmap.png", output_dir
    )  # Uses bbox_inches='tight'


def plot_performance_distribution_by_tool_count(
    combined_tools, combined_design, output_dir=None
):
    """Show the distribution of performance metrics relative to tools used."""
    # Merge performance with tool usage counts
    df = combined_design.merge(
        combined_tools[["example_id", "model", "total_tools", "unique_tools"]],
        on=["example_id", "model"],
    )

    fig, axes = plt.subplots(
        1, 2, figsize=PLOT_STYLE["figsize_full_width"], constrained_layout=True
    )
    metrics = ["score", "iou"]
    labels = ["Score", "IoU"]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # FIX: Added hue="total_tools" and legend=False to resolve the FutureWarnings
        sns.violinplot(
            data=df,
            x="total_tools",
            y=metric,
            hue="total_tools",
            legend=False,
            inner="quart",
            palette="Pastel1",
            linewidth=0.7,
            ax=ax,
            cut=0,
        )

        # Overlay Jittered Points
        sns.stripplot(
            data=df,
            x="total_tools",
            y=metric,
            color="black",
            size=2,
            alpha=0.3,
            jitter=True,
            ax=ax,
        )

        # Add mean and std annotations
        font_sizes = PLOT_STYLE["font_sizes"]
        for j, tool_count in enumerate(sorted(df["total_tools"].unique())):
            subset = df[df["total_tools"] == tool_count][metric]
            if len(subset) > 0:
                stats_text = f"$\\mu$={subset.mean():.2f}\n$\\sigma$={subset.std():.2f}"
                # Place annotation at the top of the data range (not axis limit)
                y_max = subset.max()
                y_min = subset.min()
                y_pos = y_min + (y_max - y_min) * 0.5  # Center of data range
                ax.annotate(
                    stats_text,
                    xy=(j, y_pos),
                    ha="center",
                    va="center",
                    fontsize=font_sizes["annotation"],
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "alpha": 0.85,
                        "edgecolor": "0.8",
                    },
                )

        ax.set_ylabel(labels[i], fontsize=PLOT_STYLE["font_sizes"]["axes_label"])
        ax.set_xlabel(
            "Total Tools Used", fontsize=PLOT_STYLE["font_sizes"]["axes_label"]
        )
        sns.despine(ax=ax)

    save_figure(fig, "performance_stats_distribution.png", output_dir)


def main():
    """Main execution."""
    setup_style()

    print("Loading tool usage data...")
    tool_data = load_tool_usage_data()
    combined_tools = get_combined_tool_usage_df(tool_data)

    if combined_tools is None or len(combined_tools) == 0:
        print("No tool usage data found. Please run extract_data.py first.")
        print("\nExample:")
        print("  python benchmarks/evaluations/extract_data.py \\")
        print("    --project YOUR_PROJECT \\")
        print("    --model openai:gpt-5.1 \\")
        print("    --problem beams2d")
        return

    print(f"Loaded {len(combined_tools)} tool usage records")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("TOOL USAGE STATISTICS")
    print("=" * 60)
    print(f"Total examples: {len(combined_tools)}")
    print(
        f"Average tools per example: {combined_tools['total_tools'].mean():.2f} +/- {combined_tools['total_tools'].std():.2f}"
    )
    print(
        f"Average unique tools: {combined_tools['unique_tools'].mean():.2f} +/- {combined_tools['unique_tools'].std():.2f}"
    )
    print("=" * 60)

    # Generate plots
    print("\nGenerating tool usage visualizations...")

    plot_tool_usage_frequency(combined_tools)
    plot_tool_usage_by_model(combined_tools)
    plot_tool_heatmap_by_model(combined_tools)

    # Try to correlate with performance if design data is available
    print("\nAttempting to correlate tool usage with performance...")
    data = load_data()

    # Try to get combined design data
    design_dfs = [df for key, df in data.items() if key.endswith("_design")]

    if design_dfs:
        combined_design = pd.concat(design_dfs, ignore_index=True)
        plot_tool_usage_vs_performance(combined_tools, combined_design)
        plot_performance_distribution_by_tool_count(combined_tools, combined_design)

    else:
        print(f"No design metrics found. Available keys: {list(data.keys())}")

    plot_tool_usage_delta_heatmap(combined_tools)

    print("\nAll visualizations complete!")


if __name__ == "__main__":
    main()
