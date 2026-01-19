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
    load_data,
    load_tool_usage_data,
    save_figure,
    setup_style,
)


def plot_tool_usage_frequency(tool_data, output_dir=None):
    """Plot frequency of each tool across all models.

    Args:
        tool_data: Combined tool usage DataFrame
        output_dir: Optional output directory for saving
    """
    if tool_data is None or len(tool_data) == 0:
        print("⚠️  No tool usage data available")
        return

    # Get all tool columns (columns starting with 'tool_')
    tool_columns = [col for col in tool_data.columns if col.startswith("tool_")]

    if not tool_columns:
        print("⚠️  No tool columns found in data")
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

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(tool_names, tool_counts, color=PLOT_STYLE["colors"]["GPT-4.1"])

    # Add count labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=9,
            weight="bold",
        )

    ax.set_xlabel("Total Usage Count")
    ax.set_ylabel("Tool Name")
    ax.set_title("Tool Usage Frequency Across All Models")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    return save_figure(fig, "tool_usage_frequency.png", output_dir)


def plot_tool_usage_by_model(tool_data, output_dir=None):
    """Plot tool usage comparison across models.

    Args:
        tool_data: Combined tool usage DataFrame
        output_dir: Optional output directory for saving
    """
    if tool_data is None or len(tool_data) == 0:
        print("⚠️  No tool usage data available")
        return

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Average total tools
    x = np.arange(len(model_stats))
    width = 0.35

    colors = [PLOT_STYLE["colors"][model] for model in model_stats["model"]]

    bars1 = ax1.bar(
        x,
        model_stats["avg_total"],
        width,
        yerr=model_stats["std_total"],
        color=colors,
        alpha=PLOT_STYLE["alpha"],
        capsize=5,
    )

    ax1.set_xlabel("Model / Problem")
    ax1.set_ylabel("Average Total Tool Calls")
    ax1.set_title("Average Total Tool Calls per Example")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{row['model']}\n({row['problem']})" for _, row in model_stats.iterrows()],
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
            fontsize=9,
        )

    # Plot 2: Average unique tools
    bars2 = ax2.bar(
        x,
        model_stats["avg_unique"],
        width,
        yerr=model_stats["std_unique"],
        color=colors,
        alpha=PLOT_STYLE["alpha"],
        capsize=5,
    )

    ax2.set_xlabel("Model / Problem")
    ax2.set_ylabel("Average Unique Tools Used")
    ax2.set_title("Average Unique Tools per Example")
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"{row['model']}\n({row['problem']})" for _, row in model_stats.iterrows()],
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
            fontsize=9,
        )

    plt.tight_layout()

    return save_figure(fig, "tool_usage_by_model.png", output_dir)


def plot_tool_usage_vs_performance(tool_data, design_data, output_dir=None):
    """Plot correlation between tool usage and performance.

    Args:
        tool_data: Combined tool usage DataFrame
        design_data: Combined design metrics DataFrame
        output_dir: Optional output directory for saving
    """
    if tool_data is None or design_data is None:
        print("⚠️  Missing data for correlation plot")
        return

    # Merge tool usage with design metrics
    merged = tool_data.merge(
        design_data,
        on=["example_id", "model_id", "problem_id"],
        how="inner",
        suffixes=("_tool", "_design"),
    )

    if len(merged) == 0:
        print("⚠️  No matching examples found between tool usage and design metrics")
        return

    # Create scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    metrics = [
        ("total_tools", "overall_score", "Total Tool Calls", "Overall Score"),
        ("unique_tools", "overall_score", "Unique Tools Used", "Overall Score"),
        ("total_tools", "iou", "Total Tool Calls", "IoU"),
        ("unique_tools", "iou", "Unique Tools Used", "IoU"),
    ]

    for ax, (x_col, y_col, x_label, y_label) in zip(axes.flat, metrics, strict=False):
        # Plot by model
        for model in merged["model_tool"].unique():
            model_data = merged[merged["model_tool"] == model]
            ax.scatter(
                model_data[x_col],
                model_data[y_col],
                label=model,
                marker=PLOT_STYLE["markers"].get(model, "o"),
                color=PLOT_STYLE["colors"].get(model, "gray"),
                alpha=PLOT_STYLE["alpha"],
                s=PLOT_STYLE["marker_size"],
            )

        # Add trend line
        min_points_for_trend = 2
        if len(merged) > min_points_for_trend:
            z = np.polyfit(merged[x_col], merged[y_col], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(merged[x_col].min(), merged[x_col].max(), 100)
            ax.plot(x_trend, p(x_trend), "k--", alpha=0.3, linewidth=1)

            # Calculate correlation
            corr = merged[x_col].corr(merged[y_col])
            ax.text(
                0.05,
                0.95,
                f"r = {corr:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} vs {x_label}")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    return save_figure(fig, "tool_usage_vs_performance.png", output_dir)


def plot_tool_heatmap_by_model(tool_data, output_dir=None):
    """Create heatmap showing which tools are used by which models.

    Args:
        tool_data: Combined tool usage DataFrame
        output_dir: Optional output directory for saving
    """
    if tool_data is None or len(tool_data) == 0:
        print("⚠️  No tool usage data available")
        return

    # Get tool columns
    tool_columns = [col for col in tool_data.columns if col.startswith("tool_")]

    if not tool_columns:
        print("⚠️  No tool columns found")
        return

    # Calculate average usage per model
    model_tool_usage = (
        tool_data.groupby("model")[tool_columns].sum().T
    )  # Transpose so tools are rows

    # Clean up tool names
    model_tool_usage.index = [
        idx.replace("tool_", "") for idx in model_tool_usage.index
    ]

    # Filter out tools with zero usage
    model_tool_usage = model_tool_usage[(model_tool_usage.sum(axis=1) > 0)].sort_values(
        by=model_tool_usage.columns.tolist(), ascending=False
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, max(8, len(model_tool_usage) * 0.3)))

    sns.heatmap(
        model_tool_usage,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Total Usage Count"},
        linewidths=0.5,
        ax=ax,
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Tool Name")
    ax.set_title("Tool Usage Heatmap by Model")

    plt.tight_layout()

    return save_figure(fig, "tool_usage_heatmap.png", output_dir)


def main():
    """Main execution."""
    setup_style()

    print("Loading tool usage data...")
    tool_data = load_tool_usage_data()
    combined_tools = get_combined_tool_usage_df(tool_data)

    if combined_tools is None or len(combined_tools) == 0:
        print("❌ No tool usage data found. Please run extract_data.py first.")
        print("\nExample:")
        print("  python benchmarks/evaluations/extract_data.py \\")
        print("    --project YOUR_PROJECT \\")
        print("    --model openai:gpt-5.1 \\")
        print("    --problem beams2d")
        return

    print(f"✅ Loaded {len(combined_tools)} tool usage records")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("TOOL USAGE STATISTICS")
    print("=" * 60)
    print(f"Total examples: {len(combined_tools)}")
    print(
        f"Average tools per example: {combined_tools['total_tools'].mean():.2f} ± {combined_tools['total_tools'].std():.2f}"
    )
    print(
        f"Average unique tools: {combined_tools['unique_tools'].mean():.2f} ± {combined_tools['unique_tools'].std():.2f}"
    )
    print("=" * 60)

    # Generate plots
    print("\n📊 Generating tool usage visualizations...")

    plot_tool_usage_frequency(combined_tools)
    plot_tool_usage_by_model(combined_tools)
    plot_tool_heatmap_by_model(combined_tools)

    # Try to correlate with performance if design data is available
    print("\n📊 Attempting to correlate tool usage with performance...")
    data = load_data()

    # Try to get combined design data
    design_keys = [
        "gpt_beams_design",
        "gpt_photonics_design",
        "gpt5_beams_design",
        "gpt5_photonics_design",
    ]
    design_dfs = [data[key] for key in design_keys if key in data]

    if design_dfs:
        combined_design = pd.concat(design_dfs, ignore_index=True)
        plot_tool_usage_vs_performance(combined_tools, combined_design)
    else:
        print("⚠️  No design metrics found, skipping correlation plots")

    print("\n✅ All visualizations complete!")


if __name__ == "__main__":
    main()
