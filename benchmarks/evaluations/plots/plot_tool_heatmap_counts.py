"""
Heatmap of raw tool call counts (mean +/- std) per model.

Unlike the percentage-based heatmaps in plot_tool_usage.py, this script
shows the absolute average number of times each tool was called per sample.

Usage:
    python plot_tool_heatmap_counts.py                              # Standalone
    # Or called from run_all.py as part of the full pipeline
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import (
    PLOT_STYLE,
    get_combined_design_df,
    load_data,
    save_figure,
    setup_style,
)


def plot_tool_heatmap_counts(tool_data, output_dir=None):
    """Create heatmap showing raw tool call counts (mean +/- std) by model.

    Args:
        tool_data: Combined design DataFrame with tool_* columns
        output_dir: Optional output directory for saving
    """
    if tool_data is None or len(tool_data) == 0:
        print("No tool usage data available")
        return

    setup_style()
    font_sizes = PLOT_STYLE["font_sizes"]

    # Get tool columns (exclude computed scores)
    exclude_metrics = {"tool_efficiency_score", "tool_completion_score"}
    tool_columns = [
        col
        for col in tool_data.columns
        if col.startswith("tool_") and col not in exclude_metrics
    ]

    if not tool_columns:
        print("No tool columns found in data")
        return

    # Calculate mean and std of raw counts per model
    model_tool_mean = (
        tool_data.groupby("model")[tool_columns].apply(lambda x: x.fillna(0).mean()).T
    )
    model_tool_std = (
        tool_data.groupby("model")[tool_columns].apply(lambda x: x.fillna(0).std()).T
    )

    # Abbreviate model names
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
    model_tool_mean = model_tool_mean[model_tool_mean.sum(axis=1) > 0].sort_values(
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
        fmt="",
        cmap="YlOrRd",
        cbar_kws={"label": "Avg. Calls per Sample", "shrink": 0.8},
        linewidths=0.3,
        ax=ax,
        annot_kws={"size": font_sizes["annotation"] - 1},
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    return save_figure(fig, "tool_usage_heatmap_counts.png", output_dir)


def main():
    """Standalone execution."""
    setup_style()

    print("Loading design data...")
    data = load_data()
    combined_design = get_combined_design_df(data)

    if combined_design is None or len(combined_design) == 0:
        print("No design data found. Run extract_data.py first.")
        return

    print(f"Loaded {len(combined_design)} design records")
    plot_tool_heatmap_counts(combined_design)
    print("Done!")


if __name__ == "__main__":
    main()
