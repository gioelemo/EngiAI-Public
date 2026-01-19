#!/usr/bin/env python3
"""
Generate All Figures

Run this script to generate all visualizations at once.
Individual plots can also be run separately.

Usage:
    python run_all.py           # Generate all figures
    python plot_dpp_vs_fog.py   # Generate single figure
"""

from generate_summary_table import create_summary_table
from plot_design_quality import plot_design_quality
from plot_dpp_vs_fog import plot_dpp_vs_fog
from plot_dpp_vs_mmd import plot_dpp_vs_mmd
from plot_iou_vs_objective import plot_iou_vs_objective
from plot_metrics_comparison import plot_metrics_comparison
from plot_tool_usage import (
    plot_tool_heatmap_by_model,
    plot_tool_usage_by_model,
    plot_tool_usage_frequency,
    plot_tool_usage_vs_performance,
)

from utils import (
    get_combined_design_df,
    get_combined_global_df,
    get_combined_tool_usage_df,
    get_output_dir,
    load_data,
    load_tool_usage_data,
)


def _generate_global_plots(combined_global, output_dir):
    """Generate global metric plots."""
    print("\n" + "-" * 40)
    print("Generating global metric plots...")
    print("-" * 40)

    if combined_global is None:
        return

    print(f"  Combined global metrics: {len(combined_global)} rows")

    # 1. DPP vs FOG
    print("\n[1/6] DPP vs FOG scatter...")
    plot_dpp_vs_fog(combined_global, "dpp_vs_fog.png")

    # 2. DPP vs MMD
    print("\n[2/6] DPP vs MMD scatter...")
    plot_dpp_vs_mmd(combined_global, "dpp_vs_mmd.png")

    # 3. Metrics comparison bars
    print("\n[3/6] Metrics comparison bars...")
    plot_metrics_comparison(combined_global, "metrics_comparison.png")

    # 4. Summary table
    print("\n[4/6] Summary statistics table...")
    create_summary_table(combined_global, output_dir)


def _generate_design_plots(combined_design):
    """Generate design-level plots."""
    print("\n" + "-" * 40)
    print("Generating design-level plots...")
    print("-" * 40)

    if combined_design is None:
        return

    print(f"  Combined design metrics: {len(combined_design)} rows")

    # 5. Design quality distribution
    print("\n[5/6] Design quality distribution...")
    plot_design_quality(combined_design, "design_quality_distribution.png")

    # 6. IoU vs Objective
    print("\n[6/6] IoU vs Objective score...")
    plot_iou_vs_objective(combined_design, "iou_vs_objective.png")


def _generate_tool_usage_plots(combined_tools, combined_design, output_dir):
    """Generate tool usage plots."""
    print("\n" + "-" * 40)
    print("Generating tool usage plots...")
    print("-" * 40)

    if combined_tools is None or len(combined_tools) == 0:
        print("  ⚠️  No tool usage data found. Run extract_tool_usage.py first.")
        return

    print(f"  Combined tool usage: {len(combined_tools)} records")

    # 7. Tool usage frequency
    print("\n[7/10] Tool usage frequency...")
    plot_tool_usage_frequency(combined_tools, output_dir)

    # 8. Tool usage by model
    print("\n[8/10] Tool usage by model...")
    plot_tool_usage_by_model(combined_tools, output_dir)

    # 9. Tool usage heatmap
    print("\n[9/10] Tool usage heatmap...")
    plot_tool_heatmap_by_model(combined_tools, output_dir)

    # 10. Tool usage vs performance
    if combined_design is not None:
        print("\n[10/10] Tool usage vs performance...")
        plot_tool_usage_vs_performance(combined_tools, combined_design, output_dir)


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("Diversity vs Quality Analysis")
    print("=" * 60)

    print("\nLoading data...")
    data = load_data()

    if not data:
        print("ERROR: No data found! Check file paths.")
        return

    # Data summary
    print("\nData loaded:")
    for key, df in data.items():
        print(f"  {key}: {len(df)} rows")

    # Combined dataframes
    combined_global = get_combined_global_df(data)
    combined_design = get_combined_design_df(data)

    output_dir = get_output_dir()
    print(f"\nOutput directory: {output_dir}")

    # Generate all plots using helper functions
    _generate_global_plots(combined_global, output_dir)
    _generate_design_plots(combined_design)

    # Load and generate tool usage plots
    tool_data = load_tool_usage_data()
    combined_tools = get_combined_tool_usage_df(tool_data)
    _generate_tool_usage_plots(combined_tools, combined_design, output_dir)

    print("\n" + "=" * 60)
    print(f"DONE! All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
