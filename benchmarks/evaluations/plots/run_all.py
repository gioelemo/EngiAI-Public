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

from utils import (
    get_combined_design_df,
    get_combined_global_df,
    get_output_dir,
    load_data,
)


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

    # Generate global metric plots
    print("\n" + "-" * 40)
    print("Generating global metric plots...")
    print("-" * 40)

    if combined_global is not None:
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

    # Generate design-level plots
    print("\n" + "-" * 40)
    print("Generating design-level plots...")
    print("-" * 40)

    if combined_design is not None:
        print(f"  Combined design metrics: {len(combined_design)} rows")

        # 5. Design quality distribution
        print("\n[5/6] Design quality distribution...")
        plot_design_quality(combined_design, "design_quality_distribution.png")

        # 6. IoU vs Objective
        print("\n[6/6] IoU vs Objective score...")
        plot_iou_vs_objective(combined_design, "iou_vs_objective.png")

    print("\n" + "=" * 60)
    print(f"DONE! All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
