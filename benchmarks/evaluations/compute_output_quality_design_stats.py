"""Compute statistics for per-design output quality metrics across seeds.

This script analyzes output_quality_design_metrics.csv files to compute mean ± std for
design-level metrics across multiple optimization seeds.

Usage:
    python compute_output_quality_design_stats.py <csv_file>

Example:
    python compute_output_quality_design_stats.py results/openai_gpt-4.1/beams2d/output_quality_design_metrics.csv
"""

import sys

import pandas as pd  # type: ignore[import-untyped]

# Check for CSV file argument
MIN_ARGS = 2
if len(sys.argv) < MIN_ARGS:
    print("Usage: python compute_output_quality_design_stats.py <csv_file>")
    print(
        "Example: python compute_output_quality_design_stats.py results/openai_gpt-4.1/beams2d/output_quality_design_metrics.csv"
    )
    sys.exit(1)

# Read the CSV file
csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

# Core metrics to analyze
core_metrics = [
    "overall_score",
    "iou",
    "pixel_accuracy",
    "mse",
    "constraint_score",
    "objective_score",
]

# Get all numeric columns (in case there are problem-specific metrics)
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
# Remove non-metric columns
exclude_cols = ["seed", "example_id"]
metric_cols = [col for col in numeric_cols if col not in exclude_cols]

print("=" * 60)
print(f"Per-Design Metrics Statistics for {csv_file}")
print("=" * 60)
print()

# Get number of seeds and examples
n_seeds = df["seed"].nunique()
n_examples = df["example_id"].nunique() if "example_id" in df.columns else len(df)

print(f"Number of seeds: {n_seeds}")
print(f"Number of examples per seed: {n_examples // n_seeds if n_seeds > 0 else 0}")
print(f"Total rows: {len(df)}")
print()

# Global statistics (across all seeds and examples)
print("GLOBAL STATISTICS (across all seeds and examples)")
print("-" * 60)
for metric in core_metrics:
    if metric in df.columns:
        mean = df[metric].mean()
        std = df[metric].std()
        print(f"{metric:20s}: {mean:.6f} ± {std:.6f}")
    else:
        print(f"{metric:20s}: Column not found")
print()

# Per-seed aggregated statistics
print("PER-SEED AGGREGATED STATISTICS")
print("-" * 60)
for metric in core_metrics:
    if metric in df.columns:
        # Compute mean for each seed, then mean/std of those means
        per_seed_means = df.groupby("seed")[metric].mean()
        mean_of_means = per_seed_means.mean()
        std_of_means = per_seed_means.std()
        print(f"{metric:20s}: {mean_of_means:.6f} ± {std_of_means:.6f}")
    else:
        print(f"{metric:20s}: Column not found")
print()

# Per-example statistics (variance across seeds for each example)
if "example_id" in df.columns and n_seeds > 1:
    print("PER-EXAMPLE VARIANCE (how consistent are results across seeds?)")
    print("-" * 60)
    for metric in core_metrics:
        if metric in df.columns:
            # Compute std for each example across seeds
            per_example_stds = df.groupby("example_id")[metric].std()
            mean_std = per_example_stds.mean()
            print(f"{metric:20s}: mean std = {mean_std:.6f}")
        else:
            print(f"{metric:20s}: Column not found")
    print()

# Additional metrics if present
other_metrics = [col for col in metric_cols if col not in core_metrics]
if other_metrics:
    print("ADDITIONAL PROBLEM-SPECIFIC METRICS")
    print("-" * 60)
    for metric in other_metrics:
        mean = df[metric].mean()
        std = df[metric].std()
        print(f"{metric:20s}: {mean:.6f} ± {std:.6f}")
    print()

# Summary statistics by seed
if n_seeds > 1:
    print("SUMMARY BY SEED")
    print("-" * 60)
    for seed in sorted(df["seed"].unique()):
        seed_df = df[df["seed"] == seed]
        print(f"Seed {seed}:")
        for metric in core_metrics[:3]:  # Show first 3 core metrics
            if metric in seed_df.columns:
                mean = seed_df[metric].mean()
                print(f"  {metric:18s}: {mean:.6f}")
        print()

print("=" * 60)
