"""Compute mean ± std for global output quality metrics (COG, MMD, RVC, DPP) from CSV.

Usage:
    python compute_output_quality_global_stats.py <csv_file>

Example:
    python compute_output_quality_global_stats.py results/openai_gpt-4.1/beams2d/output_quality_global_metrics.csv
"""

import sys

import pandas as pd  # type: ignore[import-untyped]

# Check for CSV file argument
MIN_ARGS = 2
if len(sys.argv) < MIN_ARGS:
    print("Usage: python compute_output_quality_global_stats.py <csv_file>")
    print(
        "Example: python compute_output_quality_global_stats.py results/openai_gpt-4.1/beams2d/output_quality_global_metrics.csv"
    )
    sys.exit(1)

# Read the CSV file
csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

# Metrics to compute statistics for
# Note: Support both 'rvc' and 'viol' column names
metrics = {
    "COG": "cog",
    "MMD": "mmd",
    "RVC": "rvc" if "rvc" in df.columns else "viol",
    "DPP": "dpp",
}

print("=" * 60)
print(f"Metrics Statistics for {csv_file}")
print("=" * 60)
print()
print(f"Number of runs: {len(df)}")
print()

# Compute and display statistics
for metric_name, column_name in metrics.items():
    if column_name in df.columns:
        mean = df[column_name].mean()
        std = df[column_name].std()
        print(f"{metric_name:4s}: {mean:.6e} ± {std:.6e}")
    else:
        print(f"{metric_name:4s}: Column '{column_name}' not found")

print()
print("=" * 60)
