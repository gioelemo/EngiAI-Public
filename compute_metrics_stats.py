"""Compute mean ± std for COG, MMD, RVC, and DPP metrics from CSV."""

import pandas as pd  # type: ignore[import-untyped]

# Read the CSV file
csv_file = "cgan_cnn_2d_beams2d_metrics.csv"
df = pd.read_csv(csv_file)

# Metrics to compute statistics for
# Note: 'viol' in the CSV corresponds to RVC (ratio of violated constraints)
metrics = {
    "COG": "cog",
    "MMD": "mmd",
    "RVC": "viol",
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
