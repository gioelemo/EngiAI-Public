#!/usr/bin/env python3
"""
Summary Statistics Table Generator

Creates summary statistics table in CSV and LaTeX formats.
Shows mean ± std for DPP, MMD, FOG, RVC.
"""

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.evaluations.plots.utils import (
    get_combined_global_df,
    get_output_dir,
    load_data,
)


def create_summary_table(combined_df, output_dir=None):
    """Create summary statistics table."""
    if output_dir is None:
        output_dir = get_output_dir()

    metrics = ["dpp", "mmd", "fog", "rvc"]
    summary_rows = []

    for (model, problem), group in combined_df.groupby(["model", "problem"]):
        row = {"Model": model, "Problem": problem, "N": len(group)}

        for metric in metrics:
            values = group[metric].dropna()
            if len(values) > 0:
                mean = values.mean()
                std = values.std()
                row[f"{metric.upper()} (mean)"] = mean
                row[f"{metric.upper()} (std)"] = std
                row[f"{metric.upper()}"] = f"{mean:.4f} ± {std:.4f}"

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Print to console
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS TABLE")
    print("=" * 80)
    print("\nMetric Definitions:")
    print("  DPP: Diversity (higher = more diverse designs)")
    print("  MMD: Distribution Match (lower = closer to ground truth)")
    print("  FOG: Final Optimality Gap (lower = better quality)")
    print("  RVC: Ratio Violated Constraints (lower = better)")
    print("\n")

    display_cols = ["Model", "Problem", "N", "DPP", "MMD", "FOG", "RVC"]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[display_cols].to_string(index=False))

    # Save CSV
    csv_path = output_dir / "summary_statistics.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Save LaTeX
    latex_path = output_dir / "summary_statistics.tex"
    latex_cols = [c for c in display_cols if c in summary_df.columns]
    latex_df = summary_df[latex_cols]

    latex_str = latex_df.to_latex(
        index=False,
        escape=False,
        column_format="l" * len(latex_cols),
        caption="Diversity vs Quality Trade-off Metrics",
        label="tab:diversity_quality",
    )

    with latex_path.open("w") as f:
        f.write(latex_str)
    print(f"Saved LaTeX: {latex_path}")

    return summary_df


def main():
    """Generate summary statistics table."""
    print("Loading data...")
    data = load_data()
    combined_df = get_combined_global_df(data)

    if combined_df is not None:
        create_summary_table(combined_df)
    else:
        print("No data available")


if __name__ == "__main__":
    main()
