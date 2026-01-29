#!/usr/bin/env python
"""
Check status of evaluation runs and display summary statistics.

Usage:
    python check_runs.py "openai:gpt-4o" beams2d
    python check_runs.py "anthropic:claude-3-5-sonnet" beams2d full no_rag
    python check_runs.py "openai:gpt-4o" beams2d full rag  # With MMORE
"""

import sys
from pathlib import Path

import pandas as pd


def check_run_status(
    model: str, problem: str, prompt_style: str = "full", rag_status: str = "no_rag"
) -> None:
    """Check which seeds have been completed and show statistics."""
    # Convert model name to safe filename format
    model_safe = model.replace("/", "_").replace(":", "_")

    # Construct results directory path
    results_dir = Path(
        f"benchmarks/evaluations/results/models/{model_safe}/{problem}/{prompt_style}/{rag_status}"
    )

    if not results_dir.exists():
        print(f"\n❌ No results directory found:")
        print(f"   {results_dir}")
        print(f"\nThis usually means no evaluations have been run yet for:")
        print(f"   Model: {model}")
        print(f"   Problem: {problem}")
        print(f"   Prompt style: {prompt_style}")
        print(f"   RAG status: {rag_status}")
        return

    # Check for design metrics CSV
    csv_file = results_dir / "output_quality_design_metrics.csv"
    if not csv_file.exists():
        print(f"\n⚠️  Results directory exists but no CSV found:")
        print(f"   {csv_file}")
        print(f"\nEvaluations may be in progress or failed.")
        return

    # Load results
    df = pd.read_csv(csv_file)

    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Model:        {model}")
    print(f"Problem:      {problem}")
    print(f"Prompt:       {prompt_style}")
    print(f"RAG:          {rag_status}")
    print(f"Location:     {results_dir}")
    print(f"{'='*70}")

    # Check if seed column exists
    if "seed" not in df.columns:
        print(f"\n✓ Total samples: {len(df)} (no seed tracking)")
        print(f"\nOverall metrics:")
        _print_metrics_summary(df)
        return

    # Analyze by seed
    seeds_completed = sorted(df["seed"].unique())
    num_samples_per_seed = df.groupby("seed").size()

    print(f"\n📊 RUNS COMPLETED")
    print(f"   Total seeds: {len(seeds_completed)}")
    print(f"   Seeds: {list(seeds_completed)}")
    print(f"\n   Samples per seed:")
    for seed, count in num_samples_per_seed.items():
        print(f"      Seed {seed}: {count:3d} samples")
    print(f"\n   Total samples across all seeds: {len(df)}")

    # Metrics by seed
    print(f"\n📈 METRICS BY SEED")
    print(f"{'='*70}")

    metrics_cols = [
        "iou",
        "pixel_accuracy",
        "constraint_score",
        "is_watertight",
        "mesh_repaired",
    ]
    available_metrics = [col for col in metrics_cols if col in df.columns]

    if available_metrics:
        seed_metrics = df.groupby("seed")[available_metrics].agg(["mean", "std"])
        print(seed_metrics.round(4))
    else:
        print("   No standard metrics found in CSV")

    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS (All Seeds Combined)")
    print(f"{'='*70}")
    _print_metrics_summary(df)

    # Watertightness analysis (new metrics from your implementation!)
    if "is_watertight" in df.columns:
        print(f"\n🔧 3D PRINTABILITY ANALYSIS (Watertightness)")
        print(f"{'='*70}")

        total_with_stl = df["is_watertight"].notna().sum()
        watertight_count = df["is_watertight"].sum()
        watertight_rate = watertight_count / total_with_stl if total_with_stl > 0 else 0

        print(f"   Designs with STL generated: {total_with_stl}")
        print(f"   Watertight meshes: {watertight_count} ({watertight_rate:.1%})")

        if "repair_attempted" in df.columns and "mesh_repaired" in df.columns:
            repair_attempted = df["repair_attempted"].sum()
            repair_success = df["mesh_repaired"].sum()
            print(
                f"   Repairs attempted: {repair_attempted} ({repair_attempted/total_with_stl:.1%})"
            )
            print(f"   Repairs successful: {repair_success}")

        # Watertightness by seed
        if len(seeds_completed) > 1:
            print(f"\n   Watertightness rate by seed:")
            wt_by_seed = df.groupby("seed")["is_watertight"].agg(["sum", "count"])
            wt_by_seed["rate"] = wt_by_seed["sum"] / wt_by_seed["count"]
            for seed, row in wt_by_seed.iterrows():
                print(
                    f"      Seed {seed}: {row['sum']:.0f}/{row['count']:.0f} ({row['rate']:.1%})"
                )

    # Check for global metrics
    global_csv = results_dir / "output_quality_global_metrics.csv"
    if global_csv.exists():
        print(f"\n🌍 GLOBAL METRICS (Distribution Analysis)")
        print(f"{'='*70}")
        global_df = pd.read_csv(global_csv)
        print(
            global_df[
                ["seed", "mean_iou", "mean_pixel_accuracy", "mmd", "dpp_diversity"]
            ].round(4)
        )


def _print_metrics_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for available metrics."""
    standard_metrics = [
        "iou",
        "pixel_accuracy",
        "constraint_score",
        "objective_score",
        "is_watertight",
    ]

    available_metrics = [col for col in standard_metrics if col in df.columns]

    if available_metrics:
        summary = df[available_metrics].agg(["mean", "std", "min", "max"])
        print(summary.round(4))
    else:
        print("   No standard metrics found")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python check_runs.py MODEL PROBLEM [PROMPT_STYLE] [RAG_STATUS]")
        print("\nExamples:")
        print('  python check_runs.py "openai:gpt-4o" beams2d')
        print('  python check_runs.py "openai:gpt-4o" beams2d full no_rag')
        print('  python check_runs.py "openai:gpt-4o" beams2d full rag')
        print('  python check_runs.py "anthropic:claude-3-5-sonnet" thermoelastic2d')
        sys.exit(1)

    model = sys.argv[1]
    problem = sys.argv[2]
    prompt_style = sys.argv[3] if len(sys.argv) > 3 else "full"
    rag_status = sys.argv[4] if len(sys.argv) > 4 else "no_rag"

    check_run_status(model, problem, prompt_style, rag_status)


if __name__ == "__main__":
    main()
