#!/usr/bin/env python
"""Compare benchmark results between agent and CGAN baselines.

This script loads results from both evaluation pipelines and produces
comparison tables and summary statistics.

Usage:
    python benchmarks/evaluations/compare_results.py --problem beams2d
    python benchmarks/evaluations/compare_results.py --problem beams2d --model gpt-4o
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Paths configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "evaluations" / "results"
BASELINES_DIR = RESULTS_DIR / "baselines"
MODELS_DIR = RESULTS_DIR / "models"

# Add project root to path to import config
sys.path.insert(0, str(PROJECT_ROOT))
from config import config  # noqa: E402

# Metrics to compare (these should be present in both CSVs)
GLOBAL_METRICS = ["mmd", "dpp", "rvc", "iog", "cog", "fog"]


def load_cgan_results(problem: str) -> pd.DataFrame | None:
    """Load CGAN evaluation results.

    Checks new location (baselines/cgan_cnn_2d/{problem}/) first, then legacy.

    Args:
        problem: Problem type (e.g., "beams2d")
    """
    # Try new location first: baselines/cgan_cnn_2d/{problem}/
    csv_path = (
        BASELINES_DIR / "cgan_cnn_2d" / problem / "output_quality_global_metrics.csv"
    )
    if not csv_path.exists():
        # Fall back to old legacy location
        csv_path = (
            RESULTS_DIR / "cgan_cnn_2d" / problem / "output_quality_global_metrics.csv"
        )
    if not csv_path.exists():
        print("Warning: CGAN results not found")
        return None
    return pd.read_csv(csv_path)


def load_agent_results(problem: str, model: str) -> pd.DataFrame | None:
    """Load agent evaluation results.

    Checks new location (models/) first, then legacy location.
    """
    model_safe = model.replace("/", "_").replace(":", "_")
    # Try new location first
    csv_path = MODELS_DIR / model_safe / problem / "output_quality_global_metrics.csv"
    if not csv_path.exists():
        # Fall back to legacy location
        csv_path = (
            RESULTS_DIR / model_safe / problem / "output_quality_global_metrics.csv"
        )
    if not csv_path.exists():
        print(f"Warning: Agent results not found for {model}")
        return None
    return pd.read_csv(csv_path)


def compute_statistics(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Compute mean and std for each metric."""
    stats = []
    for metric in metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                stats.append(
                    {
                        "metric": metric,
                        "mean": values.mean(),
                        "std": values.std(),
                        "min": values.min(),
                        "max": values.max(),
                        "n": len(values),
                    }
                )
    return pd.DataFrame(stats)


def format_metric(mean: float, std: float, scientific: bool = True) -> str:
    """Format metric value with uncertainty."""
    if scientific:
        return f"{mean:.2e} ± {std:.2e}"
    return f"{mean:.4f} ± {std:.4f}"


def compare_results(
    cgan_df: pd.DataFrame | None,
    agent_df: pd.DataFrame | None,
    metrics: list[str],
) -> None:
    """Compare and print results from both methods."""
    print("\n" + "=" * 80)
    print("METRIC COMPARISON")
    print("=" * 80)

    # Compute statistics
    cgan_stats = (
        compute_statistics(cgan_df, metrics) if cgan_df is not None else pd.DataFrame()
    )
    agent_stats = (
        compute_statistics(agent_df, metrics)
        if agent_df is not None
        else pd.DataFrame()
    )

    # Print header
    print(f"\n{'Metric':<10} {'CGAN CNN 2D':<25} {'Agent':<25} {'Better':<10}")
    print("-" * 80)

    for metric in metrics:
        # Handle empty DataFrames (when results are not available)
        cgan_row = (
            cgan_stats[cgan_stats["metric"] == metric]
            if "metric" in cgan_stats.columns
            else pd.DataFrame()
        )
        agent_row = (
            agent_stats[agent_stats["metric"] == metric]
            if "metric" in agent_stats.columns
            else pd.DataFrame()
        )

        cgan_str = "N/A"
        agent_str = "N/A"
        better = "-"
        cgan_mean = None
        agent_mean = None

        if not cgan_row.empty:
            cgan_mean = cgan_row["mean"].to_numpy()[0]
            cgan_std = cgan_row["std"].to_numpy()[0]
            scientific = metric in ["mmd", "dpp", "iog", "cog", "fog"]
            cgan_str = format_metric(cgan_mean, cgan_std, scientific)

        if not agent_row.empty:
            agent_mean = agent_row["mean"].to_numpy()[0]
            agent_std = agent_row["std"].to_numpy()[0]
            scientific = metric in ["mmd", "dpp", "iog", "cog", "fog"]
            agent_str = format_metric(agent_mean, agent_std, scientific)

        # Determine which is better (lower is better for all these metrics)
        if cgan_mean is not None and agent_mean is not None:
            if cgan_mean < agent_mean:
                better = "CGAN"
            elif agent_mean < cgan_mean:
                better = "Agent"
            else:
                better = "Tie"

        print(f"{metric:<10} {cgan_str:<25} {agent_str:<25} {better:<10}")

    print("-" * 80)

    # Print sample counts
    if cgan_df is not None:
        print(f"\nCGAN: {len(cgan_df)} evaluation runs")
    if agent_df is not None:
        print(f"Agent: {len(agent_df)} evaluation runs")


def print_detailed_results(df: pd.DataFrame, name: str, metrics: list[str]) -> None:
    """Print detailed per-seed results."""
    print(f"\n{name} Results (per seed):")
    print("-" * 60)

    cols_to_show = ["seed"] + [m for m in metrics if m in df.columns]
    if "seed" in df.columns:
        print(df[cols_to_show].to_string(index=False))
    else:
        print(df[[m for m in metrics if m in df.columns]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument(
        "--problem",
        type=str,
        default="beams2d",
        choices=["beams2d", "photonics2d", "thermoelastic2d"],
        help="Problem type (default: beams2d)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Agent model to compare (defaults to config.llm_model)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-seed results",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Save comparison to CSV file",
    )
    args = parser.parse_args()

    # Use config.llm_model if no model specified
    model_name = args.model if args.model is not None else config.llm_model

    print("=" * 80)
    print("BENCHMARK RESULTS COMPARISON")
    print("=" * 80)
    print(f"\nProblem: {args.problem}")
    print(f"Agent model: {model_name}")

    # Load results
    cgan_df = load_cgan_results(args.problem)
    agent_df = load_agent_results(args.problem, model_name)

    if cgan_df is None and agent_df is None:
        print("\nNo results found. Run evaluations first:")
        print("  python benchmarks/evaluations/run_full_benchmark.py --seeds 1 2 3 4 5")
        return

    # Compare results
    compare_results(cgan_df, agent_df, GLOBAL_METRICS)

    # Detailed results
    if args.detailed:
        if cgan_df is not None:
            print_detailed_results(cgan_df, "CGAN CNN 2D", GLOBAL_METRICS)
        if agent_df is not None:
            print_detailed_results(agent_df, f"Agent ({model_name})", GLOBAL_METRICS)

    # Save comparison to CSV
    if args.output_csv:
        comparison_rows = []
        cgan_stats = (
            compute_statistics(cgan_df, GLOBAL_METRICS)
            if cgan_df is not None
            else pd.DataFrame()
        )
        agent_stats = (
            compute_statistics(agent_df, GLOBAL_METRICS)
            if agent_df is not None
            else pd.DataFrame()
        )

        for metric in GLOBAL_METRICS:
            row = {"metric": metric, "problem": args.problem}

            # Handle empty DataFrames (when results are not available)
            if "metric" in cgan_stats.columns:
                cgan_row = cgan_stats[cgan_stats["metric"] == metric]
                if not cgan_row.empty:
                    row["cgan_mean"] = cgan_row["mean"].to_numpy()[0]
                    row["cgan_std"] = cgan_row["std"].to_numpy()[0]

            if "metric" in agent_stats.columns:
                agent_row = agent_stats[agent_stats["metric"] == metric]
                if not agent_row.empty:
                    row["agent_mean"] = agent_row["mean"].to_numpy()[0]
                    row["agent_std"] = agent_row["std"].to_numpy()[0]
                    row["agent_model"] = model_name

            comparison_rows.append(row)

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv(args.output_csv, index=False)
        print(f"\nComparison saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
