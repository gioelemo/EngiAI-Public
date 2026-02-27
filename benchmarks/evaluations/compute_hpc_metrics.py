"""
Compute and aggregate HPC training metrics from evaluate_cgan_2d.py CSV outputs.

This script reads the CSV files produced by evaluate_cgan_2d.py during HPC training
evaluations and aggregates them into a single global_metrics.json file, following
the same pattern as compute_global_metrics.py for standard workflows.

It also supports loading a pre-computed baseline CSV (from the official EngiOpt model)
for side-by-side comparison.

Usage:
    # 1. Pre-compute baseline (one-time, per seed):
    python -m engiopt.cgan_2d.evaluate_cgan_2d --problem-id beams2d --seed 1 --n-samples 50 \
        --output-csv benchmarks/problems/hpc_train_beams2d/data/baseline/seed1_metrics.csv

    # 2. Run agent evaluation (evaluate_agent.py handles this)

    # 3. Compare agent vs baseline:
    python benchmarks/evaluations/compute_hpc_metrics.py \
        --problem hpc_train_beams2d --prompt-style hpc-train --rag-status no_rag \
        --baseline-dir benchmarks/problems/hpc_train_beams2d/data/baseline/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config  # noqa: E402

# Metrics produced by evaluate_cgan_2d.py
EVAL_METRICS = ["IOG", "COG", "FOG", "MMD", "DPP", "viol"]

# Default baseline directory (resolved dynamically per problem)
DEFAULT_HPC_PROBLEM = "hpc_train_beams2d"


def get_baseline_dir(problem: str) -> Path:
    """Construct default baseline directory from problem name."""
    return PROJECT_ROOT / "benchmarks" / "problems" / problem / "data" / "baseline"


BASELINE_DIR = get_baseline_dir(DEFAULT_HPC_PROBLEM)


def find_eval_csvs(search_dir: Path) -> list[Path]:
    """Find all evaluation CSV outputs in a directory tree.

    Looks for files matching common patterns from evaluate_{algorithm}.py.
    """
    patterns = [
        "evaluate_*_metrics.csv",
        "*_seed*_metrics.csv",
        "cgan_cnn_2d_*_metrics.csv",
        "diffusion_2d_cond_*_metrics.csv",
        "seed*_metrics.csv",
    ]

    found: list[Path] = []
    for pattern in patterns:
        found.extend(search_dir.rglob(pattern))

    return sorted(set(found))


def load_eval_csvs(csv_paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate evaluation CSV files."""
    dfs = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            df["source_file"] = str(path)
            dfs.append(df)
            print(f"  Loaded: {path} ({len(df)} rows)")
        except Exception as e:
            print(f"  Failed to load {path}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def _resolve_metric_col(df: pd.DataFrame, metric: str) -> str | None:
    """Find the actual column name for a metric (case-insensitive)."""
    if metric in df.columns:
        return metric
    lower = metric.lower()
    if lower in df.columns:
        return lower
    return None


def compute_per_seed_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute per-seed metrics from a DataFrame."""
    per_seed: list[dict[str, Any]] = []

    seeds = sorted(df["seed"].unique()) if "seed" in df.columns else [0]

    for seed in seeds:
        seed_df = df[df["seed"] == seed] if "seed" in df.columns else df
        seed_metrics: dict[str, Any] = {
            "seed": int(seed),
            "n_samples": (
                int(seed_df["n_samples"].iloc[0])
                if "n_samples" in seed_df.columns
                else 0
            ),
        }

        for metric in EVAL_METRICS:
            col = _resolve_metric_col(seed_df, metric)
            if col is not None:
                values = seed_df[col].dropna()
                if len(values) > 0:
                    seed_metrics[metric] = float(np.mean(values))

        per_seed.append(seed_metrics)

    return per_seed


def compute_global_averages(
    per_seed: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute global averages across seeds."""
    global_avg: dict[str, float] = {}
    for metric in EVAL_METRICS:
        values = [
            sm[metric]
            for sm in per_seed
            if metric in sm and isinstance(sm[metric], (int, float))
        ]
        if values:
            global_avg[f"avg_{metric}"] = float(np.mean(values))
            if len(values) > 1:
                global_avg[f"std_{metric}"] = float(np.std(values, ddof=1))

    return global_avg


def print_comparison_table(
    agent_seeds: list[dict[str, Any]],
    baseline_seeds: list[dict[str, Any]],
) -> None:
    """Print a side-by-side comparison table of agent vs baseline metrics."""
    # Build lookup by seed
    baseline_by_seed = {s["seed"]: s for s in baseline_seeds}

    header = f"{'Seed':>6} | {'Metric':>6} | {'Agent':>12} | {'Baseline':>12} | {'Delta':>12}"
    print(header)
    print("-" * len(header))

    for agent_s in agent_seeds:
        seed = agent_s["seed"]
        base_s = baseline_by_seed.get(seed, {})

        for metric in EVAL_METRICS:
            agent_val = agent_s.get(metric)
            base_val = base_s.get(metric)

            a_str = f"{agent_val:.6g}" if isinstance(agent_val, (int, float)) else "N/A"
            b_str = f"{base_val:.6g}" if isinstance(base_val, (int, float)) else "N/A"

            if isinstance(agent_val, (int, float)) and isinstance(
                base_val, (int, float)
            ):
                delta = agent_val - base_val
                d_str = f"{delta:+.6g}"
            else:
                d_str = "---"

            print(f"{seed:>6} | {metric:>6} | {a_str:>12} | {b_str:>12} | {d_str:>12}")

        print("-" * len(header))


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """Save aggregated metrics to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved global metrics to: {output_path}")


def main() -> None:  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(
        description="Aggregate HPC training evaluation metrics and compare against baseline"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID (e.g. openai:gpt-5-mini). Defaults to config.llm_model",
    )
    parser.add_argument(
        "--problem",
        default="hpc_train_beams2d",
        help="Problem type (default: hpc_train_beams2d)",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="hpc-train-cgan",
        choices=[
            "hpc-train-cgan",
            "hpc-train-diff",
            "hpc-train-natural-cgan",
            "hpc-train-natural-diff",
        ],
        help="Prompt style used (default: hpc-train)",
    )
    parser.add_argument(
        "--rag-status",
        type=str,
        default="no_rag",
        choices=["rag", "no_rag", "empty_rag"],
        help="RAG status (default: no_rag)",
    )
    parser.add_argument(
        "--csv-dir",
        help="Directory to search for agent evaluation CSVs (default: auto-construct)",
    )
    parser.add_argument(
        "--baseline-dir",
        help="Directory containing baseline CSVs from official EngiOpt model "
        f"(default: {BASELINE_DIR})",
    )
    parser.add_argument(
        "--output",
        help="Output path for global metrics JSON (default: auto-construct)",
    )
    args = parser.parse_args()

    model = args.model if args.model is not None else config.llm_model

    # Construct search directory for agent evaluation CSVs
    if args.csv_dir:
        search_dir = Path(args.csv_dir)
    else:
        model_dir = model.replace("/", "_").replace(":", "_")
        search_dir = Path(
            f"benchmarks/evaluations/results/models/{model_dir}/"
            f"{args.problem}/{args.prompt_style}/{args.rag_status}"
        )

    baseline_dir = (
        Path(args.baseline_dir)
        if args.baseline_dir
        else get_baseline_dir(args.problem)
    )

    print("=" * 60)
    print("HPC TRAINING METRICS — AGENT vs BASELINE")
    print("=" * 60)
    print(f"Model:        {model}")
    print(f"Problem:      {args.problem}")
    print(f"Agent CSVs:   {search_dir}")
    print(f"Baseline dir: {baseline_dir}")
    print("=" * 60)

    # ── Load agent evaluation CSVs ──────────────────────────────────────────
    print("\n[1] Loading agent evaluation CSVs...")
    agent_csvs = find_eval_csvs(search_dir)

    if not agent_csvs:
        print("  No agent evaluation CSVs found.")
        print(f"  Searched: {search_dir}")
        print("  Run the agent evaluation first, or specify --csv-dir")
        return

    # Filter out baseline CSVs that may have been picked up
    agent_csvs = [p for p in agent_csvs if "baseline" not in str(p)]
    if not agent_csvs:
        print("  Only baseline CSVs found — no agent results.")
        print("  Run the agent evaluation first, or specify --csv-dir")
        return

    agent_df = load_eval_csvs(agent_csvs)
    if agent_df.empty:
        print("  No data loaded from agent CSVs")
        return

    agent_seeds = compute_per_seed_metrics(agent_df)
    agent_global = compute_global_averages(agent_seeds)

    # ── Load baseline CSVs ──────────────────────────────────────────────────
    baseline_seeds: list[dict[str, Any]] = []
    baseline_global: dict[str, float] = {}

    print("\n[2] Loading baseline CSVs...")
    if baseline_dir.exists():
        baseline_csvs = find_eval_csvs(baseline_dir)
        if baseline_csvs:
            baseline_df = load_eval_csvs(baseline_csvs)
            if not baseline_df.empty:
                baseline_seeds = compute_per_seed_metrics(baseline_df)
                baseline_global = compute_global_averages(baseline_seeds)
            else:
                print("  No data loaded from baseline CSVs")
        else:
            print("  No baseline CSVs found in:", baseline_dir)
            print("  To generate baseline, run for each seed:")
            print("    python -m engiopt.cgan_2d.evaluate_cgan_2d \\")
            print("      --problem-id beams2d --seed <SEED> --n-samples 50 \\")
            print(f"      --output-csv {baseline_dir}/seed<SEED>_metrics.csv")
    else:
        print(f"  Baseline directory does not exist: {baseline_dir}")
        print("  Skipping baseline comparison.")

    # ── Comparison ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)

    if baseline_seeds:
        print("AGENT vs BASELINE COMPARISON")
        print("=" * 60)
        print_comparison_table(agent_seeds, baseline_seeds)

        print("\nGlobal averages:")
        for metric in EVAL_METRICS:
            a_key = f"avg_{metric}"
            a_val = agent_global.get(a_key)
            b_val = baseline_global.get(a_key)
            a_str = f"{a_val:.6g}" if a_val is not None else "N/A"
            b_str = f"{b_val:.6g}" if b_val is not None else "N/A"
            if a_val is not None and b_val is not None:
                delta = a_val - b_val
                print(
                    f"  {metric}: agent={a_str}  baseline={b_str}  delta={delta:+.6g}"
                )
            else:
                print(f"  {metric}: agent={a_str}  baseline={b_str}")
    else:
        print("AGENT METRICS (no baseline available)")
        print("=" * 60)
        for seed_m in agent_seeds:
            seed = seed_m["seed"]
            print(f"\n  Seed {seed}:")
            for metric in EVAL_METRICS:
                val = seed_m.get(metric)
                if isinstance(val, (int, float)):
                    print(f"    {metric}: {val:.6g}")

        if agent_global:
            print("\n  Global averages:")
            for metric in EVAL_METRICS:
                key = f"avg_{metric}"
                if key in agent_global:
                    print(f"    {metric}: {agent_global[key]:.6g}")

    # ── Save results ────────────────────────────────────────────────────────
    results = {
        "problem": args.problem,
        "model": model,
        "prompt_style": args.prompt_style,
        "agent": {
            "per_seed_metrics": agent_seeds,
            "global_averages": agent_global,
            "n_seeds": len(agent_seeds),
        },
    }
    if baseline_seeds:
        results["baseline"] = {
            "per_seed_metrics": baseline_seeds,
            "global_averages": baseline_global,
            "n_seeds": len(baseline_seeds),
        }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = search_dir / "hpc_global_metrics.json"

    save_metrics(results, output_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
