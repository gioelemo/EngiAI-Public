#!/usr/bin/env python
"""Run full benchmark evaluation for both agent and CGAN baselines.

This script provides a unified way to run evaluations for both the engineering
agent and the CGAN CNN 2D baseline, ensuring they use identical samples from
the dataset for fair comparison.

Usage:
    # Run both agent and CGAN for seeds 1-5
    python benchmarks/evaluations/run_full_benchmark.py --seeds 1 2 3 4 5 --samples 10

    # Run only CGAN baseline
    python benchmarks/evaluations/run_full_benchmark.py --seeds 1 2 3 --samples 10 --cgan-only

    # Run only agent evaluation
    python benchmarks/evaluations/run_full_benchmark.py --seeds 1 2 3 --samples 10 --agent-only

    # Specify custom model for agent
    python benchmarks/evaluations/run_full_benchmark.py --seeds 1 --samples 10 --model gpt-4o
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path to import config
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from benchmarks.shared.problem_registry import PROBLEMS  # noqa: E402
from config import config  # noqa: E402

# Paths configuration (PROJECT_ROOT already defined above)
ENGIOPT_ROOT = Path.home() / "EngiOpt"
CONDA_PYTHON = (
    Path.home() / "miniforge3" / "envs" / "engineer-assistant" / "bin" / "python"
)

# Default output directory for all benchmark results
# Structure:
#   results/baselines/{baseline_type}/{problem}/                          - for baselines
#   results/models/{model_name}/{problem}/{prompt_style}/{rag_status}/    - for LLM agents
#   where rag_status is "rag" (--mmore) or "no_rag" (default)
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "evaluations" / "results"
BASELINES_DIR = RESULTS_DIR / "baselines"
MODELS_DIR = RESULTS_DIR / "models"


def run_command(
    cmd: list[str], cwd: Path | None = None, env: dict | None = None
) -> int:
    """Run a command and return the exit code."""
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    result = subprocess.run(cmd, cwd=cwd, env=merged_env, check=False)
    return result.returncode


def generate_prompts(
    problem: str,
    samples: int,
    seed: int,
    prompt_style: str = "full",
) -> int:
    """Generate prompts for a specific seed."""
    cmd = [
        str(CONDA_PYTHON),
        str(PROJECT_ROOT / "benchmarks" / "problems" / problem / "generate_prompts.py"),
        "--samples",
        str(samples),
        "--seed",
        str(seed),
        "--style",
        prompt_style,
    ]
    return run_command(cmd, cwd=PROJECT_ROOT)


def run_cgan_evaluation(
    problem: str,
    samples: int,
    seed: int,
    output_csv: Path,
    wandb_entity: str | None = None,
) -> int:
    """Run CGAN CNN 2D evaluation for a specific seed."""
    cmd = [
        str(CONDA_PYTHON),
        "-m",
        "engiopt.cgan_cnn_2d.evaluate_cgan_cnn_2d",
        "--problem_id",
        problem,
        "--seed",
        str(seed),
        "--n-samples",
        str(samples),
        "--output_csv",
        str(output_csv),
    ]
    if wandb_entity:
        cmd.extend(["--wandb_entity", wandb_entity])

    return run_command(cmd, cwd=ENGIOPT_ROOT)


def run_agent_evaluation(  # noqa: PLR0913
    problem: str,
    samples: int,
    seed: int,
    model: str | None,
    prompt_style: str = "full",
    scorers: str = "all",
    mmore_enabled: bool = False,
) -> int:
    """Run agent evaluation for a specific seed."""
    cmd = [
        str(CONDA_PYTHON),
        str(PROJECT_ROOT / "benchmarks" / "evaluations" / "evaluate_agent.py"),
        "--problem",
        problem,
        "--samples",
        str(samples),
        "--seed",
        str(seed),
        "--prompt-style",
        prompt_style,
        "--scorers",
        scorers,
    ]
    if model is not None:
        cmd.extend(["--model", model])
    if mmore_enabled:
        cmd.append("--mmore")
    return run_command(cmd, cwd=PROJECT_ROOT)


def extract_agent_data(
    problem: str,
    model: str,
    prompt_style: str,
    rag_status: str,
    seed: int,
) -> int:
    """Extract design data from Weave to JSON for a specific seed."""
    cmd = [
        str(CONDA_PYTHON),
        str(PROJECT_ROOT / "benchmarks" / "evaluations" / "extract_data.py"),
        "--problem",
        problem,
        "--model",
        model,
        "--prompt-style",
        prompt_style,
        "--rag-status",
        rag_status,
        "--seed",
        str(seed),
    ]
    return run_command(cmd, cwd=PROJECT_ROOT)


def compute_agent_metrics(
    problem: str,
    model: str,
    prompt_style: str,
    rag_status: str,
    seed: int,
) -> int:
    """Compute global metrics from extracted data for a specific seed."""
    cmd = [
        str(CONDA_PYTHON),
        str(PROJECT_ROOT / "benchmarks" / "evaluations" / "compute_global_metrics.py"),
        "--problem",
        problem,
        "--model",
        model,
        "--prompt-style",
        prompt_style,
        "--rag-status",
        rag_status,
        "--seed",
        str(seed),
    ]
    return run_command(cmd, cwd=PROJECT_ROOT)


def main() -> None:  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(
        description="Run full benchmark evaluation for agent and CGAN baselines"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="beams2d",
        choices=list(PROBLEMS.keys()),
        help="Problem type to evaluate (default: beams2d)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Random seeds to run (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples per seed (default: 10)",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="full",
        choices=["full", "approximate", "natural", "workflow"],
        help="Prompt style for agent evaluation (default: full)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for agent evaluation (defaults to config.llm_model)",
    )
    parser.add_argument(
        "--scorers",
        type=str,
        default="all",
        choices=["output_quality", "engibench", "all", "task_completion", "tool_use"],
        help="Scorer set for agent evaluation (default: all)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="engibench",
        help="Wandb entity for CGAN model artifacts (default: engibench)",
    )
    parser.add_argument(
        "--mmore",
        dest="mmore_enabled",
        action="store_true",
        default=False,
        help="Enable MMORE RAG system for document retrieval (default: disabled)",
    )
    parser.add_argument(
        "--no-mmore",
        dest="mmore_enabled",
        action="store_false",
        help="Disable MMORE RAG system (default)",
    )
    parser.add_argument(
        "--cgan-only",
        action="store_true",
        help="Run only CGAN evaluation",
    )
    parser.add_argument(
        "--agent-only",
        action="store_true",
        help="Run only agent evaluation",
    )
    parser.add_argument(
        "--skip-prompt-generation",
        action="store_true",
        help="Skip prompt generation (use existing prompts)",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.cgan_only and args.agent_only:
        print("Error: Cannot specify both --cgan-only and --agent-only")
        sys.exit(1)

    run_cgan = not args.agent_only
    run_agent = not args.cgan_only

    # Setup output directories
    # CGAN: results/baselines/cgan_cnn_2d/{problem}/
    cgan_results_dir = BASELINES_DIR / "cgan_cnn_2d" / args.problem
    cgan_results_dir.mkdir(parents=True, exist_ok=True)
    cgan_output_csv = cgan_results_dir / "output_quality_global_metrics.csv"

    print("=" * 60)
    print("FULL BENCHMARK EVALUATION")
    print("=" * 60)
    print()
    print(f"Problem: {args.problem}")
    print(f"Seeds: {args.seeds}")
    print(f"Samples per seed: {args.samples}")
    print(f"Prompt style: {args.prompt_style}")
    if run_agent:
        print(f"Agent model: {args.model or '(from config.llm_model)'}")
        print(f"MMORE RAG: {'enabled' if args.mmore_enabled else 'disabled'}")
        print(f"Scorers: {args.scorers}")
    if run_cgan:
        print(f"CGAN results: {cgan_output_csv}")
    print()

    # Track results
    results: dict[str, dict[int, str]] = {"cgan": {}, "agent": {}}
    failed_seeds: dict[str, list[int]] = {"cgan": [], "agent": []}

    for seed in args.seeds:
        print()
        print("#" * 60)
        print(f"# SEED {seed}")
        print("#" * 60)

        # Step 1: Generate prompts (needed for agent evaluation)
        if run_agent and not args.skip_prompt_generation:
            print(f"\n[Seed {seed}] Generating prompts...")
            ret = generate_prompts(
                args.problem,
                args.samples,
                seed,
                args.prompt_style,
            )
            if ret != 0:
                print(f"Warning: Prompt generation failed for seed {seed}")

        # Step 2: Run CGAN evaluation
        if run_cgan:
            print(f"\n[Seed {seed}] Running CGAN evaluation...")
            ret = run_cgan_evaluation(
                args.problem,
                args.samples,
                seed,
                cgan_output_csv,
                args.wandb_entity,
            )
            if ret == 0:
                results["cgan"][seed] = "success"
            else:
                results["cgan"][seed] = "failed"
                failed_seeds["cgan"].append(seed)

        # Step 3: Run agent evaluation
        if run_agent:
            print(f"\n[Seed {seed}] Running agent evaluation...")
            ret = run_agent_evaluation(
                args.problem,
                args.samples,
                seed,
                args.model,
                args.prompt_style,
                args.scorers,
                args.mmore_enabled,
            )
            if ret == 0:
                results["agent"][seed] = "success"

                # Step 4: Extract data from Weave to JSON
                print(f"\n[Seed {seed}] Extracting design data from Weave...")
                model_name = args.model if args.model is not None else config.llm_model
                rag_status = "rag" if args.mmore_enabled else "no_rag"
                ret = extract_agent_data(
                    args.problem,
                    model_name,
                    args.prompt_style,
                    rag_status,
                    seed,
                )
                if ret != 0:
                    print(f"Warning: Data extraction failed for seed {seed}")

                # Step 5: Compute global metrics from extracted data
                print(f"\n[Seed {seed}] Computing global metrics...")
                ret = compute_agent_metrics(
                    args.problem,
                    model_name,
                    args.prompt_style,
                    rag_status,
                    seed,
                )
                if ret != 0:
                    print(f"Warning: Metrics computation failed for seed {seed}")
            else:
                results["agent"][seed] = "failed"
                failed_seeds["agent"].append(seed)

    # Print summary
    print()
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print()

    if run_cgan:
        cgan_success = len([s for s in results["cgan"].values() if s == "success"])
        print(f"CGAN evaluations: {cgan_success}/{len(args.seeds)} successful")
        if failed_seeds["cgan"]:
            print(f"  Failed seeds: {failed_seeds['cgan']}")
        print(f"  Results saved to: {cgan_output_csv}")

    if run_agent:
        agent_success = len([s for s in results["agent"].values() if s == "success"])
        print(f"Agent evaluations: {agent_success}/{len(args.seeds)} successful")
        if failed_seeds["agent"]:
            print(f"  Failed seeds: {failed_seeds['agent']}")
        # Get actual model name (from args or config)
        model_name = args.model if args.model is not None else config.llm_model
        model_safe = model_name.replace("/", "_").replace(":", "_")
        rag_dir = "rag" if args.mmore_enabled else "no_rag"
        agent_results_dir = (
            MODELS_DIR / model_safe / args.problem / args.prompt_style / rag_dir
        )
        print(f"  Results saved to: {agent_results_dir}")

    print()
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
