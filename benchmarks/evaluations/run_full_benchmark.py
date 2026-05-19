#!/usr/bin/env python
"""Run agent benchmark evaluation across multiple seeds.

This script wraps evaluate_agent.py to run evaluations across multiple seeds
with optional prompt generation.

Usage:
    # Run agent for seeds 1-5
    python benchmarks/evaluations/run_full_benchmark.py --seeds 1 2 3 4 5 --samples 10

    # Specify custom model
    python benchmarks/evaluations/run_full_benchmark.py --seeds 1 --samples 10 --model gpt-4o

    # Skip prompt generation (use existing prompts)
    python benchmarks/evaluations/run_full_benchmark.py --seeds 1 2 3 --samples 10 --skip-prompt-generation
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

# Paths configuration
CONDA_PYTHON = Path.home() / "miniforge3" / "envs" / "engiai" / "bin" / "python"

# Default output directory for benchmark results
# Structure: results/models/{model_name}/{problem}/{prompt_style}/{rag_status}/
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "evaluations" / "results"
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


def run_agent_evaluation(  # noqa: PLR0913
    problem: str,
    samples: int,
    seed: int,
    model: str | None,
    prompt_style: str = "full",
    scorers: str = "all",
    rag_mode: str = "no_rag",
    use_run_flag: bool = False,
) -> int:
    """Run agent evaluation for a specific seed or run.

    Args:
        rag_mode: One of "rag", "no_rag", or "empty_rag".
        use_run_flag: If True, pass ``seed`` as ``--run`` instead of ``--seed``.
            Use for problems with fixed prompts (HPC, RAG) where the seed is
            only a run identifier, not an optimization seed.
    """
    cmd = [
        str(CONDA_PYTHON),
        str(PROJECT_ROOT / "benchmarks" / "evaluations" / "evaluate_agent.py"),
        "--problem",
        problem,
        "--samples",
        str(samples),
        "--run" if use_run_flag else "--seed",
        str(seed),
        "--prompt-style",
        prompt_style,
        "--scorers",
        scorers,
    ]
    if model is not None:
        cmd.extend(["--model", model])
    _rag_flags = {"rag": "--mmore", "no_rag": "--no-mmore", "empty_rag": "--empty-rag"}
    cmd.append(_rag_flags.get(rag_mode, "--no-mmore"))

    # Set SKIP_MMORE env var to prevent health check warnings when RAG is disabled
    mmore_enabled = rag_mode in ("rag", "empty_rag")
    env = {"SKIP_MMORE": "false" if mmore_enabled else "true"}
    return run_command(cmd, cwd=PROJECT_ROOT, env=env)


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="Run agent benchmark evaluation across multiple seeds"
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
        choices=[
            "full",
            "natural",
            "workflow-random",
            "workflow-derived-params",
            "workflow-distractor",
            "workflow-conditional",
            "workflow-multi-export",
            "rag-eval",
            "hpc-train-cgan",
            "hpc-train-diff",
            "hpc-train-natural-cgan",
            "hpc-train-natural-diff",
        ],
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
        choices=["output_quality", "task_completion", "tool_use", "all"],
        help="Scorer set for agent evaluation (default: all)",
    )
    rag_group = parser.add_mutually_exclusive_group()
    rag_group.add_argument(
        "--mmore",
        dest="rag_mode",
        action="store_const",
        const="rag",
        help="Enable MMORE RAG system for document retrieval",
    )
    rag_group.add_argument(
        "--no-mmore",
        dest="rag_mode",
        action="store_const",
        const="no_rag",
        help="Disable MMORE RAG system (default)",
    )
    rag_group.add_argument(
        "--empty-rag",
        dest="rag_mode",
        action="store_const",
        const="empty_rag",
        help="RAG tools available but index is empty (control condition)",
    )
    parser.set_defaults(rag_mode="no_rag")
    parser.add_argument(
        "--skip-prompt-generation",
        action="store_true",
        help="Skip prompt generation (use existing prompts)",
    )
    args = parser.parse_args()

    # Problems without a HuggingFace dataset (RAG, HPC) have fixed prompts.
    # Use --run instead of --seed so Weave traces say "run_N" (repeated eval)
    # instead of "seed_N" (different dataset sample), and no "Use seed=N"
    # instruction is appended to the prompt.
    problem_config = PROBLEMS[args.problem]
    use_run_flag = not problem_config.dataset_name
    iter_label = "RUN" if use_run_flag else "SEED"

    model_name = args.model or config.llm_model

    print("=" * 60)
    print("AGENT BENCHMARK EVALUATION")
    print("=" * 60)
    print()
    print(f"Problem: {args.problem}")
    print(f"Seeds: {args.seeds}")
    print(f"Samples per seed: {args.samples}")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Model: {model_name}")
    _rag_labels = {"rag": "enabled", "no_rag": "disabled", "empty_rag": "empty index"}
    print(f"MMORE RAG: {_rag_labels.get(args.rag_mode, args.rag_mode)}")
    print(f"Scorers: {args.scorers}")
    print()

    # Track results
    results: dict[int, str] = {}
    failed_seeds: list[int] = []

    for seed in args.seeds:
        print()
        print("#" * 60)
        print(f"# {iter_label} {seed}")
        print("#" * 60)

        # Step 1: Generate prompts
        if not args.skip_prompt_generation:
            print(f"\n[{iter_label.title()} {seed}] Generating prompts...")
            ret = generate_prompts(
                args.problem,
                args.samples,
                seed,
                args.prompt_style,
            )
            if ret != 0:
                print(
                    f"Warning: Prompt generation failed for {iter_label.lower()} {seed}"
                )

        # Step 2: Run agent evaluation
        print(f"\n[{iter_label.title()} {seed}] Running agent evaluation...")
        ret = run_agent_evaluation(
            args.problem,
            args.samples,
            seed,
            args.model,
            args.prompt_style,
            args.scorers,
            args.rag_mode,
            use_run_flag=use_run_flag,
        )
        if ret == 0:
            results[seed] = "success"
        else:
            results[seed] = "failed"
            failed_seeds.append(seed)

    # Print summary
    print()
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print()

    success_count = len([s for s in results.values() if s == "success"])
    print(f"Agent evaluations: {success_count}/{len(args.seeds)} successful")
    if failed_seeds:
        print(f"  Failed seeds: {failed_seeds}")
    model_safe = model_name.replace("/", "_").replace(":", "_")
    rag_dir = args.rag_mode
    agent_results_dir = (
        MODELS_DIR / model_safe / args.problem / args.prompt_style / rag_dir
    )
    print(f"  Results saved to: {agent_results_dir}")

    print()
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
