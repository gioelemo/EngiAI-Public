"""
Unified agent evaluation script for benchmarking across different problem types.

This script uses Weave's evaluation framework to track and visualize agent performance
on engineering design tasks across multiple problem types and LLM models.

Usage:
    python evaluate_agent.py --problem beams2d --model gpt-4o --samples 5
    python evaluate_agent.py --problem thermoelastic2d --model claude-3-5-sonnet --samples 10

Reference: https://docs.wandb.ai/weave/guides/core-types/evaluations
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, TypedDict

import weave
from langchain_core.messages import HumanMessage

# Suppress Weave serialization warnings (non-critical, caused by ModelMetaclass in LangChain)
logging.getLogger("langchain_core.callbacks.manager").setLevel(logging.ERROR)

# Set SKIP_MCP to avoid Prusa MCP server connection issues during evaluation
os.environ["SKIP_MCP"] = "true"

# Set SKIP_MMORE to avoid MMORE Docker container requirement during evaluation
os.environ["SKIP_MMORE"] = "true"

# Add project root to path to import src modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import EngiBench scorers
from benchmarks.shared.engibench_scorers import (  # noqa: E402
    compute_global_metrics,  # For global metrics after evaluation
    score_design_extracted,  # Lightweight scorer for engibench mode
)
from benchmarks.shared.generic_scorer import score_design_generic  # noqa: E402
from benchmarks.shared.problem_registry import PROBLEMS  # noqa: E402
from config import config  # noqa: E402
from src.agents.supervisor_agent import SupervisorAgent  # noqa: E402
from src.tools.engibench import clear_session_state, set_session_id  # noqa: E402
from src.utils.weave_integration import init_weave  # noqa: E402


class ProblemConfig(TypedDict):
    """Configuration for a problem type."""

    dataset_name: str
    prompt_file: str
    scorers: list[Any]


# Problem-specific configurations
# NOTE: The --scorers flag controls which metrics are computed:
# - "generic": Only per-design metrics from generic scorer
# - "engibench": Per-design metrics + global metrics (MMD, DPP, RVC, IOG, COG, FOG)
# - "all": Same as engibench (for backward compatibility)
# Build PROBLEM_CONFIGS from the central problem registry to avoid duplication
PROBLEM_CONFIGS: dict[str, ProblemConfig] = {
    name: {
        "dataset_name": problem.dataset_name,
        "prompt_file": problem.prompt_file_template.replace(
            "{problem}", name
        ),  # Replace placeholder
        "scorers": [score_design_generic],  # All problems use generic scorer
    }
    for name, problem in PROBLEMS.items()
}

# Configure logger for this module
logger = logging.getLogger(__name__)


class EngineeringAgent(weave.Model):
    """Weave Model wrapper for the multi-agent supervisor system."""

    model_name: str
    temperature: float = 0.7
    problem_type: str = "beams2d"

    @weave.op()
    def predict(self, prompt: str) -> dict[str, Any]:
        """
        Generate a response to an engineering design prompt using the multi-agent system.

        Args:
            prompt: User prompt for engineering design

        Returns:
            Dictionary with response and metadata
        """
        # Generate unique thread_id for this evaluation
        thread_id = f"eval_{uuid.uuid4().hex[:8]}"

        # Set session ID for state isolation - prevents data corruption in batch evaluations
        set_session_id(thread_id)

        try:
            # Initialize the supervisor agent with the configured model
            # Use eval_mode=True to reduce token costs with minimal prompts
            supervisor = SupervisorAgent(
                model_name=self.model_name,
                temperature=self.temperature,
                eval_mode=True,
            )

            # Convert prompt to message format
            messages = [HumanMessage(content=prompt)]
            state = {"messages": messages}

            config_dict = {"configurable": {"thread_id": thread_id}}

            # Invoke the supervisor agent
            result = supervisor.invoke(state, config_dict)

            # Extract the final response from messages
            final_message = result["messages"][-1]
            response_content = final_message.content

            # Debug logging: Extract and log tool calls to see what config is being used
            tool_calls_info = [
                {
                    "name": tool_call.get("name", "unknown"),
                    "args": tool_call.get("args", {}),
                }
                for msg in result["messages"]
                if hasattr(msg, "tool_calls") and msg.tool_calls
                for tool_call in msg.tool_calls
            ]

            # Log optimize_design calls to see what constraints are being passed
            for tc in tool_calls_info:
                if tc["name"] == "optimize_design":
                    # Check both 'constraints' (new) and 'config' (old) parameter names
                    constraints_used = tc["args"].get(
                        "constraints", tc["args"].get("config", None)
                    )
                    logger.debug(
                        "optimize_design called with constraints: %s", constraints_used
                    )
                    if constraints_used is None:
                        logger.warning(
                            "NO CONSTRAINTS - agent is not passing constraint parameters!"
                        )

            return {
                "response": response_content,
                "model": self.model_name,
                "response_length": len(response_content),
                "agent_type": getattr(final_message, "name", "unknown"),
                "messages": result[
                    "messages"
                ],  # Include full message history for design extraction
                "tool_calls_info": tool_calls_info,  # Debug info
            }
        finally:
            # Clean up session state to prevent memory leaks in batch evaluations
            clear_session_state(thread_id)


def prepare_evaluation_dataset(
    prompts: list[dict[str, Any]],
    sample_size: int,
    problem_type: str,
    dataset_name: str,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Prepare prompts for Weave evaluation format.

    Args:
        prompts: List of prompt dictionaries
        sample_size: Number of samples to include
        problem_type: Type of problem being evaluated
        dataset_name: Name of the HuggingFace dataset for ground truth
        seed: Optional seed to use for all prompts in this evaluation

    Returns:
        List of evaluation examples in Weave format
    """
    eval_dataset = []

    for i, prompt_data in enumerate(prompts[:sample_size]):
        prompt = prompt_data["prompt"]

        # Add seed instruction if provided
        if seed is not None:
            prompt = (
                f"{prompt}\n\n"
                f"IMPORTANT: Use seed={seed} when calling the optimize_design tool."
            )

        eval_dataset.append(
            {
                "prompt": prompt,
                "conditions": prompt_data["conditions"],
                "metadata": {
                    **prompt_data.get("metadata", {}),
                    "example_id": prompt_data.get("example_id", i),
                    "dataset_split": prompt_data.get("dataset_split", "test"),
                    "problem_type": problem_type,
                    "dataset_name": dataset_name,
                    "seed": seed,  # Track which seed was used
                },
                "target": prompt_data.get("target", {}),
            }
        )

    return eval_dataset


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate engineering agent on design tasks"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="beams2d",
        choices=list(PROBLEM_CONFIGS.keys()),
        help="Problem type to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model to use (defaults to config.llm_model)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Model temperature (defaults to config.llm_temperature)",
    )
    parser.add_argument(
        "--scorers",
        type=str,
        default="generic",
        choices=["generic", "engibench", "all"],
        help=(
            "Metrics to compute: "
            "'generic' (per-design metrics only), "
            "'engibench' or 'all' (per-design + global metrics: MMD, DPP, RVC, optimality gaps)"
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to use for prompts (default: test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for optimization (e.g., 1, 2, 3). Run multiple times with different seeds to collect statistics.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV file to save metrics (will append if file exists). Default: benchmarks/evaluations/results/{model}/{problem}/metrics.csv",
    )
    return parser.parse_args()


def load_prompts(problem: str, prompt_file_name: str) -> list[dict[str, Any]] | None:
    """Load prompts from problem-specific directory.

    Args:
        problem: Problem type
        prompt_file_name: Name of the prompt file

    Returns:
        List of prompts or None if file not found
    """
    prompt_file = (
        Path(__file__).parent.parent
        / "problems"
        / problem
        / "data"
        / "generated"
        / prompt_file_name
    )
    print(f"📂 Loading prompts from: {prompt_file}")

    if not prompt_file.exists():
        print(f"❌ Error: File not found: {prompt_file}")
        print(f"Please run generate_prompts.py for {problem} first.")
        return None

    with prompt_file.open() as f:
        return json.load(f)


def get_or_create_dataset(
    eval_dataset: list[dict[str, Any]], dataset_name: str, num_samples: int
) -> weave.Dataset:
    """Get existing dataset or create new one if needed.

    Args:
        eval_dataset: Prepared evaluation dataset
        dataset_name: Name for the Weave dataset
        num_samples: Expected number of samples

    Returns:
        Weave Dataset object
    """
    try:
        dataset = weave.ref(dataset_name).get()
        if len(dataset.rows) == num_samples:
            print(
                f"📦 Using existing evaluation dataset from Weave ({len(dataset.rows)} samples)"
            )
        else:
            print(
                f"⚠️  Existing dataset has {len(dataset.rows)} samples, need {num_samples}. Recreating..."
            )
            dataset = weave.Dataset(name=dataset_name, rows=eval_dataset)  # type: ignore[arg-type]
            weave.publish(dataset)
            print("📦 Published new evaluation dataset to Weave")
    except Exception:
        dataset = weave.Dataset(name=dataset_name, rows=eval_dataset)  # type: ignore[arg-type]
        weave.publish(dataset)
        print(f"📦 Published new evaluation dataset to Weave ({num_samples} samples)")
    return dataset


def print_evaluation_summary(evaluation_results: Any, scorers: list[Any]) -> None:
    """Print a summary of evaluation results."""
    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print()

    # Get scores from results
    if hasattr(evaluation_results, "rows"):
        results_data = evaluation_results.rows
        print(f"Total examples evaluated: {len(results_data)}")
        print()

        # Calculate aggregate scores
        print("Scorer Performance:")
        for scorer in scorers:
            scorer_name = scorer.__name__
            # Get scores for this scorer
            scores = [
                row.get(scorer_name, {}).get("score", 0)
                for row in results_data
                if scorer_name in row
            ]
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"  • {scorer_name}: {avg_score:.1%}")

        print()
        print("✅ Evaluation complete! View detailed results in Weave dashboard.")


async def main() -> None:  # noqa: PLR0915, PLR0912
    """Main execution function."""
    args = parse_arguments()

    # Get problem configuration
    problem_config = PROBLEM_CONFIGS[args.problem]
    model_name = args.model or config.llm_model
    temperature = (
        args.temperature if args.temperature is not None else config.llm_temperature
    )

    # Select scorers based on command line argument
    if args.scorers == "generic":
        # Only per-design metrics from generic scorer
        scorers = [score_design_generic]
    elif args.scorers == "engibench":
        # Lightweight scorer for design extraction + global metrics computed after
        scorers = [score_design_extracted]
    elif args.scorers == "all":
        # Both generic scorer and lightweight scorer for comprehensive metrics
        scorers = [score_design_generic, score_design_extracted]
    else:
        scorers = [score_design_generic]

    print("=" * 60)
    print(f"ENGINEERING AGENT EVALUATION ({args.problem})")
    print("=" * 60)
    print()
    print(f"Problem Type: {args.problem}")
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}")
    print(f"Dataset Split: {args.split}")
    print(f"Samples: {args.samples}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print(f"Scorer Set: {args.scorers}")
    print(f"Active Scorers: {[s.__name__ for s in scorers]}")  # type: ignore[attr-defined]
    print()

    # Initialize Weave
    print("🔧 Initializing Weave...")
    if not init_weave():
        print("❌ Error: Weave initialization failed")
        return
    print("✅ Weave initialized successfully!")
    print()

    # Construct prompt filename dynamically based on args
    prompt_file_template = problem_config["prompt_file"]
    prompt_file = prompt_file_template.format(
        problem=args.problem,
        split=args.split,
    )

    # Load prompts
    prompts = load_prompts(args.problem, prompt_file)
    if prompts is None:
        return

    print(f"✅ Loaded {len(prompts)} prompts")
    print(f"📊 Evaluating on {args.samples} samples")
    print()

    # Prepare evaluation dataset
    eval_dataset = prepare_evaluation_dataset(
        prompts, args.samples, args.problem, problem_config["dataset_name"], args.seed
    )

    # Get or create Weave dataset
    dataset_name = f"{args.problem}_eval_dataset"
    if args.seed is not None:
        dataset_name += f"_seed_{args.seed}"
    dataset = get_or_create_dataset(eval_dataset, dataset_name, len(eval_dataset))
    print()

    # Create agent model
    print("🤖 Creating multi-agent system...")
    agent = EngineeringAgent(
        model_name=model_name,
        temperature=temperature,
        problem_type=args.problem,
    )
    print()

    # Define evaluation
    print("🔍 Running evaluation...")
    eval_name = (
        f"{args.problem}_agent_eval_{model_name.replace('/', '_')}_{args.scorers}"
    )
    if args.seed is not None:
        eval_name += f"_seed_{args.seed}"
    evaluation = weave.Evaluation(
        name=eval_name,
        dataset=dataset,
        scorers=scorers,  # type: ignore[arg-type]
    )

    # Run evaluation (async)
    results = await evaluation.evaluate(agent)

    # Print summary
    print_evaluation_summary(results, scorers)

    await asyncio.sleep(10)  # Wait for any async logging to complete
    # Compute global metrics if using EngiBench scorers
    if args.scorers in ("engibench", "all"):
        print()
        print("=" * 60)
        print("COMPUTING GLOBAL METRICS")
        print("=" * 60)
        print()
        print("Computing metrics across all generated designs...")

        # Setup output directory for comparison images
        model_safe = model_name.replace("/", "_").replace(":", "_")
        if args.seed is not None:
            comparison_dir = f"benchmarks/evaluations/results/{model_safe}/{args.problem}/comparisons/seed_{args.seed}"
        else:
            comparison_dir = f"benchmarks/evaluations/results/{model_safe}/{args.problem}/comparisons"

        # Pass evaluation object to compute global metrics (not results!)
        global_metrics = compute_global_metrics(
            evaluation,
            dataset_name=problem_config["dataset_name"],
            sigma=10.0,
            save_comparisons=True,
            comparison_output_dir=comparison_dir,
        )

        print()
        print("Global Metrics:")
        print(
            f"  • MMD (similarity to dataset): {global_metrics.get('mmd', 'N/A'):.6e}"
            if global_metrics.get("mmd") is not None
            else "  • MMD: Failed to compute"
        )
        print(
            f"  • DPP Diversity: {global_metrics.get('dpp_diversity', 'N/A'):.6e}"
            if global_metrics.get("dpp_diversity") is not None
            else "  • DPP Diversity: Failed to compute"
        )
        print(
            f"  • RVC (Ratio of Violated Constraints): {global_metrics.get('rvc', 'N/A'):.4f}"
            if global_metrics.get("rvc") is not None
            else "  • RVC: No constraints to check"
        )

        # Print detailed RVC information if available
        rvc_details = global_metrics.get("rvc_details")
        if rvc_details and rvc_details.get("n_violations", 0) > 0:
            print(
                f"    - Designs with violations: {rvc_details['n_violations']}/{rvc_details['n_total']}"
            )
            violation_summary = rvc_details.get("violation_summary", {})
            if violation_summary:
                print("    - Most common constraint violations:")
                for constraint_name, count in sorted(
                    violation_summary.items(), key=lambda x: x[1], reverse=True
                )[:3]:  # Show top 3
                    print(f"      • {constraint_name}: {count} design(s)")
        print(
            f"  • IOG (Initial Optimality Gap): {global_metrics.get('iog', 'N/A'):.6e}"
            if global_metrics.get("iog") is not None
            else "  • IOG: No optimization history found"
        )
        print(
            f"  • COG (Cumulative Optimality Gap): {global_metrics.get('cog', 'N/A'):.6e}"
            if global_metrics.get("cog") is not None
            else "  • COG: No optimization history found"
        )
        print(
            f"  • FOG (Final Optimality Gap): {global_metrics.get('fog', 'N/A'):.6e}"
            if global_metrics.get("fog") is not None
            else "  • FOG: No optimization history found"
        )
        print(f"  • Designs evaluated: {global_metrics.get('n_designs', 0)}")
        print(f"  • Failed extractions: {global_metrics.get('n_failed', 0)}")

        # Save metrics to CSV if requested
        if args.output_csv or args.seed is not None:
            # Determine output CSV path
            if args.output_csv:
                csv_path = args.output_csv
            else:
                model_safe = model_name.replace("/", "_").replace(":", "_")
                results_dir = Path(
                    f"benchmarks/evaluations/results/{model_safe}/{args.problem}"
                )
                results_dir.mkdir(parents=True, exist_ok=True)
                csv_path = str(results_dir / "metrics.csv")

            # Prepare metrics row
            metrics_row = {
                "iog": global_metrics.get("iog"),
                "cog": global_metrics.get("cog"),
                "fog": global_metrics.get("fog"),
                "mmd": global_metrics.get("mmd"),
                "dpp": global_metrics.get("dpp_diversity"),
                "rvc": global_metrics.get("rvc"),
                "seed": args.seed if args.seed is not None else 0,
                "problem_id": args.problem,
                "model_id": model_name,
                "n_samples": args.samples,
                "sigma": 10.0,  # Default sigma used in compute_global_metrics
            }

            # Check if file exists to determine if we need header
            csv_file = Path(csv_path)
            file_exists = csv_file.exists()

            # Append to CSV
            with csv_file.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=metrics_row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(metrics_row)

            print()
            print(f"📊 Metrics saved to: {csv_path}")

    print()
    print("🎉 Evaluation complete!")
    print("📊 View detailed results in Weave dashboard")
    if args.scorers in ("engibench", "all"):
        print(f"📁 Comparison images saved to: {comparison_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
