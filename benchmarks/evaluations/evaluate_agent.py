"""
Unified agent evaluation script for benchmarking across different problem types.

This script uses Weave's evaluation framework to track and visualize agent performance
on engineering design tasks across multiple problem types and LLM models.

Usage:
    python evaluate_agent.py --problem beams2d --model gpt-4o --samples 5
    python evaluate_agent.py --problem thermoelastic2d --model claude-3-5-sonnet --samples 10

Reference: https://docs.wandb.ai/weave/guides/core-types/evaluations
"""

# Suppress Pydantic warnings BEFORE any imports that might trigger them
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*'repr' attribute.*Field.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*'frozen' attribute.*Field.*",
)

import argparse  # noqa: E402
import asyncio  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, TypedDict  # noqa: E402

import weave  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

# Suppress Weave serialization warnings (non-critical, caused by ModelMetaclass in LangChain)
logging.getLogger("langchain_core.callbacks.manager").setLevel(logging.ERROR)

# Set SKIP_MCP to avoid Prusa MCP server connection issues during evaluation
os.environ["SKIP_MCP"] = "true"

# Set SKIP_MMORE to avoid MMORE Docker container requirement during evaluation
os.environ["SKIP_MMORE"] = "true"

# Add project root and services to path to import src and prusa_mcp_server modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "services"))

# Import output quality scorers
from benchmarks.shared.output_quality_engibench_scorer import (  # noqa: E402
    compute_global_metrics,  # For global metrics after evaluation
    score_output_quality_engibench,  # EngiBench scorer for global metrics
)
from benchmarks.shared.output_quality_visual_scorer import (  # noqa: E402
    score_output_quality_visual,
)
from benchmarks.shared.problem_registry import PROBLEMS  # noqa: E402
from benchmarks.shared.efficiency_scorer import (  # noqa: E402
    score_efficiency,
)
from benchmarks.shared.task_completion_scorer import (  # noqa: E402
    score_task_completion,
)
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
# - "generic" or "all": Per-design metrics (IoU, pixel accuracy, constraints, objectives) + global metrics (MMD, DPP, RVC, IOG, COG, FOG)
# - "engibench": Lightweight design extraction only (faster, minimal metrics in evaluation table)
# Build PROBLEM_CONFIGS from the central problem registry to avoid duplication
PROBLEM_CONFIGS: dict[str, ProblemConfig] = {
    name: {
        "dataset_name": problem.dataset_name,
        "prompt_file": problem.prompt_file_template.replace(
            "{problem}", name
        ),  # Replace placeholder
        "scorers": [score_output_quality_visual],  # All problems use generic scorer
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
            supervisor = SupervisorAgent(
                model_name=self.model_name,
                temperature=self.temperature,
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

            # Log optimize_design calls to see what problem_config is being passed
            for tc in tool_calls_info:
                if tc["name"] == "optimize_design":
                    # Check 'problem_config' parameter (current name)
                    problem_config = tc["args"].get("problem_config", None)
                    logger.debug(
                        "optimize_design called with problem_config: %s", problem_config
                    )
                    if problem_config is None:
                        logger.warning(
                            "NO PROBLEM_CONFIG - agent is not passing configuration parameters!"
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
            prompt = f"{prompt}\n\nIMPORTANT: Use seed={seed} when calling tools."

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
                    # Efficiency scoring: optimal tool call info
                    "optimal_call_count": prompt_data.get("optimal_call_count", 2),
                    "optimal_tool_calls": prompt_data.get("optimal_tool_calls", [
                        {"name": "optimize_design", "count": 1},
                        {"name": "render_design", "count": 1},
                    ]),
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
        choices=["generic", "engibench", "all", "task_completion", "efficiency"],
        help=(
            "Metrics to compute: "
            "'generic' or 'all' (detailed per-design metrics + global metrics: MMD, DPP, RVC, optimality gaps), "
            "'engibench' (lightweight design extraction only, use for faster evaluations), "
            "'task_completion' (check if render_design tool was called successfully), "
            "'efficiency' (compute efficiency ratio: optimal_calls / actual_calls)"
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
        help="Output CSV file to save metrics (will append if file exists). Default: benchmarks/evaluations/results/{model}/{problem}/output_quality_global_metrics.csv",
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


def save_per_design_metrics(  # noqa: PLR0912, PLR0915
    evaluation: Any,
    csv_path: str,
    seed: int | None,
    problem_id: str,
    model_id: str,
) -> None:
    """Save per-design metrics to CSV file.

    Args:
        evaluation: Weave Evaluation object (after evaluate() has been called)
        csv_path: Path to save CSV file
        seed: Random seed used (if any)
        problem_id: Problem type
        model_id: Model name
    """
    try:
        # Use Weave's get_scores() API to access scorer outputs
        scores = evaluation.get_scores()
        if not scores:
            print("⚠️  No scorer results available for per-design metrics")
            return

        # Get the latest trace (most recent evaluation)
        latest_trace_id = list(scores.keys())[-1]
        latest_trace_scores = scores[latest_trace_id]

        # Try to get results from score_output_quality_visual first (has detailed metrics)
        # Fall back to score_output_quality_engibench if needed
        # Match both old naming (with prefixes) and new naming (without prefixes)
        scorer_outputs = []
        for key in latest_trace_scores:
            if key.endswith("_output_quality_visual") or key == "output_quality_visual":
                scorer_outputs = latest_trace_scores[key]
                break

        if not scorer_outputs:
            for key in latest_trace_scores:
                if key.endswith("_engibench") or key == "engibench":
                    scorer_outputs = latest_trace_scores[key]
                    break

        if not scorer_outputs:
            print("⚠️  No scorer outputs found for per-design metrics")
            return

        # Get dataset rows for metadata (example IDs)
        dataset_rows = (
            list(evaluation.dataset.rows) if hasattr(evaluation, "dataset") else []
        )

    except Exception as e:
        print(f"⚠️  Error accessing evaluation results: {e}")
        return

    # Extract per-design metrics from scorer outputs
    design_metrics_list = []
    for i, scorer_result in enumerate(scorer_outputs):
        if not isinstance(scorer_result, dict):
            continue

        # Get example_id from scorer result or dataset row
        example_id = scorer_result.get("example_id", i)
        if example_id == i and i < len(dataset_rows):
            # Try to get from dataset metadata
            metadata = dataset_rows[i].get("metadata", {})
            example_id = metadata.get("example_id", i)

        # Build metrics row
        metrics_row = {
            "seed": seed if seed is not None else 0,
            "example_id": example_id,
            "problem_id": problem_id,
            "model_id": model_id,
            # Main metrics
            "overall_score": scorer_result.get("score", 0.0),
            "iou": scorer_result.get("iou", 0.0),
            "pixel_accuracy": scorer_result.get("pixel_accuracy", 0.0),
            "mse": scorer_result.get("mse", 0.0),
            "constraint_score": scorer_result.get("constraint_score", 0.0),
            "objective_score": scorer_result.get("objective_score", 0.0),
        }

        # Add any additional constraint/objective specific metrics
        for key, value in scorer_result.items():
            if key not in metrics_row and isinstance(value, (int, float)):
                metrics_row[key] = value

        design_metrics_list.append(metrics_row)

    if not design_metrics_list:
        print("⚠️  No design metrics extracted")
        return

    # Check if file exists to determine if we need header
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_file.exists()

    # Get all possible fieldnames from all rows (in case some have extra fields)
    all_keys: set[str] = set()
    for row in design_metrics_list:
        all_keys.update(row.keys())

    # Ensure standard fields come first
    standard_fields = [
        "seed",
        "example_id",
        "problem_id",
        "model_id",
        "overall_score",
        "iou",
        "pixel_accuracy",
        "mse",
        "constraint_score",
        "objective_score",
    ]

    # If file exists, read existing header and merge with new keys
    if file_exists:
        with csv_file.open("r", newline="") as f:
            reader = csv.reader(f)
            existing_fieldnames = next(reader, [])
        # Use existing order and add any new fields at the end
        new_keys = [k for k in all_keys if k not in existing_fieldnames]
        fieldnames = existing_fieldnames + sorted(new_keys)
    else:
        # New file: sort and put standard fields first
        fieldnames = sorted(all_keys)
        fieldnames = [f for f in standard_fields if f in fieldnames] + [
            f for f in fieldnames if f not in standard_fields
        ]

    # Append to CSV
    with csv_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(design_metrics_list)

    print(f"📊 Per-design metrics saved to: {csv_path}")
    print(f"   ({len(design_metrics_list)} designs)")


def create_contextual_scorer(
    scorer_func: Any,
    model_name: str,  # noqa: ARG001 - Kept for signature compatibility
    problem_type: str,  # noqa: ARG001 - Kept for signature compatibility
    scorer_type: str,
) -> Any:
    """Create a scorer wrapper with evaluation context in its trace name.

    Model name and problem type are excluded to enable cross-model and cross-problem
    metric comparison in Weave UI.

    Args:
        scorer_func: Original scorer function to wrap
        model_name: Model name (e.g., "gpt-4o", "claude-3-5-sonnet") [unused, kept for signature compatibility]
        problem_type: Problem type (e.g., "beams2d", "thermoelastic2d") [unused, kept for signature compatibility]
        scorer_type: Scorer identifier (e.g., "output_quality_visual", "engibench")

    Returns:
        Wrapped scorer function with contextual trace name
    """
    # Create trace name with just the scorer type
    # Model name and problem type are excluded to enable cross-model and cross-problem comparison
    trace_name = scorer_type

    @weave.op(name=trace_name)
    def contextual_scorer(
        output: dict[str, Any],
        target: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Contextual wrapper that delegates to the original scorer."""
        return scorer_func(output, target, metadata)

    # Set the __name__ attribute so it displays correctly
    contextual_scorer.__name__ = scorer_type  # type: ignore[attr-defined]
    return contextual_scorer


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

    # Select base scorers based on command line argument
    base_scorers = []
    scorer_types = []

    if args.scorers == "generic":
        # Only per-design metrics from generic scorer
        base_scorers = [score_output_quality_visual]
        scorer_types = ["output_quality_visual"]
    elif args.scorers == "engibench":
        # Lightweight scorer for design extraction + global metrics computed after
        base_scorers = [score_output_quality_engibench]
        scorer_types = ["engibench"]
    elif args.scorers == "all":
        # Use output_quality_visual + task_completion + efficiency for comprehensive metrics
        base_scorers = [score_output_quality_visual, score_task_completion, score_efficiency]
        scorer_types = ["output_quality_visual", "task_completion", "efficiency"]
    elif args.scorers == "task_completion":
        # Task completion scorer: checks if render_design was called successfully
        base_scorers = [score_task_completion]
        scorer_types = ["task_completion"]
    elif args.scorers == "efficiency":
        # Efficiency scorer: compute efficiency ratio (optimal_calls / actual_calls)
        base_scorers = [score_efficiency]
        scorer_types = ["efficiency"]
    else:
        base_scorers = [score_output_quality_visual]
        scorer_types = ["output_quality_visual"]

    # Wrap scorers with evaluation context for better trace naming in Weave UI
    scorers = [
        create_contextual_scorer(scorer_func, model_name, args.problem, scorer_type)
        for scorer_func, scorer_type in zip(base_scorers, scorer_types, strict=False)
    ]

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

    # Debug: print the expected trace names
    safe_model = model_name.replace("/", "_").replace(":", "_")
    # Scorer names exclude model name and problem type to enable cross-model and cross-problem comparison
    expected_scorer_names = scorer_types
    expected_eval_run_name = f"{safe_model}_{args.problem}_evaluation"
    if args.seed is not None:
        expected_eval_run_name += f"_seed_{args.seed}"

    # Global metrics trace name includes model and problem for organization
    expected_global_metrics_name = f"{safe_model}_{args.problem}_global_metrics"
    if args.seed is not None:
        expected_global_metrics_name += f"_seed_{args.seed}"

    print("Expected Weave trace names:")
    print(f"  Evaluation: {expected_eval_run_name}")
    print(f"  Scorers: {expected_scorer_names}")
    print(f"  Global Metrics: {expected_global_metrics_name}")
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
    eval_type_name = (
        f"{args.problem}_agent_eval_{model_name.replace('/', '_')}_{args.scorers}"
    )
    if args.seed is not None:
        eval_type_name += f"_seed_{args.seed}"
    evaluation = weave.Evaluation(
        name=eval_type_name,
        dataset=dataset,
        scorers=scorers,  # type: ignore[arg-type]
    )

    # Create contextual evaluation run name
    safe_model = model_name.replace("/", "_").replace(":", "_")
    eval_run_name = f"{safe_model}_{args.problem}_evaluation"
    if args.seed is not None:
        eval_run_name += f"_seed_{args.seed}"

    # Wrap evaluation.evaluate() in a named weave op for custom trace naming
    @weave.op(name=eval_run_name)
    async def run_evaluation(eval_obj: Any, agent_obj: Any) -> Any:
        """Run evaluation with custom trace name."""
        return await eval_obj.evaluate(agent_obj)

    # Run evaluation (async) with custom trace name
    results = await run_evaluation(evaluation, agent)

    # Print summary
    print_evaluation_summary(results, scorers)

    print()

    await asyncio.sleep(10)  # Wait for Weave to sync evaluation results

    # Save per-design metrics to CSV
    model_safe = model_name.replace("/", "_").replace(":", "_")
    results_dir = Path(f"benchmarks/evaluations/results/{model_safe}/{args.problem}")
    results_dir.mkdir(parents=True, exist_ok=True)
    design_metrics_csv = str(results_dir / "output_quality_design_metrics.csv")
    save_per_design_metrics(
        evaluation,  # Pass evaluation object instead of results
        design_metrics_csv,
        args.seed,
        args.problem,
        model_name,
    )

    # Compute global metrics for all scorer types (all scorers now support design extraction)
    # Skip for task_completion and efficiency scorers which don't extract designs
    if args.scorers == "task_completion":
        print()
        print("=" * 60)
        print("TASK COMPLETION RESULTS")
        print("=" * 60)
        print()
        print("Task completion scorer does not compute global metrics.")
        print("Check Weave dashboard for per-example success_rate scores.")
        print()
        print("🎉 Evaluation complete!")
        print("📊 View detailed results in Weave dashboard")
        return

    if args.scorers == "efficiency":
        print()
        print("=" * 60)
        print("EFFICIENCY RESULTS")
        print("=" * 60)
        print()
        print("Efficiency scorer does not compute global metrics.")
        print("Check Weave dashboard for per-example efficiency_ratio scores.")
        print()
        print("🎉 Evaluation complete!")
        print("📊 View detailed results in Weave dashboard")
        return

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
        comparison_dir = (
            f"benchmarks/evaluations/results/{model_safe}/{args.problem}/comparisons"
        )

    # Pass evaluation object to compute global metrics (not results!)
    global_metrics = compute_global_metrics(
        evaluation,
        dataset_name=problem_config["dataset_name"],
        sigma=10.0,
        save_comparisons=True,
        comparison_output_dir=comparison_dir,
        model_name=model_name,
        problem_type=args.problem,
        seed=args.seed,
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
            csv_path = str(results_dir / "output_quality_global_metrics.csv")

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
    print(f"📁 Comparison images saved to: {comparison_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
