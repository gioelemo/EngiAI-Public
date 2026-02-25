"""
Unified agent evaluation script for benchmarking across different problem types.

This script uses Weave's evaluation framework to track and visualize agent performance
on engineering design tasks across multiple problem types and LLM models.

Usage:
    python evaluate_agent.py --problem beams2d --model gpt-4o --samples 5
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
# Suppress UI-only prompt sections (suggested_prompts) during evaluation
os.environ["EVAL_MODE"] = "true"

# Note: SKIP_MMORE is now controlled by the --mmore / --no-mmore CLI flag

# Add project root and services to path to import src and prusa_mcp_server modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "services"))

# Results directory structure:
# benchmarks/evaluations/results/models/{model_name}/{problem}/{prompt_style}/{rag_status}/
# where rag_status is "rag" (--mmore), "no_rag" (--no-mmore), or "empty_rag" (--empty-rag)
RESULTS_BASE_DIR = Path("benchmarks/evaluations/results/models")


def _get_rag_dir(rag_mode: str) -> str:
    """Return the RAG subdirectory name based on the rag mode.

    Args:
        rag_mode: One of "rag", "no_rag", or "empty_rag".
    """
    return rag_mode


# Import output quality scorers
from benchmarks.shared.problem_registry import PROBLEMS  # noqa: E402
from benchmarks.shared.scorers import (  # noqa: E402
    score_hpc_workflow,
    score_output_quality,
    score_rag_evaluation,
    score_task_completion,
    score_tool_use,
)
from config import config  # noqa: E402
from src.agents.supervisor_agent import SupervisorAgent  # noqa: E402
from src.tools.engibench import clear_session_state, set_session_id  # noqa: E402
from src.utils.weave_integration import init_weave  # noqa: E402


class ProblemConfig(TypedDict):
    """Configuration for a problem type."""

    dataset_name: str
    prompt_file: str


# Problem-specific configurations
# NOTE: The --scorers flag controls which metrics are computed during evaluation:
# - "output_quality": Per-design metrics (IoU, pixel accuracy, constraints, objectives)
# - "all": output_quality + task_completion + tool_use (comprehensive per-design metrics)
# - "task_completion": Check if render_design tool was called successfully
# - "tool_use": Tool efficiency and sequence correctness
# Global metrics (MMD, DPP, RVC, IOG, COG, FOG) are computed offline by compute_global_metrics.py
# Build PROBLEM_CONFIGS from the central problem registry to avoid duplication
PROBLEM_CONFIGS: dict[str, ProblemConfig] = {
    name: {
        "dataset_name": problem.dataset_name,
        "prompt_file": problem.prompt_file_template.replace(
            "{problem}", name
        ),  # Replace placeholder
    }
    for name, problem in PROBLEMS.items()
}

# Configure logger for this module
logger = logging.getLogger(__name__)


class EngineeringAgent(weave.Model):
    """Weave Model wrapper for the multi-agent supervisor system."""

    model_name: str
    temperature: float = 0.7
    llm_seed: int | None = None
    problem_type: str = "beams2d"
    rag_read_only: bool = False

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
                seed=self.llm_seed,
                rag_read_only=self.rag_read_only,
            )

            # Convert prompt to message format
            messages = [HumanMessage(content=prompt)]
            state = {"messages": messages}

            # Set recursion_limit to prevent infinite loops (e.g., models repeatedly
            # calling ask_human_for_clarification without stopping).
            # HPC training workflows need a higher limit due to multi-agent routing
            # and long monitoring tool calls.
            recursion_limit = 200 if self.problem_type == "hpc_train_beams2d" else 50
            config_dict = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": recursion_limit,
            }

            # Invoke the supervisor agent
            result = supervisor.invoke(state, config_dict)

            # Extract the final response from messages
            final_message = result["messages"][-1]
            response_content = final_message.content
            # content can be None, a list of content blocks, or a str; normalise
            if response_content is None:
                response_content = ""
            elif isinstance(response_content, list):
                response_content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in response_content
                )

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

            # Log optimize_design calls to see what config is being passed
            for tc in tool_calls_info:
                if tc["name"] == "optimize_design":
                    args = tc["args"]
                    # Check for config: either problem_config dict or flat parameters
                    problem_config = args.get("problem_config", None)
                    # Flat config params (LLM may use aliases)
                    flat_config_keys = {
                        "volume_fraction",
                        "volfrac",
                        "force_distribution",
                        "forcedist",
                        "rmin",
                        "lambda1",
                        "lambda2",
                        "blur_radius",
                        "weight",
                    }
                    has_flat_params = any(
                        args.get(k) is not None for k in flat_config_keys
                    )

                    logger.debug(
                        "optimize_design called with problem_config=%s, flat_params=%s",
                        problem_config,
                        {
                            k: args.get(k)
                            for k in flat_config_keys
                            if args.get(k) is not None
                        },
                    )
                    if problem_config is None and not has_flat_params:
                        logger.warning(
                            "NO CONFIG - agent is not passing configuration parameters!"
                        )

            return {
                "response": response_content,
                "model": self.model_name,
                "temperature": self.temperature,
                "llm_seed": self.llm_seed,
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
    eval_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Prepare prompts for Weave evaluation format.

    Args:
        prompts: List of prompt dictionaries
        sample_size: Number of samples to include
        eval_metadata: Evaluation metadata dict containing:
            - problem_type: Type of problem being evaluated
            - dataset_name: Name of the HuggingFace dataset for ground truth
            - seed: Optional seed to use for all prompts in this evaluation (optimization seed)
            - prompt_style: Style of prompt (full, natural, workflow-random, ...)
            - mmore_enabled: Whether MMORE RAG system is enabled
            - model_name: LLM model used for evaluation
            - temperature: Model temperature setting
            - llm_seed: LLM random seed for reproducibility

    Returns:
        List of evaluation examples in Weave format
    """
    seed = eval_metadata.get("seed")
    eval_dataset = []

    for i, prompt_data in enumerate(prompts[:sample_size]):
        prompt = prompt_data["prompt"]

        # Add seed instruction for dataset-sampled problems (beams2d, etc.).
        # Skip for fixed-prompt problems (HPC, RAG) — seeds are either
        # embedded in the prompt or irrelevant, and appending a global seed
        # creates contradictions.
        dataset_name = eval_metadata.get("dataset_name", "")
        if seed is not None and dataset_name:
            prompt = f"{prompt}\n\nIMPORTANT: Use seed={seed} when calling tools."

        eval_dataset.append(
            {
                "prompt": prompt,
                "conditions": prompt_data["conditions"],
                "metadata": {
                    **prompt_data.get("metadata", {}),
                    "example_id": prompt_data.get("example_id", i),
                    "dataset_split": prompt_data.get("dataset_split", "test"),
                    **eval_metadata,  # Include all evaluation metadata
                    # Efficiency scoring: optimal tool call info
                    # Default assumes optimize → simulate → render sequence
                    "optimal_call_count": prompt_data.get("optimal_call_count", 3),
                    "optimal_tool_calls": prompt_data.get(
                        "optimal_tool_calls",
                        [
                            {"name": "optimize_design", "count": 1},
                            {"name": "simulate_design", "count": 1},
                            {"name": "render_design", "count": 1},
                        ],
                    ),
                    # STL parameter validation for workflow-random prompts
                    "stl_expected_params": prompt_data.get("stl_expected_params"),
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
        default=10,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Model temperature (defaults to config.llm_temperature)",
    )
    parser.add_argument(
        "--llm-seed",
        type=int,
        default=None,
        help="LLM random seed for model reproducibility (defaults to config.llm_seed)",
    )
    parser.add_argument(
        "--scorers",
        type=str,
        default="all",
        choices=["output_quality", "task_completion", "tool_use", "all"],
        help=(
            "Metrics to compute: "
            "'output_quality' (per-design metrics only), "
            "'task_completion' (check if render_design tool was called successfully), "
            "'tool_use' (compute tool use efficiency and sequence correctness)"
            "'all' (output_quality + task_completion + tool_use for comprehensive metrics), "
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
        "--run",
        type=int,
        default=None,
        help="Run number for repeated evaluations (e.g., 1, 2, 3). Uses a fixed optimization seed across all runs. Combine with --seed to override the default optimization seed.",
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
        help="Prompt style to use (default: full). Determines optimal tool sequence expectations.",
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
    args = parser.parse_args()

    # --run = tracking identifier (1, 2, 3, …) with a fixed optimization seed.
    # --seed = optimization seed that also serves as tracking identifier.
    # When --run is used without --seed, default optimization seed to 1.
    if args.run is not None and args.seed is None:
        args.seed = 1
    args.run_id = args.run if args.run is not None else args.seed

    return args


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
        # Check count AND prompt content so stale cached datasets are refreshed
        existing_prompts = [row.get("prompt", "") for row in dataset.rows]
        new_prompts = [row.get("prompt", "") for row in eval_dataset]
        if existing_prompts == new_prompts:
            print(
                f"📦 Using existing evaluation dataset from Weave ({len(dataset.rows)} samples)"
            )
        else:
            print(
                f"⚠️  Dataset prompts changed (had {len(dataset.rows)}, need {num_samples}). Recreating..."
            )
            dataset = weave.Dataset(name=dataset_name, rows=eval_dataset)  # type: ignore[arg-type]
            weave.publish(dataset)
            print("📦 Published new evaluation dataset to Weave")
    except Exception:
        dataset = weave.Dataset(name=dataset_name, rows=eval_dataset)  # type: ignore[arg-type]
        weave.publish(dataset)
        print(f"📦 Published new evaluation dataset to Weave ({num_samples} samples)")
    return dataset


def create_contextual_scorer(
    scorer_func: Any,
    scorer_type: str,
) -> Any:
    """Create a scorer wrapper with evaluation context in its trace name.

    Model name and problem type are excluded to enable cross-model and cross-problem
    metric comparison in Weave UI.

    Args:
        scorer_func: Original scorer function to wrap
        scorer_type: Scorer identifier (e.g., "output_quality", "task_completion")

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
        conditions: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Contextual wrapper that delegates to the original scorer."""
        # Weave passes `conditions` from the top-level dataset row field; merge it
        # into metadata so all scorers can access it via metadata["conditions"].
        full_metadata = {**metadata, "conditions": conditions}
        return scorer_func(output, target, full_metadata)

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

    # Set SKIP_MMORE based on CLI flag (must be set before importing agent modules)
    # "rag" and "empty_rag" both need MMORE client initialized (tools available).
    # "no_rag" disables the MMORE client entirely (no RAG tools).
    mmore_enabled = args.rag_mode in ("rag", "empty_rag")
    os.environ["SKIP_MMORE"] = "false" if mmore_enabled else "true"
    # Empty-RAG mode: tools exist but retrieve/list return empty results.
    os.environ["MMORE_EMPTY_RAG"] = "true" if args.rag_mode == "empty_rag" else "false"
    # Reset MMORE cache to pick up the new env var value
    config.reset_mmore_cache()

    # For RAG evaluation problems disable ArXiv so MMORE is the only document source.
    # This keeps the RAG-on vs RAG-off comparison clean: the only variable is whether
    # MMORE (search_documents) is available, not whether the agent can reach the paper
    # via the ArXiv agent as an alternative route.
    os.environ["SKIP_ARXIV"] = "true" if args.problem == "rag_beams2d" else "false"

    # Disable SLURM email notifications for HPC training benchmarks to avoid spam
    os.environ["SKIP_SLURM_EMAIL"] = (
        "true" if args.problem == "hpc_train_beams2d" else "false"
    )

    # Set results directory for evaluate_model tool so HPC training CSVs
    # land in the per-model/prompt-style directory instead of the flat results/ root.
    model_safe_env = (
        (args.model or config.llm_model).replace("/", "_").replace(":", "_")
    )
    rag_dir_env = _get_rag_dir(args.rag_mode)
    eval_results_dir = str(
        RESULTS_BASE_DIR
        / model_safe_env
        / args.problem
        / args.prompt_style
        / rag_dir_env
    )
    os.environ["EVAL_RESULTS_DIR"] = eval_results_dir

    # Get problem configuration
    problem_config = PROBLEM_CONFIGS[args.problem]
    model_name = args.model or config.llm_model
    temperature = (
        args.temperature if args.temperature is not None else config.llm_temperature
    )
    llm_seed = args.llm_seed if args.llm_seed is not None else config.llm_seed

    # Select base scorers based on command line argument
    base_scorers = []
    scorer_types = []

    if args.scorers == "output_quality":
        # Only per-design metrics from output quality scorer
        base_scorers = [score_output_quality]
        scorer_types = ["output_quality"]
    elif args.scorers == "task_completion":
        # Task completion scorer: checks if render_design was called successfully
        base_scorers = [score_task_completion]
        scorer_types = ["task_completion"]
    elif args.scorers == "tool_use":
        # Tool use scorer: compute efficiency ratio and sequence correctness
        base_scorers = [score_tool_use]
        scorer_types = ["tool_use"]
    else:
        # "all" (the argparse default): comprehensive evaluation with all scorers
        base_scorers = [
            score_output_quality,
            score_task_completion,
            score_tool_use,
        ]
        scorer_types = ["output_quality", "task_completion", "tool_use"]

    # Substitute the primary scorer based on problem registry configuration.
    # Problems with a non-default primary_scorer (e.g., rag_evaluation, hpc_workflow)
    # replace score_output_quality with their domain-specific scorer.
    scorer_map = {
        "output_quality": score_output_quality,
        "rag_evaluation": score_rag_evaluation,
        "hpc_workflow": score_hpc_workflow,
    }
    problem_reg = PROBLEMS[args.problem]
    primary_scorer_type = problem_reg.primary_scorer
    if primary_scorer_type != "output_quality":
        primary_scorer_func = scorer_map[primary_scorer_type]
        base_scorers = [
            primary_scorer_func if s is score_output_quality else s
            for s in base_scorers
        ]
        scorer_types = [
            primary_scorer_type if t == "output_quality" else t for t in scorer_types
        ]

    # HPC training requires WandB for model download after training
    if args.problem == "hpc_train_beams2d":
        os.environ["USE_WANDB"] = "True"

    # Wrap scorers with evaluation context for better trace naming in Weave UI
    scorers = [
        create_contextual_scorer(scorer_func, scorer_type)
        for scorer_func, scorer_type in zip(base_scorers, scorer_types, strict=True)
    ]

    print("=" * 60)
    print(f"ENGINEERING AGENT EVALUATION ({args.problem})")
    print("=" * 60)
    print()
    print(f"Problem Type: {args.problem}")
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}")
    print(f"LLM Seed: {llm_seed}")
    print(f"Dataset Split: {args.split}")
    print(f"Prompt Style: {args.prompt_style}")
    _rag_mode_labels = {
        "rag": "enabled",
        "no_rag": "disabled",
        "empty_rag": "empty index",
    }
    print(f"MMORE RAG: {_rag_mode_labels.get(args.rag_mode, args.rag_mode)}")
    print(
        f"ArXiv: {'disabled (rag eval)' if args.problem == 'rag_beams2d' else 'enabled'}"
    )
    print(f"Max Samples: {args.samples}")
    if args.run_id is not None:
        run_label = "Run" if args.run is not None else "Seed"
        print(f"{run_label}: {args.run_id}")
    print(f"Scorer Set: {args.scorers}")
    print(f"Active Scorers: {[s.__name__ for s in scorers]}")  # type: ignore[attr-defined]

    # Debug: print the expected trace names
    safe_model = model_name.replace("/", "_").replace(":", "_")
    _mmore_suffixes = {
        "rag": "mmore_on",
        "no_rag": "mmore_off",
        "empty_rag": "mmore_empty",
    }
    mmore_suffix = _mmore_suffixes.get(args.rag_mode, "mmore_off")
    # Scorer names exclude model name and problem type to enable cross-model and cross-problem comparison
    expected_scorer_names = scorer_types
    expected_eval_run_name = (
        f"{safe_model}_{args.problem}_{args.prompt_style}_{mmore_suffix}_evaluation"
    )
    if args.run_id is not None:
        run_suffix = "run" if args.run is not None else "seed"
        expected_eval_run_name += f"_{run_suffix}_{args.run_id}"

    print("Expected Weave trace names:")
    print(f"  Evaluation: {expected_eval_run_name}")
    print(f"  Scorers: {expected_scorer_names}")
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
        style=args.prompt_style,
        n_samples=args.samples,
        seed=args.seed if args.seed is not None else 0,
    )

    # Load prompts
    prompts = load_prompts(args.problem, prompt_file)
    if prompts is None:
        return

    # Cap samples to actual number of prompts available
    n_eval = min(args.samples, len(prompts))
    print(f"✅ Loaded {len(prompts)} prompts")
    print(f"📊 Evaluating on {n_eval} samples")
    print()

    # Prepare evaluation dataset
    eval_metadata = {
        "problem_type": args.problem,
        "dataset_name": problem_config["dataset_name"],
        "seed": args.seed,  # Only --seed adds "Use seed=N" prompt instruction
        "run_id": args.run_id,  # Tracking identifier (from --run or --seed)
        "prompt_style": args.prompt_style,
        "mmore_enabled": args.rag_mode in ("rag", "empty_rag"),
        "rag_mode": args.rag_mode,
        "model_name": model_name,
        "temperature": temperature,
        "llm_seed": llm_seed,
    }
    eval_dataset = prepare_evaluation_dataset(prompts, args.samples, eval_metadata)

    # Get or create Weave dataset (include sample count to avoid conflicts)
    dataset_name = f"{args.problem}_{args.prompt_style}_{mmore_suffix}_eval_dataset_{safe_model}_n{args.samples}"
    if args.run_id is not None:
        run_suffix = "run" if args.run is not None else "seed"
        dataset_name += f"_{run_suffix}_{args.run_id}"
    dataset = get_or_create_dataset(eval_dataset, dataset_name, len(eval_dataset))
    print()

    # Create agent model
    print("🤖 Creating multi-agent system...")
    agent = EngineeringAgent(
        model_name=model_name,
        temperature=temperature,
        llm_seed=llm_seed,
        problem_type=args.problem,
        rag_read_only=args.problem == "rag_beams2d",
    )
    print()

    # Define evaluation
    print("🔍 Running evaluation...")
    eval_type_name = f"{args.problem}_{args.prompt_style}_{mmore_suffix}_agent_eval_{model_name.replace('/', '_')}_{args.scorers}"
    if args.run_id is not None:
        run_suffix = "run" if args.run is not None else "seed"
        eval_type_name += f"_{run_suffix}_{args.run_id}"
    evaluation = weave.Evaluation(
        name=eval_type_name,
        dataset=dataset,
        scorers=scorers,  # type: ignore[arg-type]
    )

    # Create contextual evaluation run name
    safe_model = model_name.replace("/", "_").replace(":", "_")
    eval_run_name = (
        f"{safe_model}_{args.problem}_{args.prompt_style}_{mmore_suffix}_evaluation"
    )
    if args.run_id is not None:
        run_suffix = "run" if args.run is not None else "seed"
        eval_run_name += f"_{run_suffix}_{args.run_id}"

    # Wrap evaluation.evaluate() in a named weave op for custom trace naming
    @weave.op(name=eval_run_name)
    async def run_evaluation(eval_obj: Any, agent_obj: Any) -> Any:
        """Run evaluation with custom trace name."""
        return await eval_obj.evaluate(agent_obj)

    # Run evaluation (async) with custom trace name
    results = await run_evaluation(evaluation, agent)

    # Print summary
    print_evaluation_summary(results, scorers)

    await asyncio.sleep(30)  # Wait for Weave to sync evaluation results

    # Save per-design metrics to CSV
    model_safe = model_name.replace("/", "_").replace(":", "_")
    rag_dir = _get_rag_dir(args.rag_mode)
    results_dir = (
        RESULTS_BASE_DIR / model_safe / args.problem / args.prompt_style / rag_dir
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # HPC training evaluation — no HuggingFace ground-truth, design quality scored offline.
    if args.problem == "hpc_train_beams2d":
        print()
        print("=" * 60)
        print("HPC TRAINING EVALUATION RESULTS")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Run extract_data.py to export per-example HPC workflow metrics:")
        print(
            f"     python benchmarks/evaluations/extract_data.py"
            f" --problem {args.problem} --prompt-style {args.prompt_style}"
            f" --rag-status {rag_dir}"
        )
        print("  2. Run compute_hpc_metrics.py to compute design quality vs EngiBench:")
        print(
            f"     python benchmarks/evaluations/compute_hpc_metrics.py"
            f" --problem {args.problem}"
        )
        print()
        print("Evaluation complete!")
        print("View detailed results in Weave dashboard")
        return

    # RAG evaluation — no HuggingFace ground-truth designs, so skip global metrics.
    # Per-example results are in Weave; use extract_data.py to export to JSON.
    if args.problem == "rag_beams2d":
        print()
        print("=" * 60)
        print("RAG EVALUATION RESULTS")
        print("=" * 60)
        print()
        _rag_status_labels = {
            "rag": "enabled (RAG on)",
            "no_rag": "disabled (RAG off)",
            "empty_rag": "empty index (RAG tools available, no documents)",
        }
        mmore_status = _rag_status_labels.get(args.rag_mode, args.rag_mode)
        print(f"MMORE RAG: {mmore_status}")
        print()
        print("Next steps:")
        print("  1. Run extract_data.py to export per-example RAG metrics to JSON:")
        print(
            f"     python benchmarks/evaluations/extract_data.py"
            f" --problem {args.problem} --prompt-style {args.prompt_style}"
            f" --rag-status {rag_dir}"
            f"  # filters rag_beams2d runs using RAG status/metadata; pass --eval-name to narrow if needed"
        )
        print(
            '  2. Run "python benchmarks/evaluations/plots/run_all.py'
            f' --problem {args.problem}" to generate RAG plots across models.'
        )
        print()
        print("🎉 Evaluation complete!")
        print("📊 View detailed results in Weave dashboard")
        return

    # Compute global metrics for all scorer types (all scorers now support design extraction)
    # Skip for task_completion and tool_use scorers which don't extract designs
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

    if args.scorers == "tool_use":
        print()
        print("=" * 60)
        print("TOOL USE RESULTS")
        print("=" * 60)
        print()
        print("Tool use scorer does not compute global metrics.")
        print(
            "Check Weave dashboard for per-example efficiency_ratio (tool call efficiency)."
        )
        print()
        print("🎉 Evaluation complete!")
        print("📊 View detailed results in Weave dashboard")
        return

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print()
    print("✅ Agent evaluation finished successfully!")
    print("📊 View detailed per-example results in Weave dashboard")
    print()
    print("Next steps:")
    print("  1. Run extract_data.py to export per-design metrics to JSON")
    print("  2. Run compute_global_metrics.py to calculate global metrics per seed")
    print(
        f'  3. Run "python benchmarks/evaluations/plots/run_all.py --problem {args.problem}" '
        "to create visualizations"
    )
    print()
    print("🎉 Evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())
