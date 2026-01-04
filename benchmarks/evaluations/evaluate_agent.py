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
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, TypedDict

import weave
from langchain_core.messages import HumanMessage

# Set SKIP_MCP to avoid Prusa MCP server connection issues during evaluation
os.environ["SKIP_MCP"] = "true"

# Add project root to path to import src modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import shared scorers and utilities
from benchmarks.shared.scorers import (  # noqa: E402
    score_constraint_accuracy,
    score_design_match,
    score_no_contradictions,
    score_provides_actionable_guidance,
    score_target_awareness,
    score_understands_tradeoffs,
)
from config import config  # noqa: E402
from src.agents.supervisor_agent import SupervisorAgent  # noqa: E402
from src.utils.weave_integration import init_weave  # noqa: E402


class ProblemConfig(TypedDict):
    """Configuration for a problem type."""

    dataset_name: str
    prompt_file: str
    scorers: list[Any]


# Problem-specific configurations
PROBLEM_CONFIGS: dict[str, ProblemConfig] = {
    "beams2d": {
        "dataset_name": "IDEALLab/beams_2d_50_100_v0",
        "prompt_file": "beam_prompts_50_samples.json",
        "scorers": [
            score_constraint_accuracy,
            score_target_awareness,
            score_understands_tradeoffs,
            score_provides_actionable_guidance,
            score_no_contradictions,
            score_design_match,
        ],
    },
    # Add more problem types here in the future
    # "thermoelastic2d": {  # noqa: ERA001
    #     "dataset_name": "...",  # noqa: ERA001
    #     "prompt_file": "thermoelastic_prompts_50_samples.json",  # noqa: ERA001
    #     "scorers": [...],  # noqa: ERA001
    # },
}


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
        # Initialize the supervisor agent with the configured model
        supervisor = SupervisorAgent(
            model_name=self.model_name,
            temperature=self.temperature,
        )

        # Convert prompt to message format
        messages = [HumanMessage(content=prompt)]
        state = {"messages": messages}

        # Generate unique thread_id for this evaluation
        thread_id = f"eval_{uuid.uuid4().hex[:8]}"
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

        # Log optimize_design calls to see what config is being passed
        for tc in tool_calls_info:
            if tc["name"] == "optimize_design":
                config_used = tc["args"].get("config", None)
                print("\n[DEBUG] optimize_design called with:")
                print(f"  config: {config_used}")
                if config_used is None:
                    print(
                        "  ⚠️  NO CONFIG - agent is not passing constraint parameters!"
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


def prepare_evaluation_dataset(
    prompts: list[dict[str, Any]],
    sample_size: int,
    problem_type: str,
    dataset_name: str,
) -> list[dict[str, Any]]:
    """
    Prepare prompts for Weave evaluation format.

    Args:
        prompts: List of prompt dictionaries
        sample_size: Number of samples to include
        problem_type: Type of problem being evaluated
        dataset_name: Name of the HuggingFace dataset for ground truth

    Returns:
        List of evaluation examples in Weave format
    """
    return [
        {
            "prompt": prompt_data["prompt"],
            "conditions": prompt_data["conditions"],
            "metadata": {
                **prompt_data.get("metadata", {}),
                "example_id": prompt_data.get("example_id", i),
                "problem_type": problem_type,
                "dataset_name": dataset_name,
            },
            "target": prompt_data.get("target", {}),
        }
        for i, prompt_data in enumerate(prompts[:sample_size])
    ]


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


async def main() -> None:
    """Main execution function."""
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

    args = parser.parse_args()

    # Get problem configuration
    problem_config = PROBLEM_CONFIGS[args.problem]
    model_name = args.model or config.llm_model
    temperature = (
        args.temperature if args.temperature is not None else config.llm_temperature
    )

    print("=" * 60)
    print(f"ENGINEERING AGENT EVALUATION ({args.problem})")
    print("=" * 60)
    print()
    print(f"Problem Type: {args.problem}")
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}")
    print(f"Samples: {args.samples}")
    print()

    # Initialize Weave
    print("🔧 Initializing Weave...")
    if not init_weave():
        print("❌ Error: Weave initialization failed")
        return
    print("✅ Weave initialized successfully!")
    print()

    # Load prompts from problem-specific directory
    prompt_file = (
        Path(__file__).parent.parent
        / "problems"
        / args.problem
        / "data"
        / "generated"
        / problem_config["prompt_file"]
    )
    print(f"📂 Loading prompts from: {prompt_file}")

    if not prompt_file.exists():
        print(f"❌ Error: File not found: {prompt_file}")
        print(f"Please run generate_prompts.py for {args.problem} first.")
        return

    with prompt_file.open() as f:
        prompts = json.load(f)

    print(f"✅ Loaded {len(prompts)} prompts")
    print(f"📊 Evaluating on {args.samples} samples")
    print()

    # Prepare evaluation dataset
    eval_dataset = prepare_evaluation_dataset(
        prompts, args.samples, args.problem, problem_config["dataset_name"]
    )

    # Create Weave dataset
    dataset = weave.Dataset(
        name=f"{args.problem}_eval_dataset_{model_name.replace('/', '_')}",
        rows=eval_dataset,  # type: ignore[arg-type]
    )
    weave.publish(dataset)
    print("📦 Published evaluation dataset to Weave")
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
    evaluation = weave.Evaluation(
        name=f"{args.problem}_agent_eval_{model_name.replace('/', '_')}",
        dataset=dataset,
        scorers=problem_config["scorers"],
    )

    # Run evaluation (async)
    results = await evaluation.evaluate(agent)

    # Print summary
    print_evaluation_summary(results, problem_config["scorers"])

    print()
    print("🎉 Evaluation complete!")
    print("📊 View detailed results in Weave dashboard")
    print(
        f"📁 Comparison images saved to: benchmarks/evaluations/results/{model_name}/{args.problem}/comparisons/"
    )


if __name__ == "__main__":
    asyncio.run(main())
