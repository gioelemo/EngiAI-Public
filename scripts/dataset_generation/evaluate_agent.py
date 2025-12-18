"""
Evaluate engineering agent performance on beam design prompts using Weave.

This script uses Weave's evaluation framework to properly track and visualize
agent performance on beam design tasks.

Reference: https://docs.wandb.ai/weave/guides/core-types/evaluations
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import weave
from langchain_core.messages import HumanMessage

# Set SKIP_MCP to avoid Prusa MCP server connection issues during evaluation
os.environ["SKIP_MCP"] = "true"

# Add project root to path to import src modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import config  # noqa: E402
from src.agents.supervisor_agent import SupervisorAgent  # noqa: E402
from src.utils.weave_integration import init_weave  # noqa: E402

# Evaluation configuration
SAMPLE_SIZE = 5  # Start with small sample for testing

# Quality thresholds
MIN_RESPONSE_LENGTH = 100  # Minimum characters for substantial response


class BeamDesignAgent(weave.Model):
    """Weave Model wrapper for the multi-agent supervisor system."""

    model_name: str
    temperature: float = 0.7

    @weave.op()
    def predict(self, prompt: str) -> dict[str, Any]:
        """
        Generate a response to a beam design prompt using the multi-agent system.

        Args:
            prompt: User prompt for beam design

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
        # The last message should be the agent's response
        final_message = result["messages"][-1]
        response_content = final_message.content

        return {
            "response": response_content,
            "model": self.model_name,
            "response_length": len(response_content),
            "agent_type": getattr(final_message, "name", "unknown"),
        }


# Define scorer functions for Weave evaluation
@weave.op()
def score_acknowledges_volfrac(
    prompt: str,  # noqa: ARG001
    conditions: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, Any]:
    """Score whether the agent acknowledges volume fraction constraint."""
    response = output["response"].lower()
    volfrac = conditions["volfrac"]

    # Check if volfrac appears in various formats
    acknowledges = (
        f"{volfrac * 100:.1f}" in output["response"]
        or f"{volfrac:.2f}" in output["response"]
        or f"{volfrac:.3f}" in output["response"]
        or "volume" in response
    )

    return {"score": 1.0 if acknowledges else 0.0}


@weave.op()
def score_acknowledges_rmin(
    prompt: str,  # noqa: ARG001
    conditions: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, Any]:
    """Score whether the agent acknowledges minimum radius constraint."""
    response = output["response"].lower()
    rmin = conditions["rmin"]

    acknowledges = (
        f"{rmin:.1f}" in output["response"]
        or str(rmin) in output["response"]
        or "minimum" in response
    )

    return {"score": 1.0 if acknowledges else 0.0}


@weave.op()
def score_mentions_compliance(
    prompt: str,  # noqa: ARG001
    conditions: dict[str, Any],  # noqa: ARG001
    output: dict[str, Any],
) -> dict[str, Any]:
    """Score whether the agent mentions compliance or stiffness."""
    response = output["response"].lower()
    mentions = "compliance" in response or "stiff" in response

    return {"score": 1.0 if mentions else 0.0}


@weave.op()
def score_mentions_optimization(
    prompt: str,  # noqa: ARG001
    conditions: dict[str, Any],  # noqa: ARG001
    output: dict[str, Any],
) -> dict[str, Any]:
    """Score whether the agent mentions optimization or design process."""
    response = output["response"].lower()
    mentions = "optim" in response or "design" in response

    return {"score": 1.0 if mentions else 0.0}


@weave.op()
def score_structural_understanding(
    prompt: str,  # noqa: ARG001
    conditions: dict[str, Any],  # noqa: ARG001
    output: dict[str, Any],
) -> dict[str, Any]:
    """Score whether the agent shows structural engineering understanding."""
    response = output["response"].lower()
    structural_terms = ["load", "stress", "strain", "topology", "material", "structure"]

    shows_understanding = any(term in response for term in structural_terms)

    return {"score": 1.0 if shows_understanding else 0.0}


@weave.op()
def score_provides_guidance(
    prompt: str,  # noqa: ARG001
    conditions: dict[str, Any],  # noqa: ARG001
    output: dict[str, Any],
) -> dict[str, Any]:
    """Score whether the agent provides substantial guidance."""
    response_length = output.get("response_length", 0)
    provides = response_length > MIN_RESPONSE_LENGTH

    return {"score": 1.0 if provides else 0.0}


def prepare_evaluation_dataset(
    prompts: list[dict[str, Any]], sample_size: int
) -> list[dict[str, Any]]:
    """
    Prepare prompts for Weave evaluation format.

    Args:
        prompts: List of prompt dictionaries
        sample_size: Number of samples to include

    Returns:
        List of evaluation examples in Weave format
    """
    return [
        {
            "prompt": prompt_data["prompt"],
            "conditions": prompt_data["conditions"],
            "metadata": prompt_data.get("metadata", {}),
            "target": prompt_data.get("target", {}),
        }
        for prompt_data in prompts[:sample_size]
    ]


def print_evaluation_summary(evaluation_results: Any) -> None:
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
        scorers = [
            "score_acknowledges_volfrac",
            "score_acknowledges_rmin",
            "score_mentions_compliance",
            "score_mentions_optimization",
            "score_structural_understanding",
            "score_provides_guidance",
        ]

        print("Scorer Performance:")
        for scorer in scorers:
            # Get scores for this scorer
            scores = [
                row.get(scorer, {}).get("score", 0)
                for row in results_data
                if scorer in row
            ]
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"  • {scorer}: {avg_score:.1%}")

        print()
        print("✅ Evaluation complete! View detailed results in Weave dashboard.")


async def main() -> None:
    """Main execution function."""
    print("=" * 60)
    print("ENGINEERING AGENT EVALUATION (Weave Framework)")
    print("=" * 60)
    print()

    # Initialize Weave
    print("🔧 Initializing Weave...")
    if not init_weave():
        print("❌ Error: Weave initialization failed")
        return
    print("✅ Weave initialized successfully!")
    print()

    # Load prompts
    input_file = Path(
        "data/datasets/beam_prompts/generated/beam_prompts_50_samples.json"
    )
    print(f"📂 Loading prompts from: {input_file}")

    if not input_file.exists():
        print(f"❌ Error: File not found: {input_file}")
        return

    with input_file.open() as f:
        prompts = json.load(f)

    print(f"✅ Loaded {len(prompts)} prompts")
    print(f"📊 Evaluating on {SAMPLE_SIZE} samples")
    print()

    # Prepare evaluation dataset
    eval_dataset = prepare_evaluation_dataset(prompts, SAMPLE_SIZE)

    # Create Weave dataset
    dataset = weave.Dataset(name="beam_design_eval_dataset", rows=eval_dataset)  # type: ignore[arg-type]
    weave.publish(dataset)
    print("📦 Published evaluation dataset to Weave")
    print()

    # Create agent model (using production multi-agent system)
    print(f"🤖 Creating multi-agent system with model: {config.llm_model}")
    print(f"    Temperature: {config.llm_temperature}")
    agent = BeamDesignAgent(
        model_name=config.llm_model,
        temperature=config.llm_temperature,
    )
    print()

    # Define evaluation
    print("🔍 Running evaluation...")
    evaluation = weave.Evaluation(
        name="beam_design_agent_eval",
        dataset=dataset,
        scorers=[
            score_acknowledges_volfrac,
            score_acknowledges_rmin,
            score_mentions_compliance,
            score_mentions_optimization,
            score_structural_understanding,
            score_provides_guidance,
        ],
    )

    # Run evaluation (async)
    results = await evaluation.evaluate(agent)

    # Print summary
    print_evaluation_summary(results)

    print()
    print("🎉 Evaluation complete!")
    print("📊 View detailed results in Weave dashboard")


if __name__ == "__main__":
    asyncio.run(main())
