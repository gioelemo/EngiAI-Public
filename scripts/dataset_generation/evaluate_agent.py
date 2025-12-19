"""
Evaluate engineering agent performance on beam design prompts using Weave.

This script uses Weave's evaluation framework to properly track and visualize
agent performance on beam design tasks.

Reference: https://docs.wandb.ai/weave/guides/core-types/evaluations
"""

import asyncio
import base64
import io
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import weave
from datasets import load_dataset
from langchain_core.messages import HumanMessage
from PIL import Image

# Set SKIP_MCP to avoid Prusa MCP server connection issues during evaluation
os.environ["SKIP_MCP"] = "true"

# Add project root to path to import src modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import config  # noqa: E402
from src.agents.supervisor_agent import SupervisorAgent  # noqa: E402
from src.tools.engibench import get_unified_last_design  # noqa: E402
from src.utils.weave_integration import init_weave  # noqa: E402

# Evaluation configuration
SAMPLE_SIZE = 5  # Start with small sample for testing

# Quality thresholds
MIN_RESPONSE_LENGTH = 100  # Minimum characters for substantial response
MIN_ACTIONABLE_TERMS = 2  # Minimum actionable terms to consider guidance sufficient
LOW_VOLFRAC_THRESHOLD = (
    0.3  # Volume fraction below which design is considered constrained
)
STIFF_COMPLIANCE_THRESHOLD = 30  # Compliance below which design is considered stiff
FLEXIBLE_COMPLIANCE_THRESHOLD = (
    80  # Compliance above which design is considered flexible
)
BINARY_THRESHOLD = 0.5  # Threshold for converting density to binary (material vs void)


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
# These scorers validate the agent's output against ground truth design data


@weave.op()
def score_constraint_accuracy(
    prompt: str,  # noqa: ARG001
    conditions: dict[str, Any],
    output: dict[str, Any],
    target: dict[str, Any],  # noqa: ARG001
) -> dict[str, Any]:
    """
    Score if agent correctly references ALL constraint values.

    Checks that the agent mentions the correct volfrac, rmin values from the prompt.
    """
    response = output["response"].lower()
    volfrac = conditions["volfrac"]
    rmin = conditions["rmin"]

    # Check volfrac mention (allow percentage or decimal)
    volfrac_percent = int(volfrac * 100)
    volfrac_mentioned = any(
        [
            f"{volfrac:.2f}" in response,
            f"{volfrac:.1%}".lower() in response.replace(" ", ""),
            f"{volfrac_percent}%" in response,
            f"0.{volfrac_percent}" in response,
        ]
    )

    # Check rmin mention
    rmin_mentioned = str(rmin) in response or f"{rmin:.1f}" in response

    # Both must be mentioned correctly
    score = 1.0 if (volfrac_mentioned and rmin_mentioned) else 0.0

    return {
        "score": score,
        "volfrac_mentioned": volfrac_mentioned,
        "rmin_mentioned": rmin_mentioned,
    }


@weave.op()
def score_target_awareness(
    prompt: str,  # noqa: ARG001
    conditions: dict[str, Any],  # noqa: ARG001
    output: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    """
    Score if agent references the target compliance value.

    The optimal design from the dataset has a specific compliance value.
    Check if the agent mentions working towards or achieving this target.
    """
    response = output["response"].lower()
    target_compliance = target["compliance"]

    # Check if agent mentions compliance values near the target
    # Allow some tolerance since agent might round
    mentions_target = any(
        [
            f"{target_compliance:.1f}" in response,
            f"{target_compliance:.2f}" in response,
            f"{int(target_compliance)}" in response,
        ]
    )

    # Also check if agent mentions minimizing compliance / maximizing stiffness
    mentions_objective = any(
        [
            "minimize compliance" in response,
            "minimiz" in response and "compliance" in response,
            "maximize stiffness" in response,
            "maxim" in response and "stiff" in response,
        ]
    )

    score = 1.0 if (mentions_target or mentions_objective) else 0.0

    return {
        "score": score,
        "mentions_target_value": mentions_target,
        "mentions_objective": mentions_objective,
        "target_compliance": target_compliance,
    }


@weave.op()
def score_understands_tradeoffs(
    prompt: str,  # noqa: ARG001
    conditions: dict[str, Any],
    output: dict[str, Any],
    target: dict[str, Any],  # noqa: ARG001
) -> dict[str, Any]:
    """
    Score if agent understands the design trade-offs.

    Low volume fraction + low compliance is challenging - agent should
    acknowledge the constraint of limited material.
    """
    response = output["response"].lower()
    volfrac = conditions["volfrac"]

    # Check if agent mentions material limitations or trade-offs
    mentions_tradeoff = any(
        [
            "limited material" in response,
            "constrained" in response or "constraint" in response,
            "trade" in response and "off" in response,
            "balance" in response,
            "efficient" in response and "material" in response,
        ]
    )

    # For low volume fractions, agent should emphasize efficiency
    low_volfrac = volfrac < LOW_VOLFRAC_THRESHOLD
    emphasizes_efficiency = low_volfrac and any(
        [
            "efficient" in response,
            "careful" in response,
            "strategic" in response,
            "optimal" in response,
        ]
    )

    score = 1.0 if (mentions_tradeoff or emphasizes_efficiency) else 0.0

    return {
        "score": score,
        "mentions_tradeoff": mentions_tradeoff,
        "emphasizes_efficiency": emphasizes_efficiency,
        "volfrac": volfrac,
    }


@weave.op()
def score_provides_actionable_guidance(
    prompt: str,  # noqa: ARG001
    conditions: dict[str, Any],  # noqa: ARG001
    output: dict[str, Any],
    target: dict[str, Any],  # noqa: ARG001
) -> dict[str, Any]:
    """
    Score if agent provides concrete, actionable steps.

    The agent should explain HOW to approach the design, not just
    restate the constraints.
    """
    response = output["response"].lower()

    # Check for actionable verbs and concrete guidance
    actionable_terms = [
        "start by" in response,
        "first" in response and ("step" in response or "," in response),
        "should" in response,
        "place material" in response,
        "distribute" in response,
        "connect" in response,
        "path" in response and "load" in response,
        "iterative" in response or "iterate" in response,
        "algorithm" in response,
    ]

    provides_steps = sum(actionable_terms) >= MIN_ACTIONABLE_TERMS
    sufficient_length = len(response) > MIN_RESPONSE_LENGTH

    score = 1.0 if (provides_steps and sufficient_length) else 0.0

    return {
        "score": score,
        "provides_steps": provides_steps,
        "sufficient_length": sufficient_length,
    }


@weave.op()
def score_no_contradictions(
    prompt: str,  # noqa: ARG001
    conditions: dict[str, Any],
    output: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    """
    Score if agent avoids contradicting the ground truth.

    Check that the agent doesn't give wrong information about:
    - Whether the target compliance is achievable
    - The difficulty of the problem given the constraints
    """
    response = output["response"].lower()
    volfrac = conditions["volfrac"]
    target_compliance = target["compliance"]

    # Very low compliance is good (stiff), very high is bad (flexible)
    is_stiff_design = target_compliance < STIFF_COMPLIANCE_THRESHOLD
    is_flexible_design = target_compliance > FLEXIBLE_COMPLIANCE_THRESHOLD

    # Check for contradictions
    contradictions = []

    # If design is stiff (low compliance), shouldn't say it's flexible
    if is_stiff_design and "flexible" in response:
        contradictions.append("calls_stiff_design_flexible")

    # If design is flexible (high compliance), shouldn't say it's stiff
    if is_flexible_design and ("very stiff" in response or "rigid" in response):
        contradictions.append("calls_flexible_design_stiff")

    # With low volfrac, shouldn't claim it's easy or has excess material
    if volfrac < LOW_VOLFRAC_THRESHOLD and ("easy" in response or "excess" in response):
        contradictions.append("claims_low_volfrac_is_easy")

    no_contradictions = len(contradictions) == 0
    score = 1.0 if no_contradictions else 0.0

    return {
        "score": score,
        "contradictions": contradictions,
        "target_compliance": target_compliance,
    }


# Cache for HuggingFace dataset to avoid reloading
_hf_dataset_cache = None


def get_hf_dataset():
    """Load HuggingFace dataset with caching."""
    global _hf_dataset_cache  # noqa: PLW0603
    if _hf_dataset_cache is None:
        _hf_dataset_cache = load_dataset("IDEALLab/beams_2d_50_100_v0", split="train")
    return _hf_dataset_cache


def _create_design_comparison(
    agent_design: np.ndarray, ground_truth: np.ndarray, example_id: int
) -> Image.Image | None:
    """
    Create a side-by-side comparison visualization of agent and ground truth designs.

    Args:
        agent_design: Agent's optimized design array
        ground_truth: Ground truth optimal design from dataset
        example_id: Example identifier for title

    Returns:
        PIL Image object for Weave visualization, or None if creation fails
    """
    try:
        # Create figure with 3 subplots: agent design, ground truth, difference
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot agent design
        im0 = axes[0].imshow(agent_design, cmap="gray_r", vmin=0, vmax=1)
        axes[0].set_title(
            f"Agent Design (Example {example_id})", fontsize=12, fontweight="bold"
        )
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot ground truth
        im1 = axes[1].imshow(ground_truth, cmap="gray_r", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Plot absolute difference
        diff = np.abs(agent_design - ground_truth)
        im2 = axes[2].imshow(diff, cmap="Reds", vmin=0, vmax=1)
        axes[2].set_title(
            f"Difference (MAE: {diff.mean():.3f})", fontsize=12, fontweight="bold"
        )
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        pil_image = Image.open(buf).copy()
        plt.close(fig)

    except Exception as e:
        print(f"Warning: Failed to create comparison visualization: {e}")
        return None
    else:
        return pil_image


@weave.op()
def score_design_match(
    prompt: str,  # noqa: ARG001
    conditions: dict[str, Any],  # noqa: ARG001
    output: dict[str, Any],  # noqa: ARG001
    target: dict[str, Any],  # noqa: ARG001
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Score if the agent's design output matches the ground truth optimal design.

    Extracts design array from agent's internal cache (not from text response).
    Uses multiple metrics: IoU, pixel accuracy, and topology similarity.
    """
    # Try to get the design from the internal cache
    # The optimize_design tool stores the result here
    problem_type = metadata.get("problem_type", "beams2d")
    design_array = get_unified_last_design(problem_type)

    # If no design found in cache, return 0 score
    if design_array is None:
        return {
            "score": 0.0,
            "design_found": False,
            "reason": "No design array found in internal cache",
        }

    # Load ground truth design from HuggingFace dataset
    example_id = metadata.get("example_id", 0)
    hf_dataset = get_hf_dataset()

    if example_id >= len(hf_dataset):
        return {
            "score": 0.0,
            "design_found": True,
            "reason": f"Invalid example_id: {example_id}",
        }

    ground_truth = np.array(hf_dataset[example_id]["optimal_design"])

    # Validate shapes match
    if design_array.shape != ground_truth.shape:
        return {
            "score": 0.0,
            "design_found": True,
            "reason": f"Shape mismatch: agent={design_array.shape}, ground_truth={ground_truth.shape}",
            "agent_shape": str(design_array.shape),
            "gt_shape": str(ground_truth.shape),
        }

    # Calculate similarity metrics

    # 1. Intersection over Union (IoU) - treat as binary (material vs void)
    agent_binary = (design_array > BINARY_THRESHOLD).astype(int)
    gt_binary = (ground_truth > BINARY_THRESHOLD).astype(int)

    intersection = np.logical_and(agent_binary, gt_binary).sum()
    union = np.logical_or(agent_binary, gt_binary).sum()
    iou = intersection / union if union > 0 else 0.0

    # 2. Pixel-wise accuracy
    pixel_accuracy = np.mean(agent_binary == gt_binary)

    # 3. Mean squared error of density values
    mse = np.mean((design_array - ground_truth) ** 2)

    # 4. Volume fraction match
    agent_volfrac = np.mean(design_array)
    gt_volfrac = np.mean(ground_truth)
    volfrac_error = abs(agent_volfrac - gt_volfrac)

    # Overall score: weighted combination
    # IoU is most important for topology, then pixel accuracy
    score = 0.5 * iou + 0.3 * pixel_accuracy + 0.2 * (1.0 - min(volfrac_error * 2, 1.0))

    # Create visualization for Weave UI
    comparison_image = _create_design_comparison(design_array, ground_truth, example_id)

    result = {
        "score": score,
        "design_found": True,
        "iou": float(iou),
        "pixel_accuracy": float(pixel_accuracy),
        "mse": float(mse),
        "volfrac_error": float(volfrac_error),
        "agent_volfrac": float(agent_volfrac),
        "gt_volfrac": float(gt_volfrac),
    }

    # Save and add comparison image if available
    if comparison_image is not None:
        # Save to outputs directory for local viewing
        output_dir = project_root / "outputs" / "eval_comparisons"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"comparison_example_{example_id}.png"
        comparison_image.save(image_path)
        result["comparison_image_path"] = str(image_path)

        # Also encode as base64 data URL for potential inline display
        buffered = io.BytesIO()
        comparison_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        result["comparison_image_base64"] = f"data:image/png;base64,{img_str}"

    return result


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
            "metadata": {
                **prompt_data.get("metadata", {}),
                "example_id": prompt_data.get("example_id", i),
                "problem_type": "beams2d",  # Add problem type to metadata
            },
            "target": prompt_data.get("target", {}),
        }
        for i, prompt_data in enumerate(prompts[:sample_size])
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
    input_file = (
        project_root
        / "data"
        / "datasets"
        / "beam_prompts"
        / "generated"
        / "beam_prompts_50_samples.json"
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
            score_constraint_accuracy,
            score_target_awareness,
            score_understands_tradeoffs,
            score_provides_actionable_guidance,
            score_no_contradictions,
            score_design_match,
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
