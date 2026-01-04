"""Shared scorer functions for evaluating agent performance across problem types."""

import base64
import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import weave
from datasets import load_dataset

from benchmarks.shared.utils import (
    create_design_comparison,
    extract_design_from_tool_messages,
)

# Configure logger
logger = logging.getLogger(__name__)

# Quality thresholds (can be overridden per problem type)
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

# Cache for HuggingFace datasets to avoid reloading
_hf_dataset_cache: dict[str, Any] = {}


def get_hf_dataset(dataset_name: str):
    """Load HuggingFace dataset with caching.

    Args:
        dataset_name: Name of the HuggingFace dataset to load

    Returns:
        Loaded dataset
    """
    if dataset_name not in _hf_dataset_cache:
        _hf_dataset_cache[dataset_name] = load_dataset(dataset_name, split="train")
    return _hf_dataset_cache[dataset_name]


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


@weave.op()
def score_design_match(
    prompt: str,  # noqa: ARG001
    conditions: dict[str, Any],  # noqa: ARG001
    output: dict[str, Any],
    target: dict[str, Any],  # noqa: ARG001
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Score if the agent's design output matches the ground truth optimal design.

    Extracts design array from agent's tool call results in message history.
    Uses multiple metrics: IoU, pixel accuracy, and topology similarity.

    NOTE: This scorer requires get_unified_last_design to be imported from src.tools.engibench
    and assumes a HuggingFace dataset is specified in metadata.
    """
    # Import here to avoid circular dependencies
    import sys  # noqa: PLC0415
    from pathlib import Path as PathLib  # noqa: PLC0415

    # Add project root to path
    project_root = PathLib(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.tools.engibench import get_unified_last_design  # noqa: PLC0415

    # Extract design from tool call results in message history
    messages = output.get("messages", [])
    example_id = metadata.get("example_id", 0)

    # Use helper function to extract design from messages
    design_array = extract_design_from_tool_messages(messages, example_id)

    # If no design found in messages, try the global cache as fallback
    if design_array is None:
        problem_type = metadata.get("problem_type", "beams2d")
        design_array = get_unified_last_design(problem_type)
        logger.warning("Example %s: Using global cache as fallback", example_id)

    # If still no design found, return 0 score
    if design_array is None:
        return {
            "score": 0.0,
            "design_found": False,
            "reason": "No design array found in tool results or cache",
            "num_messages": len(messages),
        }

    # Load ground truth design from HuggingFace dataset
    dataset_name = metadata.get("dataset_name", "IDEALLab/beams_2d_50_100_v0")
    hf_dataset = get_hf_dataset(dataset_name)

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
    comparison_image = create_design_comparison(design_array, ground_truth, example_id)

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
        # Determine output directory based on problem type and model
        problem_type = metadata.get("problem_type", "beams2d")
        model_name = output.get("model", "unknown")

        # Save to results directory organized by model and problem
        output_dir = (
            Path(__file__).parent.parent
            / "evaluations"
            / "results"
            / model_name
            / problem_type
            / "comparisons"
        )
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
