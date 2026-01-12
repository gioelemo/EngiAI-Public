"""Generic scorer that works for any topology optimization problem.

This scorer uses problem configuration to extract objectives, check constraints,
and compute scores without problem-specific code.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import weave
from PIL import Image

from benchmarks.shared.objective_extractor import (
    calculate_objective_score,
    extract_objectives_from_tool_messages,
)
from benchmarks.shared.problem_registry import get_problem_config
from benchmarks.shared.utils import (
    create_design_comparison,
    extract_design_from_tool_messages,
    get_hf_dataset,
)

logger = logging.getLogger(__name__)

# Quality thresholds for topology optimization
BINARY_THRESHOLD = 0.5  # Threshold for converting density to binary (material vs void)


def _get_design_array(
    output: dict[str, Any],
    _metadata: dict[str, Any],
    _problem_config: Any,
    example_id: int,
) -> np.ndarray | None:
    """Extract design array from messages.

    Args:
        output: Agent output with messages
        metadata: Metadata dict
        problem_config: Problem configuration
        example_id: Example ID for logging

    Returns:
        Design array or None if not found
    """
    messages = output.get("messages", [])
    design_array = extract_design_from_tool_messages(messages, example_id)

    if design_array is None:
        logger.warning(f"Example {example_id}: No design array found in messages")
    else:
        logger.debug(
            f"Example {example_id}: Extracted design with shape {design_array.shape}"
        )

    return design_array


def _calculate_design_metrics(
    design_array: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    """Calculate similarity metrics between agent and ground truth designs.

    These metrics are universal across all topology optimization problems.

    Args:
        design_array: Agent's optimized design
        ground_truth: Ground truth optimal design

    Returns:
        Dictionary with IoU, pixel accuracy, MSE metrics
    """
    agent_binary = (design_array > BINARY_THRESHOLD).astype(int)
    gt_binary = (ground_truth > BINARY_THRESHOLD).astype(int)

    intersection = np.logical_and(agent_binary, gt_binary).sum()
    union = np.logical_or(agent_binary, gt_binary).sum()
    iou = intersection / union if union > 0 else 0.0

    pixel_accuracy = np.mean(agent_binary == gt_binary)
    mse = np.mean((design_array - ground_truth) ** 2)

    return {
        "iou": float(iou),
        "pixel_accuracy": float(pixel_accuracy),
        "mse": float(mse),
    }


def _calculate_constraint_score(
    design_array: np.ndarray,
    conditions: dict[str, Any],
    problem_config: Any,
    example_id: int,
) -> tuple[float, dict[str, Any]]:
    """Calculate constraint matching score based on problem config.

    Args:
        design_array: Agent's design
        conditions: Target conditions from dataset
        problem_config: Problem configuration
        example_id: Example ID for logging

    Returns:
        Tuple of (constraint_score, detailed_metrics)
    """
    constraint_conditions = problem_config.get_constraint_conditions()

    if not constraint_conditions:
        logger.debug(f"Example {example_id}: No constraints to check")
        return 1.0, {}

    violations = 0
    metrics = {}

    for cond_config in constraint_conditions:
        # Get target value from conditions
        target_value = conditions.get(cond_config.field_name)
        if target_value is None:
            # Try aliases
            for alias in cond_config.aliases:
                target_value = conditions.get(alias)
                if target_value is not None:
                    break

        if target_value is None:
            logger.debug(
                f"Example {example_id}: No target value for {cond_config.name}"
            )
            continue

        # Check constraint based on type
        if cond_config.constraint_type == "equality":
            # For equality constraints on design array (e.g., volume fraction)
            if (
                cond_config.name == "volume_fraction"
                or "volume" in cond_config.name.lower()
            ):
                actual_value = np.mean(design_array)
                error = abs(actual_value - target_value)

                metrics[f"{cond_config.name}_actual"] = float(actual_value)
                metrics[f"{cond_config.name}_target"] = float(target_value)
                metrics[f"{cond_config.name}_error"] = float(error)

                # Check if within tolerance
                if error >= cond_config.tolerance:
                    violations += 1
                    logger.debug(
                        f"Example {example_id}: {cond_config.name} violated - "
                        f"actual={actual_value:.4f}, target={target_value:.4f}, "
                        f"error={error:.4f} >= tolerance={cond_config.tolerance}"
                    )

        elif cond_config.constraint_type == "inequality":
            # Placeholder for inequality constraints (can be extended)
            logger.debug(
                f"Example {example_id}: Inequality constraints not yet implemented"
            )

    # Calculate score: 1.0 if no violations, 0.0 if any violations
    # (Can be made more nuanced with partial credit)
    constraint_score = 1.0 if violations == 0 else 0.0
    metrics["constraint_violations"] = violations

    return constraint_score, metrics


def _save_comparison_image(
    comparison_image: Image.Image,
    output: dict[str, Any],
    metadata: dict[str, Any],
    example_id: int,
) -> dict[str, str]:
    """Save comparison image and return paths/encodings."""
    problem_type = metadata.get("problem_type", "unknown")
    model_name = output.get("model", "unknown")

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

    buffered = io.BytesIO()
    comparison_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {
        "comparison_image_path": str(image_path),
        "comparison_image_base64": f"data:image/png;base64,{img_str}",
    }


def _create_error_result(error_info: dict[str, Any]) -> dict[str, Any]:
    """Create a standardized error result with score=0.0."""
    return {"score": 0.0, **error_info}


def _validate_inputs(
    metadata: dict[str, Any],
) -> tuple[str, int, Any] | dict[str, Any]:
    """Validate metadata and get problem config.

    Returns (problem_name, example_id, problem_config) or error dict.
    """
    example_id = metadata.get("example_id", 0)
    problem_name = metadata.get("problem_type")

    if not problem_name:
        return _create_error_result({"error": "No problem_type in metadata"})

    try:
        problem_config = get_problem_config(problem_name)
    except ValueError as e:
        return _create_error_result({"error": str(e)})

    return problem_name, example_id, problem_config


def _load_dataset_and_ground_truth(
    metadata: dict[str, Any],
    problem_config: Any,
    example_id: int,
    design_array: np.ndarray,
) -> tuple[Any, np.ndarray] | dict[str, Any]:
    """Load dataset and ground truth, validate shape.

    Returns (hf_dataset, ground_truth) or error dict.
    """
    dataset_name = metadata.get("dataset_name", problem_config.dataset_name)
    dataset_split = metadata.get("dataset_split", "test")

    try:
        hf_dataset = get_hf_dataset(dataset_name, split=dataset_split)
    except Exception as e:
        return _create_error_result(
            {
                "design_found": True,
                "error": f"Failed to load dataset: {e}",
            }
        )

    if example_id >= len(hf_dataset):
        return _create_error_result(
            {
                "design_found": True,
                "reason": f"Invalid example_id: {example_id} for split: {dataset_split}",
            }
        )

    ground_truth = np.array(hf_dataset[example_id][problem_config.design_field])

    if design_array.shape != ground_truth.shape:
        return _create_error_result(
            {
                "design_found": True,
                "reason": f"Shape mismatch: agent={design_array.shape}, ground_truth={ground_truth.shape}",
                "agent_shape": str(design_array.shape),
                "gt_shape": str(ground_truth.shape),
            }
        )

    return hf_dataset, ground_truth


@weave.op()
def score_design_generic(
    output: dict[str, Any],
    target: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Generic scorer for any topology optimization problem.

    Uses problem configuration to:
    - Extract design from messages
    - Calculate design similarity metrics (IoU, pixel accuracy, MSE)
    - Check constraints (volume fraction, etc.)
    - Extract and score objectives (compliance, overlap, etc.)
    - Compute weighted overall score

    Args:
        output: Agent output with messages and model info
        target: Target/ground truth data from dataset
        metadata: Metadata including problem_type, dataset_name, example_id

    Returns:
        Dictionary with score and detailed metrics
    """
    # Validate inputs and get problem config
    validation_result = _validate_inputs(metadata)
    if isinstance(validation_result, dict):
        return validation_result

    problem_name, example_id, problem_config = validation_result

    # Extract design array
    design_array = _get_design_array(output, metadata, problem_config, example_id)
    if design_array is None:
        return _create_error_result(
            {
                "design_found": False,
                "reason": "No design array found in tool results",
                "num_messages": len(output.get("messages", [])),
            }
        )

    # Load dataset and ground truth
    dataset_result = _load_dataset_and_ground_truth(
        metadata, problem_config, example_id, design_array
    )
    if isinstance(dataset_result, dict):
        return dataset_result

    hf_dataset, ground_truth = dataset_result

    # Calculate design metrics (universal)
    design_metrics = _calculate_design_metrics(design_array, ground_truth)

    # Get conditions from target or dataset row
    conditions = target if isinstance(target, dict) else {}
    if not conditions and "conditions" in hf_dataset[example_id]:
        conditions = hf_dataset[example_id]["conditions"]

    # Calculate constraint matching score
    constraint_score, constraint_metrics = _calculate_constraint_score(
        design_array, conditions, problem_config, example_id
    )

    # Extract objectives from messages
    messages = output.get("messages", [])
    agent_objectives = extract_objectives_from_tool_messages(
        messages, problem_config, example_id
    )

    # Get target objectives from dataset
    target_objectives = {}
    for obj_config in problem_config.objectives:
        # Try to get from target dict
        value = (
            target.get(obj_config.target_field) if isinstance(target, dict) else None
        )

        # If not in target, try dataset row directly
        if value is None:
            value = hf_dataset[example_id].get(obj_config.target_field)

        target_objectives[obj_config.name] = value

    # Calculate objective matching score
    objective_score, objective_metrics = calculate_objective_score(
        agent_objectives, target_objectives, problem_config, example_id
    )

    # Compute weighted overall score
    weights = problem_config.design_metrics_weights
    score = (
        weights["iou"] * design_metrics["iou"]
        + weights["pixel_accuracy"] * design_metrics["pixel_accuracy"]
        + weights["constraint_match"] * constraint_score
        + weights["objective_match"] * objective_score
    )

    # Build result dictionary
    result: dict[str, Any] = {
        "score": float(score),
        "design_found": True,
        "problem_type": problem_name,
        "constraint_score": float(constraint_score),
        "objective_score": float(objective_score),
        **design_metrics,
        **constraint_metrics,
        **objective_metrics,
    }

    # Save comparison image if possible
    try:
        # Pass problem_type and conditions for physics-based visualizations
        comparison_image = create_design_comparison(
            design_array,
            ground_truth,
            example_id,
            problem_type=problem_name,
            conditions=conditions,
        )
        if comparison_image is not None:
            image_data = _save_comparison_image(
                comparison_image, output, metadata, example_id
            )
            result.update(image_data)
            result["comparison_image_generated"] = True
    except Exception as e:
        logger.debug(f"Example {example_id}: Failed to create comparison image: {e}")
        result["comparison_image_generated"] = False

    return result
