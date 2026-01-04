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

# Quality thresholds
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
