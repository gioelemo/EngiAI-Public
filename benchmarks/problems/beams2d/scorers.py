"""Beams2D-specific scorer functions for evaluating topology optimization performance."""

import base64
import io
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import weave
from langchain_core.messages import ToolMessage
from PIL import Image

from benchmarks.shared.utils import (
    create_design_comparison,
    extract_design_from_tool_messages,
    get_hf_dataset,
)

# Configure logger
logger = logging.getLogger(__name__)

# Quality thresholds for topology optimization
BINARY_THRESHOLD = 0.5  # Threshold for converting density to binary (material vs void)


def _extract_compliance_from_dict(content: dict) -> dict[str, float]:
    """Extract compliance values from dict content (for dict-type tool messages)."""
    compliance_data = {}

    if "final_compliance" in content:
        compliance_data["final_compliance"] = float(content["final_compliance"])
    elif "final_c" in content:
        compliance_data["final_compliance"] = float(content["final_c"])

    if "initial_compliance" in content:
        compliance_data["initial_compliance"] = float(content["initial_compliance"])
    elif "initial_c" in content:
        compliance_data["initial_compliance"] = float(content["initial_c"])

    if "compliance_improvement" in content:
        compliance_data["improvement"] = float(content["compliance_improvement"])
    elif "c_improvement" in content:
        compliance_data["improvement"] = float(content["c_improvement"])

    return compliance_data


def _parse_string_content(content: str, example_id: int) -> dict[str, float] | None:
    """Parse string content using regex to extract compliance values."""
    logger.info(
        "Example %s: optimize_design content length: %s chars, contains 'compliance': %s",
        example_id,
        len(content),
        "compliance" in content.lower(),
    )

    try:
        final_match = re.search(
            r"['\"]?(final_compliance|final_c)['\"]?\s*:\s*([0-9.eE+-]+)", content
        )
        initial_match = re.search(
            r"['\"]?(initial_compliance|initial_c)['\"]?\s*:\s*([0-9.eE+-]+)",
            content,
        )
        improvement_match = re.search(
            r"['\"]?(compliance_improvement|c_improvement)['\"]?\s*:\s*([0-9.eE+-]+)",
            content,
        )

        if final_match:
            compliance_data = {"final_compliance": float(final_match.group(2))}
            if initial_match:
                compliance_data["initial_compliance"] = float(initial_match.group(2))
            if improvement_match:
                compliance_data["improvement"] = float(improvement_match.group(2))

            logger.info(
                "Example %s: Extracted compliance via regex: %s",
                example_id,
                compliance_data,
            )
            return compliance_data
    except (ValueError, AttributeError) as e:
        logger.debug("Example %s: Regex extraction failed: %s", example_id, e)
    return None


def extract_compliance_from_tool_messages(
    messages: list, example_id: int
) -> dict[str, float] | None:
    """Extract compliance values from optimize_design tool message.

    Args:
        messages: List of messages from agent conversation
        example_id: Example identifier for debug logging

    Returns:
        Dictionary with initial_compliance, final_compliance, improvement if found
    """
    tool_messages_checked = 0

    message_types = [type(msg).__name__ for msg in messages]
    logger.debug(
        "Example %s: Message types in conversation: %s", example_id, message_types
    )

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        tool_messages_checked += 1
        tool_name = getattr(msg, "name", "unknown")
        content = msg.content

        logger.debug(
            "Example %s: Tool message #%s - tool name: %s, content type: %s",
            example_id,
            tool_messages_checked,
            tool_name,
            type(content).__name__,
        )

        if tool_name == "optimize_design":
            logger.info(
                "Example %s: Found optimize_design tool message! Content type: %s",
                example_id,
                type(content).__name__,
            )
            logger.debug(
                "Example %s: optimize_design content: %s",
                example_id,
                str(content)[:500],
            )

        # Handle dict content
        if isinstance(content, dict):
            compliance_data = _extract_compliance_from_dict(content)
            if compliance_data:
                logger.info(
                    "Example %s: Extracted compliance from dict: %s",
                    example_id,
                    compliance_data,
                )
                return compliance_data
            continue

        # Handle string content for optimize_design tool
        if isinstance(content, str) and tool_name == "optimize_design":
            result = _parse_string_content(content, example_id)
            if result:
                return result

    logger.warning(
        "Example %s: No compliance values found in %s tool messages",
        example_id,
        tool_messages_checked,
    )
    return None


def _get_design_array(
    output: dict[str, Any], metadata: dict[str, Any], example_id: int
) -> np.ndarray | None:
    """Extract design array from messages or cache."""
    from src.tools.engibench import get_unified_last_design  # noqa: PLC0415

    messages = output.get("messages", [])
    logger.debug("Example %s: Output dict keys: %s", example_id, list(output.keys()))
    logger.debug(
        "Example %s: Output dict (excluding messages): %s",
        example_id,
        {k: v for k, v in output.items() if k != "messages"},
    )

    design_array = extract_design_from_tool_messages(messages, example_id)

    if design_array is None:
        problem_type = metadata.get("problem_type", "beams2d")
        design_array = get_unified_last_design(problem_type)
        logger.warning("Example %s: Using global cache as fallback", example_id)

    return design_array


def _calculate_design_metrics(
    design_array: np.ndarray, ground_truth: np.ndarray
) -> dict[str, float]:
    """Calculate similarity metrics between agent and ground truth topology designs.

    Includes volume fraction metrics specific to topology optimization.
    """
    agent_binary = (design_array > BINARY_THRESHOLD).astype(int)
    gt_binary = (ground_truth > BINARY_THRESHOLD).astype(int)

    intersection = np.logical_and(agent_binary, gt_binary).sum()
    union = np.logical_or(agent_binary, gt_binary).sum()
    iou = intersection / union if union > 0 else 0.0

    pixel_accuracy = np.mean(agent_binary == gt_binary)
    mse = np.mean((design_array - ground_truth) ** 2)

    # Volume fraction metrics (topology optimization specific)
    agent_volfrac = np.mean(design_array)
    gt_volfrac = np.mean(ground_truth)
    volfrac_error = abs(agent_volfrac - gt_volfrac)

    return {
        "iou": float(iou),
        "pixel_accuracy": float(pixel_accuracy),
        "mse": float(mse),
        "volfrac_error": float(volfrac_error),
        "agent_volfrac": float(agent_volfrac),
        "gt_volfrac": float(gt_volfrac),
    }


def _calculate_compliance_score(
    messages: list,
    target: dict[str, Any],
    example_id: int,
) -> tuple[float, dict[str, float | None]]:
    """Calculate compliance score and extract compliance metrics.

    Compliance is a structural mechanics metric specific to beams2d topology optimization.
    """
    tool_msg_names = [
        getattr(msg, "name", "unknown")
        for msg in messages
        if isinstance(msg, ToolMessage)
    ]
    logger.info(
        "Example %s: Tool messages found: %s",
        example_id,
        tool_msg_names if tool_msg_names else "NONE",
    )

    compliance_data = extract_compliance_from_tool_messages(messages, example_id)

    if not compliance_data:
        logger.warning(
            "Example %s: Failed to extract compliance from %s messages (tool messages: %s)",
            example_id,
            len(messages),
            tool_msg_names,
        )
        return 0.0, {
            "agent_compliance": None,
            "target_compliance": None,
            "compliance_relative_error": None,
        }

    if "final_compliance" not in compliance_data:
        return 0.0, {
            "agent_compliance": None,
            "target_compliance": None,
            "compliance_relative_error": None,
        }

    agent_compliance = compliance_data["final_compliance"]
    target_compliance = target.get("compliance") if target else None

    if not target_compliance:
        return 0.0, {
            "agent_compliance": agent_compliance,
            "target_compliance": None,
            "compliance_relative_error": None,
        }

    compliance_relative_error = (
        abs(agent_compliance - target_compliance) / target_compliance
    )
    compliance_score = max(0.0, 1.0 - compliance_relative_error / 0.2)

    logger.debug(
        "Example %s: Compliance - Agent: %.4f, Target: %.4f, Error: %.2f%%, Score: %.3f",
        example_id,
        agent_compliance,
        target_compliance,
        compliance_relative_error * 100,
        compliance_score,
    )

    return compliance_score, {
        "agent_compliance": agent_compliance,
        "target_compliance": target_compliance,
        "compliance_relative_error": compliance_relative_error,
    }


def _save_comparison_image(
    comparison_image: Image.Image,
    output: dict[str, Any],
    metadata: dict[str, Any],
    example_id: int,
) -> dict[str, str]:
    """Save comparison image and return paths/encodings."""
    problem_type = metadata.get("problem_type", "beams2d")
    model_name = output.get("model", "unknown")

    output_dir = (
        Path(__file__).parent.parent.parent
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


@weave.op()
def score_design_match(
    output: dict[str, Any],
    target: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Score beams2d topology optimization design against ground truth optimal design.

    Extracts design array from agent's tool call results in message history.
    Uses multiple metrics specific to topology optimization:
    - IoU (Intersection over Union): Binary topology match
    - Pixel accuracy: Pixel-wise density match
    - MSE: Mean squared error of density values
    - Volume fraction error: Difference in material usage (topology optimization metric)
    - Compliance performance: Structural mechanics metric for beams2d

    Overall score weights (when compliance available):
    - 40% IoU (topology match)
    - 25% pixel accuracy
    - 15% volume fraction match
    - 20% compliance performance

    NOTE: This scorer requires get_unified_last_design to be imported from src.tools.engibench
    and assumes a HuggingFace beams2d dataset is specified in metadata.
    """
    # Import here to avoid circular dependencies
    import sys  # noqa: PLC0415

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    example_id = metadata.get("example_id", 0)
    design_array = _get_design_array(output, metadata, example_id)

    if design_array is None:
        return {
            "score": 0.0,
            "design_found": False,
            "reason": "No design array found in tool results or cache",
            "num_messages": len(output.get("messages", [])),
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

    if design_array.shape != ground_truth.shape:
        return {
            "score": 0.0,
            "design_found": True,
            "reason": f"Shape mismatch: agent={design_array.shape}, ground_truth={ground_truth.shape}",
            "agent_shape": str(design_array.shape),
            "gt_shape": str(ground_truth.shape),
        }

    # Calculate metrics
    metrics = _calculate_design_metrics(design_array, ground_truth)
    messages = output.get("messages", [])
    compliance_score, compliance_metrics = _calculate_compliance_score(
        messages, target, example_id
    )

    # Calculate overall score
    if compliance_score > 0:
        score = (
            0.4 * metrics["iou"]
            + 0.25 * metrics["pixel_accuracy"]
            + 0.15 * (1.0 - min(metrics["volfrac_error"] * 2, 1.0))
            + 0.2 * compliance_score
        )
    else:
        score = (
            0.5 * metrics["iou"]
            + 0.3 * metrics["pixel_accuracy"]
            + 0.2 * (1.0 - min(metrics["volfrac_error"] * 2, 1.0))
        )

    # Build result
    result: dict[str, Any] = {
        "score": score,
        "design_found": True,
        "compliance_score": float(compliance_score),
        **metrics,
    }

    # Add compliance metrics if available
    for key, value in compliance_metrics.items():
        if value is not None:
            result[key] = float(value)

    # Save comparison image
    comparison_image = create_design_comparison(design_array, ground_truth, example_id)
    if comparison_image is not None:
        result.update(
            _save_comparison_image(comparison_image, output, metadata, example_id)
        )

    return result
