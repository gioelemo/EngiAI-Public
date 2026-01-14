"""Shared utility functions for benchmark evaluations."""

import ast
import io
import json
import logging
import re
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image

# Use Agg backend for matplotlib to avoid threading issues in parallel evaluation
matplotlib.use("Agg")

# Debug output configuration
DEBUG_PREVIEW_LENGTH = 200  # Characters to show in debug preview for design messages
DEBUG_PREVIEW_SHORT = 150  # Characters to show in debug preview for other messages

# Configure logger
logger = logging.getLogger(__name__)

# Cache for HuggingFace datasets to avoid reloading
_hf_dataset_cache: dict[str, Any] = {}


def get_hf_dataset(dataset_name: str, split: str = "test"):
    """Load a HuggingFace dataset with split-specific caching.

    Args:
        dataset_name: Name of the HuggingFace dataset to load.
        split: Dataset split to load (for example, "train", "validation", "test").
            The split name is also part of the cache key, so each (dataset_name, split)
            combination is cached separately.

    Returns:
        The loaded dataset for the requested ``dataset_name`` and ``split``. Repeated calls
        with the same arguments reuse the cached dataset instance instead of reloading it.
    """
    cache_key = f"{dataset_name}:{split}"
    if cache_key not in _hf_dataset_cache:
        _hf_dataset_cache[cache_key] = load_dataset(dataset_name, split=split)
    return _hf_dataset_cache[cache_key]


def extract_design_from_tool_messages(  # noqa: PLR0912, PLR0915
    messages: list, example_id: int
) -> np.ndarray | None:
    """Extract optimized design array from tool message history.

    Args:
        messages: List of messages from agent conversation
        example_id: Example identifier for debug logging

    Returns:
        Design array if found, None otherwise
    """
    design_array = None
    tool_messages_count = 0

    for msg in messages:
        # Check if this is a tool message (has tool_call_id)
        if not hasattr(msg, "tool_call_id"):
            continue

        tool_messages_count += 1
        content = msg.content

        if not isinstance(content, str):
            continue

        if "optimized_design" not in content:
            # Debug: show what the content looks like
            content_preview = (
                content[:DEBUG_PREVIEW_SHORT]
                if len(content) > DEBUG_PREVIEW_SHORT
                else content
            )
            logger.debug(
                "Example %s: Tool msg #%s: %s...",
                example_id,
                tool_messages_count,
                content_preview,
            )
            continue

        # Found optimized_design in message
        logger.debug("Example %s: Found optimized_design in tool message", example_id)

        # Parse the tool response - try multiple approaches
        result = None

        # Approach 1: Try converting Python repr to JSON
        try:
            # Replace Python-specific syntax with JSON equivalents
            json_content = content.replace("'", '"')
            json_content = json_content.replace("True", "true")
            json_content = json_content.replace("False", "false")
            json_content = json_content.replace("None", "null")

            # Remove array() calls by extracting just the content inside
            json_content = re.sub(r"array\((.*?)\)", r"\1", json_content)

            result = json.loads(json_content)
            logger.debug("Example %s: Parsed with JSON conversion", example_id)
        except (json.JSONDecodeError, Exception) as e:
            logger.debug("Example %s: JSON conversion failed: %s", example_id, e)

            # Approach 2: Try ast.literal_eval on a simplified version
            try:
                # Extract just the optimized_design field using bracket balancing
                key_match = re.search(r"'optimized_design'\s*:\s*", content)
                if key_match:
                    idx = key_match.end()
                    # Skip any whitespace after the colon
                    while idx < len(content) and content[idx].isspace():
                        idx += 1

                    # Expect the optimized_design value to start with a list '['
                    if idx < len(content) and content[idx] == "[":
                        start = idx
                        bracket_count = 0
                        end = None

                        for i in range(start, len(content)):
                            ch = content[i]
                            if ch == "[":
                                bracket_count += 1
                            elif ch == "]":
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end = i + 1
                                    break

                        if end is not None and bracket_count == 0:
                            design_list_str = content[start:end]
                            design_list = ast.literal_eval(design_list_str)
                            result = {"optimized_design": design_list}
                            logger.debug(
                                "Example %s: Extracted optimized_design with bracket balancing",
                                example_id,
                            )
                        else:
                            logger.debug(
                                "Example %s: Unbalanced brackets when extracting optimized_design",
                                example_id,
                            )
                    else:
                        logger.debug(
                            "Example %s: optimized_design value does not start with '['",
                            example_id,
                        )
                else:
                    logger.debug(
                        "Example %s: Could not find optimized_design key pattern",
                        example_id,
                    )
            except (ValueError, SyntaxError) as e2:
                logger.debug(
                    "Example %s: Bracket-balanced extraction failed: %s",
                    example_id,
                    e2,
                )

        # Extract design array if parsing succeeded
        if result and isinstance(result, dict) and "optimized_design" in result:
            design_array = np.array(result["optimized_design"])
            logger.debug(
                "Example %s: Design extracted from message history (shape: %s)",
                example_id,
                design_array.shape,
            )
            # Continue iterating to get the LAST occurrence (most recent design)
            # Don't break here - we want the final design, not the first one

        else:
            logger.debug(
                "Example %s: Result parsed but no optimized_design field found",
                example_id,
            )

    if tool_messages_count == 0:
        logger.debug(
            "Example %s: No tool messages found in %s total messages",
            example_id,
            len(messages),
        )

    return design_array


def extract_optimization_history_from_tool_messages(  # noqa: PLR0912, PLR0915
    messages: list, example_id: int
) -> list[dict[str, Any]] | None:
    """Extract optimization history from tool message history.

    Args:
        messages: List of messages from agent conversation
        example_id: Example identifier for debug logging

    Returns:
        List of optimization steps if found, None otherwise
    """
    optimization_history = None
    tool_messages_count = 0

    for msg in messages:
        # Check if this is a tool message (has tool_call_id)
        if not hasattr(msg, "tool_call_id"):
            continue

        tool_messages_count += 1
        content = msg.content

        if not isinstance(content, str):
            continue

        if "optimization_info" not in content:
            # Debug: Check what keys are actually in the content
            if "optimized_design" in content or "success" in content:
                logger.debug(
                    f"Example {example_id}: Tool msg #{tool_messages_count}: Found tool result but no optimization_info"
                )
            continue

        # Found optimization_info in message
        logger.info(
            f"Example {example_id}: Found optimization_info in tool message #{tool_messages_count}"
        )

        # Log a preview of the content for debugging
        max_preview_length = 500
        content_preview = (
            content[:max_preview_length]
            if len(content) > max_preview_length
            else content
        )
        logger.debug(
            f"Example {example_id}: Message content preview: {content_preview}"
        )

        # Parse the tool response - try multiple approaches
        result = None

        # Approach 1: Try converting Python repr to JSON
        try:
            # Replace Python-specific syntax with JSON equivalents
            json_content = content.replace("'", '"')
            json_content = json_content.replace("True", "true")
            json_content = json_content.replace("False", "false")
            json_content = json_content.replace("None", "null")

            # Remove array() calls by extracting just the content inside
            json_content = re.sub(r"array\((.*?)\)", r"\1", json_content)

            result = json.loads(json_content)
            logger.debug("Example %s: Parsed with JSON conversion", example_id)
        except (json.JSONDecodeError, Exception) as e:
            logger.debug("Example %s: JSON conversion failed: %s", example_id, e)

            # Approach 2: Try ast.literal_eval on a simplified version
            try:
                # Extract just the optimization_info field using bracket balancing
                key_match = re.search(r"'optimization_info'\s*:\s*", content)
                if key_match:
                    idx = key_match.end()
                    # Skip any whitespace after the colon
                    while idx < len(content) and content[idx].isspace():
                        idx += 1

                    # Expect the optimization_info value to start with a list '[' or dict '{'
                    if idx < len(content) and content[idx] in ["[", "{"]:
                        start = idx
                        bracket_count = 0
                        end = None
                        open_char = content[idx]
                        close_char = "]" if open_char == "[" else "}"

                        for i in range(start, len(content)):
                            ch = content[i]
                            if ch == open_char:
                                bracket_count += 1
                            elif ch == close_char:
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end = i + 1
                                    break

                        if end is not None and bracket_count == 0:
                            opt_info_str = content[start:end]
                            opt_info_data = ast.literal_eval(opt_info_str)

                            # Check if it's a list (step-by-step history) or dict (summary)
                            if isinstance(opt_info_data, list):
                                result = {"optimization_info": opt_info_data}
                                logger.info(
                                    f"Example {example_id}: Extracted optimization_info list with {len(opt_info_data)} steps"
                                )
                            elif isinstance(opt_info_data, dict):
                                # It's a summary dict, not a step-by-step history
                                logger.warning(
                                    f"Example {example_id}: optimization_info is a dict (summary), not a list (step history). "
                                    f"Keys: {list(opt_info_data.keys())}"
                                )
                                result = {"optimization_info": opt_info_data}
                            else:
                                logger.warning(
                                    f"Example {example_id}: optimization_info has unexpected type: {type(opt_info_data)}"
                                )
                        else:
                            logger.debug(
                                "Example %s: Unbalanced brackets when extracting optimization_info",
                                example_id,
                            )
                    else:
                        logger.warning(
                            f"Example {example_id}: optimization_info value does not start with '[' or '{{'. "
                            f"Found: {content[idx] if idx < len(content) else 'EOF'}"
                        )
                else:
                    logger.debug(
                        "Example %s: Could not find optimization_info key pattern",
                        example_id,
                    )
            except (ValueError, SyntaxError) as e2:
                logger.debug(
                    "Example %s: Bracket-balanced extraction failed: %s",
                    example_id,
                    e2,
                )

        # Extract optimization history if parsing succeeded
        if result and isinstance(result, dict) and "optimization_info" in result:
            opt_info = result["optimization_info"]

            # If it's a list, treat it as step-by-step history
            if isinstance(opt_info, list):
                optimization_history = opt_info
                logger.info(
                    f"Example {example_id}: Extracted step-by-step optimization history ({len(optimization_history)} steps)"
                )
                break
            # If it's a dict with summary stats, it's not a step history
            elif isinstance(opt_info, dict):
                logger.warning(
                    f"Example {example_id}: optimization_info is a summary dict, not step-by-step history. "
                    f"Cannot compute optimality gap metrics without step history."
                )
                optimization_history = []
                break
            else:
                logger.warning(
                    f"Example {example_id}: optimization_info has unexpected type: {type(opt_info)}"
                )

        logger.debug(
            "Example %s: Result parsed but no optimization_info field found", example_id
        )

    if tool_messages_count == 0:
        logger.debug(
            "Example %s: No tool messages found in %s total messages",
            example_id,
            len(messages),
        )

    return optimization_history


def create_design_comparison(
    agent_design: np.ndarray,
    ground_truth: np.ndarray,
    example_id: int,
    problem_type: str | None = None,  # noqa: ARG001
    conditions: dict[str, Any] | None = None,  # noqa: ARG001
) -> Image.Image | None:
    """
    Create a side-by-side comparison visualization of agent and ground truth designs.

    Shows material distribution comparison for all problem types.

    Args:
        agent_design: Agent's optimized design array
        ground_truth: Ground truth optimal design from dataset
        example_id: Example identifier for title
        problem_type: Problem type (e.g., "photonics2d") to determine visualization method
        conditions: Problem conditions (e.g., wavelengths) needed for physics simulations

    Returns:
        PIL Image object for Weave visualization, or None if creation fails
    """
    # Use simple material-only comparison for all problem types (including photonics2d)
    try:
        # Create figure with 3 subplots: agent design, ground truth, difference
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot agent design
        im0 = axes[0].imshow(agent_design, cmap="gray_r", vmin=0, vmax=1)
        axes[0].set_title(
            f"Agent Design (Example {example_id})", fontsize=12, fontweight="bold"
        )
        axes[0].axis("off")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot ground truth
        im1 = axes[1].imshow(ground_truth, cmap="gray_r", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Plot absolute difference
        diff = np.abs(agent_design - ground_truth)
        im2 = axes[2].imshow(diff, cmap="Reds", vmin=0, vmax=1)
        axes[2].set_title(
            f"Difference (MAE: {diff.mean():.3f})", fontsize=12, fontweight="bold"
        )
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        fig.tight_layout()

        # Convert matplotlib figure to PIL Image
        # Use figure-level methods to avoid global state issues with parallel execution
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        pil_image = Image.open(buf).copy()
        buf.close()
        plt.close(fig)

    except Exception:
        logger.exception(
            "Failed to create comparison visualization for example_id %s", example_id
        )
        return None
    else:
        return pil_image
