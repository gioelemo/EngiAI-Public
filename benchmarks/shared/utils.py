"""Shared utility functions for benchmark evaluations."""

import ast
import io
import json
import logging
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from langchain_core.messages import ToolMessage
from PIL import Image

# Use Agg backend for matplotlib to avoid threading issues in parallel evaluation
matplotlib.use("Agg")

# Debug output configuration
DEBUG_PREVIEW_LENGTH = 200  # Characters to show in debug preview for design messages
DEBUG_PREVIEW_SHORT = 150  # Characters to show in debug preview for other messages

# Configure logger
logger = logging.getLogger(__name__)


def extract_design_from_tool_messages(
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
        content_preview = (
            content[:DEBUG_PREVIEW_LENGTH]
            if len(content) > DEBUG_PREVIEW_LENGTH
            else content
        )
        logger.debug("Example %s: Found optimized_design in tool message", example_id)
        logger.debug("Content preview: %s...", content_preview)

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
                # Extract just the optimized_design field using regex
                match = re.search(
                    r"'optimized_design':\s*(\[\[.*?\]\])", content, re.DOTALL
                )
                if match:
                    design_list_str = match.group(1)
                    design_list = ast.literal_eval(design_list_str)
                    result = {"optimized_design": design_list}
                    logger.debug(
                        "Example %s: Extracted optimized_design with regex", example_id
                    )
                else:
                    logger.debug(
                        "Example %s: Could not find optimized_design pattern",
                        example_id,
                    )
            except (ValueError, SyntaxError) as e2:
                logger.debug("Example %s: Regex extraction failed: %s", example_id, e2)

        # Extract design array if parsing succeeded
        if result and isinstance(result, dict) and "optimized_design" in result:
            design_array = np.array(result["optimized_design"])
            logger.debug(
                "Example %s: Design extracted from message history (shape: %s)",
                example_id,
                design_array.shape,
            )
            break

        logger.debug(
            "Example %s: Result parsed but no optimized_design field found", example_id
        )

    if tool_messages_count == 0:
        logger.debug(
            "Example %s: No tool messages found in %s total messages",
            example_id,
            len(messages),
        )

    return design_array


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

    # First, let's see what types of messages we have
    message_types = [type(msg).__name__ for msg in messages]
    logger.debug("Example %s: Message types in conversation: %s", example_id, message_types)

    for msg in messages:
        # Check if this is a tool message using isinstance
        if not isinstance(msg, ToolMessage):
            continue

        tool_messages_checked += 1

        # Debug: Log which tool this message is from
        tool_name = getattr(msg, "name", "unknown")
        logger.debug(
            "Example %s: Tool message #%s - tool name: %s, content type: %s",
            example_id,
            tool_messages_checked,
            tool_name,
            type(msg.content).__name__
        )

        content = msg.content

        # Log the raw content for optimize_design messages
        if tool_name == "optimize_design":
            logger.info(
                "Example %s: Found optimize_design tool message! Content type: %s",
                example_id,
                type(content).__name__
            )
            logger.debug("Example %s: optimize_design content: %s", example_id, str(content)[:500])

        # Handle both string and dict content
        if isinstance(content, dict):
            # Content is already a dict - extract directly
            # Try both full names and abbreviated names (c, initial_c, final_c)
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

            if compliance_data:
                logger.info(
                    "Example %s: Extracted compliance from dict: %s",
                    example_id,
                    compliance_data,
                )
                return compliance_data
            continue

        if not isinstance(content, str):
            continue

        # Log what we found for non-optimize_design tools
        if tool_name != "optimize_design":
            continue

        # At this point, we have the optimize_design tool message!
        logger.info(
            "Example %s: Processing optimize_design tool message (content type: %s)",
            example_id,
            type(content).__name__
        )

        # Check if compliance is in the content
        has_compliance = "compliance" in content.lower()
        logger.info(
            "Example %s: optimize_design content length: %s chars, contains 'compliance': %s",
            example_id,
            len(content),
            has_compliance,
        )

        # Show both start and end of content
        start_preview = content[:300]
        end_preview = content[-500:] if len(content) > 500 else content
        logger.info("Example %s: Content START: %s...", example_id, start_preview)
        logger.info("Example %s: Content END: ...%s", example_id, end_preview)

        # Try multiple parsing approaches
        # Approach 1: Try ast.literal_eval (for Python dict string representation)
        try:
            result = ast.literal_eval(content)

            # Extract compliance values if available
            # Note: The tool may use abbreviated names (c, initial_c, final_c) or full names
            if isinstance(result, dict):
                compliance_data = {}

                # Try full names first, then abbreviated names
                if "final_compliance" in result:
                    compliance_data["final_compliance"] = float(result["final_compliance"])
                elif "final_c" in result:
                    compliance_data["final_compliance"] = float(result["final_c"])

                if "initial_compliance" in result:
                    compliance_data["initial_compliance"] = float(result["initial_compliance"])
                elif "initial_c" in result:
                    compliance_data["initial_compliance"] = float(result["initial_c"])

                if "compliance_improvement" in result:
                    compliance_data["improvement"] = float(result["compliance_improvement"])
                elif "c_improvement" in result:
                    compliance_data["improvement"] = float(result["c_improvement"])

                if compliance_data:
                    logger.info(
                        "Example %s: Extracted compliance via ast.literal_eval: %s",
                        example_id,
                        compliance_data,
                    )
                    return compliance_data

        except (ValueError, SyntaxError) as e:
            logger.debug(
                "Example %s: ast.literal_eval failed: %s, trying JSON...", example_id, e
            )

        # Approach 2: Try JSON parsing (fallback)
        try:
            # Replace Python-specific syntax with JSON equivalents
            json_content = content.replace("'", '"')
            json_content = json_content.replace("True", "true")
            json_content = json_content.replace("False", "false")
            json_content = json_content.replace("None", "null")

            result = json.loads(json_content)

            # Extract compliance values if available
            if isinstance(result, dict):
                compliance_data = {}
                if "final_compliance" in result:
                    compliance_data["final_compliance"] = float(
                        result["final_compliance"]
                    )
                if "initial_compliance" in result:
                    compliance_data["initial_compliance"] = float(
                        result["initial_compliance"]
                    )
                if "improvement" in result:
                    compliance_data["improvement"] = float(result["improvement"])

                if compliance_data:
                    logger.info(
                        "Example %s: Successfully extracted compliance: %s",
                        example_id,
                        compliance_data,
                    )
                    return compliance_data

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(
                "Example %s: JSON parsing failed: %s, trying regex...", example_id, e
            )

        # Approach 3: Try regex extraction for numeric values
        # Try both full names (final_compliance) and abbreviated names (final_c)
        try:
            # Look for patterns like "final_compliance": 123.45 or 'final_c': 123.45
            final_match = re.search(
                r"['\"]?(final_compliance|final_c)['\"]?\s*:\s*([0-9.eE+-]+)", content
            )
            initial_match = re.search(
                r"['\"]?(initial_compliance|initial_c)['\"]?\s*:\s*([0-9.eE+-]+)", content
            )
            improvement_match = re.search(
                r"['\"]?(compliance_improvement|c_improvement)['\"]?\s*:\s*([0-9.eE+-]+)", content
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

    logger.warning(
        "Example %s: No compliance values found in %s tool messages",
        example_id,
        tool_messages_checked,
    )
    return None


def create_design_comparison(
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

    except Exception as e:
        print(f"Warning: Failed to create comparison visualization: {e}")
        return None
    else:
        return pil_image
