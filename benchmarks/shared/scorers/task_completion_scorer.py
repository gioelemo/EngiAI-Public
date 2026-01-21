"""Task completion scorer for engineering agent evaluations.

This scorer checks whether the render_design tool was called successfully,
indicating that the agent completed its task of generating and rendering a design.
"""

import ast
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Tool name to check for task completion
RENDER_TOOL_NAME = "render_design"


def _parse_tool_result(content: str, example_id: int) -> dict[str, Any] | None:
    """Parse tool result from message content.

    Handles both JSON and Python repr formats.

    Args:
        content: Raw tool message content string
        example_id: Example ID for logging

    Returns:
        Parsed dictionary or None if parsing fails
    """
    if not isinstance(content, str):
        return None

    # Approach 1: Try converting Python repr to JSON
    try:
        json_content = content.replace("'", '"')
        json_content = json_content.replace("True", "true")
        json_content = json_content.replace("False", "false")
        json_content = json_content.replace("None", "null")
        # Remove array() calls
        json_content = re.sub(r"array\((.*?)\)", r"\1", json_content)
        return json.loads(json_content)
    except (json.JSONDecodeError, Exception) as e:
        logger.debug("Example %s: JSON conversion failed: %s", example_id, e)

    # Approach 2: Try ast.literal_eval
    try:
        return ast.literal_eval(content)
    except (ValueError, SyntaxError) as e:
        logger.debug("Example %s: ast.literal_eval failed: %s", example_id, e)

    return None


def score_task_completion(
    output: dict[str, Any],
    target: dict[str, Any],  # noqa: ARG001 - Required by scorer interface
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Score whether the render_design tool was called successfully.

    This scorer checks for task completion by verifying that:
    1. The render_design tool was called at least once
    2. The tool returned success=True (indicating an image was saved)

    Args:
        output: Agent output with messages and model info
        target: Target/ground truth data from dataset (unused)
        metadata: Metadata including example_id, problem_type, etc.

    Returns:
        Dictionary with:
        - success_rate: 1.0 if render_design succeeded, 0.0 otherwise (main score)
        - render_called: Whether render_design was called
        - render_success: Whether render_design returned success=True
        - render_save_path: Path where image was saved (if successful)
        - render_npy_path: Path where .npy array was saved (if successful)
        - render_error: Error message (if failed)
        - render_details: Additional details from render result
        - example_id: Example identifier
    """
    example_id = metadata.get("example_id", 0)
    messages = output.get("messages", [])

    # Track render_design calls
    render_called = False
    render_success = False
    render_save_path = None
    render_npy_path = None
    render_error = None
    render_details: dict[str, Any] = {}

    # Iterate through messages looking for render_design tool results
    for msg in messages:
        # Check if this is a tool message (has tool_call_id attribute)
        if not hasattr(msg, "tool_call_id"):
            continue

        # Get tool name - try both 'name' attribute and checking content
        tool_name = getattr(msg, "name", None)

        # If no name attribute, try to infer from content
        if tool_name is None:
            content = msg.content
            if isinstance(content, str) and "save_path" in content:
                # Likely a render_design result based on content
                tool_name = RENDER_TOOL_NAME

        # Skip if not the render tool
        if tool_name != RENDER_TOOL_NAME:
            continue

        render_called = True
        logger.debug("Example %s: Found render_design tool call", example_id)

        # Parse the tool result
        content = msg.content
        result = _parse_tool_result(content, example_id)

        if result is None:
            render_error = "Failed to parse render_design result"
            logger.warning(
                "Example %s: Could not parse render_design result", example_id
            )
            continue

        # Check success flag
        if result.get("success", False):
            render_success = True
            render_save_path = result.get("save_path")
            render_npy_path = result.get("npy_path")
            render_details = {
                "problem_type": result.get("problem_type"),
                "design_shape": result.get("design_shape"),
                "design_type": result.get("design_type"),
                "seed": result.get("seed"),
                "message": result.get("message"),
            }
            logger.debug(
                "Example %s: render_design succeeded, saved to %s",
                example_id,
                render_save_path,
            )
            # Continue iterating to find the LAST successful render (most recent)
        else:
            render_error = result.get("error", "Unknown error")
            logger.debug(
                "Example %s: render_design failed with error: %s",
                example_id,
                render_error,
            )

    # Compute final score
    # success_rate is 1.0 if render was called AND succeeded, 0.0 otherwise
    success_rate = 1.0 if (render_called and render_success) else 0.0

    # Log summary
    if not render_called:
        logger.info("Example %s: render_design was never called", example_id)
    elif not render_success:
        logger.info(
            "Example %s: render_design was called but failed: %s",
            example_id,
            render_error,
        )
    else:
        logger.info(
            "Example %s: Task completed successfully (image saved to %s)",
            example_id,
            render_save_path,
        )

    return {
        "success_rate": success_rate,
        "render_called": render_called,
        "render_success": render_success,
        "render_save_path": render_save_path,
        "render_npy_path": render_npy_path,
        "render_error": render_error,
        "render_details": render_details,
        "example_id": example_id,
    }
