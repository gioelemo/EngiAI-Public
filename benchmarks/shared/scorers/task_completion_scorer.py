"""Task completion scorer for engineering agent evaluations.

This scorer checks task completion based on prompt style:
- Standard prompts (full, approximate, natural): render_design must be called successfully
- Workflow prompts: convert_design_to_stl must be called successfully
"""

import ast
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Tool names to check for task completion
RENDER_TOOL_NAME = "render_design"
STL_EXPORT_TOOL_NAME = "convert_design_to_stl"


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


def score_task_completion(  # noqa: PLR0912, PLR0915 - Complex scoring logic
    output: dict[str, Any],
    target: dict[str, Any],  # noqa: ARG001 - Required by scorer interface
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Score task completion based on prompt style.

    For standard prompts (full, approximate, natural):
    - Success = render_design called and returned success=True

    For workflow prompts:
    - Success = convert_design_to_stl called and returned success=True
    - render_design is optional and not counted toward success

    Args:
        output: Agent output with messages and model info
        target: Target/ground truth data from dataset (unused)
        metadata: Metadata including example_id, problem_type, prompt_style, etc.

    Returns:
        Dictionary with:
        - success_rate: 1.0 if task completed successfully, 0.0 otherwise (main score)
        - workflow_complete: Alias for success_rate
        - prompt_style: The prompt style being evaluated
        - success_criteria: What defines success for this prompt style

        For render-based completion:
        - render_called: Whether render_design was called
        - render_success: Whether render_design returned success=True
        - render_save_path: Path where image was saved (if successful)
        - render_npy_path: Path where .npy array was saved (if successful)
        - render_error: Error message (if failed)
        - render_details: Additional details from render result

        For STL export completion:
        - stl_called: Whether convert_design_to_stl was called
        - stl_success: Whether convert_design_to_stl returned success=True
        - stl_save_path: Path where STL was saved (if successful)
        - stl_error: Error message (if failed)
        - stl_details: Additional details from STL export result

        - example_id: Example identifier
    """
    example_id = metadata.get("example_id", 0)
    prompt_style = metadata.get("prompt_style", "full")
    messages = output.get("messages", [])

    # Determine success criteria based on prompt style
    is_workflow = prompt_style == "workflow"
    success_criteria = "stl_export" if is_workflow else "render_design"

    # Track render_design calls
    render_called = False
    render_success = False
    render_save_path = None
    render_npy_path = None
    render_error = None
    render_details: dict[str, Any] = {}

    # Track convert_design_to_stl calls
    stl_called = False
    stl_success = False
    stl_save_path = None
    stl_error = None
    stl_details: dict[str, Any] = {}

    # Iterate through messages looking for tool results
    for msg in messages:
        # Check if this is a tool message (has tool_call_id attribute)
        if not hasattr(msg, "tool_call_id"):
            continue

        # Get tool name - try both 'name' attribute and checking content
        tool_name = getattr(msg, "name", None)

        # If no name attribute, try to infer from content
        if tool_name is None:
            content = msg.content
            if isinstance(content, str):
                if "save_path" in content and "design" in content.lower():
                    tool_name = RENDER_TOOL_NAME
                elif "stl" in content.lower():
                    tool_name = STL_EXPORT_TOOL_NAME

        # Process render_design calls
        if tool_name == RENDER_TOOL_NAME:
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

        # Process convert_design_to_stl calls
        elif tool_name == STL_EXPORT_TOOL_NAME:
            stl_called = True
            logger.debug(
                "Example %s: Found convert_design_to_stl tool call", example_id
            )

            # Parse the tool result
            content = msg.content
            result = _parse_tool_result(content, example_id)

            if result is None:
                stl_error = "Failed to parse convert_design_to_stl result"
                logger.warning(
                    "Example %s: Could not parse convert_design_to_stl result",
                    example_id,
                )
                continue

            # Check success flag
            if result.get("success", False):
                stl_success = True
                stl_save_path = result.get("stl_path")
                stl_details = {
                    "mesh_vertices": result.get("mesh_vertices"),
                    "mesh_faces": result.get("mesh_faces"),
                    "scale_xy": result.get("scale_xy"),
                    "scale_z": result.get("scale_z"),
                    "mirror_y": result.get("mirror_y"),
                    "message": result.get("message"),
                }
                logger.debug(
                    "Example %s: convert_design_to_stl succeeded, saved to %s",
                    example_id,
                    stl_save_path,
                )
                # Continue iterating to find the LAST successful export (most recent)
            else:
                stl_error = result.get("error", "Unknown error")
                logger.debug(
                    "Example %s: convert_design_to_stl failed with error: %s",
                    example_id,
                    stl_error,
                )

    # Compute final score based on prompt style
    if is_workflow:
        # For workflow prompts: success = STL export succeeded
        task_completed = stl_called and stl_success
        success_rate = 1.0 if task_completed else 0.0

        # Log summary
        if not stl_called:
            logger.info(
                "Example %s (workflow): convert_design_to_stl was never called",
                example_id,
            )
        elif not stl_success:
            logger.info(
                "Example %s (workflow): convert_design_to_stl was called but failed: %s",
                example_id,
                stl_error,
            )
        else:
            logger.info(
                "Example %s (workflow): Task completed successfully (STL saved to %s)",
                example_id,
                stl_save_path,
            )
    else:
        # For standard prompts: success = render succeeded
        task_completed = render_called and render_success
        success_rate = 1.0 if task_completed else 0.0

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
        "workflow_complete": task_completed,
        "prompt_style": prompt_style,
        "success_criteria": success_criteria,
        # Render-based metrics
        "render_called": render_called,
        "render_success": render_success,
        "render_save_path": render_save_path,
        "render_npy_path": render_npy_path,
        "render_error": render_error,
        "render_details": render_details,
        # STL export metrics
        "stl_called": stl_called,
        "stl_success": stl_success,
        "stl_save_path": stl_save_path,
        "stl_error": stl_error,
        "stl_details": stl_details,
        # Common
        "example_id": example_id,
    }
