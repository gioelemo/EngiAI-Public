"""Task completion scorer for engineering agent evaluations.

This scorer checks task completion based on prompt style:
- Standard prompts (full, natural): render_design must be called successfully
- Workflow prompts: convert_design_to_stl must be called successfully
- Workflow-conditional: STL export with params resolved from compliance-based branching
- Workflow-multi-export: Two STL exports with different params, validated in order
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
CLARIFICATION_TOOL_NAME = "ask_human_for_clarification"

# Expected number of STL exports for workflow-multi-export
MULTI_EXPORT_COUNT = 2


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

    # Approach 1: Try direct JSON parsing first (handles valid JSON with apostrophes)
    try:
        return json.loads(content)
    except (json.JSONDecodeError, Exception):
        pass  # Continue to fallback approaches

    # Approach 2: Try converting Python repr to JSON
    # (only if direct JSON parsing failed)
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

    # Approach 3: Try ast.literal_eval (for Python repr)
    try:
        return ast.literal_eval(content)
    except (ValueError, SyntaxError) as e:
        logger.debug("Example %s: ast.literal_eval failed: %s", example_id, e)

    return None


def _validate_stl_parameters(
    stl_details: dict[str, Any],
    expected_params: dict[str, Any],
    example_id: int,
) -> tuple[float, dict[str, Any]]:
    """Validate STL export parameters against expected values.

    Follows constraint validation pattern from output_quality_scorer.py.

    Args:
        stl_details: Actual parameters from convert_design_to_stl tool result
        expected_params: Expected parameters from prompt metadata (stl_expected_params)
        example_id: Example ID for logging

    Returns:
        Tuple of (validation_score, metrics_dict)
        - validation_score: 1.0 if all params match within tolerance, 0.0 otherwise
        - metrics_dict: Per-parameter metrics with _actual, _expected, _error, _valid
    """
    if not expected_params:
        logger.debug("Example %s: No expected STL params to validate", example_id)
        return 1.0, {}

    float_tolerance = 0.05  # Allow LLM rounding to 1 decimal place (e.g. 16.635 → 16.6)

    violations = 0
    metrics: dict[str, Any] = {}

    # Parameter configs: (expected_key, actual_key_in_result, tolerance, type)
    param_configs = [
        ("scale_xy", "scale_xy", float_tolerance, float),
        ("scale_z", "scale_z", float_tolerance, float),
        ("threshold", "threshold", float_tolerance, float),
        ("mirror_y", "mirrored", 0, bool),  # Tool returns "mirrored"
    ]

    for param_name, result_key, tolerance, param_type in param_configs:
        expected_value = expected_params.get(param_name)
        actual_value = stl_details.get(result_key)

        if expected_value is None or actual_value is None:
            logger.warning(
                "Example %s: Missing STL param %s (expected=%s, actual=%s)",
                example_id,
                param_name,
                expected_value,
                actual_value,
            )
            violations += 1
            metrics[f"stl_{param_name}_valid"] = False
            continue

        # Validate based on type
        if param_type is float:
            actual_value = float(actual_value)
            expected_value = float(expected_value)
            error = abs(actual_value - expected_value)
            is_valid = error < tolerance
        elif param_type is bool:
            actual_value = bool(actual_value)
            expected_value = bool(expected_value)
            error = 0.0 if actual_value == expected_value else 1.0
            is_valid = actual_value == expected_value
        else:
            error = 0.0
            is_valid = True

        # Store metrics (following constraint validation pattern)
        metrics[f"stl_{param_name}_actual"] = actual_value
        metrics[f"stl_{param_name}_expected"] = expected_value
        metrics[f"stl_{param_name}_error"] = error
        metrics[f"stl_{param_name}_valid"] = is_valid

        if not is_valid:
            violations += 1
            logger.info(
                "Example %s: STL param %s validation failed - "
                "expected=%s, actual=%s, error=%s",
                example_id,
                param_name,
                expected_value,
                actual_value,
                error,
            )

    validation_score = 1.0 if violations == 0 else 0.0
    metrics["stl_param_violations"] = violations
    metrics["stl_param_validation_score"] = validation_score

    return validation_score, metrics


def _resolve_conditional_params(
    conditional_params: dict[str, Any],
    target: dict[str, Any],
    example_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve conditional STL parameters based on ground truth compliance.

    For workflow-conditional prompts, determines which branch the agent should
    have taken based on the ground truth compliance value, and returns the
    expected flat parameter dict for validation.

    Args:
        conditional_params: The conditional stl_expected_params structure with
            compliance_threshold, branch_high, branch_low, and common keys
        target: Target dict containing ground truth compliance
        example_id: Example ID for logging

    Returns:
        Tuple of (resolved_flat_params, branch_metrics)
        - resolved_flat_params: Flat dict with scale_xy, scale_z, threshold, mirror_y
        - branch_metrics: Dict with branch decision details for reporting
    """
    compliance_threshold = conditional_params["compliance_threshold"]
    gt_compliance = target.get("compliance", 0.0)

    # Determine correct branch
    if gt_compliance > compliance_threshold:
        correct_branch = "high"
        branch_params = conditional_params["branch_high"]
    else:
        correct_branch = "low"
        branch_params = conditional_params["branch_low"]

    common_params = conditional_params["common"]

    # Build flat expected params (same format as workflow-random)
    resolved = {
        "threshold": branch_params["threshold"],
        "mirror_y": branch_params["mirror_y"],
        "scale_xy": common_params["scale_xy"],
        "scale_z": common_params["scale_z"],
    }

    branch_metrics = {
        "conditional_compliance_threshold": compliance_threshold,
        "conditional_gt_compliance": gt_compliance,
        "conditional_correct_branch": correct_branch,
    }

    logger.info(
        "Example %s (workflow-conditional): gt_compliance=%.2f, threshold=%.1f, "
        "correct_branch=%s",
        example_id,
        gt_compliance,
        compliance_threshold,
        correct_branch,
    )

    return resolved, branch_metrics


def _validate_multi_export_params(
    all_stl_details: list[dict[str, Any]],
    expected_exports: list[dict[str, Any]],
    example_id: int,
) -> tuple[float, dict[str, Any]]:
    """Validate parameters for workflow-multi-export (two STL calls in order).

    Validates the first two successful STL calls against expected Export A and B.
    Order-based: first call → exports[0], second call → exports[1].
    All-or-nothing: both must pass for overall score of 1.0.

    Args:
        all_stl_details: List of detail dicts from each successful STL call
        expected_exports: List of expected param dicts (length 2)
        example_id: Example ID for logging

    Returns:
        Tuple of (overall_score, combined_metrics)
    """
    metrics: dict[str, Any] = {
        "multi_export_count": len(all_stl_details),
    }

    if len(all_stl_details) < MULTI_EXPORT_COUNT:
        logger.info(
            "Example %s (workflow-multi-export): Expected 2 STL exports, got %d",
            example_id,
            len(all_stl_details),
        )
        metrics["multi_export_both_valid"] = False
        metrics["stl_param_validation_score"] = 0.0
        metrics["stl_param_violations"] = 4  # All params missing for missing export
        return 0.0, metrics

    # Validate each export in order
    labels = ["export_a", "export_b"]
    all_valid = True

    for i, (label, expected) in enumerate(zip(labels, expected_exports, strict=True)):
        actual_details = all_stl_details[i]
        score_i, metrics_i = _validate_stl_parameters(
            stl_details=actual_details,
            expected_params=expected,
            example_id=example_id,
        )

        # Prefix metrics with export label
        for key, value in metrics_i.items():
            metrics[f"{label}_{key}"] = value

        if score_i == 0.0:
            all_valid = False
            logger.info(
                "Example %s (workflow-multi-export): %s validation failed",
                example_id,
                label,
            )

    overall_score = 1.0 if all_valid else 0.0
    metrics["multi_export_both_valid"] = all_valid
    metrics["stl_param_validation_score"] = overall_score

    return overall_score, metrics


def score_task_completion(  # noqa: PLR0912, PLR0915 - Complex scoring logic
    output: dict[str, Any],
    target: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Score task completion based on prompt style.

    For standard prompts (full):
    - Success = render_design called and returned success=True

    For workflow prompts:
    - Success = convert_design_to_stl called and returned success=True
    - render_design is optional and not counted toward success

    For natural prompts with clarification criteria:
    - Success = ask_human_for_clarification was called (regardless of question content)

    Args:
        output: Agent output with messages and model info
        target: Target/ground truth data from dataset (used for workflow-conditional)
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
    is_workflow = prompt_style in [
        "workflow",
        "workflow-random",
        "workflow-conditional",
        "workflow-multi-export",
    ]
    is_workflow_random = prompt_style == "workflow-random"
    is_workflow_conditional = prompt_style == "workflow-conditional"
    is_workflow_multi_export = prompt_style == "workflow-multi-export"
    is_clarification = metadata.get("success_criteria") == "clarification_requested"
    success_criteria = "stl_export" if is_workflow else "render_design"
    if is_workflow_random or is_workflow_conditional or is_workflow_multi_export:
        success_criteria = "stl_export_with_params"
    if is_clarification:
        success_criteria = "clarification_requested"

    # Track ask_human_for_clarification calls
    clarification_called = False
    clarification_question = None

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
    all_stl_details: list[dict[str, Any]] = []  # Accumulates all successful STL calls

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

        # Process ask_human_for_clarification calls
        elif tool_name == CLARIFICATION_TOOL_NAME:
            clarification_called = True
            logger.debug(
                "Example %s: Found ask_human_for_clarification tool call",
                example_id,
            )

            # Parse the tool result (structured JSON)
            content = msg.content
            result = _parse_tool_result(content, example_id)

            if result is not None and result.get("success", False):
                clarification_question = result.get("question")
                logger.debug(
                    "Example %s: Parsed clarification question: %s",
                    example_id,
                    clarification_question,
                )
            else:
                # Fallback: try to extract from raw content for backward compatibility
                if isinstance(content, str):
                    if content.startswith("Clarification requested: "):
                        clarification_question = content.split(
                            "Clarification requested: ", 1
                        )[1].split("\n")[0]
                    else:
                        clarification_question = content
                else:
                    clarification_question = None
                logger.debug(
                    "Example %s: Using fallback parsing for clarification question",
                    example_id,
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
                    "num_vertices": result.get("num_vertices"),
                    "num_faces": result.get("num_faces"),
                    "scale_xy": result.get("scale_xy"),
                    "scale_z": result.get("scale_z"),
                    "mirrored": result.get("mirrored"),
                    "threshold": result.get("threshold"),
                    "message": result.get("message"),
                }
                # Accumulate for multi-export validation (order matters)
                all_stl_details.append(dict(stl_details))
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
    if is_clarification:
        # For natural prompts: success = clarification was requested
        task_completed = clarification_called
        success_rate = 1.0 if task_completed else 0.0

        if not clarification_called:
            logger.info(
                "Example %s (natural): ask_human_for_clarification was never called",
                example_id,
            )
        else:
            logger.info(
                "Example %s (natural): Task completed successfully "
                "(clarification requested)",
                example_id,
            )
    elif is_workflow:
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

    # STL Parameter Validation for workflow-random, workflow-conditional,
    # and workflow-multi-export
    stl_param_validation_score = 1.0
    stl_param_metrics: dict[str, Any] = {}

    if is_workflow_multi_export and stl_called:
        expected_stl_params = metadata.get("stl_expected_params", {})
        if expected_stl_params and expected_stl_params.get("multi_export"):
            stl_param_validation_score, param_metrics = _validate_multi_export_params(
                all_stl_details=all_stl_details,
                expected_exports=expected_stl_params["exports"],
                example_id=example_id,
            )
            stl_param_metrics.update(param_metrics)

            if stl_param_validation_score == 0.0:
                success_rate = 0.0
                task_completed = False
                logger.info(
                    "Example %s (workflow-multi-export): Task incomplete due to "
                    "multi-export validation failure",
                    example_id,
                )
        else:
            logger.warning(
                "Example %s (workflow-multi-export): No stl_expected_params "
                "in metadata",
                example_id,
            )
    elif (is_workflow_random or is_workflow_conditional) and stl_success:
        expected_stl_params = metadata.get("stl_expected_params", {})

        if expected_stl_params:
            # For workflow-conditional: resolve branch before validation
            if is_workflow_conditional and expected_stl_params.get("conditional"):
                resolved_params, branch_metrics = _resolve_conditional_params(
                    conditional_params=expected_stl_params,
                    target=target,
                    example_id=example_id,
                )
                stl_param_metrics.update(branch_metrics)
                expected_stl_params = resolved_params

            stl_param_validation_score, param_metrics = _validate_stl_parameters(
                stl_details=stl_details,
                expected_params=expected_stl_params,
                example_id=example_id,
            )
            stl_param_metrics.update(param_metrics)

            # Task ONLY complete if params valid
            if stl_param_validation_score == 0.0:
                success_rate = 0.0
                task_completed = False
                style_label = (
                    "workflow-conditional"
                    if is_workflow_conditional
                    else "workflow-random"
                )
                logger.info(
                    "Example %s (%s): Task incomplete due to "
                    "STL parameter validation failure "
                    "(%s violations)",
                    example_id,
                    style_label,
                    stl_param_metrics.get("stl_param_violations", 0),
                )
        else:
            style_label = (
                "workflow-conditional" if is_workflow_conditional else "workflow-random"
            )
            logger.warning(
                "Example %s (%s): No stl_expected_params in metadata",
                example_id,
                style_label,
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
        # Clarification metrics (natural prompts)
        "clarification_called": clarification_called,
        "clarification_question": clarification_question,
        # STL parameter validation metrics (workflow-random)
        "stl_param_validation_score": stl_param_validation_score,
        **stl_param_metrics,  # Unpacks all per-parameter metrics
        # Common
        "example_id": example_id,
    }
