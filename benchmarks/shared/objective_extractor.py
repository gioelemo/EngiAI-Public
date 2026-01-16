"""Generic objective extraction from tool messages.

This module provides generic extraction of objective values from tool outputs
based on problem configuration, eliminating the need for problem-specific parsers.
"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import ToolMessage

from benchmarks.shared.problem_config import ObjectiveConfig, ProblemConfig

logger = logging.getLogger(__name__)


def extract_objectives_from_tool_messages(
    messages: list,
    problem_config: ProblemConfig,
    example_id: int,
) -> dict[str, float | None]:
    """Extract objective values from tool messages based on problem config.

    This function searches through tool messages for the configured tool_name
    and extracts objective values using the field names specified in the
    problem configuration.

    Args:
        messages: List of messages from agent conversation
        problem_config: Problem configuration with objective definitions
        example_id: Example identifier for debug logging

    Returns:
        Dictionary mapping objective names to their values (or None if not found)
    """
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        tool_name = getattr(msg, "name", "unknown")
        if tool_name != problem_config.tool_name:
            continue

        content = msg.content

        # Handle dict content (if tool returns structured data)
        if isinstance(content, dict):
            result = _extract_objectives_from_dict(
                content, problem_config.objectives, example_id
            )
            if result:
                return result

        # Handle string content (parse with regex)
        elif isinstance(content, str):
            result = _extract_objectives_from_string(
                content, problem_config.objectives, example_id
            )
            if result:
                return result

    logger.debug(f"Example {example_id}: No objectives found in tool messages")

    # Return dict with all objectives set to None
    return {obj.name: None for obj in problem_config.objectives}


def _extract_objectives_from_dict(
    content: dict,
    objectives: list[ObjectiveConfig],
    example_id: int,
) -> dict[str, float | None] | None:
    """Extract objectives from dictionary content.

    Args:
        content: Dictionary content from tool message
        objectives: List of objective configurations
        example_id: Example ID for logging

    Returns:
        Dictionary of objective values, or None if no objectives found
    """
    result: dict[str, float | None] = {}

    for obj_config in objectives:
        value = None

        # Try primary field name
        if obj_config.field_name in content:
            value = float(content[obj_config.field_name])
        else:
            # Try aliases
            for alias in obj_config.aliases:
                if alias in content:
                    value = float(content[alias])
                    break

        if value is not None:
            result[obj_config.name] = value
            logger.debug(
                f"Example {example_id}: Extracted {obj_config.name}={value} from dict"
            )

    return result if result else None


def _extract_objectives_from_string(
    content: str,
    objectives: list[ObjectiveConfig],
    example_id: int,
) -> dict[str, float | None] | None:
    """Extract objectives from string content using regex.

    Tries to parse the string as JSON first, then falls back to regex extraction.

    Args:
        content: String content from tool message
        objectives: List of objective configurations
        example_id: Example ID for logging

    Returns:
        Dictionary of objective values, or None if no objectives found
    """
    # Approach 1: Try parsing as JSON (or Python dict converted to JSON)
    try:
        # Replace Python-specific syntax with JSON equivalents
        json_content = content.replace("'", '"')
        json_content = json_content.replace("True", "true")
        json_content = json_content.replace("False", "false")
        json_content = json_content.replace("None", "null")

        parsed = json.loads(json_content)
        if isinstance(parsed, dict):
            return _extract_objectives_from_dict(parsed, objectives, example_id)
    except (json.JSONDecodeError, Exception):
        # Fall through to regex approach
        pass

    # Approach 2: Regex extraction
    result: dict[str, float | None] = {}

    for obj_config in objectives:
        # Try all possible field names (primary + aliases)
        field_patterns = [obj_config.field_name, *obj_config.aliases]

        for field in field_patterns:
            # PRIORITY 1: Try to find "Final" or "final" prefixed versions first
            # (e.g., "Final total_overlap: 1.851" instead of "Initial total_overlap: 0.291")
            final_patterns = [
                rf"[Ff]inal\s+{re.escape(field)}\s*:\s*([0-9.eE+-]+)",  # "Final field: value"
                rf"['\"]?final_{re.escape(field)}['\"]?\s*:\s*([0-9.eE+-]+)",  # "final_field": value
            ]

            found_final = False
            for final_pattern in final_patterns:
                match = re.search(final_pattern, content)
                if match:
                    result[obj_config.name] = float(match.group(1))
                    logger.debug(
                        f"Example {example_id}: Extracted {obj_config.name}="
                        f"{result[obj_config.name]} from Final {field} via regex"
                    )
                    found_final = True
                    break

            if found_final:
                break  # Found final value for this objective, move to next

            # PRIORITY 2: Fall back to regular field name (finds FIRST occurrence)
            pattern = rf"['\"]?{re.escape(field)}['\"]?\s*:\s*([0-9.eE+-]+)"
            match = re.search(pattern, content)

            if match:
                result[obj_config.name] = float(match.group(1))
                logger.debug(
                    f"Example {example_id}: Extracted {obj_config.name}="
                    f"{result[obj_config.name]} from field '{field}' via regex "
                    f"(warning: may be initial value)"
                )
                break  # Found this objective, move to next

    return result if result else None


def calculate_objective_score(
    agent_objectives: dict[str, float | None],
    target_objectives: dict[str, float | None],
    problem_config: ProblemConfig,
    example_id: int,
) -> tuple[float, dict[str, Any]]:
    """Calculate objective matching score.

    Compares agent's achieved objective values against target/ground truth values
    and computes a normalized score.

    Args:
        agent_objectives: Objective values extracted from agent output
        target_objectives: Target objective values from dataset
        problem_config: Problem configuration
        example_id: Example ID for logging

    Returns:
        Tuple of (overall_score, detailed_metrics_dict)
    """
    if not agent_objectives or all(v is None for v in agent_objectives.values()):
        logger.warning(
            f"Example {example_id}: No agent objectives found, objective_score=0.0"
        )
        return 0.0, {f"agent_{obj.name}": None for obj in problem_config.objectives}

    scores = []
    metrics = {}

    for obj_config in problem_config.objectives:
        agent_value = agent_objectives.get(obj_config.name)
        target_value = target_objectives.get(obj_config.name)

        # Store raw values
        metrics[f"agent_{obj_config.name}"] = agent_value
        metrics[f"target_{obj_config.name}"] = target_value

        if agent_value is None or target_value is None:
            logger.debug(
                f"Example {example_id}: Missing {obj_config.name} "
                f"(agent={agent_value}, target={target_value})"
            )
            scores.append(0.0)
            metrics[f"{obj_config.name}_relative_error"] = None
            metrics[f"{obj_config.name}_score"] = 0.0
            continue

        # Calculate relative error
        if target_value == 0:
            # Avoid division by zero - use absolute error
            relative_error = abs(agent_value - target_value)
        else:
            relative_error = abs(agent_value - target_value) / abs(target_value)

        metrics[f"{obj_config.name}_relative_error"] = relative_error

        # Compute score based on threshold
        obj_score = max(0.0, 1.0 - relative_error / obj_config.relative_error_threshold)
        metrics[f"{obj_config.name}_score"] = obj_score

        # Weight and add to scores
        weighted_score = obj_score * obj_config.weight
        scores.append(weighted_score)

        logger.debug(
            f"Example {example_id}: {obj_config.name} - "
            f"Agent: {agent_value:.4f}, Target: {target_value:.4f}, "
            f"Error: {relative_error:.2%}, Score: {obj_score:.3f}"
        )

    # Normalize by total weight
    total_weight = sum(obj.weight for obj in problem_config.objectives)
    overall_score = sum(scores) / total_weight if total_weight > 0 else 0.0

    return overall_score, metrics
