"""Tool use scorer for engineering agent evaluations.

This scorer evaluates how well the agent uses tools compared to the optimal set,
measuring both correctness (right tools) and efficiency (right number of calls).
Tool ordering is NOT scored as multiple valid orderings exist.

Metrics:
- Efficiency Ratio = correctly_matched_calls / max(optimal_calls, actual_calls)
  - 1.0 = agent called exactly the right tools the right number of times
  - <1.0 = wrong tools called, missing tools, or extra tools
  - A tool swap (one correct tool missing, one wrong tool added) gives <1.0
    even when total call counts match, unlike the old optimal/actual formula.
"""

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_weave_value(value: Any) -> Any:
    """Resolve Weave references to actual values.

    Weave sometimes passes ObjectRef references instead of actual values.
    This function extracts the underlying value.

    Args:
        value: A value that might be a Weave reference or actual value

    Returns:
        The resolved actual value
    """
    # Check if it's a Weave ObjectRef (has a special attribute or is a string starting with weave:///)
    if isinstance(value, str) and value.startswith("weave:///"):
        # This is a Weave reference string - try to extract the tool name from it
        # Format: weave:///.../optimal_tool_calls/index/N/key/name
        # The actual value might be embedded in the path
        # For now, return None to indicate we need to use defaults
        logger.debug("Received Weave reference string: %s", value[:100])
        return None

    # Try to check for Weave ObjectRef type
    try:
        # Check if value has a Weave-specific method to get the actual value
        if hasattr(value, "__weave_ref__") or hasattr(value, "_ref"):
            # Try to get the actual value
            if hasattr(value, "val"):
                return value.val
            if hasattr(value, "get"):
                return value.get()
    except Exception:
        pass

    return value


def _is_weave_reference(value: Any) -> bool:
    """Check if a value is a Weave reference that couldn't be resolved.

    Args:
        value: Value to check

    Returns:
        True if value is an unresolved Weave reference
    """
    if isinstance(value, str) and value.startswith("weave:///"):
        return True
    if value is None:
        return False
    try:
        if hasattr(value, "__weave_ref__") or hasattr(value, "_ref"):
            return True
    except Exception:
        pass
    return False


def _deep_resolve_weave(obj: Any) -> Any:
    """Recursively resolve all Weave references in a nested structure.

    Args:
        obj: A potentially nested structure with Weave references

    Returns:
        The structure with all Weave references resolved
    """
    obj = _resolve_weave_value(obj)

    if isinstance(obj, dict):
        return {k: _deep_resolve_weave(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_resolve_weave(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_deep_resolve_weave(item) for item in obj)

    return obj


def _extract_optimal_sequence(
    metadata: dict[str, Any],
) -> tuple[list[str], int]:
    """Extract and resolve optimal tool sequence from metadata.

    Args:
        metadata: Metadata containing optimal_call_count and optimal_tool_calls

    Returns:
        Tuple of (optimal_sequence, optimal_call_count)
    """
    default_sequence = ["optimize_design", "simulate_design", "render_design"]
    default_count = 3

    # Get optimal call count
    optimal_call_count_raw = metadata.get("optimal_call_count")
    optimal_call_count = _resolve_weave_value(optimal_call_count_raw)
    if optimal_call_count is None or _is_weave_reference(optimal_call_count):
        optimal_call_count = default_count

    # Get optimal tool calls
    default_optimal_tool_calls = [
        {"name": "optimize_design", "count": 1},
        {"name": "simulate_design", "count": 1},
        {"name": "render_design", "count": 1},
    ]
    optimal_tool_calls_raw = metadata.get(
        "optimal_tool_calls", default_optimal_tool_calls
    )
    optimal_tool_calls = _deep_resolve_weave(optimal_tool_calls_raw)

    # Build optimal sequence
    optimal_sequence = []
    has_weave_refs = False

    for tc in optimal_tool_calls:
        tool_name_raw = tc.get("name", "unknown") if isinstance(tc, dict) else None
        count_raw = tc.get("count", 1) if isinstance(tc, dict) else 1

        tool_name = _resolve_weave_value(tool_name_raw)
        count = _resolve_weave_value(count_raw)

        if tool_name is None or _is_weave_reference(tool_name):
            has_weave_refs = True
            break
        if count is None or _is_weave_reference(count):
            count = 1

        optimal_sequence.extend([str(tool_name)] * int(count))

    # Fall back to defaults if needed
    if has_weave_refs or not optimal_sequence:
        logger.debug("Using default optimal sequence due to Weave references")
        return default_sequence, default_count

    return optimal_sequence, int(optimal_call_count)


def _rebuild_optimal_tool_calls(optimal_sequence: list[str]) -> list[dict[str, Any]]:
    """Rebuild optimal_tool_calls from resolved sequence.

    Groups consecutive identical tools and counts them.

    Args:
        optimal_sequence: List of tool names in order

    Returns:
        List of {"name": str, "count": int} dicts
    """
    if not optimal_sequence:
        return []

    result = []
    current_tool = optimal_sequence[0]
    current_count = 1

    for tool in optimal_sequence[1:]:
        if tool == current_tool:
            current_count += 1
        else:
            result.append({"name": current_tool, "count": current_count})
            current_tool = tool
            current_count = 1

    result.append({"name": current_tool, "count": current_count})
    return result


def score_tool_use(
    output: dict[str, Any],
    target: dict[str, Any],  # noqa: ARG001 - Required by scorer interface
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Score the efficiency of tool usage compared to optimal.

    This scorer computes efficiency ratio (optimal_calls / actual_calls).
    Tool call ordering is NOT scored as multiple valid orderings exist.

    Args:
        output: Agent output containing 'tool_calls_info' list
        target: Target/ground truth data from dataset (unused but required by interface)
        metadata: Metadata containing 'optimal_call_count' and 'optimal_tool_calls'

    Returns:
        Dictionary with:
        - efficiency_ratio: optimal_calls / actual_calls (1.0 = perfect)
        - optimal_call_count: Expected number of tool calls
        - actual_call_count: Actual number of tool calls made
        - tool_call_breakdown: Counter of tool calls by name
        - optimal_tool_calls: Expected tool calls
        - actual_tool_sequence: Actual sequence of tool names
        - missing_tools: Tools in optimal but not called
        - extra_tools: Tools called but not in optimal
        - excess_calls: Number of calls beyond optimal
        - example_id: Example identifier
    """
    example_id = metadata.get("example_id", 0)

    # Extract optimal sequence from metadata (handles Weave references)
    optimal_sequence, optimal_call_count = _extract_optimal_sequence(metadata)

    # Rebuild optimal_tool_calls from resolved sequence
    resolved_optimal_tool_calls = _rebuild_optimal_tool_calls(optimal_sequence)

    # Extract actual tool calls from agent output
    tool_calls_info = output.get("tool_calls_info", [])
    actual_call_count = len(tool_calls_info)

    # Build actual sequence of tool names
    actual_sequence = [tc.get("name", "unknown") for tc in tool_calls_info]

    # Build breakdown of tool calls by name
    tool_call_breakdown = Counter(actual_sequence)
    optimal_tool_breakdown = Counter(optimal_sequence)

    # Count correctly-matched tool calls (multiset intersection: min per tool)
    correctly_matched = sum(
        min(optimal_tool_breakdown[t], tool_call_breakdown[t])
        for t in optimal_tool_breakdown
    )

    # Efficiency ratio = correctly matched / max(optimal, actual)
    # - 1.0: agent called exactly the right tools the right number of times
    # - <1.0: wrong tools, missing tools, or extra tools (all penalised)
    denominator = max(optimal_call_count, actual_call_count)
    efficiency_ratio = correctly_matched / denominator if denominator > 0 else 0.0

    # Missing tools: one entry per unique tool with its expected vs actual count
    missing_tools_counter = optimal_tool_breakdown - tool_call_breakdown
    missing_tools = [
        {
            "name": tool,
            "expected": optimal_tool_breakdown[tool],
            "actual": tool_call_breakdown.get(tool, 0),
            "deficit": count,
        }
        for tool, count in missing_tools_counter.items()
    ]

    # Extra tools: one entry per unique tool with its expected vs actual count
    extra_tools_counter = tool_call_breakdown - optimal_tool_breakdown
    extra_tools = [
        {
            "name": tool,
            "expected": optimal_tool_breakdown.get(tool, 0),
            "actual": tool_call_breakdown[tool],
            "excess": count,
        }
        for tool, count in extra_tools_counter.items()
    ]

    # Compute excess calls
    excess_calls = max(0, actual_call_count - optimal_call_count)

    # Log summary
    logger.info(
        "Example %s: efficiency_ratio=%.2f (optimal: %d, actual: %d, missing: %s, extra: %s)",
        example_id,
        efficiency_ratio,
        optimal_call_count,
        actual_call_count,
        missing_tools or "none",
        extra_tools or "none",
    )

    return {
        # Main score
        "efficiency_ratio": float(efficiency_ratio),
        # Counts
        "optimal_call_count": int(optimal_call_count),
        "actual_call_count": int(actual_call_count),
        "excess_calls": int(excess_calls),
        # Sequences
        "actual_tool_sequence": list(actual_sequence),
        # Breakdown
        "tool_call_breakdown": dict(tool_call_breakdown),
        "optimal_tool_calls": resolved_optimal_tool_calls,
        "missing_tools": missing_tools,
        "extra_tools": extra_tools,
        # Metadata
        "example_id": int(example_id)
        if isinstance(example_id, (int, float))
        else example_id,
    }
