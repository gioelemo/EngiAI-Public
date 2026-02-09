"""Tool use scorer for engineering agent evaluations.

This scorer evaluates how well the agent uses tools compared to the optimal sequence,
measuring both efficiency (number of calls) and correctness (order of calls).

Metrics:
- Efficiency Ratio = optimal_calls / actual_calls
  - 1.0 = perfectly efficient (agent used exactly the optimal number of calls)
  - <1.0 = less efficient (agent used more calls than optimal)

- Sequence Score = LCS length / optimal sequence length
  - 1.0 = tools called in perfect order
  - <1.0 = some tools out of order or missing
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


def _longest_common_subsequence(seq1: list[str], seq2: list[str]) -> list[str]:
    """Compute the longest common subsequence between two sequences.

    Uses dynamic programming to find the longest subsequence that appears
    in both sequences in the same order (but not necessarily contiguous).

    Args:
        seq1: First sequence of tool names
        seq2: Second sequence of tool names

    Returns:
        The longest common subsequence as a list
    """
    m, n = len(seq1), len(seq2)

    # Build DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find the LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            lcs.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return list(reversed(lcs))


def _compute_sequence_metrics(
    optimal_sequence: list[str],
    actual_sequence: list[str],
) -> dict[str, Any]:
    """Compute sequence order metrics.

    Args:
        optimal_sequence: Expected sequence of tool names in order
        actual_sequence: Actual sequence of tool calls made

    Returns:
        Dictionary with sequence metrics
    """
    if not optimal_sequence:
        return {
            "sequence_score": 1.0 if not actual_sequence else 0.0,
            "correct_order": not actual_sequence,
            "lcs": [],
            "lcs_length": 0,
            "missing_tools": [],
            "extra_tools": [],
            "out_of_order_tools": [],
        }

    # Compute LCS
    lcs = _longest_common_subsequence(optimal_sequence, actual_sequence)
    lcs_length = len(lcs)

    # Sequence score: proportion of optimal sequence matched in order
    sequence_score = lcs_length / len(optimal_sequence)

    # Check if all optimal tools are present in correct order
    correct_order = lcs == optimal_sequence

    # Find missing tools (in optimal but not matched in LCS)
    optimal_counts = Counter(optimal_sequence)
    lcs_counts = Counter(lcs)
    missing_tools = []
    for tool, count in optimal_counts.items():
        missing_count = count - lcs_counts.get(tool, 0)
        missing_tools.extend([tool] * missing_count)

    # Find extra tools (in actual but not in optimal)
    actual_counts = Counter(actual_sequence)
    extra_tools = []
    for tool, count in actual_counts.items():
        extra_count = count - optimal_counts.get(tool, 0)
        if extra_count > 0:
            extra_tools.extend([tool] * extra_count)

    # Find out-of-order tools
    # These are tools that are in actual_sequence and optimal_sequence
    # but appear in a different relative order
    out_of_order_tools = []
    if not correct_order and lcs_length > 0:
        # Tools that were called but in wrong position relative to others
        optimal_set = set(optimal_sequence)
        actual_filtered = [t for t in actual_sequence if t in optimal_set]
        if actual_filtered != optimal_sequence:
            # Find which tools broke the order
            for i, tool in enumerate(actual_filtered):
                if (
                    i < len(optimal_sequence)
                    and tool != optimal_sequence[i]
                    and tool not in out_of_order_tools
                ):
                    out_of_order_tools.append(tool)

    return {
        "sequence_score": sequence_score,
        "correct_order": correct_order,
        "lcs": lcs,
        "lcs_length": lcs_length,
        "missing_tools": missing_tools,
        "extra_tools": extra_tools,
        "out_of_order_tools": out_of_order_tools,
    }


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


def _log_tool_use_summary(
    example_id: Any,
    metrics: dict[str, Any],
    seq_metrics: dict[str, Any],
) -> None:
    """Log a summary of tool use metrics.

    Args:
        example_id: Example identifier
        metrics: Dict with efficiency_ratio, sequence_score, combined_score,
                 optimal_call_count, actual_call_count
        seq_metrics: Sequence metrics from _compute_sequence_metrics
    """
    if metrics["actual_call_count"] == 0:
        logger.info("Example %s: No tool calls made (efficiency: 0.0)", example_id)
        return

    order_status = "correct" if seq_metrics["correct_order"] else "incorrect"
    logger.info(
        "Example %s: efficiency=%.2f, sequence=%.2f, combined=%.2f, order=%s "
        "(optimal: %d, actual: %d)",
        example_id,
        metrics["efficiency_ratio"],
        metrics["sequence_score"],
        metrics["combined_score"],
        order_status,
        metrics["optimal_call_count"],
        metrics["actual_call_count"],
    )
    if seq_metrics["missing_tools"]:
        logger.debug(
            "Example %s: Missing tools: %s", example_id, seq_metrics["missing_tools"]
        )
    if seq_metrics["extra_tools"]:
        logger.debug(
            "Example %s: Extra tools: %s", example_id, seq_metrics["extra_tools"]
        )
    if seq_metrics["out_of_order_tools"]:
        logger.debug(
            "Example %s: Out of order: %s",
            example_id,
            seq_metrics["out_of_order_tools"],
        )


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

    # Compute efficiency ratio
    # optimal / actual so that 1.0 = perfect efficiency
    if actual_call_count > 0:
        efficiency_ratio = optimal_call_count / actual_call_count
    else:
        efficiency_ratio = 0.0

    # Cap efficiency at 1.0
    efficiency_ratio = min(efficiency_ratio, 1.0)

    # Compute tool coverage (which tools were called vs expected)
    optimal_tools_set = set(optimal_sequence)
    actual_tools_set = set(actual_sequence)

    # Missing tools: in optimal but not called
    missing_tools = list(optimal_tools_set - actual_tools_set)

    # Extra tools: called but not in optimal
    extra_tools = list(actual_tools_set - optimal_tools_set)

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
