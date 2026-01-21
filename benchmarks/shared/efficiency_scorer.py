"""Efficiency scorer for engineering agent evaluations.

This scorer computes the efficiency ratio between optimal and actual tool calls,
measuring how efficiently the agent solves the design problem.

Efficiency Ratio = optimal_calls / actual_calls
- 1.0 = perfectly efficient (agent used exactly the optimal number of calls)
- <1.0 = less efficient (agent used more calls than optimal)
- Should not exceed 1.0 (would indicate the optimal count is wrong)
"""

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def score_efficiency(
    output: dict[str, Any],
    target: dict[str, Any],  # noqa: ARG001 - Required by scorer interface
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Score the efficiency of tool usage compared to optimal.

    This scorer computes the ratio of optimal tool calls to actual tool calls,
    providing a measure of how efficiently the agent solved the problem.

    Args:
        output: Agent output containing 'tool_calls_info' list
        target: Target/ground truth data from dataset (unused but required by interface)
        metadata: Metadata containing 'optimal_call_count' and 'optimal_tool_calls'

    Returns:
        Dictionary with:
        - efficiency_ratio: optimal_calls / actual_calls (main score, 1.0 = perfect)
        - optimal_call_count: Expected number of tool calls
        - actual_call_count: Actual number of tool calls made
        - tool_call_breakdown: Counter of tool calls by name
        - optimal_tool_calls: Expected tool call sequence
        - excess_calls: Number of calls beyond optimal (0 if efficient)
        - example_id: Example identifier
    """
    example_id = metadata.get("example_id", 0)

    # Get optimal call info from metadata (passed through from prompt data)
    optimal_call_count = metadata.get("optimal_call_count", 2)  # Default: optimize + render
    optimal_tool_calls = metadata.get("optimal_tool_calls", [
        {"name": "optimize_design", "count": 1},
        {"name": "render_design", "count": 1},
    ])

    # Extract actual tool calls from agent output
    tool_calls_info = output.get("tool_calls_info", [])
    actual_call_count = len(tool_calls_info)

    # Build breakdown of tool calls by name
    tool_call_breakdown = Counter(tc.get("name", "unknown") for tc in tool_calls_info)

    # Compute efficiency ratio
    # optimal / actual so that 1.0 = perfect efficiency
    if actual_call_count > 0:
        efficiency_ratio = optimal_call_count / actual_call_count
    else:
        # No tool calls made - agent failed to use tools
        efficiency_ratio = 0.0

    # Cap efficiency at 1.0 (if agent somehow used fewer calls than optimal,
    # it means our optimal estimate was wrong, not that agent is super-efficient)
    efficiency_ratio = min(efficiency_ratio, 1.0)

    # Compute excess calls
    excess_calls = max(0, actual_call_count - optimal_call_count)

    # Log summary
    if actual_call_count == 0:
        logger.info("Example %s: No tool calls made (efficiency: 0.0)", example_id)
    elif efficiency_ratio == 1.0:
        logger.info(
            "Example %s: Perfect efficiency (%d calls = optimal)",
            example_id,
            actual_call_count,
        )
    else:
        logger.info(
            "Example %s: Efficiency %.2f (optimal: %d, actual: %d, excess: %d)",
            example_id,
            efficiency_ratio,
            optimal_call_count,
            actual_call_count,
            excess_calls,
        )
        logger.debug("Example %s: Tool breakdown: %s", example_id, dict(tool_call_breakdown))

    return {
        "efficiency_ratio": efficiency_ratio,
        "optimal_call_count": optimal_call_count,
        "actual_call_count": actual_call_count,
        "tool_call_breakdown": dict(tool_call_breakdown),
        "optimal_tool_calls": optimal_tool_calls,
        "excess_calls": excess_calls,
        "example_id": example_id,
    }
