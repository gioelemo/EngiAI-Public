"""RAG evaluation scorer for rag_beams2d benchmark.

Measures whether the agent uses RAG-retrieved information to make better
engineering decisions.

Supports two scoring modes depending on which conditions are present:

Single-parameter mode (expected_volfrac only):
    Dimensions and weights:
    1. effective_volfrac_accuracy  0.50  (gated on rag_tool_called)
    2. rag_tool_called             0.30
    3. source_cited                0.20

Two-parameter mode (expected_volfrac + expected_forcedist):
    Dimensions and weights:
    1. effective_volfrac_accuracy  0.35  (gated on rag_tool_called)
    2. effective_forcedist_accuracy 0.35 (gated on rag_tool_called)
    3. rag_tool_called             0.20
    4. source_cited                0.10

In both modes, parameter accuracy dimensions only contribute when
search_documents was called (rag_tool_called=True). This prevents agents
from earning credit by retrieving the correct value through engineering tools
(e.g. get_problem_details) instead of the indexed paper.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Tool names
SEARCH_TOOL_NAME = "search_documents"
OPTIMIZE_TOOL_NAME = "optimize_design"

# Keywords that suggest the agent cited a source in its final response
CITATION_KEYWORDS = [
    "engibench",
    "according to",
    "the paper",
    "the document",
    "retrieved",
    "found in",
    "states that",
    "mentions",
    "references",
    "source",
]

# Weights for single-parameter mode (volfrac only)
_W1_VOLFRAC = 0.50
_W1_RAG_TOOL = 0.30
_W1_CITED = 0.20

# Weights for two-parameter mode (volfrac + forcedist)
_W2_VOLFRAC = 0.35
_W2_FORCEDIST = 0.35
_W2_RAG_TOOL = 0.20
_W2_CITED = 0.10


def _extract_param_from_args(
    args: dict[str, Any], aliases: list[str]
) -> float | None:
    """Extract a float parameter from optimize_design tool args.

    Checks flat keys first, then nested in problem_config dict.
    """
    for alias in aliases:
        value = args.get(alias)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    # Nested in problem_config dict
    problem_config = args.get("problem_config")
    if isinstance(problem_config, dict):
        for alias in aliases:
            value = problem_config.get(alias)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    return None


def _extract_volfrac_from_args(args: dict[str, Any]) -> float | None:
    """Extract volfrac from optimize_design tool args."""
    return _extract_param_from_args(
        args, ["volfrac", "volume_fraction", "vol_frac", "volume"]
    )


def _extract_forcedist_from_args(args: dict[str, Any]) -> float | None:
    """Extract forcedist from optimize_design tool args."""
    return _extract_param_from_args(
        args, ["forcedist", "force_distribution", "force_dist", "forcedistribution"]
    )


def _score_param_accuracy(
    actual: float | None,
    expected: float,
    tolerance: float,
) -> float:
    """Score how close the actual value is to the expected.

    Returns:
        1.0 if within tolerance, soft linear decay to 0.0 beyond tolerance.
    """
    if actual is None:
        return 0.0

    error = abs(actual - expected)
    if error <= tolerance:
        return 1.0

    # Soft decay: full penalty at 3x the tolerance
    decay_range = 3.0 * tolerance
    return float(max(0.0, 1.0 - (error - tolerance) / decay_range))


def _check_source_cited(response: str) -> bool:
    """Return True if the final response contains citation keywords."""
    lowered = response.lower()
    return any(kw in lowered for kw in CITATION_KEYWORDS)


def score_rag_evaluation(
    output: dict[str, Any],
    target: dict[str, Any],  # noqa: ARG001 — required by scorer interface
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Score a RAG-dependent evaluation prompt.

    Evaluates whether the agent used RAG to retrieve information from the
    EngiBench paper and applied that information accurately to the design task.

    Supports single-parameter prompts (volfrac only) and two-parameter prompts
    (volfrac + forcedist). Weights are adjusted automatically based on which
    conditions are present.

    Args:
        output: Agent output dict containing:
            - tool_calls_info: list of {name, args} dicts
            - response: final response text from the agent
            - messages: full message history
        target: Ground-truth data (unused for RAG eval)
        metadata: Evaluation metadata containing:
            - conditions.expected_volfrac: expected volume fraction (float)
            - conditions.expected_volfrac_tolerance: tolerance window (float)
            - conditions.expected_forcedist: expected force distribution (float, optional)
            - conditions.expected_forcedist_tolerance: tolerance window (float, optional)
            - example_id: int

    Returns:
        Dict with:
        - rag_benefit_score: Composite score (0.0-1.0), main metric
        - rag_tool_called: Whether search_documents was invoked (bool)
        - volfrac_accuracy: Raw closeness score for volfrac (0.0-1.0)
        - effective_volfrac_accuracy: volfrac_accuracy gated on rag_tool_called (0.0-1.0)
        - volfrac_actual: Volume fraction the agent used (float | None)
        - volfrac_expected: Expected volume fraction from conditions (float)
        - volfrac_error: Absolute error |actual - expected| (float | None)
        - volfrac_within_tolerance: Whether error <= tolerance (bool)
        - forcedist_accuracy: Raw closeness score for forcedist (0.0-1.0), 0.0 if not tested
        - effective_forcedist_accuracy: forcedist_accuracy gated on rag_tool_called (0.0-1.0)
        - forcedist_actual: Force distribution the agent used (float | None)
        - forcedist_expected: Expected forcedist from conditions (float | None)
        - forcedist_error: Absolute error |actual - expected| (float | None)
        - forcedist_within_tolerance: Whether error <= tolerance (bool)
        - forcedist_tested: Whether forcedist scoring was active for this prompt (bool)
        - source_cited: Whether the response references the source (bool)
        - example_id: int
    """
    example_id = metadata.get("example_id", 0)
    conditions = metadata.get("conditions", {})

    volfrac_expected: float = float(conditions.get("expected_volfrac", 0.35))
    volfrac_tolerance: float = float(conditions.get("expected_volfrac_tolerance", 0.05))

    # forcedist is optional — present only in two-parameter prompts
    forcedist_expected_raw = conditions.get("expected_forcedist")
    forcedist_tested = forcedist_expected_raw is not None
    forcedist_expected: float | None = (
        float(forcedist_expected_raw) if forcedist_tested else None
    )
    forcedist_tolerance: float = float(
        conditions.get("expected_forcedist_tolerance", 0.05)
    )

    tool_calls_info: list[dict[str, Any]] = output.get("tool_calls_info", [])
    response: str = str(output.get("response", ""))

    # --- Dimension: RAG tool usage ---
    rag_tool_called = any(tc.get("name") == SEARCH_TOOL_NAME for tc in tool_calls_info)
    logger.debug("Example %s: search_documents called = %s", example_id, rag_tool_called)

    # --- Dimension: volfrac accuracy ---
    volfrac_actual: float | None = None
    forcedist_actual: float | None = None

    for tc in tool_calls_info:
        if tc.get("name") == OPTIMIZE_TOOL_NAME:
            args = tc.get("args", {})
            if volfrac_actual is None:
                volfrac_actual = _extract_volfrac_from_args(args)
            if forcedist_actual is None:
                forcedist_actual = _extract_forcedist_from_args(args)
            if volfrac_actual is not None and (
                not forcedist_tested or forcedist_actual is not None
            ):
                break  # All needed values extracted

    volfrac_error: float | None = (
        abs(volfrac_actual - volfrac_expected) if volfrac_actual is not None else None
    )
    volfrac_within_tolerance = (
        volfrac_error is not None and volfrac_error <= volfrac_tolerance
    )
    volfrac_accuracy = _score_param_accuracy(
        volfrac_actual, volfrac_expected, volfrac_tolerance
    )

    logger.info(
        "Example %s: volfrac actual=%s expected=%.3f error=%s within_tol=%s",
        example_id,
        f"{volfrac_actual:.3f}" if volfrac_actual is not None else "N/A",
        volfrac_expected,
        f"{volfrac_error:.3f}" if volfrac_error is not None else "N/A",
        volfrac_within_tolerance,
    )

    # --- Dimension: forcedist accuracy (two-parameter prompts only) ---
    forcedist_error: float | None = None
    forcedist_within_tolerance = False
    forcedist_accuracy = 0.0

    if forcedist_tested and forcedist_expected is not None:
        forcedist_error = (
            abs(forcedist_actual - forcedist_expected)
            if forcedist_actual is not None
            else None
        )
        forcedist_within_tolerance = (
            forcedist_error is not None and forcedist_error <= forcedist_tolerance
        )
        forcedist_accuracy = _score_param_accuracy(
            forcedist_actual, forcedist_expected, forcedist_tolerance
        )

        logger.info(
            "Example %s: forcedist actual=%s expected=%.3f error=%s within_tol=%s",
            example_id,
            f"{forcedist_actual:.3f}" if forcedist_actual is not None else "N/A",
            forcedist_expected,
            f"{forcedist_error:.3f}" if forcedist_error is not None else "N/A",
            forcedist_within_tolerance,
        )

    # --- Dimension: source citation ---
    source_cited = _check_source_cited(response)
    logger.debug("Example %s: source_cited = %s", example_id, source_cited)

    # --- Composite score ---
    # Parameter accuracy dimensions only contribute when RAG was called.
    # This ensures agents that reach correct values via get_problem_details
    # (rather than the paper) receive no parameter accuracy credit.
    effective_volfrac_accuracy = volfrac_accuracy if rag_tool_called else 0.0
    effective_forcedist_accuracy = forcedist_accuracy if rag_tool_called else 0.0

    if forcedist_tested:
        # Two-parameter mode
        rag_benefit_score = (
            _W2_VOLFRAC * effective_volfrac_accuracy
            + _W2_FORCEDIST * effective_forcedist_accuracy
            + _W2_RAG_TOOL * float(rag_tool_called)
            + _W2_CITED * float(source_cited)
        )
    else:
        # Single-parameter mode (backward compatible)
        rag_benefit_score = (
            _W1_VOLFRAC * effective_volfrac_accuracy
            + _W1_RAG_TOOL * float(rag_tool_called)
            + _W1_CITED * float(source_cited)
        )

    logger.info(
        "Example %s: rag_benefit_score=%.3f "
        "(eff_volfrac=%.2f, eff_forcedist=%.2f, rag_called=%s, cited=%s, two_param=%s)",
        example_id,
        rag_benefit_score,
        effective_volfrac_accuracy,
        effective_forcedist_accuracy,
        rag_tool_called,
        source_cited,
        forcedist_tested,
    )

    return {
        # Main metric
        "rag_benefit_score": float(rag_benefit_score),
        # Dimension breakdown
        "rag_tool_called": bool(rag_tool_called),
        "volfrac_accuracy": float(volfrac_accuracy),
        "effective_volfrac_accuracy": float(effective_volfrac_accuracy),
        "forcedist_accuracy": float(forcedist_accuracy),
        "effective_forcedist_accuracy": float(effective_forcedist_accuracy),
        "forcedist_tested": bool(forcedist_tested),
        "source_cited": bool(source_cited),
        # volfrac detail
        "volfrac_actual": volfrac_actual,
        "volfrac_expected": volfrac_expected,
        "volfrac_error": volfrac_error,
        "volfrac_within_tolerance": bool(volfrac_within_tolerance),
        # forcedist detail
        "forcedist_actual": forcedist_actual,
        "forcedist_expected": forcedist_expected,
        "forcedist_error": forcedist_error,
        "forcedist_within_tolerance": bool(forcedist_within_tolerance),
        # Metadata
        "example_id": int(example_id)
        if isinstance(example_id, (int, float))
        else example_id,
    }
