"""RAG evaluation scorer for rag_beams2d benchmark.

Measures whether the agent uses RAG-retrieved information to make better
engineering decisions. Three dimensions are scored:

1. Parameter accuracy (weight 0.50) — conditional on RAG being called
   Did the agent call optimize_design with the expected volfrac AND use RAG?
   volfrac_accuracy is only credited when search_documents was also called.
   This prevents agents from earning credit by reaching the correct value via
   engineering tools (e.g. get_problem_details) instead of the paper.

2. RAG tool usage (weight 0.30)
   Was search_documents called at all?

3. Source citation (weight 0.20)
   Does the final response reference the retrieved source?

Composite:
    effective_volfrac_accuracy = volfrac_accuracy if rag_tool_called else 0.0
    rag_benefit_score = 0.50 * effective_volfrac_accuracy
                      + 0.30 * rag_tool_called
                      + 0.20 * source_cited
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

# Scoring weights
WEIGHT_VOLFRAC_ACCURACY = 0.50
WEIGHT_RAG_TOOL_CALLED = 0.30
WEIGHT_SOURCE_CITED = 0.20


def _extract_volfrac_from_args(args: dict[str, Any]) -> float | None:
    """Extract volfrac from optimize_design tool args.

    Checks three paths that the LLM may use:
    - Flat key: args["volfrac"] or args["volume_fraction"] / aliases
    - Nested: args["problem_config"]["volfrac"]
    """
    flat_aliases = ["volfrac", "volume_fraction", "vol_frac", "volume"]
    for alias in flat_aliases:
        value = args.get(alias)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    # Nested in problem_config dict
    problem_config = args.get("problem_config")
    if isinstance(problem_config, dict):
        for alias in flat_aliases:
            value = problem_config.get(alias)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    return None


def _score_volfrac_accuracy(
    volfrac_actual: float | None,
    volfrac_expected: float,
    tolerance: float,
) -> float:
    """Score how close the actual volfrac is to the expected value.

    Returns:
        1.0 if within tolerance, soft linear decay to 0.0 beyond tolerance.
    """
    if volfrac_actual is None:
        return 0.0

    error = abs(volfrac_actual - volfrac_expected)
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

    Args:
        output: Agent output dict containing:
            - tool_calls_info: list of {name, args} dicts
            - response: final response text from the agent
            - messages: full message history
        target: Ground-truth data (unused for RAG eval — no HF design to compare)
        metadata: Evaluation metadata containing:
            - conditions.expected_volfrac: expected volume fraction (float)
            - conditions.expected_volfrac_tolerance: tolerance window (float)
            - example_id: int

    Returns:
        Dict with:
        - rag_benefit_score: Composite score (0.0-1.0), main metric
        - rag_tool_called: Whether search_documents was invoked (bool)
        - volfrac_accuracy: Raw closeness score for volfrac parameter (0.0-1.0)
        - effective_volfrac_accuracy: volfrac_accuracy gated on rag_tool_called (0.0-1.0)
        - volfrac_actual: Volume fraction the agent used (float | None)
        - volfrac_expected: Expected volume fraction from conditions (float)
        - volfrac_error: Absolute error |actual - expected| (float | None)
        - volfrac_within_tolerance: Whether error <= tolerance (bool)
        - source_cited: Whether the response references the source (bool)
        - example_id: int
    """
    example_id = metadata.get("example_id", 0)
    conditions = metadata.get("conditions", {})

    volfrac_expected: float = float(conditions.get("expected_volfrac", 0.35))
    volfrac_tolerance: float = float(conditions.get("expected_volfrac_tolerance", 0.05))

    tool_calls_info: list[dict[str, Any]] = output.get("tool_calls_info", [])
    response: str = str(output.get("response", ""))

    # --- Dimension 1: RAG tool usage ---
    rag_tool_called = any(tc.get("name") == SEARCH_TOOL_NAME for tc in tool_calls_info)
    logger.debug(
        "Example %s: search_documents called = %s", example_id, rag_tool_called
    )

    # --- Dimension 2: Parameter accuracy ---
    volfrac_actual: float | None = None
    for tc in tool_calls_info:
        if tc.get("name") == OPTIMIZE_TOOL_NAME:
            extracted = _extract_volfrac_from_args(tc.get("args", {}))
            if extracted is not None:
                volfrac_actual = extracted
                break  # Use the first optimize_design call

    volfrac_error: float | None = (
        abs(volfrac_actual - volfrac_expected) if volfrac_actual is not None else None
    )
    volfrac_within_tolerance = (
        volfrac_error is not None and volfrac_error <= volfrac_tolerance
    )
    volfrac_accuracy = _score_volfrac_accuracy(
        volfrac_actual, volfrac_expected, volfrac_tolerance
    )

    logger.info(
        "Example %s: volfrac actual=%.3f expected=%.3f error=%s within_tol=%s",
        example_id,
        volfrac_actual if volfrac_actual is not None else float("nan"),
        volfrac_expected,
        f"{volfrac_error:.3f}" if volfrac_error is not None else "N/A",
        volfrac_within_tolerance,
    )

    # --- Dimension 3: Source citation ---
    source_cited = _check_source_cited(response)
    logger.debug("Example %s: source_cited = %s", example_id, source_cited)

    # --- Composite score ---
    # volfrac_accuracy only contributes when the agent used RAG.
    # This ensures an agent that reaches the correct value via get_problem_details
    # (rather than the paper) does not receive parameter accuracy credit.
    effective_volfrac_accuracy = volfrac_accuracy if rag_tool_called else 0.0
    rag_benefit_score = (
        WEIGHT_VOLFRAC_ACCURACY * effective_volfrac_accuracy
        + WEIGHT_RAG_TOOL_CALLED * float(rag_tool_called)
        + WEIGHT_SOURCE_CITED * float(source_cited)
    )

    logger.info(
        "Example %s: rag_benefit_score=%.3f (eff_volfrac=%.2f, rag_called=%s, cited=%s)",
        example_id,
        rag_benefit_score,
        effective_volfrac_accuracy,
        rag_tool_called,
        source_cited,
    )

    return {
        # Main metric
        "rag_benefit_score": float(rag_benefit_score),
        # Dimension breakdown
        "rag_tool_called": bool(rag_tool_called),
        "volfrac_accuracy": float(volfrac_accuracy),
        "effective_volfrac_accuracy": float(effective_volfrac_accuracy),
        "source_cited": bool(source_cited),
        # Parameter detail
        "volfrac_actual": volfrac_actual,
        "volfrac_expected": volfrac_expected,
        "volfrac_error": volfrac_error,
        "volfrac_within_tolerance": bool(volfrac_within_tolerance),
        # Metadata
        "example_id": int(example_id)
        if isinstance(example_id, (int, float))
        else example_id,
    }
