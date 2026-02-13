"""RAG evaluation scorer for rag_beams2d benchmark.

Measures whether the agent uses RAG-retrieved information to make better
engineering decisions.

Supports four scoring modes depending on which conditions are present:

Single-parameter mode (expected_volfrac only):
    Dimensions and weights:
    1. effective_volfrac_accuracy  0.60  (gated on rag_tool_called)
    2. rag_tool_called             0.40

Two-parameter mode (expected_volfrac + expected_forcedist):
    Dimensions and weights:
    1. effective_volfrac_accuracy  0.40  (gated on rag_tool_called)
    2. effective_forcedist_accuracy 0.40 (gated on rag_tool_called)
    3. rag_tool_called             0.20

Rmin two-parameter mode (expected_volfrac + expected_rmin):
    Dimensions and weights:
    1. effective_volfrac_accuracy  0.40  (gated on rag_tool_called)
    2. effective_rmin_accuracy     0.40  (gated on rag_tool_called)
    3. rag_tool_called             0.20

Triple-parameter mode (expected_volfrac + expected_forcedist + expected_rmin):
    Dimensions and weights:
    1. effective_volfrac_accuracy   0.30 (gated on rag_tool_called)
    2. effective_forcedist_accuracy 0.30 (gated on rag_tool_called)
    3. effective_rmin_accuracy      0.30 (gated on rag_tool_called)
    4. rag_tool_called              0.10

In all modes, parameter accuracy dimensions only contribute when
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

# Scorer weights keyed by output column name.
# Imported by the plotting layer to keep weights as a single source of truth.
COMPONENT_WEIGHTS_SINGLE: dict[str, float] = {
    "effective_volfrac_accuracy": 0.60,
    "effective_forcedist_accuracy": 0.00,
    "rag_tool_called": 0.40,
}
COMPONENT_WEIGHTS_DOUBLE: dict[str, float] = {
    "effective_volfrac_accuracy": 0.40,
    "effective_forcedist_accuracy": 0.40,
    "rag_tool_called": 0.20,
}
COMPONENT_WEIGHTS_RMIN: dict[str, float] = {
    "effective_volfrac_accuracy": 0.40,
    "effective_rmin_accuracy": 0.40,
    "rag_tool_called": 0.20,
}
COMPONENT_WEIGHTS_TRIPLE: dict[str, float] = {
    "effective_volfrac_accuracy": 0.30,
    "effective_forcedist_accuracy": 0.30,
    "effective_rmin_accuracy": 0.30,
    "rag_tool_called": 0.10,
}

# Private aliases kept for internal use
_W1_VOLFRAC = COMPONENT_WEIGHTS_SINGLE["effective_volfrac_accuracy"]
_W1_RAG_TOOL = COMPONENT_WEIGHTS_SINGLE["rag_tool_called"]
_W2_VOLFRAC = COMPONENT_WEIGHTS_DOUBLE["effective_volfrac_accuracy"]
_W2_FORCEDIST = COMPONENT_WEIGHTS_DOUBLE["effective_forcedist_accuracy"]
_W2_RAG_TOOL = COMPONENT_WEIGHTS_DOUBLE["rag_tool_called"]
_WR_VOLFRAC = COMPONENT_WEIGHTS_RMIN["effective_volfrac_accuracy"]
_WR_RMIN = COMPONENT_WEIGHTS_RMIN["effective_rmin_accuracy"]
_WR_RAG_TOOL = COMPONENT_WEIGHTS_RMIN["rag_tool_called"]
_WT_VOLFRAC = COMPONENT_WEIGHTS_TRIPLE["effective_volfrac_accuracy"]
_WT_FORCEDIST = COMPONENT_WEIGHTS_TRIPLE["effective_forcedist_accuracy"]
_WT_RMIN = COMPONENT_WEIGHTS_TRIPLE["effective_rmin_accuracy"]
_WT_RAG_TOOL = COMPONENT_WEIGHTS_TRIPLE["rag_tool_called"]

# All fields emitted by score_rag_evaluation — shared with extract/plot layers.
RAG_OUTPUT_FIELDS: tuple[str, ...] = (
    "rag_benefit_score",
    "rag_tool_called",
    # volfrac
    "volfrac_accuracy",
    "effective_volfrac_accuracy",
    "volfrac_actual",
    "volfrac_expected",
    "volfrac_error",
    "volfrac_within_tolerance",
    # forcedist
    "forcedist_accuracy",
    "effective_forcedist_accuracy",
    "forcedist_actual",
    "forcedist_expected",
    "forcedist_error",
    "forcedist_within_tolerance",
    "forcedist_tested",
    # rmin
    "rmin_accuracy",
    "effective_rmin_accuracy",
    "rmin_actual",
    "rmin_expected",
    "rmin_error",
    "rmin_within_tolerance",
    "rmin_tested",
    # metadata
    "example_id",
)


def _extract_param_from_args(args: dict[str, Any], aliases: list[str]) -> float | None:
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


def _extract_rmin_from_args(args: dict[str, Any]) -> float | None:
    """Extract rmin (filter radius) from optimize_design tool args."""
    return _extract_param_from_args(args, ["rmin", "filter_radius", "r_min"])


def _score_param_accuracy(
    actual: float | None,
    expected: float,
    tolerance: float,
) -> float:
    """Score whether the actual value matches the expected within tolerance.

    Returns:
        1.0 if within tolerance, 0.0 otherwise (binary).
    """
    if actual is None:
        return 0.0

    error = abs(actual - expected)
    return 1.0 if error <= tolerance else 0.0


def _score_optional_param(
    actual: float | None,
    expected: float | None,
    tolerance: float,
    tested: bool,
) -> tuple[float | None, bool, float]:
    """Compute (error, within_tolerance, accuracy) for an optional parameter.

    Returns (None, False, 0.0) when the parameter is not tested.
    """
    if not tested or expected is None:
        return None, False, 0.0
    error = abs(actual - expected) if actual is not None else None
    within_tol = error is not None and error <= tolerance
    return error, within_tol, _score_param_accuracy(actual, expected, tolerance)


def score_rag_evaluation(
    output: dict[str, Any],
    target: dict[str, Any],  # noqa: ARG001 — required by scorer interface
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Score a RAG-dependent evaluation prompt.

    Evaluates whether the agent used RAG to retrieve information from the
    EngiBench paper and applied that information accurately to the design task.

    Supports single-parameter prompts (volfrac only), two-parameter prompts
    (volfrac + forcedist), and rmin two-parameter prompts (volfrac + rmin).
    Weights are adjusted automatically based on which conditions are present.

    Args:
        output: Agent output dict containing:
            - tool_calls_info: list of {name, args} dicts
            - messages: full message history
        target: Ground-truth data (unused for RAG eval)
        metadata: Evaluation metadata containing:
            - conditions.expected_volfrac: expected volume fraction (float)
            - conditions.expected_volfrac_tolerance: tolerance window (float)
            - conditions.expected_forcedist: expected force distribution (float, optional)
            - conditions.expected_forcedist_tolerance: tolerance window (float, optional)
            - conditions.expected_rmin: expected filter radius (float, optional)
            - conditions.expected_rmin_tolerance: tolerance window (float, optional)
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
        - rmin_accuracy: Raw closeness score for rmin (0.0-1.0), 0.0 if not tested
        - effective_rmin_accuracy: rmin_accuracy gated on rag_tool_called (0.0-1.0)
        - rmin_actual: Filter radius the agent used (float | None)
        - rmin_expected: Expected rmin from conditions (float | None)
        - rmin_error: Absolute error |actual - expected| (float | None)
        - rmin_within_tolerance: Whether error <= tolerance (bool)
        - rmin_tested: Whether rmin scoring was active for this prompt (bool)
        - example_id: int
    """
    example_id = metadata.get("example_id", 0)
    conditions = metadata.get("conditions", {})

    volfrac_expected: float = float(conditions.get("expected_volfrac", 0.35))
    volfrac_tolerance: float = float(conditions.get("expected_volfrac_tolerance", 0.05))

    # forcedist is optional — present only in forcedist two-parameter prompts
    forcedist_expected_raw = conditions.get("expected_forcedist")
    forcedist_tested = forcedist_expected_raw is not None
    forcedist_expected: float | None = (
        float(forcedist_expected_raw) if forcedist_tested else None
    )
    forcedist_tolerance: float = float(
        conditions.get("expected_forcedist_tolerance", 0.05)
    )

    # rmin is optional — present only in rmin two-parameter prompts
    rmin_expected_raw = conditions.get("expected_rmin")
    rmin_tested = rmin_expected_raw is not None
    rmin_expected: float | None = float(rmin_expected_raw) if rmin_tested else None
    rmin_tolerance: float = float(conditions.get("expected_rmin_tolerance", 0.05))

    tool_calls_info: list[dict[str, Any]] = output.get("tool_calls_info", [])

    # --- Dimension: RAG tool usage ---
    rag_tool_called = any(tc.get("name") == SEARCH_TOOL_NAME for tc in tool_calls_info)
    logger.debug(
        "Example %s: search_documents called = %s", example_id, rag_tool_called
    )

    # --- Dimension: volfrac, forcedist, rmin accuracy ---
    volfrac_actual: float | None = None
    forcedist_actual: float | None = None
    rmin_actual: float | None = None

    for tc in tool_calls_info:
        if tc.get("name") == OPTIMIZE_TOOL_NAME:
            args = tc.get("args", {})
            if volfrac_actual is None:
                volfrac_actual = _extract_volfrac_from_args(args)
            if forcedist_actual is None:
                forcedist_actual = _extract_forcedist_from_args(args)
            if rmin_actual is None:
                rmin_actual = _extract_rmin_from_args(args)
            if (
                volfrac_actual is not None
                and (not forcedist_tested or forcedist_actual is not None)
                and (not rmin_tested or rmin_actual is not None)
            ):
                break  # All needed values extracted

    volfrac_error, volfrac_within_tolerance, volfrac_accuracy = _score_optional_param(
        volfrac_actual, volfrac_expected, volfrac_tolerance, tested=True
    )
    logger.info(
        "Example %s: volfrac actual=%s expected=%.3f error=%s within_tol=%s",
        example_id,
        f"{volfrac_actual:.3f}" if volfrac_actual is not None else "N/A",
        volfrac_expected,
        f"{volfrac_error:.3f}" if volfrac_error is not None else "N/A",
        volfrac_within_tolerance,
    )

    # --- Dimension: forcedist accuracy (forcedist two-parameter prompts only) ---
    forcedist_error, forcedist_within_tolerance, forcedist_accuracy = (
        _score_optional_param(
            forcedist_actual, forcedist_expected, forcedist_tolerance, forcedist_tested
        )
    )
    if forcedist_tested:
        logger.info(
            "Example %s: forcedist actual=%s expected=%.3f error=%s within_tol=%s",
            example_id,
            f"{forcedist_actual:.3f}" if forcedist_actual is not None else "N/A",
            forcedist_expected,
            f"{forcedist_error:.3f}" if forcedist_error is not None else "N/A",
            forcedist_within_tolerance,
        )

    # --- Dimension: rmin accuracy (rmin two-parameter prompts only) ---
    rmin_error, rmin_within_tolerance, rmin_accuracy = _score_optional_param(
        rmin_actual, rmin_expected, rmin_tolerance, rmin_tested
    )
    if rmin_tested:
        logger.info(
            "Example %s: rmin actual=%s expected=%.3f error=%s within_tol=%s",
            example_id,
            f"{rmin_actual:.3f}" if rmin_actual is not None else "N/A",
            rmin_expected,
            f"{rmin_error:.3f}" if rmin_error is not None else "N/A",
            rmin_within_tolerance,
        )

    # --- Composite score ---
    # Parameter accuracy dimensions only contribute when RAG was called.
    # This ensures agents that reach correct values via get_problem_details
    # (rather than the paper) receive no parameter accuracy credit.
    effective_volfrac_accuracy = volfrac_accuracy if rag_tool_called else 0.0
    effective_forcedist_accuracy = forcedist_accuracy if rag_tool_called else 0.0
    effective_rmin_accuracy = rmin_accuracy if rag_tool_called else 0.0

    if forcedist_tested and rmin_tested:
        # Triple-parameter mode (volfrac + forcedist + rmin)
        rag_benefit_score = (
            _WT_VOLFRAC * effective_volfrac_accuracy
            + _WT_FORCEDIST * effective_forcedist_accuracy
            + _WT_RMIN * effective_rmin_accuracy
            + _WT_RAG_TOOL * float(rag_tool_called)
        )
    elif forcedist_tested:
        # Two-parameter mode (volfrac + forcedist)
        rag_benefit_score = (
            _W2_VOLFRAC * effective_volfrac_accuracy
            + _W2_FORCEDIST * effective_forcedist_accuracy
            + _W2_RAG_TOOL * float(rag_tool_called)
        )
    elif rmin_tested:
        # Two-parameter mode (volfrac + rmin)
        rag_benefit_score = (
            _WR_VOLFRAC * effective_volfrac_accuracy
            + _WR_RMIN * effective_rmin_accuracy
            + _WR_RAG_TOOL * float(rag_tool_called)
        )
    else:
        # Single-parameter mode
        rag_benefit_score = (
            _W1_VOLFRAC * effective_volfrac_accuracy
            + _W1_RAG_TOOL * float(rag_tool_called)
        )

    logger.info(
        "Example %s: rag_benefit_score=%.3f "
        "(eff_volfrac=%.2f, eff_forcedist=%.2f, eff_rmin=%.2f, "
        "rag_called=%s, forcedist_mode=%s, rmin_mode=%s, triple_mode=%s)",
        example_id,
        rag_benefit_score,
        effective_volfrac_accuracy,
        effective_forcedist_accuracy,
        effective_rmin_accuracy,
        rag_tool_called,
        forcedist_tested,
        rmin_tested,
        forcedist_tested and rmin_tested,
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
        "rmin_accuracy": float(rmin_accuracy),
        "effective_rmin_accuracy": float(effective_rmin_accuracy),
        "rmin_tested": bool(rmin_tested),
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
        # rmin detail
        "rmin_actual": rmin_actual,
        "rmin_expected": rmin_expected,
        "rmin_error": rmin_error,
        "rmin_within_tolerance": bool(rmin_within_tolerance),
        # Metadata
        "example_id": int(example_id)
        if isinstance(example_id, (int, float))
        else example_id,
    }
