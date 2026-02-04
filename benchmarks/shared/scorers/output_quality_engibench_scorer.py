"""Constraint violation computation for EngiBench evaluation.

This module provides the compute_rvc function to compute the Ratio of Violated
Constraints (RVC) for generated designs. This is used by compute_global_metrics.py
for offline global metric computation.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _check_design_constraints(
    design: np.ndarray,
    conditions: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Check if a design violates any constraints.

    This function checks volume fraction constraints using the same tolerance-based
    approach as the EngiBench paper's metrics.py implementation (>= tolerance means violation).
    This differs from EngiBench's check_constraints which uses <= tolerance (no violation).

    Args:
        design: Design array to check
        conditions: Conditions dict for the design
    Returns:
        Tuple of (has_violations, list_of_violated_constraint_names)
    """
    constraint_names = []

    # Check volume fraction constraint with tolerance (matching metrics.py from paper)
    # Using >= tolerance for violation (paper's convention), not <= tolerance (EngiBench's convention)
    if conditions:
        tol = 0.01  # Tolerance for equality constraint deviation
        target_vol = conditions.get("volfrac") or conditions.get("volume")
        if target_vol is not None:
            actual_vol = np.mean(design)
            viol = np.abs(actual_vol - target_vol) >= tol
            if viol:
                constraint_names.append(
                    f"volume_fraction_bound: Volume fraction of the design {actual_vol:.4f} "
                    f"does not match target {target_vol:.4f} specified in the conditions. "
                    f"While the optimizer might fix it, this is likely to affect objective "
                    f"values as the initial design is not feasible given the constraints."
                )

    # Note: We do NOT call EngiBench's check_constraints for volume_fraction_bound
    # because it uses a different tolerance convention (<= vs >=) and would give
    # inconsistent results with the paper's metrics.py implementation.
    # If you need to check other constraints beyond volume fraction, add them here.

    has_violations = len(constraint_names) > 0
    return has_violations, constraint_names


def compute_rvc(
    designs: list[np.ndarray],
    conditions_list: list[dict[str, Any]],
    example_ids: list[int] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute Ratio of Violated Constraints (RVC) with detailed violation information.

    RVC measures the fraction of generated designs that violate at least one constraint.
    Lower RVC is better (closer to 0 = no violations).

    Args:
        designs: List of generated designs
        conditions_list: List of conditions dicts, one per design
        example_ids: Optional list of example IDs for detailed logging

    Returns:
        Tuple of (RVC value between 0 and 1, detailed violation info dict)
    """
    if len(designs) == 0:
        return 0.0, {"violations": [], "violation_summary": {}}

    if len(designs) != len(conditions_list):
        logger.warning(
            f"Mismatch: {len(designs)} designs but {len(conditions_list)} conditions, using first condition for all"
        )
        # Use the first condition for all designs if mismatch
        conditions_list = [conditions_list[0]] * len(designs)

    if example_ids is None:
        example_ids = list(range(len(designs)))

    violations = 0
    violation_details = []
    constraint_counter: dict[str, int] = {}

    for idx, (design, conditions, example_id) in enumerate(
        zip(designs, conditions_list, example_ids, strict=False)
    ):
        has_violations, violated_constraints = _check_design_constraints(
            design, conditions
        )

        if has_violations:
            violations += 1
            violation_info = {
                "example_id": example_id,
                "design_index": idx,
                "violated_constraints": violated_constraints,
            }
            violation_details.append(violation_info)

            # Log detailed violation information
            logger.info(
                f"Example {example_id} (index {idx}): VIOLATED {len(violated_constraints)} constraint(s): {', '.join(violated_constraints)}"
            )

            # Count constraint types
            for constraint_name in violated_constraints:
                constraint_counter[constraint_name] = (
                    constraint_counter.get(constraint_name, 0) + 1
                )
        else:
            logger.debug(
                f"Example {example_id} (index {idx}): No constraint violations"
            )

    rvc = violations / len(designs)

    # Create detailed violation info dictionary
    violation_info_dict = {
        "violations": violation_details,
        "violation_summary": constraint_counter,
        "n_violations": violations,
        "n_total": len(designs),
    }

    # Log summary
    if violations > 0:
        logger.info("Constraint violation summary:")
        logger.info(f"  Total designs with violations: {violations}/{len(designs)}")
        logger.info("  Constraint violation counts:")
        for constraint_name, count in sorted(
            constraint_counter.items(), key=lambda x: x[1], reverse=True
        ):
            logger.info(f"    - {constraint_name}: {count} design(s)")
    else:
        logger.info("No constraint violations detected across all designs")

    return rvc, violation_info_dict
