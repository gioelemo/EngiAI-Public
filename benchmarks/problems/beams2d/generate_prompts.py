"""
Generate prompts from beam design conditions.

This script transforms beam design parameters from the HuggingFace dataset
into natural language prompts that can be used to evaluate the engineering agent.

Supports multiple prompt styles:
- full: Exact numerical parameters
- natural: Natural language descriptions only
- workflow: Full workflow with export steps
- workflow-derived-params: Workflow with STL parameters derived from optimization inputs
- workflow-distractor: Workflow with distractor parameters mixed with real STL params
- workflow-conditional: Workflow with if/then branching based on simulation results
- workflow-multi-export: Workflow requiring two STL exports with different parameters

Dataset: https://huggingface.co/datasets/IDEALLab/beams_2d_50_100_v0
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path to import src modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.shared.prompt_generation import (  # noqa: E402
    run_prompt_generation_workflow,
)

# Force distribution thresholds for natural language descriptions
FORCE_CONCENTRATED = 0.2
FORCE_LOWER_LEFT = 0.4
FORCE_MIDDLE = 0.6
FORCE_UPPER_RIGHT = 0.8

# Compliance thresholds for stiffness descriptions
COMPLIANCE_VERY_STIFF = 30
COMPLIANCE_STIFF = 60
COMPLIANCE_MODERATE = 100
COMPLIANCE_FLEXIBLE = 200

# Volume fraction thresholds for descriptions
VOLFRAC_LIGHTWEIGHT = 0.3
VOLFRAC_MODERATE = 0.5

# Derivation rule constants for workflow-derived-params
DERIVED_SCALE_XY_MULTIPLIER = 2
DERIVED_SCALE_Z_MULTIPLIER = 40
DERIVED_MIRROR_VOLFRAC_THRESHOLD = 0.4

# Seed offset for distractor parameter generation (workflow-distractor style)
DISTRACTOR_SEED_OFFSET = 10000

# STL parameter ranges for random generation (used across all workflow styles)
STL_THRESHOLD_MIN = 0.3  # Minimum density threshold for solid/void conversion
STL_THRESHOLD_MAX = 0.7  # Maximum density threshold
STL_SCALE_XY_MIN = 0.5  # Minimum XY dimension scaling factor
STL_SCALE_XY_MAX = 5.0  # Maximum XY dimension scaling factor
STL_SCALE_Z_MIN = 5.0  # Minimum Z extrusion height
STL_SCALE_Z_MAX = 20.0  # Maximum Z extrusion height

# Compliance threshold range for workflow-conditional branching
COMPLIANCE_THRESHOLD_MIN = 100.0  # Minimum compliance threshold value
COMPLIANCE_THRESHOLD_MAX = 300.0  # Maximum compliance threshold value

# Gap requirements for parameter distinctness
MIN_THRESHOLD_GAP = 0.1  # Minimum gap between threshold values for distinguishability
MIN_DISTRACTOR_SCALE_GAP = (
    0.5  # Larger gap for distractor vs real (clearer distinction)
)
MIN_MULTI_EXPORT_SCALE_GAP = 0.2  # Smaller gap for two valid exports (both are real)

# Fallback thresholds for distractor parameter generation (when random sampling fails)
DISTRACTOR_THRESHOLD_MIDPOINT = (STL_THRESHOLD_MIN + STL_THRESHOLD_MAX) / 2
DISTRACTOR_SCALE_XY_MIDPOINT = (STL_SCALE_XY_MIN + STL_SCALE_XY_MAX) / 2

# Prompt styles configuration with their optimal tool sequences
PROMPT_STYLES: dict[str, dict[str, Any]] = {
    "full": {
        "description": "Exact numerical parameters",
        "optimal_tool_calls": [
            {"name": "optimize_design", "count": 1},
            {"name": "simulate_design", "count": 1},
            {"name": "render_design", "count": 1},
        ],
        "optimal_call_count": 3,
    },
    "natural": {
        "description": "Natural language descriptions only",
        "optimal_tool_calls": [
            {"name": "ask_human_for_clarification", "count": 1},
        ],
        "optimal_call_count": 1,
        "success_criteria": "clarification_requested",
    },
    "workflow-random": {
        "description": "Full workflow with random STL parameters",
        "optimal_tool_calls": [
            {"name": "optimize_design", "count": 1},
            {"name": "simulate_design", "count": 1},
            {"name": "convert_design_to_stl", "count": 1},
        ],
        "optimal_call_count": 3,
        "success_criteria": "stl_export_with_params",
        "validate_stl_params": True,
    },
    "workflow-derived-params": {
        "description": "Workflow with STL parameters derived from optimization inputs",
        "optimal_tool_calls": [
            {"name": "optimize_design", "count": 1},
            {"name": "convert_design_to_stl", "count": 1},
        ],
        "optimal_call_count": 2,
        "success_criteria": "stl_export_with_params",
        "validate_stl_params": True,
    },
    "workflow-distractor": {
        "description": "Workflow with distractor parameters mixed with real STL params",
        "optimal_tool_calls": [
            {"name": "optimize_design", "count": 1},
            {"name": "convert_design_to_stl", "count": 1},
        ],
        "optimal_call_count": 2,
        "success_criteria": "stl_export_with_params",
        "validate_stl_params": True,
    },
    "workflow-conditional": {
        "description": "Workflow with conditional branching based on simulation results",
        "optimal_tool_calls": [
            {"name": "optimize_design", "count": 1},
            {"name": "simulate_design", "count": 1},
            {"name": "convert_design_to_stl", "count": 1},
        ],
        "optimal_call_count": 3,
        "success_criteria": "stl_export_with_params",
        "validate_stl_params": True,
    },
    "workflow-multi-export": {
        "description": "Workflow requiring two STL exports with different parameters",
        "optimal_tool_calls": [
            {"name": "optimize_design", "count": 1},
            {"name": "simulate_design", "count": 1},
            {"name": "convert_design_to_stl", "count": 2},
        ],
        "optimal_call_count": 4,
        "success_criteria": "stl_export_with_params",
        "validate_stl_params": True,
    },
}


def describe_force_distribution(forcedist: float) -> str:
    """Convert forcedist value to natural language description."""
    if forcedist < FORCE_CONCENTRATED:
        return "a concentrated force at the bottom left"
    if forcedist < FORCE_LOWER_LEFT:
        return "a force distributed in the lower left region"
    if forcedist < FORCE_MIDDLE:
        return "a force distributed in the middle region"
    if forcedist < FORCE_UPPER_RIGHT:
        return "a force distributed in the upper right region"
    return "a uniformly distributed force"


def describe_expected_stiffness(compliance: float) -> str:
    """Convert compliance value to natural language stiffness description."""
    if compliance < COMPLIANCE_VERY_STIFF:
        return "very stiff (very low compliance)"
    if compliance < COMPLIANCE_STIFF:
        return "stiff (low compliance)"
    if compliance < COMPLIANCE_MODERATE:
        return "moderately stiff"
    if compliance < COMPLIANCE_FLEXIBLE:
        return "somewhat flexible"
    return "flexible (high compliance)"


def describe_volfrac(volfrac: float) -> str:
    """Convert volume fraction to natural language description."""
    if volfrac < VOLFRAC_LIGHTWEIGHT:
        return "lightweight (low material usage)"
    if volfrac < VOLFRAC_MODERATE:
        return "moderate material usage"
    return "dense (high material usage)"


def _create_full_prompt(volfrac: float, forcedist: float, rmin: float) -> str:
    """Create prompt with exact numerical parameters."""
    return (
        f"Design a 2D beam structure.\n\n"
        f"Design requirements:\n"
        f"- Use a material volume fraction of {volfrac}\n"
        f"- Force distribution parameter: {forcedist}\n"
        f"- Minimum filter radius (rmin): {rmin}\n\n"
        f"Optimize the structure to minimize compliance while respecting the volume constraint."
    )


def _create_natural_prompt(volfrac: float, forcedist: float, compliance: float) -> str:
    """Create prompt with natural language descriptions only."""
    force_desc = describe_force_distribution(forcedist)
    stiffness_desc = describe_expected_stiffness(compliance)
    volfrac_desc = describe_volfrac(volfrac)

    return (
        f"Design a 2D beam structure.\n\n"
        f"Design requirements:\n"
        f"- The design should be {volfrac_desc}\n"
        f"- Apply {force_desc}\n"
        f"- The resulting structure should be {stiffness_desc}\n"
        f"Optimize the structure to minimize compliance while respecting the volume constraint."
    )


def _generate_random_stl_params(seed: int | None = None) -> dict[str, Any]:
    """Generate random STL parameters for workflow-random prompts.

    Uses deterministic seed for reproducibility across evaluations.
    Ranges chosen based on tool defaults and 3D printing constraints.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Dict with: mirror_y (bool), scale_xy (float), scale_z (float), threshold (float)
    """
    rng = np.random.default_rng(seed)

    return {
        "mirror_y": bool(rng.choice([True, False])),
        "scale_xy": float(rng.uniform(STL_SCALE_XY_MIN, STL_SCALE_XY_MAX)),
        "scale_z": float(rng.uniform(STL_SCALE_Z_MIN, STL_SCALE_Z_MAX)),
        "threshold": float(rng.uniform(STL_THRESHOLD_MIN, STL_THRESHOLD_MAX)),
    }


def _generate_distractor_params(
    rng: np.random.Generator,
    real_params: dict[str, Any],
) -> dict[str, float]:
    """Generate competing distractor values for real STL parameters.

    Produces alternative values for ``threshold`` and ``scale_xy`` that look
    plausible but belong to a non-export context (preview / analysis).  The
    agent must pick the correct (export) values, not these distractors.

    Each distractor is guaranteed to differ from the real value by at least
    the minimum gap so the ±0.05 scorer tolerance can always distinguish them.

    Args:
        rng: NumPy random generator instance.
        real_params: The real STL parameters (from ``_generate_random_stl_params``).

    Returns:
        Dict with ``distractor_threshold`` (float) and
        ``distractor_scale_xy`` (float).
    """
    max_attempts = 100

    # Generate distractor threshold with guaranteed gap from real value
    real_threshold = real_params["threshold"]
    dt = None
    for _ in range(max_attempts):
        candidate = float(rng.uniform(STL_THRESHOLD_MIN, STL_THRESHOLD_MAX))
        if abs(candidate - real_threshold) >= MIN_THRESHOLD_GAP:
            dt = candidate
            break

    # Fallback: if random sampling fails, use deterministic value with guaranteed gap
    if dt is None:
        # Place distractor at opposite end of range from real value
        dt = (
            STL_THRESHOLD_MIN
            if real_threshold > DISTRACTOR_THRESHOLD_MIDPOINT
            else STL_THRESHOLD_MAX
        )

    # Generate distractor scale_xy with guaranteed gap from real value
    real_scale_xy = real_params["scale_xy"]
    ds = None
    for _ in range(max_attempts):
        candidate = float(rng.uniform(STL_SCALE_XY_MIN, STL_SCALE_XY_MAX))
        if abs(candidate - real_scale_xy) >= MIN_DISTRACTOR_SCALE_GAP:
            ds = candidate
            break

    # Fallback: if random sampling fails, use deterministic value with guaranteed gap
    if ds is None:
        # Place distractor at opposite end of range from real value
        ds = (
            STL_SCALE_XY_MIN
            if real_scale_xy > DISTRACTOR_SCALE_XY_MIDPOINT
            else STL_SCALE_XY_MAX
        )

    return {
        "distractor_threshold": dt,
        "distractor_scale_xy": ds,
    }


def _generate_random_conditional_params(seed: int | None = None) -> dict[str, Any]:
    """Generate random parameters for workflow-conditional prompts.

    Generates a compliance threshold for branching, two distinct sets of
    branch-specific params (threshold, mirror_y), and common params (scale_xy, scale_z).

    Args:
        seed: Random seed for reproducibility

    Returns:
        Dict with: conditional flag, compliance_threshold, branch_high, branch_low, common
    """
    rng = np.random.default_rng(seed)

    # Compliance threshold: 100-300 range (centered on COMPLIANCE_FLEXIBLE=200)
    compliance_threshold = float(
        rng.uniform(COMPLIANCE_THRESHOLD_MIN, COMPLIANCE_THRESHOLD_MAX)
    )

    # Branch-specific parameters: threshold and mirror_y
    threshold_high = float(rng.uniform(STL_THRESHOLD_MIN, STL_THRESHOLD_MAX))
    threshold_low = float(rng.uniform(STL_THRESHOLD_MIN, STL_THRESHOLD_MAX))

    # Ensure the two thresholds are distinguishable
    for _ in range(100):
        if abs(threshold_high - threshold_low) >= MIN_THRESHOLD_GAP:
            break
        threshold_low = float(rng.uniform(STL_THRESHOLD_MIN, STL_THRESHOLD_MAX))
    else:
        # Deterministic fallback: place at opposite end of range
        midpoint = (STL_THRESHOLD_MIN + STL_THRESHOLD_MAX) / 2
        threshold_low = (
            STL_THRESHOLD_MIN if threshold_high > midpoint else STL_THRESHOLD_MAX
        )

    # Mirror: one branch mirrors, the other does not (guaranteed distinct)
    mirror_high = bool(rng.choice([True, False]))
    mirror_low = not mirror_high

    # Common parameters (same for both branches)
    scale_xy = float(rng.uniform(STL_SCALE_XY_MIN, STL_SCALE_XY_MAX))
    scale_z = float(rng.uniform(STL_SCALE_Z_MIN, STL_SCALE_Z_MAX))

    return {
        "conditional": True,
        "compliance_threshold": round(compliance_threshold, 1),
        "branch_high": {
            "threshold": threshold_high,
            "mirror_y": mirror_high,
        },
        "branch_low": {
            "threshold": threshold_low,
            "mirror_y": mirror_low,
        },
        "common": {
            "scale_xy": scale_xy,
            "scale_z": scale_z,
        },
    }


def _generate_random_multi_export_params(seed: int | None = None) -> dict[str, Any]:
    """Generate random parameters for workflow-multi-export prompts.

    Generates TWO distinct param sets for two STL exports from the same optimization.
    Distinctness guarantees: mirror_y opposite, threshold gap >= MIN_THRESHOLD_GAP,
    scale_xy gap >= MIN_MULTI_EXPORT_SCALE_GAP, scale_z gap >= MIN_MULTI_EXPORT_SCALE_GAP.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Dict with: multi_export flag, exports list of two param dicts
    """
    rng = np.random.default_rng(seed)

    # Mirror: guaranteed opposite
    mirror_a = bool(rng.choice([True, False]))
    mirror_b = not mirror_a

    # Thresholds: both in [0.3, 0.7], gap >= MIN_THRESHOLD_GAP
    threshold_a = float(rng.uniform(STL_THRESHOLD_MIN, STL_THRESHOLD_MAX))
    threshold_b = float(rng.uniform(STL_THRESHOLD_MIN, STL_THRESHOLD_MAX))
    for _ in range(100):
        if abs(threshold_a - threshold_b) >= MIN_THRESHOLD_GAP:
            break
        threshold_b = float(rng.uniform(STL_THRESHOLD_MIN, STL_THRESHOLD_MAX))
    else:
        midpoint = (STL_THRESHOLD_MIN + STL_THRESHOLD_MAX) / 2
        threshold_b = STL_THRESHOLD_MIN if threshold_a > midpoint else STL_THRESHOLD_MAX

    # Scale XY: both in [0.5, 5.0], gap >= MIN_MULTI_EXPORT_SCALE_GAP
    scale_xy_a = float(rng.uniform(STL_SCALE_XY_MIN, STL_SCALE_XY_MAX))
    scale_xy_b = float(rng.uniform(STL_SCALE_XY_MIN, STL_SCALE_XY_MAX))
    for _ in range(100):
        if abs(scale_xy_a - scale_xy_b) >= MIN_MULTI_EXPORT_SCALE_GAP:
            break
        scale_xy_b = float(rng.uniform(STL_SCALE_XY_MIN, STL_SCALE_XY_MAX))
    else:
        midpoint = (STL_SCALE_XY_MIN + STL_SCALE_XY_MAX) / 2
        scale_xy_b = STL_SCALE_XY_MIN if scale_xy_a > midpoint else STL_SCALE_XY_MAX

    # Scale Z: both in [5.0, 20.0], gap >= MIN_MULTI_EXPORT_SCALE_GAP
    scale_z_a = float(rng.uniform(STL_SCALE_Z_MIN, STL_SCALE_Z_MAX))
    scale_z_b = float(rng.uniform(STL_SCALE_Z_MIN, STL_SCALE_Z_MAX))
    for _ in range(100):
        if abs(scale_z_a - scale_z_b) >= MIN_MULTI_EXPORT_SCALE_GAP:
            break
        scale_z_b = float(rng.uniform(STL_SCALE_Z_MIN, STL_SCALE_Z_MAX))
    else:
        midpoint = (STL_SCALE_Z_MIN + STL_SCALE_Z_MAX) / 2
        scale_z_b = STL_SCALE_Z_MIN if scale_z_a > midpoint else STL_SCALE_Z_MAX

    return {
        "multi_export": True,
        "exports": [
            {
                "label": "A",
                "mirror_y": mirror_a,
                "scale_xy": scale_xy_a,
                "scale_z": scale_z_a,
                "threshold": threshold_a,
            },
            {
                "label": "B",
                "mirror_y": mirror_b,
                "scale_xy": scale_xy_b,
                "scale_z": scale_z_b,
                "threshold": threshold_b,
            },
        ],
    }


def _compute_derived_stl_params(volfrac: float, rmin: float) -> dict[str, Any]:
    """Compute STL parameters from optimization inputs using derivation rules.

    Rules:
    - threshold = volfrac
    - scale_xy = 2 * rmin
    - scale_z = threshold * 40 = volfrac * 40
    - mirror_y = True if volfrac > 0.4

    Args:
        volfrac: Volume fraction from optimization
        rmin: Minimum filter radius from optimization

    Returns:
        Flat dict with: mirror_y (bool), scale_xy (float), scale_z (float), threshold (float)
    """
    threshold = volfrac
    return {
        "mirror_y": volfrac > DERIVED_MIRROR_VOLFRAC_THRESHOLD,
        "scale_xy": float(DERIVED_SCALE_XY_MULTIPLIER * rmin),
        "scale_z": float(DERIVED_SCALE_Z_MULTIPLIER * threshold),
        "threshold": float(threshold),
    }


def _create_workflow_derived_params_prompt(
    volfrac: float, forcedist: float, rmin: float
) -> tuple[str, dict[str, Any]]:
    """Create workflow prompt where STL params are derived from optimization inputs.

    The prompt gives derivation RULES (not final values). The agent must compute
    the correct parameters from the optimization inputs.

    Args:
        volfrac: Volume fraction for optimization
        forcedist: Force distribution parameter
        rmin: Minimum filter radius

    Returns:
        Tuple of (prompt_text, stl_expected_params_flat_dict)
    """
    stl_params = _compute_derived_stl_params(volfrac, rmin)

    prompt = (
        f"Execute a 2D topology optimization and export the resulting geometry "
        f"as a 3D-printable STL file.\n\n"
        f"1. Optimization Configuration\n"
        f"   - Volume Fraction: {volfrac}\n"
        f"   - Force Distribution: {forcedist}\n"
        f"   - Filter Radius (rmin): {rmin}\n"
        f"   - Objective: Minimize compliance\n\n"
        f"2. Post-processing & Export\n"
        f"   The STL export parameters must be derived from the optimization inputs:\n"
        f"   - Thresholding: Use the volume fraction value as the density threshold\n"
        f"   - Mirror: Mirror the design across the y-axis only if the volume fraction "
        f"is greater than 0.4\n"
        f"   - XY Scaling: Scale the X and Y dimensions by twice the filter radius\n"
        f"   - Extrusion: Extrude the 2D result in the Z-axis by the threshold value "
        f"multiplied by 40\n"
        f"   - Export: Save the final geometry as an STL file with these derived parameters"
    )

    return prompt, stl_params


def _create_workflow_multi_export_prompt(
    volfrac: float,
    forcedist: float,
    rmin: float,
    example_id: int,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create workflow prompt requiring two STL exports with different parameters.

    The prompt instructs the agent to:
    1. Optimize and simulate
    2. Export STL twice with completely different parameter sets (A and B)

    Args:
        volfrac: Volume fraction for optimization
        forcedist: Force distribution parameter
        rmin: Minimum filter radius
        example_id: Unique example identifier
        seed: Base seed for random generation

    Returns:
        Tuple of (prompt_text, multi_export_params_dict)
    """
    unique_seed = (seed if seed is not None else 0) + example_id
    params = _generate_random_multi_export_params(unique_seed)

    export_a = params["exports"][0]
    export_b = params["exports"][1]

    mirror_a_instr = (
        "Mirror the design across the y-axis"
        if export_a["mirror_y"]
        else "Do NOT mirror the design"
    )
    mirror_b_instr = (
        "Mirror the design across the y-axis"
        if export_b["mirror_y"]
        else "Do NOT mirror the design"
    )

    prompt = (
        f"Execute a 2D topology optimization and export the resulting geometry "
        f"as TWO separate 3D-printable STL files with different parameters.\n\n"
        f"1. Optimization Configuration\n"
        f"   - Volume Fraction: {volfrac}\n"
        f"   - Force Distribution: {forcedist}\n"
        f"   - Filter Radius (rmin): {rmin}\n"
        f"   - Objective: Minimize compliance\n\n"
        f"2. Post-processing & Export\n\n"
        f"   Export A:\n"
        f"   - Thresholding: Apply a {export_a['threshold']:.2f} density threshold "
        f"to convert the continuous density map into binary geometry\n"
        f"   - Mirror: {mirror_a_instr} for the final geometry\n"
        f"   - XY Scaling: Scale the X and Y dimensions by {export_a['scale_xy']:.2f}\n"
        f"   - Extrusion: Extrude the 2D result by {export_a['scale_z']:.1f} units "
        f"in the Z-axis to create a 3D volume\n"
        f"   - Export: Save the final geometry as an STL file with these exact parameters\n\n"
        f"   Export B:\n"
        f"   - Thresholding: Apply a {export_b['threshold']:.2f} density threshold "
        f"to convert the continuous density map into binary geometry\n"
        f"   - Mirror: {mirror_b_instr} for the final geometry\n"
        f"   - XY Scaling: Scale the X and Y dimensions by {export_b['scale_xy']:.2f}\n"
        f"   - Extrusion: Extrude the 2D result by {export_b['scale_z']:.1f} units "
        f"in the Z-axis to create a 3D volume\n"
        f"   - Export: Save the final geometry as an STL file with these exact parameters"
    )

    return prompt, params


def _create_workflow_conditional_prompt(
    volfrac: float,
    forcedist: float,
    rmin: float,
    example_id: int,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create workflow prompt with conditional branching based on compliance.

    The prompt instructs the agent to:
    1. Optimize and simulate to get compliance
    2. Branch based on compliance vs. a randomized threshold
    3. Apply branch-specific STL params (threshold, mirror_y)
    4. Apply common STL params (scale_xy, scale_z)

    Args:
        volfrac: Volume fraction for optimization
        forcedist: Force distribution parameter
        rmin: Minimum filter radius
        example_id: Unique example identifier
        seed: Base seed for random generation

    Returns:
        Tuple of (prompt_text, conditional_stl_expected_params_dict)
    """
    unique_seed = (seed if seed is not None else 0) + example_id
    params = _generate_random_conditional_params(unique_seed)

    ct = params["compliance_threshold"]
    bh = params["branch_high"]
    bl = params["branch_low"]
    common = params["common"]

    mirror_high_instr = (
        "Mirror the design across the y-axis"
        if bh["mirror_y"]
        else "Do NOT mirror the design"
    )
    mirror_low_instr = (
        "Mirror the design across the y-axis"
        if bl["mirror_y"]
        else "Do NOT mirror the design"
    )

    prompt = (
        f"Execute a 2D topology optimization, simulate the result, then export "
        f"the geometry as a 3D-printable STL file with parameters that depend on "
        f"the simulation outcome.\n\n"
        f"1. Optimization Configuration\n"
        f"   - Volume Fraction: {volfrac}\n"
        f"   - Force Distribution: {forcedist}\n"
        f"   - Filter Radius (rmin): {rmin}\n"
        f"   - Objective: Minimize compliance\n\n"
        f"2. Simulation\n"
        f"   - After optimization, simulate the design to obtain the compliance value\n\n"
        f"3. Post-processing & Export (conditional on compliance)\n"
        f"   - If compliance > {ct:.1f}:\n"
        f"     - Thresholding: Apply a {bh['threshold']:.2f} density threshold "
        f"to convert the continuous density map into binary geometry\n"
        f"     - Mirror: {mirror_high_instr} for the final geometry\n"
        f"   - If compliance <= {ct:.1f}:\n"
        f"     - Thresholding: Apply a {bl['threshold']:.2f} density threshold "
        f"to convert the continuous density map into binary geometry\n"
        f"     - Mirror: {mirror_low_instr} for the final geometry\n"
        f"   - In both cases:\n"
        f"     - XY Scaling: Scale the X and Y dimensions by {common['scale_xy']:.2f}\n"
        f"     - Extrusion: Extrude the 2D result by {common['scale_z']:.1f} units "
        f"in the Z-axis to create a 3D volume\n"
        f"   - Export: Save the final geometry as an STL file with these exact parameters"
    )

    return prompt, params


def _create_workflow_random_prompt(
    volfrac: float,
    forcedist: float,
    rmin: float,
    example_id: int,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create workflow prompt with random STL parameters.

    Args:
        volfrac: Volume fraction for optimization
        forcedist: Force distribution parameter
        rmin: Minimum filter radius
        example_id: Unique example identifier
        seed: Base seed for random generation

    Returns:
        Tuple of (prompt_text, stl_expected_params_dict)
    """
    # Generate random params (use seed + example_id for uniqueness per prompt)
    unique_seed = (seed if seed is not None else 0) + example_id
    stl_params = _generate_random_stl_params(unique_seed)

    # Format clear instructions
    mirror_instruction = (
        "Mirror the design across the y-axis"
        if stl_params["mirror_y"]
        else "Do NOT mirror the design"
    )

    prompt = (
        f"Execute a 2D topology optimization and export the resulting geometry "
        f"as a 3D-printable STL file.\n\n"
        f"1. Optimization Configuration\n"
        f"   - Volume Fraction: {volfrac}\n"
        f"   - Force Distribution: {forcedist}\n"
        f"   - Filter Radius (rmin): {rmin}\n"
        f"   - Objective: Minimize compliance\n\n"
        f"2. Post-processing & Export\n"
        f"   - Thresholding: Apply a {stl_params['threshold']:.2f} density threshold "
        f"to convert the continuous density map into binary geometry\n"
        f"   - Mirror: {mirror_instruction} for the final geometry\n"
        f"   - XY Scaling: Scale the X and Y dimensions by {stl_params['scale_xy']:.2f}\n"
        f"   - Extrusion: Extrude the 2D result by {stl_params['scale_z']:.1f} units "
        f"in the Z-axis to create a 3D volume\n"
        f"   - Export: Save the final geometry as an STL file with these exact parameters"
    )

    return prompt, stl_params


def _create_workflow_distractor_prompt(
    volfrac: float,
    forcedist: float,
    rmin: float,
    example_id: int,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create workflow prompt with competing parameter values.

    The prompt presents two plausible values for ``threshold`` and
    ``scale_xy`` — one in a preview/analysis context and one in the
    manufacturing/export context.  The agent must pick the export-context
    values.  Because both values map to valid tool-schema keys, LangChain
    will not reject the call; only the parameter-validation scorer catches
    the wrong choice.

    Args:
        volfrac: Volume fraction for optimization
        forcedist: Force distribution parameter
        rmin: Minimum filter radius
        example_id: Unique example identifier
        seed: Base seed for random generation

    Returns:
        Tuple of (prompt_text, stl_expected_params_dict) where expected params
        contains only the 4 real STL parameters (export-context values)
    """
    # Generate real params (same as workflow-random)
    unique_seed = (seed if seed is not None else 0) + example_id
    stl_params = _generate_random_stl_params(unique_seed)

    # Generate competing distractor values with a derived seed
    distractor_rng = np.random.default_rng(unique_seed + DISTRACTOR_SEED_OFFSET)
    distractors = _generate_distractor_params(distractor_rng, stl_params)

    # Format mirror instruction
    mirror_instruction = (
        "Mirror the design across the y-axis"
        if stl_params["mirror_y"]
        else "Do NOT mirror the design"
    )

    prompt = (
        f"Execute a 2D topology optimization and export the resulting geometry "
        f"as a 3D-printable STL file.\n\n"
        f"1. Optimization Configuration\n"
        f"   - Volume Fraction: {volfrac}\n"
        f"   - Force Distribution: {forcedist}\n"
        f"   - Filter Radius (rmin): {rmin}\n"
        f"   - Objective: Minimize compliance\n\n"
        f"2. Post-processing & Export\n"
        f"   - Threshold the density field at "
        f"{distractors['distractor_threshold']:.2f} to preview the design "
        f"topology\n"
        f"   - Apply a {stl_params['threshold']:.2f} density threshold to "
        f"produce the final solid/void geometry\n"
        f"   - Scale the preview display by "
        f"{distractors['distractor_scale_xy']:.2f}x in XY for quick "
        f"inspection\n"
        f"   - Scale the X and Y dimensions of the part by "
        f"{stl_params['scale_xy']:.2f} for manufacturing\n"
        f"   - {mirror_instruction} for the final geometry\n"
        f"   - Extrude the 2D result by {stl_params['scale_z']:.1f} units "
        f"in the Z-axis to create a 3D volume\n"
        f"   - Export: Save the final geometry as an STL file with these exact parameters"
    )

    return prompt, stl_params


def create_prompt_from_conditions(
    example: dict[str, Any],
    include_target: bool = True,
    prompt_style: str = "full",
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Create a benchmark prompt from beam design conditions.

    Args:
        example: Single example from the HuggingFace dataset
        include_target: Whether to include target compliance for validation
        prompt_style: Style of prompt to generate ('full', 'natural',
            'workflow-random', 'workflow-derived-params',
            'workflow-distractor', 'workflow-conditional', 'workflow-multi-export')
        seed: Random seed for reproducible random parameter generation
            (workflow-random, workflow-distractor, workflow-conditional,
            workflow-multi-export)

    Returns:
        Dictionary with prompt, conditions, and optional target values
    """
    # Validate prompt style
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(
            f"Unknown prompt style: {prompt_style}. "
            f"Available styles: {list(PROMPT_STYLES.keys())}"
        )

    # Extract conditions from example
    volfrac = example.get("volfrac", 0.5)
    forcedist = example.get("forcedist", 0.0)
    rmin = example.get("rmin", 1.5)

    # Extract target compliance value
    compliance = example.get("c", 0.0)

    # Get natural language descriptions for metadata
    force_desc = describe_force_distribution(forcedist)
    compliance_desc = describe_expected_stiffness(compliance)

    # Create style-specific prompt
    stl_expected_params = None  # Initialize for workflow-random

    if prompt_style == "full":
        prompt = _create_full_prompt(volfrac, forcedist, rmin)
    elif prompt_style == "natural":
        prompt = _create_natural_prompt(volfrac, forcedist, compliance)
    elif prompt_style == "workflow-random":
        prompt, stl_expected_params = _create_workflow_random_prompt(
            volfrac, forcedist, rmin, example.get("example_id", 0), seed
        )
    elif prompt_style == "workflow-derived-params":
        prompt, stl_expected_params = _create_workflow_derived_params_prompt(
            volfrac, forcedist, rmin
        )
    elif prompt_style == "workflow-distractor":
        prompt, stl_expected_params = _create_workflow_distractor_prompt(
            volfrac, forcedist, rmin, example.get("example_id", 0), seed
        )
    elif prompt_style == "workflow-multi-export":
        prompt, stl_expected_params = _create_workflow_multi_export_prompt(
            volfrac, forcedist, rmin, example.get("example_id", 0), seed
        )
    elif prompt_style == "workflow-conditional":
        prompt, stl_expected_params = _create_workflow_conditional_prompt(
            volfrac, forcedist, rmin, example.get("example_id", 0), seed
        )
    else:
        prompt = _create_full_prompt(volfrac, forcedist, rmin)

    # Get style configuration
    style_config = PROMPT_STYLES[prompt_style]

    # Create structured data for benchmarking
    prompt_data = {
        "prompt": prompt,
        "prompt_style": prompt_style,
        "conditions": {
            "volfrac": float(volfrac),
            "forcedist": float(forcedist),
            "rmin": float(rmin),
            "overhang_constraint": False,  # Required by cGAN model (4th condition)
        },
        "metadata": {
            "force_description": force_desc,
            "expected_stiffness": compliance_desc,
            "prompt_style": prompt_style,
            "prompt_style_description": style_config["description"],
            "success_criteria": style_config.get("success_criteria", "render_design"),
        },
        # Optimal tool call sequence for efficiency scoring (style-specific)
        "optimal_tool_calls": style_config["optimal_tool_calls"],
        "optimal_call_count": style_config["optimal_call_count"],
    }

    if include_target:
        prompt_data["target"] = {
            "compliance": float(compliance),
            "optimal_design": example["optimal_design"],
        }

    # Add STL expected params for workflow styles that validate parameters
    if stl_expected_params is not None:
        prompt_data["stl_expected_params"] = stl_expected_params

    return prompt_data


def main() -> None:
    """Main execution function."""
    run_prompt_generation_workflow(
        problem_name="beams2d",
        prompt_creator_func=create_prompt_from_conditions,
        target_keys_to_keep=["compliance"],
        available_styles=list(PROMPT_STYLES.keys()),
    )


if __name__ == "__main__":
    main()
