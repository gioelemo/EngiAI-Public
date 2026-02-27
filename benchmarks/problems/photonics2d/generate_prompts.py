"""
Generate prompts from photonics design conditions.

This script transforms photonic design parameters from the HuggingFace dataset
into natural language prompts that can be used to evaluate the engineering agent.

Supports multiple prompt styles:
- full: Exact numerical parameters
- natural: Natural language descriptions only
- workflow-random: Full workflow with randomized STL export parameters
- workflow-derived-params: Workflow with STL parameters derived from optimization inputs
- workflow-distractor: Workflow with distractor parameters mixed with real STL params
- workflow-conditional: Workflow with if/then branching based on simulation results
- workflow-multi-export: Workflow requiring two STL exports with different parameters

Dataset: https://huggingface.co/datasets/IDEALLab/photonics_2d_120_120_v0
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

# STL parameter ranges for random generation (problem-independent)
STL_THRESHOLD_MIN = 0.3  # Minimum density threshold for solid/void conversion
STL_THRESHOLD_MAX = 0.7  # Maximum density threshold
STL_SCALE_XY_MIN = 0.5  # Minimum XY dimension scaling factor
STL_SCALE_XY_MAX = 5.0  # Maximum XY dimension scaling factor
STL_SCALE_Z_MIN = 5.0  # Minimum Z extrusion height
STL_SCALE_Z_MAX = 20.0  # Maximum Z extrusion height

# Overlap threshold range for workflow-conditional branching
# Dataset total_overlap: min=-0.03, P25=0.14, median=0.29, P75=0.56, max=2.10
# Range chosen to give roughly 50/50 branch split across dataset samples.
OVERLAP_THRESHOLD_MIN = 0.1
OVERLAP_THRESHOLD_MAX = 0.8

# Lambda thresholds for natural language descriptions
LAMBDA_SHORT = 0.7   # Short wavelength upper bound
LAMBDA_MEDIUM = 1.0  # Medium wavelength upper bound

# Blur radius thresholds for natural language descriptions
BLUR_NONE = 0       # No blur
BLUR_LIGHT = 1      # Light smoothing
BLUR_MODERATE = 2   # Moderate smoothing

# Derivation rule constants for workflow-derived-params
# threshold = blur_radius * 0.1 + 0.3  (maps 0→0.3, 4→0.7)
DERIVED_THRESHOLD_SCALE = 0.1
DERIVED_THRESHOLD_OFFSET = 0.3
# scale_xy = lambda1 + lambda2  (maps ~1.3 to ~2.7)
# scale_z = (lambda1 + lambda2) * 5.0  (maps ~6.5 to ~13.5)
DERIVED_SCALE_Z_MULTIPLIER = 5.0
# mirror_y = True if lambda1 > 1.0  (roughly 50/50 split)
DERIVED_MIRROR_LAMBDA1_THRESHOLD = 1.0

# Seed offset for distractor parameter generation (workflow-distractor style)
DISTRACTOR_SEED_OFFSET = 10000

# Fallback thresholds for distractor parameter generation
DISTRACTOR_THRESHOLD_MIDPOINT = (STL_THRESHOLD_MIN + STL_THRESHOLD_MAX) / 2
DISTRACTOR_SCALE_XY_MIDPOINT = (STL_SCALE_XY_MIN + STL_SCALE_XY_MAX) / 2

# Gap requirements for parameter distinctness
MIN_THRESHOLD_GAP = 0.1  # Minimum gap between threshold values
MIN_DISTRACTOR_SCALE_GAP = (
    0.5  # Larger gap for distractor vs real (clearer distinction)
)
MIN_MULTI_EXPORT_SCALE_GAP = 0.2  # Smaller gap for two valid exports (both are real)

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
            {"name": "simulate_design", "count": 1},
            {"name": "convert_design_to_stl", "count": 1},
        ],
        "optimal_call_count": 3,
        "success_criteria": "stl_export_with_params",
        "validate_stl_params": True,
    },
    "workflow-distractor": {
        "description": "Workflow with distractor parameters mixed with real STL params",
        "optimal_tool_calls": [
            {"name": "optimize_design", "count": 1},
            {"name": "simulate_design", "count": 1},
            {"name": "convert_design_to_stl", "count": 1},
        ],
        "optimal_call_count": 3,
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


def _create_full_prompt(lambda1: float, lambda2: float, blur_radius: float) -> str:
    """Create prompt with exact numerical parameters."""
    return (
        f"Design a 2D photonic structure to maximize field overlap with the "
        f"following parameters:\n"
        f"- lambda1: {lambda1:.6f}\n"
        f"- lambda2: {lambda2:.6f}\n"
        f"- blur_radius: {blur_radius:.6f}\n"
        f"- Target: Maximize total_overlap (field overlap integral)\n\n"
        f"Use binary material distribution (0=air, 1=dielectric material)."
    )


def _describe_wavelength(value: float) -> str:
    """Convert a wavelength parameter to a natural language description."""
    if value < LAMBDA_SHORT:
        return "a short wavelength"
    if value < LAMBDA_MEDIUM:
        return "a medium wavelength"
    return "a long wavelength"


def _describe_blur_radius(value: float) -> str:
    """Convert a blur radius to a natural language description."""
    r = round(value)
    if r <= BLUR_NONE:
        return "no spatial smoothing"
    if r <= BLUR_LIGHT:
        return "light spatial smoothing"
    if r <= BLUR_MODERATE:
        return "moderate spatial smoothing"
    return "heavy spatial smoothing"


def _create_natural_prompt(
    lambda1: float, lambda2: float, blur_radius: float
) -> str:
    """Create prompt with natural language descriptions only."""
    l1_desc = _describe_wavelength(lambda1)
    l2_desc = _describe_wavelength(lambda2)
    blur_desc = _describe_blur_radius(blur_radius)

    return (
        f"Design a 2D photonic structure.\n\n"
        f"Design requirements:\n"
        f"- Use {l1_desc} for the first mode\n"
        f"- Use {l2_desc} for the second mode\n"
        f"- Apply {blur_desc} to the design\n"
        f"Optimize the structure and simulate the result to obtain the "
        f"total overlap value."
    )


def _compute_derived_stl_params(
    lambda1: float, lambda2: float, blur_radius: float
) -> dict[str, Any]:
    """Compute STL parameters from optimization inputs using derivation rules.

    Rules:
    - threshold = blur_radius * 0.1 + 0.3  (maps 0→0.3, 4→0.7)
    - scale_xy = lambda1 + lambda2  (maps ~1.3 to ~2.7)
    - scale_z = (lambda1 + lambda2) * 5.0  (maps ~6.5 to ~13.5)
    - mirror_y = True if lambda1 > 1.0  (roughly 50/50 split)
    """
    threshold = DERIVED_THRESHOLD_SCALE * blur_radius + DERIVED_THRESHOLD_OFFSET
    return {
        "mirror_y": lambda1 > DERIVED_MIRROR_LAMBDA1_THRESHOLD,
        "scale_xy": float(lambda1 + lambda2),
        "scale_z": float(DERIVED_SCALE_Z_MULTIPLIER * (lambda1 + lambda2)),
        "threshold": float(threshold),
    }


def _create_workflow_derived_params_prompt(
    lambda1: float, lambda2: float, blur_radius: float
) -> tuple[str, dict[str, Any]]:
    """Create workflow prompt where STL params are derived from optimization inputs.

    The prompt gives derivation RULES (not final values). The agent must compute
    the correct parameters from the optimization inputs.
    """
    stl_params = _compute_derived_stl_params(lambda1, lambda2, blur_radius)

    prompt = (
        f"Execute a 2D photonic structure optimization, simulate the result, and "
        f"export the geometry as a 3D-printable STL file.\n\n"
        f"1. Optimization Configuration\n"
        f"   - lambda1: {lambda1:.6f}\n"
        f"   - lambda2: {lambda2:.6f}\n"
        f"   - blur_radius: {blur_radius:.6f}\n"
        f"   - Objective: Maximize total field overlap\n\n"
        f"2. Simulation\n"
        f"   - After optimization, simulate the design to obtain the total_overlap value\n\n"
        f"3. Post-processing & Export\n"
        f"   The STL export parameters must be derived from the optimization inputs:\n"
        f"   - Thresholding: Compute the density threshold as the blur radius "
        f"multiplied by 0.1 plus 0.3\n"
        f"   - Mirror: Mirror the design across the y-axis only if lambda1 "
        f"is greater than 1.0\n"
        f"   - XY Scaling: Scale the X and Y dimensions by the sum of lambda1 and "
        f"lambda2\n"
        f"   - Extrusion: Extrude the 2D result in the Z-axis by the sum of "
        f"lambda1 and lambda2 multiplied by 5\n"
        f"   - Export: Save the final geometry as an STL file with these derived "
        f"parameters"
    )

    return prompt, stl_params


def _generate_distractor_params(
    rng: np.random.Generator,
    real_params: dict[str, Any],
) -> dict[str, float]:
    """Generate competing distractor values for real STL parameters.

    Produces alternative values for ``threshold`` and ``scale_xy`` that look
    plausible but belong to a non-export context (preview / analysis).
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
    if dt is None:
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
    if ds is None:
        ds = (
            STL_SCALE_XY_MIN
            if real_scale_xy > DISTRACTOR_SCALE_XY_MIDPOINT
            else STL_SCALE_XY_MAX
        )

    return {
        "distractor_threshold": dt,
        "distractor_scale_xy": ds,
    }


def _create_workflow_distractor_prompt(
    lambda1: float,
    lambda2: float,
    blur_radius: float,
    example_id: int,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create workflow prompt with competing parameter values.

    The prompt presents two plausible values for ``threshold`` and
    ``scale_xy`` — one in a preview/analysis context and one in the
    manufacturing/export context.  The agent must pick the export-context values.
    """
    unique_seed = (seed if seed is not None else 0) + example_id
    stl_params = _generate_random_stl_params(unique_seed)

    distractor_rng = np.random.default_rng(unique_seed + DISTRACTOR_SEED_OFFSET)
    distractors = _generate_distractor_params(distractor_rng, stl_params)

    mirror_instruction = (
        "Mirror the design across the y-axis"
        if stl_params["mirror_y"]
        else "Do NOT mirror the design"
    )

    prompt = (
        f"Execute a 2D photonic structure optimization, simulate the result, and "
        f"export the geometry as a 3D-printable STL file.\n\n"
        f"1. Optimization Configuration\n"
        f"   - lambda1: {lambda1:.6f}\n"
        f"   - lambda2: {lambda2:.6f}\n"
        f"   - blur_radius: {blur_radius:.6f}\n"
        f"   - Objective: Maximize total field overlap\n\n"
        f"2. Simulation\n"
        f"   - After optimization, simulate the design to obtain the total_overlap "
        f"value\n\n"
        f"3. Post-processing & Export\n"
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
        f"   - Export: Save the final geometry as an STL file with these exact "
        f"parameters"
    )

    return prompt, stl_params


def _generate_random_multi_export_params(seed: int | None = None) -> dict[str, Any]:
    """Generate random parameters for workflow-multi-export prompts.

    Generates TWO distinct param sets for two STL exports from the same optimization.
    Distinctness guarantees: mirror_y opposite, threshold gap >= MIN_THRESHOLD_GAP,
    scale_xy gap >= MIN_MULTI_EXPORT_SCALE_GAP, scale_z gap >= MIN_MULTI_EXPORT_SCALE_GAP.
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


def _create_workflow_multi_export_prompt(
    lambda1: float,
    lambda2: float,
    blur_radius: float,
    example_id: int,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create workflow prompt requiring two STL exports with different parameters."""
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
        f"Execute a 2D photonic structure optimization, simulate the result, and "
        f"export the geometry as TWO separate 3D-printable STL files with different "
        f"parameters.\n\n"
        f"1. Optimization Configuration\n"
        f"   - lambda1: {lambda1:.6f}\n"
        f"   - lambda2: {lambda2:.6f}\n"
        f"   - blur_radius: {blur_radius:.6f}\n"
        f"   - Objective: Maximize total field overlap\n\n"
        f"2. Simulation\n"
        f"   - After optimization, simulate the design to obtain the total_overlap "
        f"value\n\n"
        f"3. Post-processing & Export\n\n"
        f"   Export A:\n"
        f"   - Thresholding: Apply a {export_a['threshold']:.2f} density threshold "
        f"to convert the continuous density map into binary geometry\n"
        f"   - Mirror: {mirror_a_instr} for the final geometry\n"
        f"   - XY Scaling: Scale the X and Y dimensions by "
        f"{export_a['scale_xy']:.2f}\n"
        f"   - Extrusion: Extrude the 2D result by {export_a['scale_z']:.1f} units "
        f"in the Z-axis to create a 3D volume\n"
        f"   - Export: Save the final geometry as an STL file with these exact "
        f"parameters\n\n"
        f"   Export B:\n"
        f"   - Thresholding: Apply a {export_b['threshold']:.2f} density threshold "
        f"to convert the continuous density map into binary geometry\n"
        f"   - Mirror: {mirror_b_instr} for the final geometry\n"
        f"   - XY Scaling: Scale the X and Y dimensions by "
        f"{export_b['scale_xy']:.2f}\n"
        f"   - Extrusion: Extrude the 2D result by {export_b['scale_z']:.1f} units "
        f"in the Z-axis to create a 3D volume\n"
        f"   - Export: Save the final geometry as an STL file with these exact "
        f"parameters"
    )

    return prompt, params


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


def _generate_random_conditional_params(seed: int | None = None) -> dict[str, Any]:
    """Generate random parameters for workflow-conditional prompts.

    Generates an overlap threshold for branching, two distinct sets of
    branch-specific params (threshold, mirror_y), and common params (scale_xy, scale_z).

    Args:
        seed: Random seed for reproducibility

    Returns:
        Dict with: conditional flag, objective_field, objective_threshold,
        branch_high, branch_low, common
    """
    rng = np.random.default_rng(seed)

    # Overlap threshold for branching (centered on dataset median ~0.29)
    overlap_threshold = float(rng.uniform(OVERLAP_THRESHOLD_MIN, OVERLAP_THRESHOLD_MAX))

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
        "objective_field": "total_overlap",
        "objective_threshold": round(overlap_threshold, 2),
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


def _create_workflow_random_prompt(
    lambda1: float,
    lambda2: float,
    blur_radius: float,
    example_id: int,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create workflow prompt with random STL parameters.

    Args:
        lambda1: First wavelength parameter
        lambda2: Second wavelength parameter
        blur_radius: Blur radius for the optimization
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
        f"Execute a 2D photonic structure optimization, simulate the result, and "
        f"export the geometry as a 3D-printable STL file.\n\n"
        f"1. Optimization Configuration\n"
        f"   - lambda1: {lambda1:.6f}\n"
        f"   - lambda2: {lambda2:.6f}\n"
        f"   - blur_radius: {blur_radius:.6f}\n"
        f"   - Objective: Maximize total field overlap\n\n"
        f"2. Simulation\n"
        f"   - After optimization, simulate the design to obtain the total_overlap value\n\n"
        f"3. Post-processing & Export\n"
        f"   - Thresholding: Apply a {stl_params['threshold']:.2f} density threshold "
        f"to convert the continuous density map into binary geometry\n"
        f"   - Mirror: {mirror_instruction} for the final geometry\n"
        f"   - XY Scaling: Scale the X and Y dimensions by {stl_params['scale_xy']:.2f}\n"
        f"   - Extrusion: Extrude the 2D result by {stl_params['scale_z']:.1f} units "
        f"in the Z-axis to create a 3D volume\n"
        f"   - Export: Save the final geometry as an STL file with these exact parameters"
    )

    return prompt, stl_params


def _create_workflow_conditional_prompt(
    lambda1: float,
    lambda2: float,
    blur_radius: float,
    example_id: int,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create workflow prompt with conditional branching based on total_overlap.

    The prompt instructs the agent to:
    1. Optimize and simulate to get total_overlap
    2. Branch based on total_overlap vs. a randomized threshold
    3. Apply branch-specific STL params (threshold, mirror_y)
    4. Apply common STL params (scale_xy, scale_z)

    Args:
        lambda1: First wavelength parameter
        lambda2: Second wavelength parameter
        blur_radius: Blur radius for the optimization
        example_id: Unique example identifier
        seed: Base seed for random generation

    Returns:
        Tuple of (prompt_text, conditional_stl_expected_params_dict)
    """
    unique_seed = (seed if seed is not None else 0) + example_id
    params = _generate_random_conditional_params(unique_seed)

    ot = params["objective_threshold"]
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
        f"Execute a 2D photonic structure optimization, simulate the result, then "
        f"export the geometry as a 3D-printable STL file with parameters that depend "
        f"on the simulation outcome.\n\n"
        f"1. Optimization Configuration\n"
        f"   - lambda1: {lambda1:.6f}\n"
        f"   - lambda2: {lambda2:.6f}\n"
        f"   - blur_radius: {blur_radius:.6f}\n"
        f"   - Objective: Maximize total field overlap\n\n"
        f"2. Simulation\n"
        f"   - After optimization, simulate the design to obtain the total_overlap value\n\n"
        f"3. Post-processing & Export (conditional on total_overlap)\n"
        f"   - If total_overlap > {ot}:\n"
        f"     - Thresholding: Apply a {bh['threshold']:.2f} density threshold "
        f"to convert the continuous density map into binary geometry\n"
        f"     - Mirror: {mirror_high_instr} for the final geometry\n"
        f"   - If total_overlap <= {ot}:\n"
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


def create_prompt_from_conditions(
    example: dict[str, Any],
    include_target: bool = True,
    prompt_style: str = "full",
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Create a benchmark prompt from photonics design conditions.

    Args:
        example: Single example from the HuggingFace dataset
        include_target: Whether to include target overlap for validation
        prompt_style: Style of prompt to generate ('full', 'workflow-random',
            'workflow-conditional')
        seed: Random seed for reproducible random parameter generation

    Returns:
        Dictionary with prompt, conditions, and optional target values
    """
    # Validate prompt style
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(
            f"Unknown prompt style: {prompt_style}. "
            f"Available styles: {list(PROMPT_STYLES.keys())}"
        )

    # Extract conditions directly from example (not nested)
    lambda1 = example.get("lambda1", 0.0)
    lambda2 = example.get("lambda2", 0.0)
    blur_radius = example.get("blur_radius", 0.0)
    total_overlap = example.get("total_overlap", 0.0)

    # Create style-specific prompt
    stl_expected_params = None
    example_id = example.get("example_id", 0)

    if prompt_style == "full":
        prompt = _create_full_prompt(lambda1, lambda2, blur_radius)
    elif prompt_style == "natural":
        prompt = _create_natural_prompt(lambda1, lambda2, blur_radius)
    elif prompt_style == "workflow-random":
        prompt, stl_expected_params = _create_workflow_random_prompt(
            lambda1, lambda2, blur_radius, example_id, seed
        )
    elif prompt_style == "workflow-derived-params":
        prompt, stl_expected_params = _create_workflow_derived_params_prompt(
            lambda1, lambda2, blur_radius
        )
    elif prompt_style == "workflow-distractor":
        prompt, stl_expected_params = _create_workflow_distractor_prompt(
            lambda1, lambda2, blur_radius, example_id, seed
        )
    elif prompt_style == "workflow-conditional":
        prompt, stl_expected_params = _create_workflow_conditional_prompt(
            lambda1, lambda2, blur_radius, example_id, seed
        )
    elif prompt_style == "workflow-multi-export":
        prompt, stl_expected_params = _create_workflow_multi_export_prompt(
            lambda1, lambda2, blur_radius, example_id, seed
        )
    else:
        prompt = _create_full_prompt(lambda1, lambda2, blur_radius)

    # Get style configuration
    style_config = PROMPT_STYLES[prompt_style]

    # Create structured data for benchmarking
    prompt_data = {
        "prompt": prompt,
        "prompt_style": prompt_style,
        "conditions": {
            "lambda1": float(lambda1),
            "lambda2": float(lambda2),
            "blur_radius": float(blur_radius),
        },
        "metadata": {
            "problem_type": "photonics2d",
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
            "total_overlap": float(total_overlap),
            "optimal_design": example["optimal_design"],
        }

    # Add STL expected params for workflow styles that validate parameters
    if stl_expected_params is not None:
        prompt_data["stl_expected_params"] = stl_expected_params

    return prompt_data


def main() -> None:
    """Main execution function."""
    run_prompt_generation_workflow(
        problem_name="photonics2d",
        prompt_creator_func=create_prompt_from_conditions,
        target_keys_to_keep=["total_overlap"],
        available_styles=list(PROMPT_STYLES.keys()),
    )


if __name__ == "__main__":
    main()
