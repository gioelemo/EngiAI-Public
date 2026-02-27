"""
Generate prompts from photonics design conditions.

This script transforms photonic design parameters from the HuggingFace dataset
into natural language prompts that can be used to evaluate the engineering agent.

Supports multiple prompt styles:
- full: Exact numerical parameters
- workflow-random: Full workflow with randomized STL export parameters
- workflow-conditional: Workflow with if/then branching based on simulation results

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

# Gap requirements for parameter distinctness
MIN_THRESHOLD_GAP = 0.1  # Minimum gap between threshold values

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
}


def _create_full_prompt(
    lambda1: float, lambda2: float, blur_radius: float
) -> str:
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
    overlap_threshold = float(
        rng.uniform(OVERLAP_THRESHOLD_MIN, OVERLAP_THRESHOLD_MAX)
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

    if prompt_style == "full":
        prompt = _create_full_prompt(lambda1, lambda2, blur_radius)
    elif prompt_style == "workflow-random":
        prompt, stl_expected_params = _create_workflow_random_prompt(
            lambda1, lambda2, blur_radius, example.get("example_id", 0), seed
        )
    elif prompt_style == "workflow-conditional":
        prompt, stl_expected_params = _create_workflow_conditional_prompt(
            lambda1, lambda2, blur_radius, example.get("example_id", 0), seed
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
