"""
Generate prompts from beam design conditions.

This script transforms beam design parameters from the HuggingFace dataset
into natural language prompts that can be used to evaluate the engineering agent.

Supports multiple prompt styles:
- full: Exact numerical parameters
- approximate: Rounded/approximate values
- natural: Natural language descriptions only
- workflow: Full workflow with export steps
- workflow-conditional: Workflow with if/then branching based on simulation results

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
    "approximate": {
        "description": "Rounded/approximate values",
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
    "workflow": {
        "description": "Full workflow with export steps",
        "optimal_tool_calls": [
            {"name": "optimize_design", "count": 1},
            {"name": "simulate_design", "count": 1},
            {"name": "convert_design_to_stl", "count": 1},
        ],
        "optimal_call_count": 3,
        "success_criteria": "stl_export",  # Success = STL file generated correctly
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


def _create_approximate_prompt(volfrac: float, forcedist: float, rmin: float) -> str:
    """Create prompt with rounded/approximate values."""
    # Round to 1 decimal place
    volfrac_approx = round(volfrac, 1)
    forcedist_approx = round(forcedist, 1)
    rmin_approx = round(rmin, 1)

    return (
        f"Design a 2D beam structure.\n\n"
        f"Design requirements:\n"
        f"- Use a material volume fraction of approximately {volfrac_approx}\n"
        f"- Force distribution parameter: around {forcedist_approx}\n"
        f"- Minimum filter radius (rmin): {rmin_approx}\n\n"
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


def _create_workflow_prompt(volfrac: float, forcedist: float, rmin: float) -> str:
    """Create detailed workflow prompt with export steps."""
    return (
        f"Execute a 2D topology optimization and export the resulting geometry "
        f"as a 3D-printable STL file.\n\n"
        f"1. Optimization Configuration\n"
        f"   - Volume Fraction: {volfrac}\n"
        f"   - Force Distribution: {forcedist}\n"
        f"   - Filter Radius (rmin): {rmin}\n"
        f"   - Objective: Minimize compliance\n\n"
        f"2. Post-processing & Export\n"
        f"   - Thresholding: Apply a 0.5 density threshold to convert the continuous "
        f"density map into a binary 'solid vs. void' geometry\n"
        f"   - Mirror: Mirror the design across the y-axis for symmetry\n"
        f"   - Extrusion: Extrude the 2D result by 5 units in the Z-axis to create "
        f"a 3D volume\n"
        f"   - Export: Save the final geometry as an STL file"
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
        "scale_xy": float(rng.uniform(0.5, 5.0)),
        "scale_z": float(rng.uniform(5.0, 20.0)),
        "threshold": float(rng.uniform(0.3, 0.7)),
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
    compliance_threshold = float(rng.uniform(100.0, 300.0))

    # Branch-specific parameters: threshold and mirror_y
    threshold_high = float(rng.uniform(0.3, 0.7))
    threshold_low = float(rng.uniform(0.3, 0.7))

    # Ensure the two thresholds are distinguishable
    min_threshold_gap = 0.1
    while abs(threshold_high - threshold_low) < min_threshold_gap:
        threshold_low = float(rng.uniform(0.3, 0.7))

    # Mirror: one branch mirrors, the other does not (guaranteed distinct)
    mirror_high = bool(rng.choice([True, False]))
    mirror_low = not mirror_high

    # Common parameters (same for both branches)
    scale_xy = float(rng.uniform(0.5, 5.0))
    scale_z = float(rng.uniform(5.0, 20.0))

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
        f"     - Thresholding: Apply a {bh['threshold']:.2f} density threshold\n"
        f"     - Mirror: {mirror_high_instr}\n"
        f"   - If compliance <= {ct:.1f}:\n"
        f"     - Thresholding: Apply a {bl['threshold']:.2f} density threshold\n"
        f"     - Mirror: {mirror_low_instr}\n"
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
        prompt_style: Style of prompt to generate ('full', 'approximate', 'natural',
            'workflow', 'workflow-random', 'workflow-conditional')
        seed: Random seed for reproducible random parameter generation
            (workflow-random and workflow-conditional)

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
    elif prompt_style == "approximate":
        prompt = _create_approximate_prompt(volfrac, forcedist, rmin)
    elif prompt_style == "natural":
        prompt = _create_natural_prompt(volfrac, forcedist, compliance)
    elif prompt_style == "workflow-random":
        prompt, stl_expected_params = _create_workflow_random_prompt(
            volfrac, forcedist, rmin, example.get("example_id", 0), seed
        )
    elif prompt_style == "workflow-conditional":
        prompt, stl_expected_params = _create_workflow_conditional_prompt(
            volfrac, forcedist, rmin, example.get("example_id", 0), seed
        )
    elif prompt_style == "workflow":
        prompt = _create_workflow_prompt(volfrac, forcedist, rmin)
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

    # Add STL expected params for workflow-random
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
