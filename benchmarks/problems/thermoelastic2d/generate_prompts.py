"""
Generate prompts from thermoelastic design conditions.

This script transforms thermoelastic design parameters from the HuggingFace dataset
into natural language prompts that can be used to evaluate the engineering agent.

Dataset: https://huggingface.co/datasets/IDEALLab/thermoelastic_2d_v0
"""

import sys
from pathlib import Path
from typing import Any

import weave

# Add project root to path to import src modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.shared.prompt_generation import (  # noqa: E402
    run_prompt_generation_workflow,
)


@weave.op()
def create_prompt_from_conditions(
    example: dict[str, Any], include_target: bool = True
) -> dict[str, Any]:
    """
    Create a benchmark prompt from thermoelastic design conditions.

    Args:
        example: Single example from the HuggingFace dataset
        include_target: Whether to include target values for validation

    Returns:
        Dictionary with prompt, conditions, and optional target values
    """
    # Extract conditions from example
    volfrac = example.get("volfrac", 0.3)
    rmin = example.get("rmin", 1.1)
    weight = example.get("weight", 0.5)

    # Extract objectives
    structural_compliance = example.get("structural_compliance", 0.0)
    thermal_compliance = example.get("thermal_compliance", 0.0)
    volume_fraction = example.get("volume_fraction", 0.0)

    # Create natural language prompt
    prompt = (
        f"Design a 2D thermoelastic structure with coupled structural and thermal optimization:\n\n"
        f"Parameters:\n"
        f"- Initial volume fraction: {volfrac:.2f} (target material usage)\n"
        f"- Filter radius (rmin): {rmin:.2f}\n"
        f"- Objective weight: {weight:.2f} (balance between structural and thermal compliance)\n\n"
        f"Objectives (all to minimize):\n"
        f"1. Structural compliance: Minimize structural deformation under mechanical loads\n"
        f"2. Thermal compliance: Minimize thermal resistance for heat dissipation\n"
        f"3. Volume fraction: Minimize material usage\n\n"
        f"The design should be on a 64x64 grid with binary material distribution (0=void, 1=material).\n"
        f"The weight parameter ({weight:.2f}) controls the trade-off between structural (weight) "
        f"and thermal (1-weight) objectives."
    )

    # Build conditions dict with optional boundary conditions
    conditions: dict[str, Any] = {
        "volfrac": float(volfrac),
        "rmin": float(rmin),
        "weight": float(weight),
    }

    # Include boundary conditions if available
    if "fixed_elements" in example:
        conditions["fixed_elements"] = example["fixed_elements"]
    if "force_elements_x" in example:
        conditions["force_elements_x"] = example["force_elements_x"]
    if "force_elements_y" in example:
        conditions["force_elements_y"] = example["force_elements_y"]
    if "heatsink_elements" in example:
        conditions["heatsink_elements"] = example["heatsink_elements"]

    # Create structured data for benchmarking
    prompt_data = {
        "prompt": prompt,
        "conditions": conditions,
        "metadata": {
            "problem_type": "thermoelastic2d",
            "multi_objective": True,  # Flag for multi-objective problem
        },
    }

    if include_target:
        prompt_data["target"] = {
            "structural_compliance": float(structural_compliance),
            "thermal_compliance": float(thermal_compliance),
            "volume_fraction": float(volume_fraction),
            "optimal_design": example["optimal_design"],
        }

    return prompt_data


def main() -> None:
    """Main execution function."""
    run_prompt_generation_workflow(
        problem_name="thermoelastic2d",
        prompt_creator_func=create_prompt_from_conditions,
        target_keys_to_keep=[
            "structural_compliance",
            "thermal_compliance",
            "volume_fraction",
        ],
    )


if __name__ == "__main__":
    main()
