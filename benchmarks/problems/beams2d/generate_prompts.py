"""
Generate prompts from beam design conditions.

This script transforms beam design parameters from the HuggingFace dataset
into natural language prompts that can be used to evaluate the engineering agent.

Dataset: https://huggingface.co/datasets/IDEALLab/beams_2d_50_100_v0
"""

import sys
from pathlib import Path
from typing import Any

# Add project root to path to import src modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.shared.prompt_generation import (  # noqa: E402
    run_prompt_generation_workflow,
)  # Force distribution thresholds for natural language descriptions

FORCE_CONCENTRATED = 0.2
FORCE_LOWER_LEFT = 0.4
FORCE_MIDDLE = 0.6
FORCE_UPPER_RIGHT = 0.8

# Compliance thresholds for stiffness descriptions
COMPLIANCE_VERY_STIFF = 30
COMPLIANCE_STIFF = 60
COMPLIANCE_MODERATE = 100
COMPLIANCE_FLEXIBLE = 200


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


def create_prompt_from_conditions(
    example: dict[str, Any], include_target: bool = True
) -> dict[str, Any]:
    """
    Create a benchmark prompt from beam design conditions.

    Args:
        example: Single example from the HuggingFace dataset
        include_target: Whether to include target compliance for validation

    Returns:
        Dictionary with prompt, conditions, and optional target values
    """
    # Extract conditions from example
    volfrac = example.get("volfrac", 0.5)
    forcedist = example.get("forcedist", 0.0)
    rmin = example.get("rmin", 1.5)

    # Extract target compliance value
    compliance = example.get("c", 0.0)

    # Get natural language descriptions
    force_desc = describe_force_distribution(forcedist)
    compliance_desc = describe_expected_stiffness(compliance)

    # Create natural language prompt
    prompt = (
        f"Design a 2D beam structure that can support {force_desc}.\n\n"
        f"Design requirements:\n"
        f"- The structure should be approximately {compliance_desc}\n"
        f"- Use a material volume fraction of {volfrac:.1%}\n"
        f"- The design should be on a 50x100 grid\n"
        f"- Minimum filter radius (rmin): {rmin:.1f}\n\n"
        f"The force is applied on the left side, and the beam is fixed on the right side.\n"
        f"Optimize the structure to minimize compliance while respecting the volume constraint."
    )

    # Create structured data for benchmarking
    prompt_data = {
        "prompt": prompt,
        "conditions": {
            "volfrac": float(volfrac),
            "forcedist": float(forcedist),
            "rmin": float(rmin),
        },
        "metadata": {
            "force_description": force_desc,
            "expected_stiffness": compliance_desc,
        },
    }

    if include_target:
        prompt_data["target"] = {
            "compliance": float(compliance),
            "optimal_design": example["optimal_design"],
        }

    return prompt_data


def main() -> None:
    """Main execution function."""
    run_prompt_generation_workflow(
        problem_name="beams2d",
        prompt_creator_func=create_prompt_from_conditions,
        target_keys_to_keep=["compliance"],
    )


if __name__ == "__main__":
    main()
