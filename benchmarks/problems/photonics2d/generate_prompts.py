"""
Generate prompts from photonics design conditions.

This script transforms photonic design parameters from the HuggingFace dataset
into natural language prompts that can be used to evaluate the engineering agent.

Dataset: https://huggingface.co/datasets/IDEALLab/photonics_2d_120_120_v0
"""

import sys
from pathlib import Path
from typing import Any

# Add project root to path to import src modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.shared.prompt_generation import (  # noqa: E402
    run_prompt_generation_workflow,
)


def create_prompt_from_conditions(
    example: dict[str, Any], include_target: bool = True
) -> dict[str, Any]:
    """
    Create a benchmark prompt from photonics design conditions.

    Args:
        example: Single example from the HuggingFace dataset
        include_target: Whether to include target overlap for validation

    Returns:
        Dictionary with prompt, conditions, and optional target values
    """
    # Extract conditions directly from example (not nested)
    lambda1 = example.get("lambda1", 0.0)
    lambda2 = example.get("lambda2", 0.0)
    blur_radius = example.get("blur_radius", 0.0)
    total_overlap = example.get("total_overlap", 0.0)

    # Create natural language prompt
    prompt = (
        f"Design a 2D photonic structure to maximize field overlap with the following parameters:\n"
        f"- lambda1: {lambda1:.6f}\n"
        f"- lambda2: {lambda2:.6f}\n"
        f"- blur_radius: {blur_radius:.6f}\n"
        f"- Target: Maximize total_overlap (field overlap integral)\n\n"
        f"The photonic structure should be designed on a 120x120 grid.\n"
        f"Use binary material distribution (0=air, 1=dielectric material)."
    )

    # Create structured data for benchmarking
    prompt_data = {
        "prompt": prompt,
        "conditions": {
            "lambda1": float(lambda1),
            "lambda2": float(lambda2),
            "blur_radius": float(blur_radius),
        },
        "metadata": {
            "problem_type": "photonics2d",
        },
    }

    if include_target:
        prompt_data["target"] = {
            "total_overlap": float(total_overlap),
            "optimal_design": example["optimal_design"],
        }

    return prompt_data


def main() -> None:
    """Main execution function."""
    run_prompt_generation_workflow(
        problem_name="photonics2d",
        prompt_creator_func=create_prompt_from_conditions,
        target_keys_to_keep=["total_overlap"],
    )


if __name__ == "__main__":
    main()
