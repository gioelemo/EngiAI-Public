"""
Generate prompts for Weave benchmarking from thermoelastic design conditions.

This script transforms thermoelastic design parameters from the HuggingFace dataset
into natural language prompts that can be used to evaluate the engineering agent.

Dataset: https://huggingface.co/datasets/IDEALLab/thermoelastic_2d_v0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import weave
from datasets import load_dataset

# Add project root to path to import src modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.shared.problem_registry import get_problem_config  # noqa: E402
from src.utils.weave_integration import init_weave, is_weave_enabled  # noqa: E402


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


@weave.op()
def generate_prompt_dataset(
    num_samples: int = 50,
    dataset_split: str = "test",
    include_targets: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate a dataset of prompts from the HuggingFace thermoelastic dataset.

    Args:
        num_samples: Number of samples to generate
        dataset_split: Dataset split to use ('train', 'val', or 'test')
        include_targets: Whether to include target values

    Returns:
        List of prompt dictionaries ready for Weave evaluation
    """
    # Get dataset name from problem registry
    problem_config = get_problem_config("thermoelastic2d")
    dataset_name = problem_config.dataset_name

    print(f"🔄 Loading {dataset_name} dataset (split: {dataset_split})...")
    dataset = load_dataset(dataset_name, split=dataset_split)
    print(f"✅ Loaded {len(dataset)} examples")

    # Limit to num_samples if specified
    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
        print(f"📊 Using {num_samples} samples")

    prompts = []
    for i, example in enumerate(dataset):
        prompt_data = create_prompt_from_conditions(
            example, include_target=include_targets
        )
        prompt_data["example_id"] = i
        prompt_data["dataset_split"] = dataset_split
        prompts.append(prompt_data)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(dataset)} prompts...")

    print(f"✅ Generated {len(prompts)} prompts successfully!")
    return prompts


def save_prompts_locally(prompts: list[dict[str, Any]], output_file: Path) -> None:
    """Save generated prompts to a local JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert any numpy arrays to lists for JSON serialization
    serializable_prompts = []
    for prompt in prompts:
        prompt_copy = prompt.copy()
        if "target" in prompt_copy and "optimal_design" in prompt_copy["target"]:
            # Don't save the full optimal_design array locally (too large)
            # Just save the objective values
            prompt_copy["target"] = {
                "structural_compliance": prompt_copy["target"]["structural_compliance"],
                "thermal_compliance": prompt_copy["target"]["thermal_compliance"],
                "volume_fraction": prompt_copy["target"]["volume_fraction"],
            }
        serializable_prompts.append(prompt_copy)

    with output_file.open("w") as f:
        json.dump(serializable_prompts, f, indent=2)

    print(f"💾 Saved prompts to: {output_file}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate thermoelastic design prompts from HuggingFace dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to generate (default: 50)",
    )
    parser.add_argument(
        "--no-targets",
        action="store_true",
        help="Exclude target values from prompts",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_arguments()

    print("=" * 60)
    print("THERMOELASTIC2D PROMPT GENERATION FOR WEAVE BENCHMARKING")
    print("=" * 60)
    print()
    print(f"Dataset split: {args.split}")
    print(f"Number of samples: {args.samples}")
    print(f"Include targets: {not args.no_targets}")
    print()

    # Initialize Weave
    print("🔧 Initializing Weave...")
    if init_weave():
        print("✅ Weave initialized successfully!")
    else:
        print("⚠️  Weave not available, continuing without tracing...")
    print()

    # Generate prompts using command-line arguments
    prompts = generate_prompt_dataset(
        num_samples=args.samples,
        dataset_split=args.split,
        include_targets=not args.no_targets,
    )

    # Save locally with split in filename
    output_dir = Path(__file__).parent / "data" / "generated"
    output_file = (
        output_dir / f"thermoelastic2d_prompts_{args.samples}_samples_{args.split}.json"
    )
    save_prompts_locally(prompts, output_file)

    print()
    print("✨ Prompt generation complete!")
    print()

    # Publish to Weave if enabled
    if is_weave_enabled():
        print("📤 Publishing dataset to Weave...")
        dataset_name = f"thermoelastic2d_prompts_{args.samples}_samples_{args.split}"
        weave.publish(prompts, name=dataset_name)
        print(f"✅ Published dataset: {dataset_name}")


if __name__ == "__main__":
    main()
