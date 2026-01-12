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
    dataset_split: str = "train",
    sample_size: int | None = None,
    include_targets: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate prompts from the thermoelastic HuggingFace dataset.

    Args:
        dataset_split: Dataset split to use (train/val/test)
        sample_size: Number of samples to generate (None = all)
        include_targets: Whether to include target values

    Returns:
        List of prompt dictionaries ready for Weave evaluation
    """
    print(f"🔄 Loading thermoelastic_2d_v0 dataset (split: {dataset_split})...")
    dataset = load_dataset("IDEALLab/thermoelastic_2d_v0", split=dataset_split)
    print(f"✅ Loaded {len(dataset)} examples")

    # Limit to sample_size if specified
    if sample_size and sample_size < len(dataset):
        dataset = dataset.select(range(sample_size))
        print(f"📊 Using {sample_size} samples")

    prompts = []
    for i, example in enumerate(dataset):
        prompt_data = create_prompt_from_conditions(example, include_targets)
        prompt_data["example_id"] = i
        prompt_data["dataset_split"] = dataset_split
        prompts.append(prompt_data)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(dataset)} prompts...")

    return prompts


def save_prompts(
    prompts: list[dict[str, Any]], output_dir: Path, split: str, sample_size: int
) -> None:
    """Save prompts to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir / f"thermoelastic2d_prompts_{sample_size}_samples_{split}.json"
    )

    with output_file.open("w") as f:
        json.dump(prompts, f, indent=2)

    print(f"\n💾 Saved {len(prompts)} prompts to: {output_file}")


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
    print("THERMOELASTIC2D PROMPT GENERATION")
    print("=" * 60)
    print()
    print(f"Dataset split: {args.split}")
    print(f"Sample size: {args.samples}")
    print(f"Include targets: {not args.no_targets}")
    print()

    # Initialize Weave if available
    print("🔧 Initializing Weave...")
    if init_weave():
        print("✅ Weave initialized successfully!")
    else:
        print("⚠️  Weave not available, continuing without tracing...")
    print()

    # Generate prompts
    prompts = generate_prompt_dataset(
        dataset_split=args.split,
        sample_size=args.samples,
        include_targets=not args.no_targets,
    )

    # Save prompts
    output_dir = Path(__file__).parent / "data" / "generated"
    save_prompts(prompts, output_dir, args.split, args.samples)

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
