"""
Generate prompts for Weave benchmarking from photonics design conditions.

This script transforms photonic design parameters from the HuggingFace dataset
into natural language prompts that can be used to evaluate the engineering agent.

Dataset: https://huggingface.co/datasets/IDEALLab/photonics_2d_120_120_v0
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


@weave.op()
def generate_prompt_dataset(
    num_samples: int = 50,
    dataset_split: str = "test",
    include_targets: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate a dataset of prompts from the HuggingFace photonics dataset.

    Args:
        num_samples: Number of examples to generate prompts for
        dataset_split: Dataset split to use ('train', 'val', or 'test')
        include_targets: Whether to include target values for validation

    Returns:
        List of prompt dictionaries
    """
    print(f"🔄 Loading {num_samples} examples from {dataset_split} split...")
    dataset = load_dataset("IDEALLab/photonics_2d_120_120_v0")
    data = dataset[dataset_split]

    # Limit to requested number of samples
    num_samples = min(num_samples, len(data))
    print(f"📝 Generating prompts for {num_samples} photonics designs...")

    prompts = []
    for i in range(num_samples):
        example = data[i]
        prompt_data = create_prompt_from_conditions(
            example, include_target=include_targets
        )
        prompt_data["example_id"] = i
        prompt_data["dataset_split"] = dataset_split  # Store split information
        prompts.append(prompt_data)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_samples} prompts...")

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
            # Just save the total_overlap value
            prompt_copy["target"] = {
                "total_overlap": prompt_copy["target"]["total_overlap"]
            }
        serializable_prompts.append(prompt_copy)

    with output_file.open("w") as f:
        json.dump(serializable_prompts, f, indent=2)

    print(f"💾 Saved prompts to: {output_file}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate photonics design prompts from HuggingFace dataset"
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
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    print("=" * 60)
    print("PHOTONICS PROMPT GENERATION FOR WEAVE BENCHMARKING")
    print("=" * 60)
    print()
    print(f"Dataset split: {args.split}")
    print(f"Number of samples: {args.samples}")
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
        include_targets=True,
    )

    # Show a sample prompt
    print()
    print("=" * 60)
    print("SAMPLE PROMPT")
    print("=" * 60)
    print(prompts[0]["prompt"])
    print()
    print("Conditions:", json.dumps(prompts[0]["conditions"], indent=2))
    print()
    print("Target total_overlap:", prompts[0]["target"]["total_overlap"])
    print(f"Dataset split: {prompts[0]['dataset_split']}")
    print()

    # Save locally with split in filename
    output_dir = Path(__file__).parent / "data" / "generated"
    output_file = (
        output_dir / f"photonics2d_prompts_{args.samples}_samples_{args.split}.json"
    )
    save_prompts_locally(prompts, output_file)

    # Publish to Weave if available
    if is_weave_enabled():
        print()
        print("📤 Publishing dataset to Weave...")
        dataset = weave.Dataset(
            name=f"photonics2d_design_prompts_v1_{args.samples}_{args.split}",
            rows=prompts,
        )
        weave.publish(dataset)
        print("✅ Dataset published to Weave!")
        print("🔗 View your dataset in the Weave UI")

    print()
    print("🎉 Prompt generation complete!")


if __name__ == "__main__":
    main()
