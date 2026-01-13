"""Shared utilities for generating prompts from HuggingFace datasets.

This module provides common functionality used across all problem-specific
prompt generation scripts, reducing code duplication and ensuring consistency.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import load_dataset

from benchmarks.shared.problem_registry import get_problem_config


def generate_prompts_from_huggingface(
    problem_name: str,
    num_samples: int,
    dataset_split: str,
    include_targets: bool,
    prompt_creator_func: Callable[[dict[str, Any], bool], dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Generate prompts from a HuggingFace dataset using a problem-specific creator function.

    This function handles all the common logic for prompt generation:
    - Loading the dataset from HuggingFace
    - Sampling the requested number of examples
    - Iterating and creating prompts with progress updates
    - Adding metadata (example_id, dataset_split)

    Args:
        problem_name: Name of the problem (e.g., 'beams2d', 'photonics2d')
        num_samples: Number of samples to generate
        dataset_split: Dataset split to use ('train', 'val', or 'test')
        include_targets: Whether to include target values
        prompt_creator_func: Function that creates a prompt from an example.
            Signature: (example: dict, include_target: bool) -> dict

    Returns:
        List of prompt dictionaries ready for evaluation
    """
    # Get dataset name from problem registry
    problem_config = get_problem_config(problem_name)
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
        prompt_data = prompt_creator_func(example, include_targets)
        prompt_data["example_id"] = i
        prompt_data["dataset_split"] = dataset_split
        prompts.append(prompt_data)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(dataset)} prompts...")

    print(f"✅ Generated {len(prompts)} prompts successfully!")
    return prompts


def save_prompts_locally(
    prompts: list[dict[str, Any]],
    output_file: Path,
    target_keys_to_keep: list[str] | None = None,
) -> None:
    """
    Save generated prompts to a local JSON file.

    Filters out large optimal_design arrays from target data to reduce file size,
    keeping only the specified target keys.

    Args:
        prompts: List of prompt dictionaries to save
        output_file: Path to the output JSON file
        target_keys_to_keep: List of target keys to keep (e.g., ['compliance']).
            If None, keeps all target data except optimal_design.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert any numpy arrays to lists for JSON serialization
    serializable_prompts = []
    for prompt in prompts:
        prompt_copy = prompt.copy()
        if "target" in prompt_copy and "optimal_design" in prompt_copy["target"]:
            # Don't save the full optimal_design array locally (too large)
            if target_keys_to_keep:
                # Keep only specified keys
                prompt_copy["target"] = {
                    key: prompt_copy["target"][key] for key in target_keys_to_keep
                }
            else:
                # Keep all keys except optimal_design
                prompt_copy["target"] = {
                    k: v
                    for k, v in prompt_copy["target"].items()
                    if k != "optimal_design"
                }
        serializable_prompts.append(prompt_copy)

    with output_file.open("w") as f:
        json.dump(serializable_prompts, f, indent=2)

    print(f"💾 Saved prompts to: {output_file}")


def create_argument_parser(problem_name: str) -> argparse.ArgumentParser:
    """
    Create a standard argument parser for prompt generation scripts.

    Args:
        problem_name: Name of the problem for the description

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=f"Generate {problem_name} design prompts from HuggingFace dataset"
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
    return parser


def run_prompt_generation_workflow(
    problem_name: str,
    prompt_creator_func: Callable[[dict[str, Any], bool], dict[str, Any]],
    target_keys_to_keep: list[str] | None = None,
) -> None:
    """
    Run the complete prompt generation workflow.

    This function orchestrates the entire process:
    - Parse arguments
    - Generate prompts
    - Save prompts locally

    Args:
        problem_name: Name of the problem (e.g., 'beams2d')
        prompt_creator_func: Function to create prompts from examples
        target_keys_to_keep: List of target keys to keep when saving locally
    """
    # Parse arguments
    parser = create_argument_parser(problem_name)
    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print(f"{problem_name.upper()} PROMPT GENERATION")
    print("=" * 60)
    print()
    print(f"Dataset split: {args.split}")
    print(f"Number of samples: {args.samples}")
    print(f"Include targets: {not args.no_targets}")
    print()

    # Generate prompts using command-line arguments
    prompts = generate_prompts_from_huggingface(
        problem_name=problem_name,
        num_samples=args.samples,
        dataset_split=args.split,
        include_targets=not args.no_targets,
        prompt_creator_func=prompt_creator_func,
    )

    # Save locally with split in filename
    output_dir = (
        Path(__file__).parent.parent / "problems" / problem_name / "data" / "generated"
    )
    output_file = (
        output_dir / f"{problem_name}_prompts_{args.samples}_samples_{args.split}.json"
    )
    save_prompts_locally(prompts, output_file, target_keys_to_keep)

    print()
    print("✨ Prompt generation complete!")
