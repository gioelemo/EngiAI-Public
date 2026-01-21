"""Shared utilities for generating prompts from HuggingFace datasets.

This module provides common functionality used across all problem-specific
prompt generation scripts, reducing code duplication and ensuring consistency.
"""

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import load_dataset

from benchmarks.shared.problem_registry import get_problem_config


def generate_prompts_from_huggingface(
    problem_name: str,
    prompt_creator_func: Callable[..., dict[str, Any]],
    *,
    num_samples: int,
    dataset_split: str = "test",
    include_targets: bool = True,
    seed: int | None = None,
    prompt_style: str = "full",
) -> list[dict[str, Any]]:
    """
    Generate prompts from a HuggingFace dataset using a problem-specific creator function.

    This function handles all the common logic for prompt generation:
    - Loading the dataset from HuggingFace
    - Randomly sampling the requested number of examples (for variability)
    - Iterating and creating prompts with progress updates
    - Adding metadata (example_id, dataset_split)

    Args:
        problem_name: Name of the problem (e.g., 'beams2d', 'photonics2d')
        prompt_creator_func: Function that creates a prompt from an example.
            Signature: (example: dict, include_target: bool, prompt_style: str) -> dict
        num_samples: Number of samples to generate
        dataset_split: Dataset split to use ('train', 'val', or 'test')
        include_targets: Whether to include target values
        seed: Random seed for reproducible sampling (default: None for random)
        prompt_style: Style of prompt to generate (e.g., 'full', 'natural', 'workflow')

    Returns:
        List of prompt dictionaries ready for evaluation
    """
    # Get dataset name from problem registry
    problem_config = get_problem_config(problem_name)
    dataset_name = problem_config.dataset_name

    print(f"🔄 Loading {dataset_name} dataset (split: {dataset_split})...")
    dataset = load_dataset(dataset_name, split=dataset_split)
    print(f"✅ Loaded {len(dataset)} examples")

    # Randomly sample num_samples if specified
    if num_samples < len(dataset):
        # Set random seed for reproducibility if provided
        if seed is not None:
            random.seed(seed)
            print(f"🎲 Using random seed: {seed}")

        # Generate random indices for sampling
        indices = random.sample(range(len(dataset)), num_samples)
        indices.sort()  # Sort for consistent iteration order
        dataset = dataset.select(indices)
        print(f"📊 Randomly selected {num_samples} samples")
    else:
        print(f"📊 Using all {len(dataset)} samples")

    prompts = []
    for i, example in enumerate(dataset):
        prompt_data = prompt_creator_func(example, include_targets, prompt_style)
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


def create_argument_parser(
    problem_name: str,
    available_styles: list[str] | None = None,
) -> argparse.ArgumentParser:
    """
    Create a standard argument parser for prompt generation scripts.

    Args:
        problem_name: Name of the problem for the description
        available_styles: List of available prompt styles (default: ['full'])

    Returns:
        Configured ArgumentParser instance
    """
    if available_styles is None:
        available_styles = ["full"]

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
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: None for random)",
    )
    parser.add_argument(
        "--no-targets",
        action="store_true",
        help="Exclude target values from prompts",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="full",
        choices=available_styles,
        help=f"Prompt style to generate (default: full). Available: {', '.join(available_styles)}",
    )
    return parser


def run_prompt_generation_workflow(
    problem_name: str,
    prompt_creator_func: Callable[..., dict[str, Any]],
    target_keys_to_keep: list[str] | None = None,
    available_styles: list[str] | None = None,
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
        available_styles: List of available prompt styles for this problem
    """
    # Parse arguments
    parser = create_argument_parser(problem_name, available_styles)
    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print(f"{problem_name.upper()} PROMPT GENERATION")
    print("=" * 60)
    print()
    print(f"Dataset split: {args.split}")
    print(f"Number of samples: {args.samples}")
    print(f"Random seed: {args.seed if args.seed is not None else 'None (random)'}")
    print(f"Include targets: {not args.no_targets}")
    print(f"Prompt style: {args.style}")
    print()

    # Generate prompts using command-line arguments
    prompts = generate_prompts_from_huggingface(
        problem_name=problem_name,
        prompt_creator_func=prompt_creator_func,
        num_samples=args.samples,
        dataset_split=args.split,
        include_targets=not args.no_targets,
        seed=args.seed,
        prompt_style=args.style,
    )

    # Save locally with split and style in filename
    output_dir = (
        Path(__file__).parent.parent / "problems" / problem_name / "data" / "generated"
    )
    output_file = (
        output_dir
        / f"{problem_name}_prompts_{args.samples}_samples_{args.split}_{args.style}.json"
    )
    save_prompts_locally(prompts, output_file, target_keys_to_keep)

    print()
    print("✨ Prompt generation complete!")
