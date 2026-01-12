"""
Generic dataset exploration tool for EngiBench problems.

This script downloads and explores any EngiBench dataset from HuggingFace
to understand its structure before generating prompts for benchmarking.

Usage:
    # Using problem name (reads dataset from registry)
    python -m benchmarks.shared.explore_dataset --problem beams2d
    python -m benchmarks.shared.explore_dataset --problem photonics2d --split test

    # Direct dataset name (for datasets not in registry)
    python -m benchmarks.shared.explore_dataset --dataset IDEALLab/beams_2d_50_100_v0
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from benchmarks.shared.problem_registry import PROBLEMS

# Constants for display formatting
ARRAY_LENGTH_THRESHOLD = 10  # Length above which to show summary stats instead of full array
MULTIDIM_THRESHOLD = 2  # Dimensions threshold for multi-dimensional arrays
STRING_PREVIEW_LENGTH = 100  # Max length for string previews


def print_dataset_info(dataset, data, dataset_name: str):
    """Print basic dataset information."""
    print("=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Splits: {list(dataset.keys())}")
    print(f"Number of examples: {len(data)}")
    print(f"Features: {list(data.features.keys())}")
    print()


def print_feature_details(data):
    """Print details about dataset features."""
    print("=" * 80)
    print("FEATURE DETAILS")
    print("=" * 80)
    for feature_name, feature_type in data.features.items():
        print(f"  • {feature_name}: {feature_type}")
    print()


def print_sample_examples(data, num_samples=3):
    """Print a few sample examples from the dataset."""
    print("=" * 80)
    print(f"SAMPLE EXAMPLES (first {num_samples})")
    print("=" * 80)

    for i in range(min(num_samples, len(data))):
        example = data[i]
        print(f"\n--- Example {i} ---")
        for key, value in example.items():
            if isinstance(value, (list, np.ndarray)):
                arr = np.array(value)
                if arr.ndim == 1:
                    # 1D array - could be optimization history or single value
                    if len(arr) > ARRAY_LENGTH_THRESHOLD:
                        print(f"  {key}: array of length {len(arr)}")
                        print(f"    Range: [{arr.min():.6f}, {arr.max():.6f}]")
                        print(f"    Mean: {arr.mean():.6f}")
                    else:
                        print(f"  {key}: {arr.tolist()}")
                elif arr.ndim >= MULTIDIM_THRESHOLD:
                    # Multi-dimensional array (designs, images, etc.)
                    print(f"  {key}: {arr.ndim}D array")
                    print(f"    Shape: {arr.shape}")
                    print(f"    Range: [{arr.min():.6f}, {arr.max():.6f}]")
                    print(f"    Mean: {arr.mean():.6f}")
                    print(f"    Non-zero: {np.count_nonzero(arr)} / {arr.size}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())}")
            elif isinstance(value, (int, float)):
                print(f"  {key}: {value}")
            else:
                # String or other types
                value_str = str(value)
                if len(value_str) > STRING_PREVIEW_LENGTH:
                    print(f"  {key}: {value_str[:STRING_PREVIEW_LENGTH]}...")
                else:
                    print(f"  {key}: {value_str}")


def collect_statistics(data):
    """Collect statistics for numerical scalar fields."""
    # Identify numerical scalar fields dynamically
    stats = {}

    # Sample first example to determine field types
    if len(data) > 0:
        example = data[0]
        for key, value in example.items():
            # Only track scalar numerical fields
            if isinstance(value, (int, float)):
                stats[key] = []
            elif isinstance(value, np.ndarray):
                arr = np.array(value)
                if arr.ndim == 0 or (arr.ndim == 1 and len(arr) == 1):
                    # Scalar or single-element array
                    stats[key] = []

    # Collect values
    for example in data:
        for key, value_list in stats.items():
            if key in example:
                value = example[key]
                if isinstance(value, (int, float)):
                    value_list.append(value)
                elif isinstance(value, (list, np.ndarray)):
                    arr = np.array(value)
                    if arr.size == 1:
                        value_list.append(float(arr.flat[0]))

    return stats


def print_statistics(stats):
    """Print statistical analysis of the dataset."""
    if not stats:
        print("\nNo scalar numerical fields found for statistics.")
        return

    print()
    print("=" * 80)
    print("DATASET STATISTICS (Scalar Numerical Fields)")
    print("=" * 80)

    for key, value_list in stats.items():
        if len(value_list) == 0:
            continue
        values_array = np.array(value_list)
        print(f"\n{key}:")
        print(f"  Count: {len(values_array)}")
        print(f"  Min: {values_array.min():.6f}")
        print(f"  Max: {values_array.max():.6f}")
        print(f"  Mean: {values_array.mean():.6f}")
        print(f"  Median: {np.median(values_array):.6f}")
        print(f"  Std: {values_array.std():.6f}")


def save_sample_data(data, output_dir: Path, problem_name: str, num_samples=10):
    """Save sample examples to JSON file."""
    sample_data = []
    for i in range(min(num_samples, len(data))):
        example = dict(data[i])
        # Convert arrays to lists for JSON serialization
        for key, value in example.items():
            if isinstance(value, np.ndarray):
                example[key] = value.tolist()
        sample_data.append(example)

    output_file = output_dir / f"{problem_name}_sample_{num_samples}_examples.json"
    with output_file.open("w") as f:
        json.dump(sample_data, f, indent=2)

    print(f"\n✅ Saved {num_samples} sample examples to: {output_file}")
    return output_file


def save_dataset_info(
    data, stats, output_dir: Path, dataset_name: str, problem_name: str
):
    """Save dataset information to text file."""
    info_file = output_dir / f"{problem_name}_dataset_info.txt"
    with info_file.open("w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total examples: {len(data)}\n")
        f.write(f"Features: {list(data.features.keys())}\n\n")

        if stats:
            f.write("Statistics (Scalar Numerical Fields):\n")
            for key, value_list in stats.items():
                if len(value_list) == 0:
                    continue
                values_array = np.array(value_list)
                f.write(f"  {key}:\n")
                f.write(f"    Count: {len(values_array)}\n")
                f.write(f"    Min: {values_array.min():.6f}\n")
                f.write(f"    Max: {values_array.max():.6f}\n")
                f.write(f"    Mean: {values_array.mean():.6f}\n")
                f.write(f"    Median: {np.median(values_array):.6f}\n")
                f.write(f"    Std: {values_array.std():.6f}\n")
        else:
            f.write("No scalar numerical fields found.\n")

    print(f"✅ Saved dataset info to: {info_file}")
    return info_file


def explore_dataset(
    dataset_name: str,
    split: str = "train",
    num_samples: int = 3,
    save_samples: int = 10,
    output_dir: Path | None = None,
):
    """Load and explore any EngiBench dataset.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "IDEALLab/beams_2d_50_100_v0")
        split: Dataset split to explore (default: "train")
        num_samples: Number of examples to print (default: 3)
        save_samples: Number of examples to save to JSON (default: 10)
        output_dir: Directory to save outputs (default: data/raw under problem folder)
    """
    print("🔍 Loading dataset from HuggingFace...")
    print(f"Dataset: {dataset_name}")
    print(f"Split: {split}")
    print()

    # Load dataset
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    print("✅ Dataset loaded successfully!")
    print()

    # Get the requested split or first available
    if split in dataset:
        data = dataset[split]
    else:
        available_splits = list(dataset.keys())
        print(f"⚠️  Split '{split}' not found. Available: {available_splits}")
        split = available_splits[0]
        data = dataset[split]
        print(f"Using split: {split}")
        print()

    # Display dataset information
    print_dataset_info(dataset, data, dataset_name)
    print_feature_details(data)
    print_sample_examples(data, num_samples=num_samples)

    # Collect and display statistics
    stats = collect_statistics(data)
    print_statistics(stats)

    # Determine problem name from dataset
    problem_name = dataset_name.split("/")[-1].replace("_v0", "").replace("_", "")

    # Set output directory
    if output_dir is None:
        # Try to determine from problem name
        benchmarks_dir = Path(__file__).parent.parent
        output_dir = benchmarks_dir / "problems" / problem_name / "data" / "raw"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    print()
    print("=" * 80)
    print("SAVING SAMPLE DATA")
    print("=" * 80)

    save_sample_data(data, output_dir, problem_name, num_samples=save_samples)
    save_dataset_info(data, stats, output_dir, dataset_name, problem_name)

    print()
    print("🎉 Exploration complete!")


def main():
    """Command-line interface for dataset exploration."""
    parser = argparse.ArgumentParser(
        description="Explore EngiBench datasets from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore using problem name (reads dataset from registry)
  python -m benchmarks.shared.explore_dataset --problem beams2d
  python -m benchmarks.shared.explore_dataset --problem photonics2d --split test

  # Show more examples
  python -m benchmarks.shared.explore_dataset --problem beams2d --num-samples 5

  # Direct dataset name (for datasets not in registry)
  python -m benchmarks.shared.explore_dataset --dataset IDEALLab/beams_2d_50_100_v0

  # List available problems
  python -m benchmarks.shared.explore_dataset --list-problems
        """,
    )

    # Create mutually exclusive group for problem vs dataset
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--problem",
        type=str,
        help="Problem name (e.g., beams2d, photonics2d) - reads dataset from registry",
    )
    input_group.add_argument(
        "--dataset",
        type=str,
        help="HuggingFace dataset name (e.g., IDEALLab/beams_2d_50_100_v0)",
    )
    input_group.add_argument(
        "--list-problems",
        action="store_true",
        help="List all available problems in the registry",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to explore (default: train)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of examples to print (default: 3)",
    )
    parser.add_argument(
        "--save-samples",
        type=int,
        default=10,
        help="Number of examples to save to JSON (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save outputs (default: auto-detect from problem name)",
    )

    args = parser.parse_args()

    # Handle list-problems
    if args.list_problems:
        print("Available problems in registry:")
        print("=" * 80)
        for problem_name, config in PROBLEMS.items():
            print(f"\n{problem_name}:")
            print(f"  Dataset: {config.dataset_name}")
            print(f"  Objectives: {', '.join(obj.name for obj in config.objectives)}")
            print(f"  Conditions: {', '.join(cond.name for cond in config.conditions)}")
        return

    # Determine dataset name
    if args.problem:
        if args.problem not in PROBLEMS:
            print(f"❌ Error: Problem '{args.problem}' not found in registry.")
            print(f"Available problems: {', '.join(PROBLEMS.keys())}")
            print("\nUse --list-problems to see details.")
            return

        problem_config = PROBLEMS[args.problem]
        dataset_name = problem_config.dataset_name
        problem_name = args.problem
        print(f"📋 Using problem: {problem_name}")
        print(f"📦 Dataset: {dataset_name}")
        print()
    elif args.dataset:
        dataset_name = args.dataset
        problem_name = dataset_name.split("/")[-1].replace("_v0", "").replace("_", "")
    else:
        parser.error("Must provide either --problem or --dataset (or --list-problems)")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None

    # If using problem name, set output dir to problem folder
    if args.problem and output_dir is None:
        benchmarks_dir = Path(__file__).parent.parent
        output_dir = benchmarks_dir / "problems" / args.problem / "data" / "raw"

    explore_dataset(
        dataset_name=dataset_name,
        split=args.split,
        num_samples=args.num_samples,
        save_samples=args.save_samples,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
