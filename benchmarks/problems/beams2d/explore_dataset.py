"""
Explore the IDEALLab beams_2d_50_100_v0 dataset from HuggingFace.

This script downloads and explores the beam design dataset to understand
its structure before generating prompts for Weave benchmarking.

Dataset: https://huggingface.co/datasets/IDEALLab/beams_2d_50_100_v0
Rows: ~3.88k
"""

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset


def print_dataset_info(dataset, data):
    """Print basic dataset information."""
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Splits: {list(dataset.keys())}")
    print(f"Number of examples: {len(data)}")
    print(f"Features: {list(data.features.keys())}")
    print()


def print_feature_details(data):
    """Print details about dataset features."""
    print("=" * 60)
    print("FEATURE DETAILS")
    print("=" * 60)
    for feature_name, feature_type in data.features.items():
        print(f"  • {feature_name}: {feature_type}")
    print()


def print_sample_examples(data, num_samples=3):
    """Print a few sample examples from the dataset."""
    print("=" * 60)
    print(f"SAMPLE EXAMPLES (first {num_samples})")
    print("=" * 60)

    for i in range(min(num_samples, len(data))):
        example = data[i]
        print(f"\n--- Example {i + 1} ---")
        for key, value in example.items():
            if isinstance(value, (list, np.ndarray)):
                print(f"  {key}: array of length {len(value)}")
                if key == "optimal_design":
                    # Show a few values
                    arr = np.array(value)
                    print(f"    Shape: {arr.shape}")
                    print(f"    Range: [{arr.min():.3f}, {arr.max():.3f}]")
                    print(f"    Mean: {arr.mean():.3f}")
            else:
                print(f"  {key}: {value}")


def collect_statistics(data):
    """Collect statistics for numerical fields."""
    stats = {
        "volfrac": [],
        "rmin": [],
        "forcedist": [],
        "overhang_constraint": [],
        "c": [],
    }

    for example in data:
        for key, value_list in stats.items():
            if key in example:
                value_list.append(example[key])

    return stats


def print_statistics(stats):
    """Print statistical analysis of the dataset."""
    print()
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    for key, value_list in stats.items():
        values_array = np.array(value_list)
        print(f"\n{key}:")
        print(f"  Min: {values_array.min():.4f}")
        print(f"  Max: {values_array.max():.4f}")
        print(f"  Mean: {values_array.mean():.4f}")
        print(f"  Std: {values_array.std():.4f}")


def save_sample_data(data, output_dir, num_samples=10):
    """Save sample examples to JSON file."""
    sample_data = []
    for i in range(min(num_samples, len(data))):
        example = dict(data[i])
        # Convert arrays to lists for JSON serialization
        for key, value in example.items():
            if isinstance(value, np.ndarray):
                example[key] = value.tolist()
        sample_data.append(example)

    output_file = output_dir / "beams2d_sample_10_examples.json"
    with output_file.open("w") as f:
        json.dump(sample_data, f, indent=2)

    print(f"✅ Saved {num_samples} sample examples to: {output_file}")
    return output_file


def save_dataset_info(data, stats, output_dir):
    """Save dataset information to text file."""
    info_file = output_dir / "beams2d_dataset_info.txt"
    with info_file.open("w") as f:
        f.write("Dataset: IDEALLab/beams_2d_50_100_v0\n")
        f.write(f"Total examples: {len(data)}\n")
        f.write(f"Features: {list(data.features.keys())}\n\n")
        f.write("Statistics:\n")
        for key, value_list in stats.items():
            values_array = np.array(value_list)
            f.write(f"  {key}:\n")
            f.write(f"    Min: {values_array.min():.4f}\n")
            f.write(f"    Max: {values_array.max():.4f}\n")
            f.write(f"    Mean: {values_array.mean():.4f}\n")
            f.write(f"    Std: {values_array.std():.4f}\n")

    print(f"✅ Saved dataset info to: {info_file}")
    return info_file


def explore_dataset():
    """Load and explore the beams dataset."""
    print("🔍 Loading dataset from HuggingFace...")
    print("Dataset: IDEALLab/beams_2d_50_100_v0")
    print()

    # Load dataset
    dataset = load_dataset("IDEALLab/beams_2d_50_100_v0")

    print("✅ Dataset loaded successfully!")
    print()

    # Get the train split (or first available split)
    split_name = next(iter(dataset.keys()))
    data = dataset[split_name]

    # Display dataset information
    print_dataset_info(dataset, data)
    print_feature_details(data)
    print_sample_examples(data, num_samples=3)

    # Collect and display statistics
    stats = collect_statistics(data)
    print_statistics(stats)

    # Save results
    print()
    print("=" * 60)
    print("SAVING SAMPLE DATA")
    print("=" * 60)

    output_dir = Path(__file__).parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_sample_data(data, output_dir, num_samples=10)
    save_dataset_info(data, stats, output_dir)

    print()
    print("🎉 Exploration complete!")


if __name__ == "__main__":
    explore_dataset()
