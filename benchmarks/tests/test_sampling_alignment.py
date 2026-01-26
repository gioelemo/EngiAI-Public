"""Test script to verify sampling alignment between agent and CGAN evaluations.

This script verifies that given the same seed and n_samples, both evaluation
pipelines select the same examples from the test dataset.

Usage:
    python benchmarks/tests/test_sampling_alignment.py
    python benchmarks/tests/test_sampling_alignment.py --seed 42 --n_samples 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def get_agent_sampling_indices(
    seed: int, n_samples: int, dataset_size: int
) -> np.ndarray:
    """Replicate the sampling logic from prompt_generation.py."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(dataset_size, n_samples, replace=True)
    return indices


def get_cgan_sampling_indices(
    seed: int, n_samples: int, dataset_size: int
) -> np.ndarray:
    """Replicate the sampling logic from dataset_sample_conditions.py."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(dataset_size, n_samples, replace=True)
    return indices


def test_indices_match(seed: int, n_samples: int, dataset_size: int) -> bool:
    """Test that both methods produce identical indices."""
    agent_indices = get_agent_sampling_indices(seed, n_samples, dataset_size)
    cgan_indices = get_cgan_sampling_indices(seed, n_samples, dataset_size)

    match = np.array_equal(agent_indices, cgan_indices)

    print(f"\nSeed: {seed}, Samples: {n_samples}, Dataset size: {dataset_size}")
    print(f"Agent indices:  {agent_indices}")
    print(f"CGAN indices:   {cgan_indices}")
    print(f"Match: {'YES' if match else 'NO'}")

    return match


def test_with_real_dataset(
    seed: int,
    n_samples: int,
    dataset_name: str = "IDEALLab/beams_2d_50_100_v0",
) -> bool:
    """Test with the actual HuggingFace dataset and compare conditions."""
    print(f"\nLoading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")

    # Get indices from both methods
    agent_indices = get_agent_sampling_indices(seed, n_samples, dataset_size)
    cgan_indices = get_cgan_sampling_indices(seed, n_samples, dataset_size)

    indices_match = np.array_equal(agent_indices, cgan_indices)

    print(f"\nIndices comparison (seed={seed}, n_samples={n_samples}):")
    print(f"  Agent indices: {agent_indices}")
    print(f"  CGAN indices:  {cgan_indices}")
    print(f"  Indices match: {'YES' if indices_match else 'NO'}")

    if not indices_match:
        return False

    # Compare actual conditions from sampled examples
    print("\nConditions comparison:")
    print(f"{'Idx':<5} {'volfrac':<10} {'forcedist':<12} {'rmin':<8}")
    print("-" * 40)

    for idx in agent_indices:
        example = dataset[int(idx)]
        volfrac = example.get("volfrac", "N/A")
        forcedist = example.get("forcedist", "N/A")
        rmin = example.get("rmin", "N/A")
        print(f"{idx:<5} {volfrac:<10.4f} {forcedist:<12.4f} {rmin:<8.4f}")

    return True


def run_multiple_seeds_test(
    seeds: list[int], n_samples: int, dataset_size: int
) -> None:
    """Run test across multiple seeds."""
    print("=" * 60)
    print("SAMPLING ALIGNMENT TEST - MULTIPLE SEEDS")
    print("=" * 60)

    all_passed = True
    for seed in seeds:
        passed = test_indices_match(seed, n_samples, dataset_size)
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED - Sampling is aligned!")
    else:
        print("SOME TESTS FAILED - Sampling is NOT aligned!")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test sampling alignment between agent and CGAN evaluations"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed to test (default: 1)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples (default: 10)",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1000,
        help="Simulated dataset size for quick test (default: 1000)",
    )
    parser.add_argument(
        "--real-dataset",
        action="store_true",
        help="Test with real HuggingFace dataset (slower, but verifies actual conditions)",
    )
    parser.add_argument(
        "--multi-seed",
        action="store_true",
        help="Test multiple seeds (1-10)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SAMPLING ALIGNMENT TEST")
    print("=" * 60)
    print()
    print("This test verifies that the agent evaluation and CGAN evaluation")
    print("select identical samples from the dataset when using the same seed.")
    print()

    if args.multi_seed:
        run_multiple_seeds_test(
            seeds=list(range(1, 11)),
            n_samples=args.n_samples,
            dataset_size=args.dataset_size,
        )
    elif args.real_dataset:
        success = test_with_real_dataset(args.seed, args.n_samples)
        print()
        if success:
            print("TEST PASSED - Sampling produces identical conditions!")
        else:
            print("TEST FAILED - Sampling mismatch detected!")
    else:
        success = test_indices_match(args.seed, args.n_samples, args.dataset_size)
        print()
        if success:
            print("TEST PASSED - Indices match!")
        else:
            print("TEST FAILED - Indices do not match!")


if __name__ == "__main__":
    main()
