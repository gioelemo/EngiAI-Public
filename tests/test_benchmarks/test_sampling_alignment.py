"""Test sampling alignment between agent and CGAN evaluations.

Verifies that given the same seed and n_samples, both evaluation
pipelines select the same examples from the test dataset.
"""

import numpy as np
import pytest


def _get_agent_sampling_indices(
    seed: int, n_samples: int, dataset_size: int
) -> np.ndarray:
    """Replicate the sampling logic from prompt_generation.py."""
    rng = np.random.default_rng(seed)
    return rng.choice(dataset_size, n_samples, replace=True)


def _get_cgan_sampling_indices(
    seed: int, n_samples: int, dataset_size: int
) -> np.ndarray:
    """Replicate the sampling logic from dataset_sample_conditions.py."""
    rng = np.random.default_rng(seed)
    return rng.choice(dataset_size, n_samples, replace=True)


# ── unit tests ────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("seed", [1, 2, 3, 5, 10, 42])
def test_indices_match_for_seed(seed: int) -> None:
    """Agent and CGAN sampling produce identical indices for a given seed."""
    n_samples, dataset_size = 10, 1000
    agent = _get_agent_sampling_indices(seed, n_samples, dataset_size)
    cgan = _get_cgan_sampling_indices(seed, n_samples, dataset_size)
    np.testing.assert_array_equal(agent, cgan)


@pytest.mark.unit
@pytest.mark.parametrize("n_samples", [1, 5, 10, 50])
def test_indices_match_for_varying_sample_counts(n_samples: int) -> None:
    """Alignment holds across different sample counts."""
    seed, dataset_size = 42, 1000
    agent = _get_agent_sampling_indices(seed, n_samples, dataset_size)
    cgan = _get_cgan_sampling_indices(seed, n_samples, dataset_size)
    np.testing.assert_array_equal(agent, cgan)


# ── integration test (real dataset) ───────────────────────────────────────────


@pytest.mark.slow
def test_indices_match_with_real_dataset() -> None:
    """Sampling alignment holds when using the actual HuggingFace dataset."""
    from datasets import load_dataset

    dataset = load_dataset("IDEALLab/beams_2d_50_100_v0", split="test")
    dataset_size = len(dataset)
    seed, n_samples = 1, 10

    agent = _get_agent_sampling_indices(seed, n_samples, dataset_size)
    cgan = _get_cgan_sampling_indices(seed, n_samples, dataset_size)
    np.testing.assert_array_equal(agent, cgan)
