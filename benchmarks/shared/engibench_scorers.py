"""Global metrics computation for EngiBench evaluation.

This module provides a function to compute MMD (Maximum Mean Discrepancy)
between generated designs and optimal designs from a dataset after evaluation completes.
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import weave
from scipy.spatial.distance import pdist  # type: ignore[import-untyped]

from benchmarks.shared.metrics import mmd
from benchmarks.shared.utils import (
    create_design_comparison,
    extract_design_from_tool_messages,
    get_hf_dataset,
)

logger = logging.getLogger(__name__)

# ======================================================================
# Helper functions for design extraction
# ======================================================================


def _get_generated_design(
    output: dict[str, Any],
    example_id: int,
) -> np.ndarray | None:
    """Extract generated design from agent output.

    IMPORTANT: Do NOT use global cache fallback in batch evaluation mode
    as it only stores one design at a time and causes data loss.
    """
    messages = output.get("messages", [])
    design_array = extract_design_from_tool_messages(messages, example_id)

    if design_array is None:
        logger.warning(f"Example {example_id}: Failed to extract design from messages")
        logger.debug(f"Example {example_id}: Messages count: {len(messages)}")

        # Count tool messages for debugging
        tool_msg_count = sum(1 for msg in messages if hasattr(msg, "tool_call_id"))
        logger.debug(f"Example {example_id}: Tool messages count: {tool_msg_count}")

    return design_array


def _get_ground_truth_design(
    dataset_name: str,
    example_id: int,
    design_field: str = "optimal_design",
    split: str = "test",
) -> np.ndarray | None:
    """Load ground truth design from HuggingFace dataset."""
    try:
        hf_dataset = get_hf_dataset(dataset_name, split=split)

        if example_id >= len(hf_dataset):
            logger.error(
                f"Invalid example_id {example_id} for dataset {dataset_name} split {split}"
            )
            return None
        else:
            design = np.array(hf_dataset[example_id][design_field])
            return design
    except Exception:
        logger.exception(f"Failed to load ground truth design from split {split}")
        return None


# ======================================================================
# Lightweight scorer to enable results access
# ======================================================================


@weave.op()
def score_design_extracted(
    output: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Lightweight scorer that extracts and stores the design in Weave.

    This scorer extracts the design from model output and stores it
    in the score results so it can be accessed later for global metrics.

    Args:
        output: Agent output with messages containing generated design
        metadata: Must contain example_id

    Returns:
        dict with design_found flag, shape, and the actual design array
    """
    example_id = metadata.get("example_id", 0)
    gen_design = _get_generated_design(output, example_id)

    if gen_design is None:
        return {
            "design_found": False,
            "design_shape": None,
            "design": None,
        }

    return {
        "design_found": True,
        "design_shape": gen_design.shape,
        "design": gen_design.tolist(),  # Convert to list for JSON serialization
    }


# ======================================================================
# Global metrics computation (after evaluation completes)
# ======================================================================


def compute_global_metrics(  # noqa: PLR0912, PLR0913, PLR0915
    evaluation: Any,
    dataset_name: str,
    sigma: float = 1.0,
    num_expected_designs: int | None = None,
    save_comparisons: bool = True,
    comparison_output_dir: str | None = None,
) -> dict[str, Any]:
    """Compute global MMD metric from evaluation object.

    This function retrieves generated designs from scorer results
    and computes MMD (similarity to dataset) across the full set.

    Args:
        evaluation: Weave Evaluation object (after evaluate() has been called)
        dataset_name: HuggingFace dataset name for ground truth
        sigma: Kernel bandwidth for MMD
        num_expected_designs: Number of designs in current evaluation (to filter from history)
        save_comparisons: Whether to save comparison images (default: True)
        comparison_output_dir: Directory to save comparison images (default: benchmarks/evaluations/results/mmd_comparisons)

    Returns:
        dict with:
        - mmd: float (similarity to dataset distribution)
        - n_designs: int (number of valid designs)
        - n_failed: int (number of failed extractions)
    """
    # Use the evaluation API to get scores
    try:
        # Give Weave time to flush all calls if needed
        time.sleep(1)

        # Try to get score calls - might need scorer name
        try:
            score_calls = evaluation.get_score_calls("score_design_extracted")
            logger.info("Retrieved score calls using scorer name 'score_design_extracted'")
        except (TypeError, AttributeError):
            score_calls = evaluation.get_score_calls()
            logger.info("Retrieved score calls without scorer name")

        logger.info(
            f"Retrieved score calls: type={type(score_calls)}, len={len(score_calls)}"
        )

        # Debug: log the actual structure
        if isinstance(score_calls, dict):
            logger.info(f"Score calls keys: {list(score_calls.keys())}")
            for key, value in score_calls.items():
                logger.info(f"Key '{key}': type={type(value)}, value={value}")

        # Flatten dict structure to list of Call objects
        if isinstance(score_calls, dict):
            score_calls_list = []
            for key, call_list in score_calls.items():
                logger.info(
                    f"Processing key={key}, value type={type(call_list)}, "
                    f"len={len(call_list) if isinstance(call_list, list) else 'N/A'}"
                )
                if isinstance(call_list, list):
                    score_calls_list.extend(call_list)
                else:
                    score_calls_list.append(call_list)
            logger.info(
                f"Flattened {len(score_calls)} dict entries to {len(score_calls_list)} Call objects"
            )
        else:
            score_calls_list = list(score_calls)

        # FILTER: Take only the last N designs matching current evaluation size
        if num_expected_designs and len(score_calls_list) > num_expected_designs:
            logger.info(
                f"Filtering to last {num_expected_designs} designs (found {len(score_calls_list)} total)"
            )
            score_calls_list = score_calls_list[-num_expected_designs:]

        # Extract generated designs from Call objects
        generated_designs = []
        example_ids = []
        dataset_split = "test"  # Default split
        n_failed = 0

        for idx, score_call in enumerate(score_calls_list):
            try:
                output = score_call.output if hasattr(score_call, "output") else None

                if output is None or not isinstance(output, dict):
                    n_failed += 1
                    continue

                if not output.get("design_found", False):
                    n_failed += 1
                    continue

                design_list = output.get("design")
                if design_list is None:
                    n_failed += 1
                    continue

                gen_design = np.array(design_list)

                # Extract example_id and dataset_split for tracking/visualization
                example_id = idx  # Use index as fallback
                if hasattr(score_call, "inputs") and isinstance(
                    score_call.inputs, dict
                ):
                    metadata = score_call.inputs.get("metadata", {})
                    if isinstance(metadata, dict):
                        example_id = metadata.get("example_id", idx)
                        # Extract dataset_split from first valid example
                        if idx == 0:
                            dataset_split = metadata.get("dataset_split", "test")

                logger.info(f"Processing design {idx}: example_id={example_id}")

                generated_designs.append(gen_design)
                example_ids.append(example_id)

            except Exception:
                logger.exception(f"Score call {idx}: Error processing")
                n_failed += 1
                continue

        logger.info(
            f"Successfully retrieved {len(generated_designs)} designs ({n_failed} failed)"
        )
        logger.info(f"Using dataset split: {dataset_split}")

        if len(generated_designs) == 0:
            logger.warning("No valid designs extracted, cannot compute MMD")
            return {
                "mmd": None,
                "n_designs": 0,
                "n_failed": n_failed,
                "error": "No valid designs extracted",
            }

        # Load ALL ground truth designs from the dataset for MMD computation
        # MMD compares the distribution of generated designs vs. distribution of all GT designs
        logger.info(
            f"Loading all ground truth designs from dataset {dataset_name} (split: {dataset_split})"
        )
        try:
            hf_dataset = get_hf_dataset(dataset_name, split=dataset_split)
            gt_designs = [
                np.array(hf_dataset[i]["optimal_design"]) for i in range(len(hf_dataset))
            ]
            logger.info(
                f"Loaded {len(gt_designs)} ground truth designs from full dataset ({dataset_split} split)"
            )
        except Exception:
            logger.exception(
                f"Failed to load ground truth designs from dataset (split: {dataset_split})"
            )
            return {
                "mmd": None,
                "n_designs": len(generated_designs),
                "n_failed": n_failed,
                "error": "Failed to load ground truth dataset",
            }

        # For visualization, load GT designs corresponding to generated examples
        gt_designs_for_viz = []
        for example_id in example_ids:
            gt_design = _get_ground_truth_design(
                dataset_name, example_id, split=dataset_split
            )
            if gt_design is not None:
                gt_designs_for_viz.append(gt_design)
            else:
                logger.warning(
                    f"Failed to load GT design for visualization: example {example_id} (split: {dataset_split})"
                )
                gt_designs_for_viz.append(np.zeros_like(generated_designs[0]))  # Placeholder

    except Exception:
        logger.exception("Failed to process evaluation results")
        return {
            "mmd": None,
            "n_designs": 0,
            "n_failed": 0,
            "error": "Failed to process results",
        }

    # Compute MMD between generated and ground truth designs
    gen_batch = np.stack(generated_designs)
    gt_batch = np.stack(gt_designs)
    gt_batch_viz = np.stack(gt_designs_for_viz) if gt_designs_for_viz else gt_batch[:len(generated_designs)]

    logger.info(f"Generated batch shape: {gen_batch.shape}")
    logger.info(f"Ground truth batch shape (full dataset): {gt_batch.shape}")
    logger.info(
        f"Generated designs stats - min: {gen_batch.min():.4f}, max: {gen_batch.max():.4f}, mean: {gen_batch.mean():.4f}"
    )
    logger.info(
        f"Ground truth designs stats - min: {gt_batch.min():.4f}, max: {gt_batch.max():.4f}, mean: {gt_batch.mean():.4f}"
    )

    # Compute L2 distance between generated and their corresponding GT designs (for logging)
    for i in range(min(len(generated_designs), len(gt_designs_for_viz))):
        l2_dist = np.linalg.norm(gen_batch[i] - gt_batch_viz[i])
        logger.info(f"L2 distance for design {i} vs its GT pair: {l2_dist:.4f}")

    # Compute appropriate sigma based on median pairwise distance
    # Flatten designs for distance computation
    gen_flat = gen_batch.reshape(gen_batch.shape[0], -1)
    gt_flat = gt_batch.reshape(gt_batch.shape[0], -1)

    all_flat = np.vstack([gen_flat, gt_flat])
    pairwise_dists = pdist(all_flat, "euclidean")
    median_dist = np.median(pairwise_dists)

    # Use median heuristic for sigma
    auto_sigma = median_dist
    logger.info(f"Median pairwise distance: {median_dist:.4f}")
    logger.info(
        f"Auto-computed sigma would be: {auto_sigma:.4f} (using provided sigma={sigma} for comparison with paper)"
    )

    try:
        # Use provided sigma for comparison with original paper
        mmd_value = mmd(gen_batch, gt_batch, sigma=sigma)
        logger.info(f"Computed MMD with sigma={sigma:.4f}: {mmd_value:.6f}")
    except Exception:
        logger.exception("Failed to compute MMD")
        return {
            "mmd": None,
            "n_designs": len(generated_designs),
            "n_failed": n_failed,
            "error": "Failed to compute MMD",
        }

    # Save comparison visualizations if requested
    if save_comparisons:
        try:
            output_dir = Path(
                comparison_output_dir
                or "benchmarks/evaluations/results/mmd_comparisons"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            for i in range(len(gen_batch)):
                comparison_img = create_design_comparison(
                    gen_batch[i], gt_batch_viz[i], example_ids[i]
                )
                if comparison_img is not None:
                    output_path = output_dir / f"comparison_example_{example_ids[i]}.png"
                    comparison_img.save(output_path)
                    logger.info(f"Saved comparison image: {output_path}")
        except Exception:
            logger.exception("Failed to save comparison visualizations")

    return {
        "mmd": mmd_value,
        "n_designs": len(generated_designs),
        "n_failed": n_failed,
    }
