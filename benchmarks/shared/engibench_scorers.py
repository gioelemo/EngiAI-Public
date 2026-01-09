"""Global metrics computation for EngiBench evaluation.

This module provides a function to compute MMD (Maximum Mean Discrepancy)
between generated designs and optimal designs from a dataset after evaluation completes.
"""

import logging
from typing import Any

import numpy as np
import weave

from benchmarks.shared.utils import extract_design_from_tool_messages, get_hf_dataset
from src.tools.metrics import mmd

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
) -> np.ndarray | None:
    """Load ground truth design from HuggingFace dataset."""
    try:
        hf_dataset = get_hf_dataset(dataset_name)

        if example_id >= len(hf_dataset):
            logger.error(f"Invalid example_id {example_id} for dataset {dataset_name}")
            return None
        else:
            design = np.array(hf_dataset[example_id][design_field])
            return design
    except Exception:
        logger.exception("Failed to load ground truth design")
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


def compute_global_metrics(  # noqa: PLR0912, PLR0915
    evaluation: Any,
    dataset_name: str,
    sigma: float = 1.0,
    num_expected_designs: int | None = None,
) -> dict[str, Any]:
    """Compute global MMD metric from evaluation object.

    This function retrieves generated designs from scorer results
    and computes MMD (similarity to dataset) across the full set.

    Args:
        evaluation: Weave Evaluation object (after evaluate() has been called)
        dataset_name: HuggingFace dataset name for ground truth
        sigma: Kernel bandwidth for MMD
        num_expected_designs: Number of designs in current evaluation (to filter from history)

    Returns:
        dict with:
        - mmd: float (similarity to dataset distribution)
        - n_designs: int (number of valid designs)
        - n_failed: int (number of failed extractions)
    """
    # Use the evaluation API to get scores
    try:
        score_calls = evaluation.get_score_calls()
        logger.info(
            f"Retrieved score calls: type={type(score_calls)}, len={len(score_calls)}"
        )

        # Flatten dict structure to list of Call objects
        if isinstance(score_calls, dict):
            score_calls_list = []
            for call_list in score_calls.values():
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

        # Extract designs from Call objects
        generated_designs = []
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
                generated_designs.append(gen_design)

            except Exception:
                logger.exception(f"Score call {idx}: Error processing")
                n_failed += 1
                continue

        logger.info(
            f"Successfully retrieved {len(generated_designs)} designs ({n_failed} failed)"
        )

        if len(generated_designs) == 0:
            logger.warning("No valid designs extracted, cannot compute MMD")
            return {
                "mmd": None,
                "n_designs": 0,
                "n_failed": n_failed,
                "error": "No valid designs extracted",
            }

    except Exception:
        logger.exception("Failed to process evaluation results")
        return {
            "mmd": None,
            "n_designs": 0,
            "n_failed": 0,
            "error": "Failed to process results",
        }

    # Load ground truth designs from dataset
    try:
        gt_designs = _load_all_ground_truth_designs(dataset_name)
        logger.info(f"Loaded {len(gt_designs)} ground truth designs from dataset")

    except Exception:
        logger.exception("Failed to load ground truth")
        return {
            "mmd": None,
            "n_designs": len(generated_designs),
            "n_failed": n_failed,
            "error": "Failed to load ground truth",
        }

    # Compute MMD between generated and ground truth designs
    gen_batch = np.stack(generated_designs)
    gt_batch = np.stack(gt_designs)

    try:
        mmd_value = mmd(gen_batch, gt_batch, sigma=sigma)
        logger.info(f"Computed MMD: {mmd_value:.4f}")
    except Exception:
        logger.exception("Failed to compute MMD")
        return {
            "mmd": None,
            "n_designs": len(generated_designs),
            "n_failed": n_failed,
            "error": "Failed to compute MMD",
        }

    return {
        "mmd": mmd_value,
        "n_designs": len(generated_designs),
        "n_failed": n_failed,
    }


def _load_all_ground_truth_designs(dataset_name: str) -> list[np.ndarray]:
    """Load all ground truth designs from HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset name

    Returns:
        List of numpy arrays containing ground truth designs
    """
    hf_dataset = get_hf_dataset(dataset_name)
    designs = []

    for example in hf_dataset:
        design = np.array(example["optimal_design"])
        designs.append(design)

    return designs
