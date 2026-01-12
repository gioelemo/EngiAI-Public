"""Global metrics computation for EngiBench evaluation.

This module provides a function to compute MMD (Maximum Mean Discrepancy) and
DPP diversity metrics between generated designs and optimal designs from a dataset
after evaluation completes.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import weave

from benchmarks.shared.metrics import dpp_diversity, mmd, optimality_gap
from benchmarks.shared.utils import (
    create_design_comparison,
    extract_design_from_tool_messages,
    extract_optimization_history_from_tool_messages,
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


def _get_reference_objective_value(
    dataset_name: str,
    example_id: int,
    obj_field: str = "c",  # Default to 'c' for beams2d compliance
    split: str = "test",
) -> float | None:
    """Load reference objective value from HuggingFace dataset."""
    try:
        hf_dataset = get_hf_dataset(dataset_name, split=split)

        if example_id >= len(hf_dataset):
            logger.error(
                f"Invalid example_id {example_id} for dataset {dataset_name} split {split}"
            )
            return None

        if obj_field not in hf_dataset[example_id]:
            logger.debug(
                f"Field '{obj_field}' not found in dataset for example {example_id}, trying fallbacks..."
            )
            # Try common fallback field names
            fallback_fields = ["optimal_objective_value", "compliance", "objective"]
            for fallback in fallback_fields:
                if fallback in hf_dataset[example_id]:
                    obj_field = fallback
                    logger.debug(f"Using fallback field: {fallback}")
                    break
            else:
                logger.warning(
                    f"No objective value field found for example {example_id}. Available fields: {list(hf_dataset[example_id].keys())}"
                )
                return None

        obj_value = hf_dataset[example_id][obj_field]
        # Handle array values (extract first element if array)
        if isinstance(obj_value, (list, np.ndarray)):
            return float(obj_value[0]) if len(obj_value) > 0 else None

        return float(obj_value)
    except Exception:
        logger.exception(f"Failed to load reference objective value from split {split}")
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

    # Also extract optimization history
    messages = output.get("messages", [])
    opt_history = extract_optimization_history_from_tool_messages(messages, example_id)

    if gen_design is None:
        return {
            "design_found": False,
            "design_shape": None,
            "design": None,
            "optimization_history": None,
        }

    return {
        "design_found": True,
        "design_shape": gen_design.shape,
        "design": gen_design.tolist(),  # Convert to list for JSON serialization
        "optimization_history": opt_history,  # Store the optimization history
    }


# ======================================================================
# Global metrics computation (after evaluation completes)
# ======================================================================


@weave.op()
def compute_global_metrics(  # noqa: PLR0912, PLR0915
    evaluation: Any,
    dataset_name: str,
    sigma: float = 1.0,
    save_comparisons: bool = True,
    comparison_output_dir: str | None = None,
) -> dict[str, Any]:
    """Compute global MMD, DPP diversity, and optimality gap metrics from evaluation object.

    This function retrieves generated designs and optimization histories from scorer results
    and computes MMD (similarity to dataset), DPP diversity (design variability),
    and optimality gap metrics (IOG, COG, FOG) across the full set.

    Args:
        evaluation: Weave Evaluation object (after evaluate() has been called)
        dataset_name: HuggingFace dataset name for ground truth
        sigma: Kernel bandwidth for MMD and DPP diversity
        save_comparisons: Whether to save comparison images (default: True)
        comparison_output_dir: Directory to save comparison images (default: benchmarks/evaluations/results/mmd_comparisons)

    Returns:
        dict with:
        - mmd: float (similarity to dataset distribution)
        - dpp_diversity: float (diversity of generated designs)
        - iog: float (average Initial Optimality Gap)
        - cog: float (average Cumulative Optimality Gap)
        - fog: float (average Final Optimality Gap)
        - n_designs: int (number of valid designs)
        - n_failed: int (number of failed extractions)
    """
    # Use evaluation.get_scores() to extract scorer outputs
    try:
        # Call get_scores() to get organized scorer outputs
        scores = evaluation.get_scores()
        logger.info(f"Retrieved scores from evaluation: {len(scores)} trace(s)")

        # Get dataset rows to access metadata
        dataset_rows = (
            list(evaluation.dataset.rows) if hasattr(evaluation, "dataset") else []
        )
        logger.info(f"Retrieved {len(dataset_rows)} dataset rows for metadata")

        # Extract generated designs from scorer outputs
        generated_designs: list[np.ndarray] = []
        optimization_histories: list[list[dict[str, Any]]] = []
        example_ids: list[int] = []
        dataset_split: str = "test"  # Default split
        n_failed: int = 0

        # Get only the LATEST trace (most recent evaluation)
        # scores dict keys are trace IDs - we want the last one
        if not scores:
            logger.error("No scores found in evaluation")
            return {
                "mmd": None,
                "n_designs": 0,
                "n_failed": 0,
                "error": "No scores found",
            }

        # Get the last trace (most recent evaluation)
        logger.info(f"Score keys: {list(scores.keys())}")
        latest_trace_id = list(scores.keys())[-1]
        latest_trace_scores = scores[latest_trace_id]
        logger.info(
            f"Using latest trace {latest_trace_id}: scorers={list(latest_trace_scores.keys())}"
        )

        # Collect outputs from the latest trace only
        all_outputs = []
        if "score_design_extracted" in latest_trace_scores:
            all_outputs = latest_trace_scores["score_design_extracted"]
            logger.info(
                f"Found {len(all_outputs)} outputs for score_design_extracted in latest trace"
            )
        else:
            logger.warning(
                f"No score_design_extracted in latest trace {latest_trace_id}"
            )

        logger.info(f"Total outputs from current evaluation: {len(all_outputs)}")

        # Process each output
        for idx, scorer_output in enumerate(all_outputs):
            try:
                if not isinstance(scorer_output, dict):
                    logger.warning(
                        f"Output {idx}: Not a dict, type={type(scorer_output)}"
                    )
                    n_failed += 1
                    continue

                if not scorer_output.get("design_found", False):
                    logger.debug(f"Output {idx}: design_found=False")
                    n_failed += 1
                    continue

                design_list = scorer_output.get("design")
                if design_list is None:
                    logger.warning(f"Output {idx}: design field is None")
                    n_failed += 1
                    continue

                gen_design = np.array(design_list)

                # Extract example_id and dataset_split from corresponding dataset row
                example_id = idx  # Use index as fallback
                if idx < len(dataset_rows):
                    dataset_row = dataset_rows[idx]
                    metadata = None

                    # Try to access metadata from dataset row
                    if hasattr(dataset_row, "metadata"):
                        metadata = dataset_row.metadata
                    elif isinstance(dataset_row, dict) and "metadata" in dataset_row:
                        metadata = dataset_row["metadata"]

                    if isinstance(metadata, dict):
                        example_id = metadata.get("example_id", idx)
                        # Extract dataset_split from first valid example
                        if len(generated_designs) == 0:
                            dataset_split = metadata.get("dataset_split", "test")
                            logger.info(f"Extracted dataset_split: {dataset_split}")

                logger.info(
                    f"Output {idx}: Successfully extracted design for example_id={example_id}"
                )

                generated_designs.append(gen_design)
                example_ids.append(example_id)

                # Also extract optimization history
                opt_history = scorer_output.get("optimization_history")
                if opt_history is not None:
                    optimization_histories.append(opt_history)
                    logger.info(
                        f"Output {idx}: Extracted optimization history with {len(opt_history)} steps"
                    )
                else:
                    optimization_histories.append([])
                    logger.info(
                        f"Output {idx}: No optimization history found in scorer output"
                    )

            except Exception:
                logger.exception(f"Output {idx}: Error processing")
                n_failed += 1
                continue

        logger.info(
            f"Successfully retrieved {len(generated_designs)} designs ({n_failed} failed)"
        )
        logger.info(f"Using dataset split: {dataset_split}")

        if len(generated_designs) == 0:
            logger.warning(
                "No valid designs extracted, cannot compute MMD and DPP diversity"
            )
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
                np.array(hf_dataset[i]["optimal_design"])
                for i in range(len(hf_dataset))
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
                gt_designs_for_viz.append(
                    np.zeros_like(generated_designs[0])
                )  # Placeholder

    except Exception:
        logger.exception("Failed to process evaluation results")
        return {
            "mmd": None,
            "n_designs": 0,
            "n_failed": 0,
            "error": "Failed to process results",
        }

    # Compute MMD and DPP diversity metrics
    gen_batch = np.stack(generated_designs)
    gt_batch = np.stack(gt_designs)
    gt_batch_viz = (
        np.stack(gt_designs_for_viz)
        if gt_designs_for_viz
        else gt_batch[: len(generated_designs)]
    )

    logger.info(f"Generated batch shape: {gen_batch.shape}")
    logger.info(f"Ground truth batch shape (full dataset): {gt_batch.shape}")
    logger.info(
        f"Generated designs stats - min: {gen_batch.min():.4f}, max: {gen_batch.max():.4f}, mean: {gen_batch.mean():.4f}"
    )
    logger.info(
        f"Ground truth designs stats - min: {gt_batch.min():.4f}, max: {gt_batch.max():.4f}, mean: {gt_batch.mean():.4f}"
    )

    try:
        # Use provided sigma (1.0) for MMD to match original paper
        mmd_value = mmd(gen_batch, gt_batch, sigma=sigma)
        logger.info(f"Computed MMD with sigma={sigma:.4f}: {mmd_value:.6f}")

        # Compute DPP diversity for generated designs with provided sigma
        dpp_value = dpp_diversity(gen_batch, sigma=sigma)
        logger.info(f"Computed DPP diversity with sigma={sigma:.4f}: {dpp_value:.6e}")
    except Exception:
        logger.exception("Failed to compute MMD or DPP")
        return {
            "mmd": None,
            "dpp_diversity": None,
            "n_designs": len(generated_designs),
            "n_failed": n_failed,
            "error": "Failed to compute metrics",
        }

    # Compute optimality gap metrics (IOG, COG, FOG)
    iog_list = []
    cog_list = []
    fog_list = []

    for i, opt_history in enumerate(optimization_histories):
        if not opt_history or len(opt_history) == 0:
            logger.debug(
                f"Example {example_ids[i]}: No optimization history, skipping gap computation"
            )
            continue

        # Get reference objective value from dataset
        reference_obj = _get_reference_objective_value(
            dataset_name, example_ids[i], split=dataset_split
        )

        if reference_obj is None:
            logger.debug(
                f"Example {example_ids[i]}: No reference objective value found, skipping gap computation"
            )
            continue

        # Create a simple object to mimic OptiStep for compatibility with optimality_gap function
        class OptiStep:
            def __init__(self, obj_values):
                self.obj_values = obj_values

        # Convert optimization history dicts to OptiStep objects
        opt_steps = []
        for step_dict in opt_history:
            obj_values = step_dict.get("obj_values")
            if obj_values is not None:
                # Handle both array and scalar objective values
                if isinstance(obj_values, (list, np.ndarray)):
                    obj_values = float(obj_values[0]) if len(obj_values) > 0 else 0.0
                else:
                    obj_values = float(obj_values)
                opt_steps.append(OptiStep(obj_values))

        if len(opt_steps) == 0:
            logger.debug(f"Example {example_ids[i]}: No valid optimization steps found")
            continue

        # Compute optimality gaps
        gaps = optimality_gap(opt_steps, reference_obj)

        # Compute IOG, COG, FOG
        iog_list.append(gaps[0])  # Initial optimality gap
        cog_list.append(sum(gaps))  # Cumulative optimality gap
        fog_list.append(gaps[-1])  # Final optimality gap

        logger.debug(
            f"Example {example_ids[i]}: IOG={gaps[0]:.4f}, COG={sum(gaps):.4f}, FOG={gaps[-1]:.4f}"
        )

    # Compute average IOG, COG, FOG
    average_iog = float(np.mean(iog_list)) if len(iog_list) > 0 else None
    average_cog = float(np.mean(cog_list)) if len(cog_list) > 0 else None
    average_fog = float(np.mean(fog_list)) if len(fog_list) > 0 else None

    if average_iog is not None:
        logger.info(
            f"Computed optimality gap metrics: IOG={average_iog:.4f}, COG={average_cog:.4f}, FOG={average_fog:.4f}"
        )
    else:
        logger.warning(
            "No valid optimization histories found, optimality gap metrics not computed"
        )

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
                    output_path = (
                        output_dir / f"comparison_example_{example_ids[i]}.png"
                    )
                    comparison_img.save(output_path)
                    logger.info(f"Saved comparison image: {output_path}")
        except Exception:
            logger.exception("Failed to save comparison visualizations")

    return {
        "mmd": mmd_value,
        "dpp_diversity": dpp_value,
        "iog": average_iog,
        "cog": average_cog,
        "fog": average_fog,
        "n_designs": len(generated_designs),
        "n_failed": n_failed,
    }
