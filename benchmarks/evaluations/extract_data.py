"""
Extract complete per-design data from Weave evaluation traces.

This script fetches evaluation results from Weave and extracts ALL scorer outputs
including metrics, design arrays, optimization histories, and metadata.
Saves to JSON format for offline global metrics computation.
"""

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

import weave

# Add project root to path to import config
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from benchmarks.shared.problem_registry import get_problem_config  # noqa: E402
from benchmarks.shared.scorers.rag_scorer import RAG_OUTPUT_FIELDS  # noqa: E402
from config import config  # noqa: E402

logger = logging.getLogger(__name__)

# Constants
REF_EXTRA_MIN_LENGTH = 3

# HPC workflow scorer output fields to extract
HPC_WORKFLOW_OUTPUT_FIELDS = [
    "hpc_workflow_score",
    "step_completion_rate",
    "steps_completed_count",
    "steps_total",
    "step_generate_training_command",
    "step_submit_slurm_job",
    "step_monitor_job_until_complete",
    "step_evaluate_model",
    "eval_metrics",
    "eval_metrics_count",
    "eval_metrics_score",
    "eval_IOG",
    "eval_COG",
    "eval_FOG",
    "eval_MMD",
    "eval_DPP",
    "eval_viol",
    "training_config_correct",
    "config_score",
]


def _extract_metadata_from_example(
    example,
) -> tuple[int | None, int | None, str | None]:
    """Extract example_id, seed, and problem_id from a Weave example object.

    Weave returns different object types depending on the SDK version and how
    the evaluation was stored (plain dict, WeaveDict with ``__getitem__``,
    ObjectRef with ``_val``/``_extra`` internals, or plain-attribute objects).
    The multiple fallback paths below cover all observed variants so that
    extraction works across Weave versions without requiring a specific API.

    Precedence for **seed**: ``metadata.run_id`` > ``metadata.seed`` >
    dataset-name suffix (``_seed_N``).

    Returns:
        Tuple of (example_id, seed, problem_id)
    """
    example_id = None
    seed = None
    problem_id = None

    # Try to access example data directly (WeaveDict supports dict access)
    if hasattr(example, "__getitem__"):
        with contextlib.suppress(KeyError, TypeError):
            metadata = example.get("metadata", {})
            if metadata:
                example_id = metadata.get("example_id")
                # run_id is the unique tracking identifier (from --run or --seed).
                # Prefer run_id so that multiple --run values don't collapse
                # during dedup (all runs share seed=1 by default).
                seed = metadata.get("run_id") or metadata.get("seed")

    # Fallback: Extract from ObjectRef if direct access failed
    if (
        example_id is None
        and hasattr(example, "__dict__")
        and "ref" in example.__dict__
    ):
        ref = example.__dict__["ref"]
        # Extract seed from dataset name like "beams2d_eval_dataset_seed_1"
        dataset_name = ref.name if hasattr(ref, "name") else ""
        if "_seed_" in dataset_name and seed is None:
            with contextlib.suppress(ValueError, IndexError):
                seed = int(dataset_name.split("_seed_")[1])

        # Use row hash as fallback example_id
        if (
            example_id is None
            and hasattr(ref, "_extra")
            and len(ref._extra) > REF_EXTRA_MIN_LENGTH
        ):
            example_id = ref._extra[REF_EXTRA_MIN_LENGTH]

    # Additional fallback to dict/attribute access (only if still None)
    if example_id is None and isinstance(example, dict):
        example_id = example.get("example_id")
        seed = example.get("seed") if seed is None else seed
        problem_id = example.get("problem_id")
    elif (
        example_id is None
        and hasattr(example, "_val")
        and isinstance(example._val, dict)
    ):
        # Weave object with _val dict
        example_id = example._val.get("example_id")
        seed = example._val.get("seed") if seed is None else seed
        problem_id = example._val.get("problem_id")
    elif example_id is None and hasattr(example, "example_id"):
        # Direct attributes
        example_id = getattr(example, "example_id", None)
        seed = getattr(example, "seed", None) if seed is None else seed
        problem_id = getattr(example, "problem_id", None)

    return example_id, seed, problem_id


def _extract_model_id_from_score_call(score_call, score_output: dict) -> str:
    """Extract model_id from score call output or child predict calls."""
    model_id = score_output.get("model", "")

    # Try to get model from child predict call if not in score output
    if not model_id:
        pred_calls = list(score_call.children())
        for pred_call in pred_calls:
            if hasattr(pred_call, "output") and isinstance(pred_call.output, dict):
                model_id = pred_call.output.get("model", "")
                if model_id:
                    break

    return model_id


def _compute_combined_overall_score(  # noqa: PLR0912
    output_quality: dict,
    task_completion: dict,
    tool_use: dict,
    problem_type: str | None = None,
) -> float | None:
    """Compute weighted overall score combining all three scorers.

    Derives category weights from the problem registry configuration to ensure
    consistency with the evaluation settings. If problem_type is not provided,
    falls back to beams2d weights.

    Category weights typically include:
    - design_quality: (IoU, pixel_accuracy, constraint, objective, connectivity, watertightness)
    - tool_efficiency: (efficiency_ratio)
    - task_completion: (success_rate)

    Note: Printability metrics (connectivity, watertightness) are now part of design_quality.
    Tool ordering is NOT scored as multiple valid orderings exist.

    Args:
        output_quality: Output quality scorer results
        task_completion: Task completion scorer results
        tool_use: Tool use scorer results
        problem_type: Problem type identifier (e.g., 'beams2d', 'photonics2d')

    Returns:
        Combined weighted score [0.0, 1.0], or None if insufficient data
    """
    # Derive category weights from problem registry
    # Fallback to beams2d if problem_type not provided or not found
    try:
        if problem_type:
            problem_config = get_problem_config(problem_type)
            weights = {
                category: category_cfg["weight"]
                for category, category_cfg in problem_config.score_categories.items()
            }
        else:
            # Fallback weights (beams2d)
            weights = {
                "design_quality": 0.65,
                "tool_efficiency": 0.20,
                "task_completion": 0.15,
            }
    except (ValueError, KeyError):
        # If problem config not found or malformed, use beams2d fallback
        weights = {
            "design_quality": 0.65,
            "tool_efficiency": 0.20,
            "task_completion": 0.15,
        }

    category_scores = {}

    # 1. Design Quality / domain-specific primary score
    #    - Workflow (beams2d): output_quality_scorer → "design_quality_score"
    #    - RAG (rag_beams2d): rag_scorer → "rag_benefit_score"  (category: rag_accuracy)
    #    - HPC (hpc_train_beams2d): hpc_workflow_scorer → "hpc_workflow_score"
    #      (category: workflow_completion)
    if isinstance(output_quality, dict):
        dq_score = output_quality.get("design_quality_score")
        if dq_score is not None:
            category_scores["design_quality"] = float(dq_score)
        elif (
            not output_quality.get("design_found", True)
            and isinstance(task_completion, dict)
            and task_completion.get("success_rate", 0) == 1.0
        ):
            # Model correctly abstained from producing a design (e.g., asked for
            # clarification on a natural prompt).  Treat design quality as perfect
            # so the combined score stays comparable across models.
            category_scores["design_quality"] = 1.0

        # RAG evaluation: rag_benefit_score maps to "rag_accuracy" category
        rag_score = output_quality.get("rag_benefit_score")
        if rag_score is not None:
            category_scores["rag_accuracy"] = float(rag_score)

        # HPC evaluation: hpc_workflow_score maps to "workflow_completion" category
        hpc_score = output_quality.get("hpc_workflow_score")
        if hpc_score is not None:
            category_scores["workflow_completion"] = float(hpc_score)

    # 2. Tool Efficiency (from tool_use scorer)
    if isinstance(tool_use, dict):
        efficiency_ratio = tool_use.get("efficiency_ratio")
        if efficiency_ratio is not None:
            category_scores["tool_efficiency"] = float(efficiency_ratio)

    # 3. Task Completion (from task_completion scorer)
    if isinstance(task_completion, dict):
        success_rate = task_completion.get("success_rate")
        if success_rate is not None:
            category_scores["task_completion"] = float(success_rate)

    # If no categories available, return None
    if not category_scores:
        return None

    # Compute weighted sum using category weights from the problem registry.
    # Missing categories contribute zero and do not cause renormalization.
    total_score = 0.0
    for category_name, weight in weights.items():
        category_score = category_scores.get(category_name)
        if category_score is None:
            continue
        total_score += weight * category_score

    return total_score


def _extract_metrics_from_scorers(
    output_quality: dict,
    task_completion: dict,
    tool_use: dict,
    problem_type: str | None = None,
) -> dict:
    """Extract all metrics from scorer outputs.

    Args:
        output_quality: Output quality scorer results
        task_completion: Task completion scorer results
        tool_use: Tool use scorer results
        problem_type: Problem type identifier for weight derivation

    Returns:
        Dictionary of extracted metrics including combined_overall_score
    """
    result = {}

    # Extract category scores (hierarchical score components)
    if isinstance(output_quality, dict):
        result.update(
            {
                "design_found": output_quality.get("design_found", False),
                "design_quality_score": output_quality.get("design_quality_score"),
                # Note: tool_efficiency_score is extracted from tool_use scorer below
                # Design metrics
                "iou": output_quality.get("iou"),
                "pixel_accuracy": output_quality.get("pixel_accuracy"),
                "mse": output_quality.get("mse"),
                "constraint_score": output_quality.get("constraint_score"),
                "objective_score": output_quality.get("objective_score"),
                "constraint_violations": output_quality.get("constraint_violations"),
                "constraint_score_components": output_quality.get(
                    "constraint_score_components"
                ),  # Number of constraints evaluated
                # Printability metrics (use actual field names from scorer)
                "connected_design": output_quality.get("connected_design"),
                "num_components": output_quality.get("num_components"),
                "is_watertight": output_quality.get("is_watertight"),
                "num_vertices": output_quality.get("num_vertices"),
                "num_faces": output_quality.get("num_faces"),
                "volume_mm3": output_quality.get("volume_mm3"),
                "surface_area_mm2": output_quality.get("surface_area_mm2"),
                "watertight_check_available": output_quality.get(
                    "watertight_check_available"
                ),
                "mesh_validation_time": output_quality.get("mesh_validation_time"),
                "comparison_image_generated": output_quality.get(
                    "comparison_image_generated"
                ),
            }
        )

        # Also extract per-constraint partial credit metrics (dynamic fields)
        # These have suffixes: _actual, _target, _error, _normalized_error, _partial_score
        constraint_suffixes = (
            "_actual",
            "_target",
            "_error",
            "_normalized_error",
            "_partial_score",
        )
        result.update(
            {
                key: value
                for key, value in output_quality.items()
                if key.endswith(constraint_suffixes)
                and isinstance(value, (int, float, bool))
            }
        )

        # Extract per-objective metrics (dynamic fields generated by objective scorer)
        # e.g., agent_compliance, target_total_overlap, structural_compliance_score, etc.
        objective_prefixes = ("agent_", "target_")
        objective_suffixes = ("_relative_error", "_score")
        result.update(
            {
                key: value
                for key, value in output_quality.items()
                if (
                    key.startswith(objective_prefixes)
                    or key.endswith(objective_suffixes)
                )
                and key not in result  # Don't overwrite already-extracted fields
                and isinstance(value, (int, float, bool, type(None)))
            }
        )

    # Extract tool efficiency metrics and detailed tool usage
    if isinstance(tool_use, dict):
        # Always extract these standard metrics
        result.update(
            {
                "efficiency_ratio": tool_use.get("efficiency_ratio"),
                "tool_efficiency_score": tool_use.get("efficiency_ratio"),
                "optimal_call_count": tool_use.get("optimal_call_count"),
                "total_tools": tool_use.get("actual_call_count"),
                "excess_calls": tool_use.get("excess_calls"),
                "unique_tools": len(tool_use.get("tool_call_breakdown", {})),
            }
        )

        # Extract individual tool call counts from tool_call_breakdown
        # This preserves the actual number of times each tool was called
        tool_breakdown = tool_use.get("tool_call_breakdown", {})
        if isinstance(tool_breakdown, dict):
            for tool_name, count in tool_breakdown.items():
                # Prefix with "tool_" and store the actual count
                # Plots can convert to binary (>0) if they need usage rates
                result[f"tool_{tool_name}"] = count

        # Also extract any individual tool usage fields (tool_*)
        result.update(
            {
                key: value
                for key, value in tool_use.items()
                if key.startswith("tool_") and isinstance(value, (int, float, bool))
            }
        )

    # Extract task completion metrics
    if isinstance(task_completion, dict):
        result["success_rate"] = task_completion.get("success_rate")
        result["task_completion_score"] = task_completion.get("success_rate")

        # Extract STL parameter validation metrics (workflow-random, workflow-derived-params, etc.)
        stl_validation = task_completion.get("stl_param_validation_score")
        if stl_validation is not None:
            result["stl_param_validation_score"] = stl_validation
            result["stl_param_violations"] = task_completion.get("stl_param_violations")
            # Extract per-parameter details (stl_{param}_{valid,expected,actual,error})
            _stl_internal = {
                "stl_called",
                "stl_success",
                "stl_save_path",
                "stl_error",
                "stl_details",
                "stl_param_validation_score",
            }
            result.update(
                {
                    k: v
                    for k, v in task_completion.items()
                    if k.startswith("stl_") and k not in _stl_internal
                }
            )

        # Extract workflow-conditional branch resolution metrics
        result.update(
            {
                k: v
                for k, v in task_completion.items()
                if k.startswith("conditional_") and isinstance(v, (int, float, str))
            }
        )

        # Extract workflow-multi-export validation metrics
        result.update(
            {
                k: v
                for k, v in task_completion.items()
                if k.startswith(("multi_export_", "export_a_", "export_b_"))
                and isinstance(v, (int, float, bool))
            }
        )

        # Extract HPC training workflow step completion metrics (hpc-train prompts)
        result.update(
            {
                k: v
                for k, v in task_completion.items()
                if k.startswith("hpc_step_") and isinstance(v, (int, float, bool))
            }
        )
        for field in ("hpc_steps_completed", "hpc_steps_total"):
            if field in task_completion:
                result[field] = task_completion[field]

    # Compute true weighted overall score combining all three scorers
    # Use problem_type to derive weights from registry
    result["combined_overall_score"] = _compute_combined_overall_score(
        output_quality, task_completion, tool_use, problem_type
    )

    return result


def _mmore_matches(example, mmore_filter: bool | None) -> bool:
    """Return False if example.metadata.mmore_enabled contradicts mmore_filter."""
    if mmore_filter is None:
        return True
    try:
        ex_meta = example.get("metadata", {}) if hasattr(example, "get") else {}
        mmore_enabled = (ex_meta or {}).get("mmore_enabled")
        if mmore_enabled is None:
            return False  # No tag — exclude from filtered runs to avoid duplicates
        return bool(mmore_enabled) == mmore_filter
    except (AttributeError, TypeError, KeyError) as exc:
        logger.warning("Could not read mmore_enabled from example metadata: %s", exc)
        return True  # Cannot read field — do not filter out


def _prompt_style_matches(example, prompt_style_filter: str | None) -> bool:
    """Return False if example.metadata.prompt_style contradicts prompt_style_filter."""
    if prompt_style_filter is None:
        return True
    try:
        ex_meta = example.get("metadata", {}) if hasattr(example, "get") else {}
        prompt_style = (ex_meta or {}).get("prompt_style")
        if prompt_style is None:
            return False  # No tag — exclude to avoid mixing styles
    except (AttributeError, TypeError, KeyError) as exc:
        logger.warning("Could not read prompt_style from example metadata: %s", exc)
        return True  # Cannot read field — do not filter out
    else:
        return prompt_style == prompt_style_filter


def _process_score_call_for_complete_data(  # noqa: PLR0911, PLR0912, PLR0915
    score_call,
    model_filter: str | None,
    seen_models: set,
    mmore_filter: bool | None = None,
    prompt_style_filter: str | None = None,
) -> dict | None:
    """Process a predict_and_score call and extract ALL data for offline processing.

    Args:
        score_call: The predict_and_score call (has example metadata and scorer outputs)
        model_filter: Optional model filter
        seen_models: Set to track seen models
        mmore_filter: If True, only include calls where example.metadata.mmore_enabled
            is True; if False, only include calls where it is False; None = no filter.
        prompt_style_filter: If set, only include calls where
            example.metadata.prompt_style matches this value; None = no filter.

    Returns:
        Complete design data dict if successful, None otherwise
    """
    try:
        # Extract example data from score_call inputs
        score_inputs = score_call.inputs or {}
        example = score_inputs.get("example", {})

        # Filter by mmore_enabled (stored in inputs.example.metadata.mmore_enabled)
        if not _mmore_matches(example, mmore_filter):
            return None

        # Filter by prompt_style (stored in inputs.example.metadata.prompt_style)
        if not _prompt_style_matches(example, prompt_style_filter):
            return None
        example_id, seed, problem_id = _extract_metadata_from_example(example)

        # Extract scorer outputs from score_call output (scores are nested under "scores")
        score_output = score_call.output
        scores = score_output.get("scores") if isinstance(score_output, dict) else None
        if not scores or not isinstance(scores, dict):
            return None

        output_quality = scores.get("output_quality", {})
        task_completion = scores.get("task_completion", {})
        tool_use = scores.get("tool_use", {})
        rag_evaluation = scores.get("rag_evaluation", {})
        hpc_workflow = scores.get("hpc_workflow", {})

        if (
            not output_quality
            and not task_completion
            and not tool_use
            and not rag_evaluation
            and not hpc_workflow
        ):
            return None

        # Extract model_id and apply filter
        model_id = _extract_model_id_from_score_call(score_call, score_output)
        if model_id:
            seen_models.add(model_id)
        if model_filter and model_id != model_filter:
            return None

        # Extract prompt_style from metadata for result dict
        prompt_style = None
        try:
            ex_meta = example.get("metadata", {}) if hasattr(example, "get") else {}
            prompt_style = (ex_meta or {}).get("prompt_style")
        except (AttributeError, TypeError, KeyError):
            pass

        # Build result dict with metadata
        result = {
            "example_id": example_id,
            "seed": seed,
            "problem_id": problem_id,
            "model_id": model_id,
            "prompt_style": prompt_style,
        }

        # Extract root-level metrics
        result["response_length"] = score_output.get("response_length")
        result["model_latency"] = score_output.get("model_latency")

        # Determine problem_type for weight derivation.
        # Try output_quality first (standard workflows), then example metadata
        # (RAG/HPC where output_quality is empty).
        problem_type = None
        if isinstance(output_quality, dict):
            problem_type = output_quality.get("problem_type")
        if problem_type is None:
            try:
                ex_meta = example.get("metadata", {}) if hasattr(example, "get") else {}
                problem_type = (ex_meta or {}).get("problem_type")
            except (AttributeError, TypeError, KeyError):
                pass

        # For RAG and HPC problems, the primary scorer output lives under a
        # different Weave key (rag_evaluation / hpc_workflow) instead of
        # output_quality.  Merge it so _compute_combined_overall_score can
        # find domain-specific score fields (rag_benefit_score, hpc_workflow_score).
        primary_scorer_output = dict(output_quality) if output_quality else {}
        if isinstance(rag_evaluation, dict) and rag_evaluation:
            primary_scorer_output.update(rag_evaluation)
        if isinstance(hpc_workflow, dict) and hpc_workflow:
            primary_scorer_output.update(hpc_workflow)

        # Extract all metrics from scorers (pass problem_type for weight derivation)
        metrics = _extract_metrics_from_scorers(
            primary_scorer_output, task_completion, tool_use, problem_type
        )
        result.update(metrics)

        # Extract RAG evaluation metrics (rag_beams2d problems)
        if isinstance(rag_evaluation, dict) and rag_evaluation:
            for field in RAG_OUTPUT_FIELDS:
                if field in rag_evaluation:
                    result[field] = rag_evaluation[field]

        # Extract HPC workflow metrics (hpc_train_beams2d problems)
        if isinstance(hpc_workflow, dict) and hpc_workflow:
            for field in HPC_WORKFLOW_OUTPUT_FIELDS:
                if field in hpc_workflow:
                    result[field] = hpc_workflow[field]

        # Extract complete data from output_quality scorer for global metrics
        if isinstance(output_quality, dict):
            # Design array (as list for JSON serialization)
            design = output_quality.get("design")
            result["design"] = design if design is not None else None

            # Design found flag
            result["design_found"] = output_quality.get("design_found", False)

            # Optimization history
            opt_history = output_quality.get("optimization_history")
            result["optimization_history"] = opt_history or []  # type: ignore[assignment]

            # Conditions (constraints, loads, etc.)
            conditions = output_quality.get("conditions")
            result["conditions"] = conditions or {}  # type: ignore[assignment]

            # Problem type
            result["problem_type"] = problem_type

            # Ground truth design (if available)
            gt_design = output_quality.get("gt_design")
            result["gt_design"] = gt_design if gt_design is not None else None

        # Skip if no actual data was extracted
        has_data = any(
            v is not None
            for k, v in result.items()
            if k not in ["example_id", "seed", "problem_id", "model_id"]
        )

    except Exception:
        return None
    else:
        return result if has_data else None


def _find_eval_ids_by_name_pattern(
    client,
    project: str,
    name_pattern: str,
    limit: int = 50,
) -> list[str]:
    """Find Evaluation.evaluate call IDs for runs whose name contains name_pattern.

    evaluate_agent.py wraps evaluation.evaluate() in a @weave.op(name=eval_run_name)
    where eval_run_name embeds the mmore_suffix (e.g. "mmore_on" or "mmore_off").
    The call hierarchy is:
        run_evaluation (op_name has mmore_on/off)
          └─ Evaluation.evaluate
               └─ predict_and_score  ← what we want to filter

    Strategy:
        1. Fetch recent Evaluation.evaluate calls.
        2. For each, look up its parent call (the run_evaluation wrapper) and check
           whether its op_name contains name_pattern.
        3. Return the Evaluation.evaluate call IDs for matching parents.

    Args:
        client: Weave client
        project: Weave project name
        name_pattern: Substring to match (e.g. "mmore_on", "mmore_off")
        limit: Max Evaluation.evaluate calls to scan

    Returns:
        List of Evaluation.evaluate call IDs whose parent op matches the pattern.
    """
    print(f"  Searching for evaluations matching '{name_pattern}'...")

    eval_calls = list(
        client.get_calls(
            filter={"op_names": [f"weave:///{project}/op/Evaluation.evaluate:*"]},
            sort_by=[{"field": "started_at", "direction": "desc"}],
            limit=limit,
        )
    )

    if not eval_calls:
        print("  No Evaluation.evaluate calls found")
        return []

    matched: list[str] = []
    debug_names: list[str] = []

    for call in eval_calls:
        # Check 1: display_name directly on the call (works in some Weave versions)
        display_name = getattr(call, "display_name", "") or ""
        if name_pattern in display_name:
            matched.append(call.id)
            continue

        # Check 2: look up the parent call (run_evaluation wrapper) and check its op_name
        parent_id = getattr(call, "parent_id", None)
        if parent_id:
            try:
                parent_calls = list(
                    client.get_calls(
                        filter={"call_ids": [parent_id]},
                        limit=1,
                    )
                )
                if parent_calls:
                    parent_op = getattr(parent_calls[0], "op_name", "") or ""
                    debug_names.append(parent_op)
                    if name_pattern in parent_op:
                        matched.append(call.id)
                        continue
            except Exception as exc:
                # Parent lookup failures are non-fatal; log at debug level and continue.
                logger.debug(
                    "Failed to fetch parent call %s while searching for evaluations "
                    "matching pattern '%s': %s",
                    parent_id,
                    name_pattern,
                    exc,
                    exc_info=True,
                )

        # Collect display names for debug output when nothing matches
        if display_name:
            debug_names.append(display_name)

    if matched:
        print(f"  Found {len(matched)} matching evaluation(s)")
    else:
        print(f"  No evaluations found matching '{name_pattern}'")
        if debug_names:
            shown = sorted(set(debug_names))[:5]
            print("  Available evaluation names (first 5):")
            for n in shown:
                print(f"    {n[:100]}")
        print("  💡 Tip: pass --eval-id <id> to target a specific evaluation run")
    return matched


def extract_complete_design_data_from_evaluation(  # noqa: PLR0913
    project: str,
    model_filter: str | None = None,
    limit: int = 100,
    eval_id: str | None = None,
    mmore_filter: bool | None = None,
    prompt_style_filter: str | None = None,
) -> list[dict]:
    """Extract complete per-design data from Weave evaluations.

    Args:
        project: Weave project name (e.g., "entity/project")
        model_filter: Optional model ID to filter by (e.g., "openai:gpt-5.1")
        limit: Maximum number of calls to retrieve
        eval_id: Optional evaluation ID (or comma-separated list) to filter by
        mmore_filter: If True/False, only include calls where
            example.metadata.mmore_enabled matches.  Used to separate RAG-on
            (True) from RAG-off (False) runs — the value lives in the Weave UI
            at inputs.example.metadata.mmore_enabled.
        prompt_style_filter: If set, only include calls where
            example.metadata.prompt_style matches this value.  Used to separate
            different prompt styles within a single Weave project.

    Returns:
        List of dictionaries with complete design data (metrics, arrays, histories) per example
    """
    client = weave.init(project)

    # Get predict_and_score calls (individual examples) with scorer outputs
    print(f"Fetching up to {limit} predict_and_score calls from Weave...")
    if mmore_filter is not None:
        print(f"  Filtering by mmore_enabled={mmore_filter} (from example metadata)")
    if prompt_style_filter is not None:
        print(
            f"  Filtering by prompt_style='{prompt_style_filter}' (from example metadata)"
        )

    filter_dict = {
        "op_names": [f"weave:///{project}/op/Evaluation.predict_and_score:*"],
    }

    if eval_id:
        # Support comma-separated list of IDs (e.g. from --eval-name resolution)
        ids = [i.strip() for i in eval_id.split(",") if i.strip()]
        filter_dict["parent_ids"] = ids
        print(f"  Filtering by evaluation ID(s): {ids}")

    score_calls = client.get_calls(
        filter=filter_dict,
        sort_by=[{"field": "started_at", "direction": "desc"}],
        limit=limit,
    )

    score_calls_list = list(score_calls)
    print(f"Found {len(score_calls_list)} predict_and_score call(s)")

    if not score_calls_list:
        print("❌ No predict_and_score calls found")
        return []

    results: list[dict] = []
    seen_models: set[str] = set()

    # Process each predict_and_score call to extract metrics
    for idx, score_call in enumerate(score_calls_list, 1):
        if idx % 50 == 0:
            print(
                f"  Processed {idx}/{len(score_calls_list)} calls, extracted {len(results)} designs..."
            )

        result = _process_score_call_for_complete_data(
            score_call, model_filter, seen_models, mmore_filter, prompt_style_filter
        )
        if result:
            results.append(result)

    print(f"✅ Extracted complete data from {len(results)} designs")
    if len(results) == 0 and seen_models:
        print(f"   Models found in data: {', '.join(sorted(seen_models))}")
        if model_filter:
            print(f"   Filter looking for: {model_filter}")
    return results


def save_complete_design_data_json(
    design_data: list[dict],
    output_path: str,
) -> None:
    """Save complete per-design data to JSON file.

    Args:
        design_data: List of design data dictionaries
        output_path: Path to output JSON file
    """
    if not design_data:
        print("No data to save")
        return

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON with indentation for readability
    with output_file.open("w") as f:
        json.dump(design_data, f, indent=2)

    print(f"✅ Saved to: {output_path}")
    print(f"   Designs: {len(design_data)}")

    # Print data summary
    total_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"   File size: {total_size_mb:.2f} MB")

    # Count designs with optimization histories
    with_histories = sum(
        1
        for d in design_data
        if d.get("optimization_history") and len(d.get("optimization_history", [])) > 0
    )
    print(f"   With optimization histories: {with_histories}/{len(design_data)}")


def print_summary(metrics_data: list[dict]) -> None:
    """Print summary statistics."""
    if not metrics_data:
        print("No data")
        return

    total_designs = len(metrics_data)

    # Calculate average scores (filter and cast to float for type safety)
    combined_overall_scores: list[float] = [
        float(score)
        for d in metrics_data
        if (score := d.get("combined_overall_score")) is not None
    ]
    design_quality_scores: list[float] = [
        float(score)
        for d in metrics_data
        if (score := d.get("design_quality_score")) is not None
    ]
    tool_efficiency_scores: list[float] = [
        float(score)
        for d in metrics_data
        if (score := d.get("tool_efficiency_score")) is not None
    ]
    task_completion_scores: list[float] = [
        float(score)
        for d in metrics_data
        if (score := d.get("task_completion_score")) is not None
    ]
    print("\n" + "=" * 60)
    print(f"Designs: {total_designs}")
    print("\nAverage Scores:")
    if combined_overall_scores:
        print(
            f"  Combined Overall:      {sum(combined_overall_scores) / len(combined_overall_scores):.3f}"
        )
    if design_quality_scores:
        print(
            f"  Design Quality:        {sum(design_quality_scores) / len(design_quality_scores):.3f}"
        )
    if tool_efficiency_scores:
        print(
            f"  Tool Efficiency:       {sum(tool_efficiency_scores) / len(tool_efficiency_scores):.3f}"
        )
    if task_completion_scores:
        print(
            f"  Task Completion:       {sum(task_completion_scores) / len(task_completion_scores):.3f}"
        )
    # Note: Printability metrics are now included in Design Quality score
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract complete per-design data (metrics + arrays + histories) from Weave"
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Weave project (entity/project). Defaults to config.weave_project",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID (openai:gpt-5.1). Defaults to config.llm_model",
    )
    parser.add_argument("--problem", required=True, help="Problem (beams2d)")
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="full",
        choices=[
            "full",
            "natural",
            "workflow",
            "workflow-random",
            "workflow-derived-params",
            "workflow-distractor",
            "workflow-conditional",
            "workflow-multi-export",
            "rag-eval",
            "hpc-train-cgan",
            "hpc-train-diff",
            "hpc-train-natural-cgan",
            "hpc-train-natural-diff",
        ],
        help="Prompt style used (default: full)",
    )
    parser.add_argument(
        "--rag-status",
        type=str,
        default="no_rag",
        choices=["rag", "no_rag"],
        help="RAG status (default: no_rag)",
    )
    parser.add_argument("--eval-id", help="Evaluation ID to filter by")
    parser.add_argument(
        "--eval-name",
        default=None,
        help=(
            "Substring to match against evaluation display names in Weave. "
            "For rag_beams2d this is auto-set from --rag-status (mmore_on / mmore_off)."
        ),
    )
    parser.add_argument("--output", help="Output path")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max calls to fetch (default: 200, increase if model not found)",
    )

    args = parser.parse_args()

    # Use config values if not specified
    project = args.project if args.project is not None else config.weave_project
    model = args.model if args.model is not None else config.llm_model

    print(f"Project: {project} | Model: {model} | Problem: {args.problem}")

    # Resolve --eval-name to a comma-separated list of eval IDs (fallback mechanism).
    # Primary mechanism for rag_beams2d is mmore_filter (reads example metadata directly).
    resolved_eval_id = args.eval_id
    if args.eval_name and not resolved_eval_id:
        tmp_client = weave.init(project)
        matched = _find_eval_ids_by_name_pattern(tmp_client, project, args.eval_name)
        if matched:
            resolved_eval_id = ",".join(matched)

    # For rag_beams2d, filter by mmore_enabled in example metadata (most reliable).
    mmore_filter: bool | None = None
    if args.problem == "rag_beams2d":
        mmore_filter = args.rag_status == "rag"
        print(f"  mmore_filter={mmore_filter} (rag_status='{args.rag_status}')")

    prompt_style_filter = args.prompt_style
    print(f"  prompt_style_filter='{prompt_style_filter}'")

    data = extract_complete_design_data_from_evaluation(
        project, model, args.limit, resolved_eval_id, mmore_filter, prompt_style_filter
    )

    if not data:
        print("\n💡 Tip: Try increasing --limit if this is an older model.")
        print("   Example: --limit 500")
        return

    print_summary(data)

    # Default output path for complete design data (JSON format)
    output_path = (
        args.output
        or f"benchmarks/evaluations/results/models/{model.replace('/', '_').replace(':', '_')}/{args.problem}/{args.prompt_style}/{args.rag_status}/design_data.json"
    )

    save_complete_design_data_json(data, output_path)


if __name__ == "__main__":
    main()
