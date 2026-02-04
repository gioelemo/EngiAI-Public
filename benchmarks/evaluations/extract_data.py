"""
Extract complete per-design data from Weave evaluation traces.

This script fetches evaluation results from Weave and extracts ALL scorer outputs
including metrics, design arrays, optimization histories, and metadata.
Saves to JSON format for offline global metrics computation.
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

import weave

# Add project root to path to import config
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import config  # noqa: E402

# Constants
REF_EXTRA_MIN_LENGTH = 3


def _extract_metadata_from_example(
    example,
) -> tuple[int | None, int | None, str | None]:
    """Extract example_id, seed, and problem_id from example object.

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
                seed = metadata.get("seed")

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


def _extract_metrics_from_scorers(
    output_quality: dict,
    task_completion: dict,
    tool_use: dict,
) -> dict:
    """Extract all metrics from scorer outputs."""
    result = {}

    # Extract category scores (hierarchical score components)
    if isinstance(output_quality, dict):
        result.update(
            {
                "design_found": output_quality.get("design_found", False),
                "overall_score": output_quality.get("score"),
                "design_quality_score": output_quality.get("design_quality_score"),
                "tool_efficiency_score": output_quality.get("tool_efficiency_score"),
                "task_completion_score": output_quality.get("task_completion_score"),
                "printability_score": output_quality.get("printability_score"),
                # Design metrics
                "iou": output_quality.get("iou"),
                "pixel_accuracy": output_quality.get("pixel_accuracy"),
                "mse": output_quality.get("mse"),
                "ssim": output_quality.get("ssim"),
                "constraint_score": output_quality.get("constraint_score"),
                "objective_score": output_quality.get("objective_score"),
                "constraint_violations": output_quality.get("constraint_violations"),
                # Objective metrics (compliance, etc.)
                "agent_compliance": output_quality.get("agent_compliance"),
                "target_compliance": output_quality.get("target_compliance"),
                "compliance_relative_error": output_quality.get(
                    "compliance_relative_error"
                ),
                "compliance_score": output_quality.get("compliance_score"),
                # Printability metrics (use actual field names from scorer)
                "connected_design": output_quality.get("connected_design"),
                "num_components": output_quality.get("num_components"),
                "is_watertight": output_quality.get("is_watertight"),
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

    # Extract tool efficiency metrics and detailed tool usage
    if isinstance(tool_use, dict):
        # Always extract these standard metrics
        result.update(
            {
                "efficiency_ratio": tool_use.get("efficiency_ratio"),
                "sequence_score": tool_use.get("sequence_score"),
                "total_tools": tool_use.get("actual_call_count"),  # Map to total_tools
                "unique_tools": len(
                    tool_use.get("tool_call_breakdown", {})
                ),  # Count unique tools
            }
        )

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

    return result


def _process_score_call_for_complete_data(
    score_call,
    model_filter: str | None,
    seen_models: set,
) -> dict | None:
    """Process a predict_and_score call and extract ALL data for offline processing.

    Args:
        score_call: The predict_and_score call (has example metadata and scorer outputs)
        model_filter: Optional model filter
        seen_models: Set to track seen models

    Returns:
        Complete design data dict if successful, None otherwise
    """
    try:
        # Extract example data from score_call inputs
        score_inputs = score_call.inputs or {}
        example = score_inputs.get("example", {})
        example_id, seed, problem_id = _extract_metadata_from_example(example)

        # Extract scorer outputs from score_call output
        score_output = score_call.output
        if not score_output or not isinstance(score_output, dict):
            return None

        # Scorer outputs are nested under "scores" key
        scores = score_output.get("scores", {})

        # Validate scores exist and have at least one scorer output
        if not scores or not isinstance(scores, dict):
            return None

        output_quality = scores.get("output_quality", {})
        task_completion = scores.get("task_completion", {})
        tool_use = scores.get("tool_use", {})

        if not output_quality and not task_completion and not tool_use:
            return None

        # Extract model_id and apply filter
        model_id = _extract_model_id_from_score_call(score_call, score_output)
        if model_id:
            seen_models.add(model_id)
        if model_filter and model_id != model_filter:
            return None

        # Build result dict with metadata
        result = {
            "example_id": example_id,
            "seed": seed,
            "problem_id": problem_id,
            "model_id": model_id,
        }

        # Extract root-level metrics
        result["response_length"] = score_output.get("response_length")
        result["model_latency"] = score_output.get("model_latency")

        # Extract all metrics from scorers
        metrics = _extract_metrics_from_scorers(
            output_quality, task_completion, tool_use
        )
        result.update(metrics)

        # Extract complete data from output_quality scorer for global metrics
        if isinstance(output_quality, dict):
            # Design array (as list for JSON serialization)
            design = output_quality.get("design")
            result["design"] = design if design is not None else None

            # Design found flag
            result["design_found"] = output_quality.get("design_found", False)

            # Optimization history
            opt_history = output_quality.get("optimization_history")
            result["optimization_history"] = opt_history if opt_history else []  # type: ignore[assignment]

            # Conditions (constraints, loads, etc.)
            conditions = output_quality.get("conditions")
            result["conditions"] = conditions if conditions else {}  # type: ignore[assignment]

            # Problem type
            result["problem_type"] = output_quality.get("problem_type")

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


def extract_complete_design_data_from_evaluation(
    project: str,
    model_filter: str | None = None,
    limit: int = 100,
    eval_id: str | None = None,
) -> list[dict]:
    """Extract complete per-design data from Weave evaluations.

    Args:
        project: Weave project name (e.g., "entity/project")
        model_filter: Optional model ID to filter by (e.g., "openai:gpt-5.1")
        limit: Maximum number of calls to retrieve
        eval_id: Optional evaluation ID to fetch predict_and_score calls from

    Returns:
        List of dictionaries with complete design data (metrics, arrays, histories) per example
    """
    client = weave.init(project)

    # Get predict_and_score calls (individual examples) with scorer outputs
    print(f"Fetching up to {limit} predict_and_score calls from Weave...")

    filter_dict = {
        "op_names": [f"weave:///{project}/op/Evaluation.predict_and_score:*"],
    }

    # If eval_id provided, filter by parent_ids
    if eval_id:
        filter_dict["parent_ids"] = [eval_id]
        print(f"  Filtering by evaluation ID: {eval_id}")

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
            score_call, model_filter, seen_models
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
    overall_scores: list[float] = [
        float(score)
        for d in metrics_data
        if (score := d.get("overall_score")) is not None
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
    printability_scores: list[float] = [
        float(score)
        for d in metrics_data
        if (score := d.get("printability_score")) is not None
    ]

    print("\n" + "=" * 60)
    print(f"Designs: {total_designs}")
    print("\nAverage Scores:")
    if overall_scores:
        print(
            f"  Overall Score:         {sum(overall_scores) / len(overall_scores):.3f}"
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
    if printability_scores:
        print(
            f"  Printability:          {sum(printability_scores) / len(printability_scores):.3f}"
        )
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
        choices=["full", "approximate", "natural", "workflow"],
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

    data = extract_complete_design_data_from_evaluation(
        project, model, args.limit, args.eval_id
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
