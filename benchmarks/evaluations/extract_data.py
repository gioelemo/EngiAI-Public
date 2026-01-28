"""
Extract tool usage statistics from Weave evaluation traces.

This script fetches traces from Weave and extracts tool call information
for analysis and visualization.
"""

import argparse
import contextlib
import csv
import sys
from collections import Counter
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


def _extract_tokens_and_latency(
    summary: dict,
) -> tuple[int | None, int | None, int | None, float | None]:
    """Extract token usage and latency from call summary.

    Returns:
        Tuple of (total_tokens, prompt_tokens, completion_tokens, latency_ms)
    """
    usage_dict = summary.get("usage", {})

    total_tokens = None
    prompt_tokens = None
    completion_tokens = None

    # Usage is nested by model name, get first available
    if usage_dict:
        for usage_data in usage_dict.values():
            if isinstance(usage_data, dict):
                total_tokens = usage_data.get("total_tokens")
                prompt_tokens = usage_data.get("prompt_tokens")
                completion_tokens = usage_data.get("completion_tokens")
                break

    # Extract latency from summary['weave']['latency_ms']
    latency_ms = None
    weave_info = summary.get("weave", {})
    if isinstance(weave_info, dict):
        latency_ms = weave_info.get("latency_ms")

    return total_tokens, prompt_tokens, completion_tokens, latency_ms


def _process_score_and_predict_call(
    score_call,
    pred_call,
    model_filter: str | None,
    seen_models: set,
) -> dict | None:
    """Process a predict_and_score call and its child predict call.

    Args:
        score_call: The predict_and_score call (has example metadata)
        pred_call: The child predict call (has tool_calls_info)
        model_filter: Optional model filter
        seen_models: Set to track seen models

    Returns:
        Tool usage dict if successful, None otherwise
    """
    try:
        # Extract example data from score_call inputs
        score_inputs = score_call.inputs or {}
        example = score_inputs.get("example", {})
        example_id, seed, problem_id = _extract_metadata_from_example(example)

        # Extract tool usage from predict call output
        pred_output = pred_call.output
        if not pred_output or not isinstance(pred_output, dict):
            return None

        tool_calls_info = pred_output.get("tool_calls_info", [])
        if not tool_calls_info:
            return None

        model_id = pred_output.get("model", "")
        seen_models.add(model_id)

        if model_filter and model_id != model_filter:
            return None

        # Extract token usage and latency from predict call summary
        summary = pred_call.summary or {}
        total_tokens, prompt_tokens, completion_tokens, latency_ms = (
            _extract_tokens_and_latency(summary)
        )

        tool_names = [
            tc.get("name")
            for tc in tool_calls_info
            if isinstance(tc, dict) and tc.get("name")
        ]
        if not tool_names:
            return None

        tool_counter = Counter(tool_names)

        result = {
            "call_id": score_call.id,
            "example_id": example_id,
            "seed": seed,
            "problem_id": problem_id,
            "model_id": model_id,
            "total_tools": len(tool_names),
            "unique_tools": len(tool_counter),
            "tool_counts": dict(tool_counter),
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
        }

        for tool_name, count in tool_counter.items():
            result[f"tool_{tool_name}"] = count

    except Exception:
        return None
    else:
        return result


def _process_prediction_call(
    pred_call,
    model_filter: str | None,
    seen_models: set,
) -> dict | None:
    """Process a single prediction call and extract tool usage.

    Returns:
        Tool usage dict if successful, None otherwise
    """
    try:
        output = pred_call.output
        if not output or not isinstance(output, dict):
            return None

        tool_calls_info = output.get("tool_calls_info", [])
        if not tool_calls_info:
            return None

        inputs = pred_call.inputs or {}

        # Extract example_id and seed from inputs['self'] (evaluation example)
        example_id = None
        seed = None
        if "self" in inputs:
            self_obj = inputs["self"]
            # Try different possible structures
            if isinstance(self_obj, dict):
                example_id = self_obj.get("example_id")
                seed = self_obj.get("seed")
            elif hasattr(self_obj, "example_id"):
                example_id = getattr(self_obj, "example_id", None)
                seed = getattr(self_obj, "seed", None)
            # Also check if it's a Weave object with _val
            elif hasattr(self_obj, "_val"):
                val = self_obj._val
                if isinstance(val, dict):
                    example_id = val.get("example_id")
                    seed = val.get("seed")

        # Extract token usage and latency from summary
        summary = pred_call.summary or {}
        total_tokens, prompt_tokens, completion_tokens, latency_ms = (
            _extract_tokens_and_latency(summary)
        )

        model_id = output.get("model", "")
        seen_models.add(model_id)

        if model_filter and model_id != model_filter:
            return None

        tool_names = [
            tc.get("name")
            for tc in tool_calls_info
            if isinstance(tc, dict) and tc.get("name")
        ]
        if not tool_names:
            return None

        tool_counter = Counter(tool_names)

        result = {
            "call_id": pred_call.id,
            "example_id": example_id,
            "seed": seed,
            "model_id": model_id,
            "total_tools": len(tool_names),
            "unique_tools": len(tool_counter),
            "tool_counts": dict(tool_counter),
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
        }

        for tool_name, count in tool_counter.items():
            result[f"tool_{tool_name}"] = count

    except Exception:
        return None
    else:
        return result


def extract_tool_usage_from_evaluation(
    project: str,
    model_filter: str | None = None,
    limit: int = 100,
    eval_id: str | None = None,
) -> list[dict]:
    """Extract tool usage from Weave evaluations.

    Args:
        project: Weave project name (e.g., "entity/project")
        model_filter: Optional model ID to filter by (e.g., "openai:gpt-5.1")
        limit: Maximum number of calls to retrieve
        eval_id: Optional evaluation ID to fetch predict_and_score calls from

    Returns:
        List of dictionaries with tool usage per example
    """
    client = weave.init(project)

    # Get predict_and_score calls (individual examples) instead of predict calls
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

    # Now for each predict_and_score call, get its child predict call
    for idx, score_call in enumerate(score_calls_list, 1):
        if idx % 50 == 0:
            print(
                f"  Checked {idx}/{len(score_calls_list)} calls, found {len(results)} matching..."
            )

        # Get the child predict call
        predict_calls = client.get_calls(
            filter={"parent_ids": [score_call.id]},
            limit=10,  # Should only be 1-2 child calls
        )

        for pred_call in predict_calls:
            result = _process_score_and_predict_call(
                score_call, pred_call, model_filter, seen_models
            )
            if result:
                results.append(result)
                break  # Only need one predict call per score call

    print(f"✅ Extracted tool usage from {len(results)} calls")
    if len(results) == 0 and seen_models:
        print(f"   Models found in data: {', '.join(sorted(seen_models))}")
        if model_filter:
            print(f"   Filter looking for: {model_filter}")
    return results


def save_tool_usage_csv(
    tool_usage_data: list[dict],
    output_path: str,
    model_id: str,
    problem_id: str,
    seed: int | None = None,
) -> None:
    """Save tool usage statistics to CSV."""
    rows = []
    for data in tool_usage_data:
        row = {
            "example_id": data.get("example_id"),
            "model_id": data.get("model_id", model_id),
            "problem_id": problem_id,
            "seed": data.get("seed", seed),  # Use extracted seed, fallback to arg
            "total_tools": data["total_tools"],
            "unique_tools": data["unique_tools"],
            "total_tokens": data.get("total_tokens"),
            "prompt_tokens": data.get("prompt_tokens"),
            "completion_tokens": data.get("completion_tokens"),
            "latency_ms": data.get("latency_ms"),
            **{
                key: value
                for key, value in data.items()
                if key.startswith("tool_") and key not in ["tool_list", "tool_counts"]
            },
        }

        rows.append(row)

    if not rows:
        print("No data to save")
        return

    all_columns: set[str] = set()
    for row in rows:
        all_columns.update(row.keys())

    fixed_cols = [
        "seed",
        "example_id",
        "model_id",
        "problem_id",
        "total_tools",
        "unique_tools",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "latency_ms",
    ]
    tool_cols = sorted([col for col in all_columns if col.startswith("tool_")])
    columns = fixed_cols + tool_cols

    # Fill missing tool columns with 0
    for row in rows:
        for tool_col in tool_cols:
            if tool_col not in row:
                row[tool_col] = 0

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved to: {output_path}")
    print(f"   Examples: {len(rows)}, Unique tools: {len(tool_cols)}")


def print_summary(tool_usage_data: list[dict]) -> None:
    """Print summary statistics."""
    if not tool_usage_data:
        print("No data")
        return

    total_calls = len(tool_usage_data)
    total_tools = sum(d["total_tools"] for d in tool_usage_data)
    avg_tools = total_tools / total_calls if total_calls > 0 else 0

    all_tools: Counter[str] = Counter()
    for data in tool_usage_data:
        all_tools.update(data["tool_counts"])

    print("\n" + "=" * 60)
    print(
        f"Examples: {total_calls} | Total calls: {total_tools} | Avg: {avg_tools:.1f}"
    )
    print("\nTop tools:")
    for tool_name, count in all_tools.most_common(10):
        pct = (count / total_tools) * 100 if total_tools > 0 else 0
        print(f"  {tool_name:30s}: {count:4d} ({pct:5.1f}%)")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Extract tool usage from Weave")
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
    parser.add_argument("--seed", type=int, help="Seed")
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

    data = extract_tool_usage_from_evaluation(project, model, args.limit, args.eval_id)

    if not data:
        print("\n💡 Tip: Try increasing --limit if this is an older model.")
        print("   Example: --limit 500")

    print_summary(data)

    output_path = (
        args.output
        or f"benchmarks/evaluations/results/models/{model.replace('/', '_').replace(':', '_')}/{args.problem}/data.csv"
    )

    save_tool_usage_csv(data, output_path, model, args.problem, args.seed)


if __name__ == "__main__":
    main()
