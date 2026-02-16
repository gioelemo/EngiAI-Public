"""
Compute global metrics from extracted design data.

This script reads design_data.json files (created by extract_data.py) and computes
global metrics (MMD, DPP, IOG/COG/FOG, RVC) offline without needing Weave.

Usage:
    python benchmarks/evaluations/compute_global_metrics.py \\
        --problem beams2d \
        --prompt-style workflow \
        --rag-status no_rag

The input path is auto-constructed from model/problem/prompt-style/rag-status.
The dataset is auto-detected from the problem config (e.g., IDEALLab/beams_2d_50_100_v0).
You can override with --dataset if needed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.shared.metrics import (  # noqa: E402
    compute_rvc,
    dpp_diversity,
    mmd,
    optimality_gap,
)
from benchmarks.shared.problem_registry import get_problem_config  # noqa: E402
from benchmarks.shared.utils import get_hf_dataset  # noqa: E402
from config import config  # noqa: E402


def load_design_data(json_path: str) -> list[dict[str, Any]]:
    """Load design data from JSON file.

    Args:
        json_path: Path to design_data.json file

    Returns:
        List of design data dictionaries
    """
    json_file = Path(json_path)
    with json_file.open() as f:
        data = json.load(f)

    print(f"Loaded {len(data)} designs from {json_path}")
    return data


def extract_valid_designs(
    design_data: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[int], list[dict], list[list[dict]], int]:
    """Extract valid designs and associated data.

    Args:
        design_data: List of design data dictionaries

    Returns:
        Tuple of (designs, example_ids, conditions_list, optimization_histories, n_failed)
    """
    designs = []
    example_ids = []
    conditions_list = []
    optimization_histories = []
    n_failed = 0

    for entry in design_data:
        # Check if design was found
        if not entry.get("design_found", False):
            n_failed += 1
            continue

        # Extract design array
        design = entry.get("design")
        if design is None:
            n_failed += 1
            continue

        # Convert to numpy array
        design_array = np.array(design)
        designs.append(design_array)

        # Extract metadata
        example_id = entry.get("example_id", 0)
        example_ids.append(example_id)

        # Extract conditions
        conditions = entry.get("conditions", {})
        conditions_list.append(conditions)

        # Extract optimization history
        opt_history = entry.get("optimization_history", [])
        optimization_histories.append(opt_history)

    print(f"Extracted {len(designs)} valid designs ({n_failed} failed)")
    return designs, example_ids, conditions_list, optimization_histories, n_failed


def compute_mmd_and_dpp(
    generated_designs: list[np.ndarray],
    dataset_name: str,
    sigma: float = 10.0,
    split: str = "test",
) -> tuple[float | None, float | None]:
    """Compute MMD and DPP diversity metrics.

    Args:
        generated_designs: List of generated design arrays
        dataset_name: HuggingFace dataset name for ground truth
        sigma: Kernel bandwidth for MMD and DPP
        split: Dataset split to use (default: "test")

    Returns:
        Tuple of (mmd_value, dpp_value)
    """
    if len(generated_designs) == 0:
        print("⚠️  No valid designs, cannot compute MMD and DPP")
        return None, None

    # Load ground truth designs from dataset
    print(f"Loading ground truth designs from {dataset_name} (split: {split})...")
    try:
        hf_dataset = get_hf_dataset(dataset_name, split=split)
        gt_designs = [
            np.array(hf_dataset[i]["optimal_design"]) for i in range(len(hf_dataset))
        ]
        print(f"Loaded {len(gt_designs)} ground truth designs")
    except Exception as e:
        print(f"❌ Failed to load ground truth dataset: {e}")
        return None, None

    # Stack into batches
    gen_batch = np.stack(generated_designs)
    gt_batch = np.stack(gt_designs)

    print(f"Generated batch shape: {gen_batch.shape}")
    print(f"Ground truth batch shape: {gt_batch.shape}")

    # Compute MMD
    try:
        mmd_value = mmd(gen_batch, gt_batch, sigma=sigma)
        print(f"✅ MMD (sigma={sigma}): {mmd_value:.6f}")
    except Exception as e:
        print(f"❌ Failed to compute MMD: {e}")
        mmd_value = None

    # Compute DPP diversity
    try:
        dpp_value = dpp_diversity(gen_batch, sigma=sigma)
        print(f"✅ DPP Diversity (sigma={sigma}): {dpp_value:.6e}")
    except Exception as e:
        print(f"❌ Failed to compute DPP: {e}")
        dpp_value = None

    return mmd_value, dpp_value


def compute_optimality_gaps(  # noqa: PLR0912, PLR0915
    optimization_histories: list[list[dict]],
    example_ids: list[int],
    dataset_name: str,
    problem_name: str,
    split: str = "test",
) -> tuple[float | None, float | None, float | None]:
    """Compute IOG, COG, FOG metrics from optimization histories.

    Args:
        optimization_histories: List of optimization histories per design
        example_ids: List of example IDs corresponding to designs
        dataset_name: HuggingFace dataset name for reference objectives
        problem_name: Problem name to get objective field names
        split: Dataset split to use (default: "test")

    Returns:
        Tuple of (average_iog, average_cog, average_fog)
    """
    if not optimization_histories or all(not h for h in optimization_histories):
        print("⚠️  No optimization histories found, skipping gap computation")
        return None, None, None

    print(f"Computing optimality gaps from {len(optimization_histories)} histories...")

    # Get problem config to know objective field names
    try:
        problem_config = get_problem_config(problem_name)
        obj_config = problem_config.objectives[0]
    except Exception as e:
        print(f"❌ Failed to get problem config: {e}")
        return None, None, None

    # Load dataset for reference objective values
    try:
        hf_dataset = get_hf_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"❌ Failed to load dataset for reference objectives: {e}")
        return None, None, None

    # Resolve the actual dataset field name by probing a sample row.
    # target_field may differ from the raw dataset key (e.g., target_field="compliance"
    # but dataset uses "c"), so try target_field first, then each alias.
    sample_row = hf_dataset[0]
    dataset_field = obj_config.target_field  # default fallback
    if obj_config.target_field in sample_row:
        dataset_field = obj_config.target_field
    else:
        for alias in obj_config.aliases:
            if alias in sample_row:
                dataset_field = alias
                break
    print(
        f"Using objective field: '{dataset_field}' (from config: '{obj_config.target_field}')"
    )

    iog_list = []
    cog_list = []
    fog_list = []

    # Simple class to mimic OptiStep for compatibility
    class OptiStep:
        def __init__(self, obj_values):
            self.obj_values = obj_values

    for i, opt_history in enumerate(optimization_histories):
        if not opt_history or len(opt_history) == 0:
            continue

        example_id = example_ids[i]

        # Get reference objective value from dataset
        try:
            if example_id >= len(hf_dataset):
                continue
            reference_obj = hf_dataset[example_id].get(dataset_field)
            if reference_obj is None:
                continue
        except Exception:
            continue

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
            continue

        # Compute optimality gaps
        try:
            gaps = optimality_gap(opt_steps, reference_obj)
            iog_list.append(gaps[0])  # Initial optimality gap
            cog_list.append(sum(gaps))  # Cumulative optimality gap
            fog_list.append(gaps[-1])  # Final optimality gap
        except Exception:
            continue

    # Compute averages
    average_iog = float(np.mean(iog_list)) if len(iog_list) > 0 else None
    average_cog = float(np.mean(cog_list)) if len(cog_list) > 0 else None
    average_fog = float(np.mean(fog_list)) if len(fog_list) > 0 else None

    if average_iog is not None:
        print(
            f"✅ Optimality Gaps: IOG={average_iog:.4f}, COG={average_cog:.4f}, FOG={average_fog:.4f}"
        )
        print(f"   (computed from {len(iog_list)} histories)")
    else:
        print("⚠️  No valid optimization histories for gap computation")

    return average_iog, average_cog, average_fog


def compute_rvc_metric(
    generated_designs: list[np.ndarray],
    conditions_list: list[dict],
    example_ids: list[int],
) -> tuple[float | None, dict | None]:
    """Compute RVC (Ratio of Violated Constraints) metric.

    Args:
        generated_designs: List of generated design arrays
        conditions_list: List of condition dictionaries per design
        example_ids: List of example IDs

    Returns:
        Tuple of (rvc_value, rvc_details)
    """
    if len(generated_designs) == 0:
        print("⚠️  No valid designs, cannot compute RVC")
        return None, None

    print("Computing RVC (Ratio of Violated Constraints)...")
    try:
        rvc_value, rvc_details = compute_rvc(
            generated_designs, conditions_list, example_ids
        )
        print(f"✅ RVC: {rvc_value:.4f} ({rvc_value * 100:.2f}% violations)")
    except Exception as e:
        print(f"❌ Failed to compute RVC: {e}")
        return None, None
    else:
        return rvc_value, rvc_details


def save_global_metrics(
    metrics: dict[str, Any],
    output_path: str,
) -> None:
    """Save global metrics to JSON file.

    Args:
        metrics: Dictionary of computed metrics
        output_path: Path to output JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Global metrics saved to: {output_path}")


def main() -> None:  # noqa: PLR0912, PLR0915
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute global metrics from extracted design data"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID (openai:gpt-5.1). Defaults to config.llm_model",
    )
    parser.add_argument(
        "--problem",
        required=True,
        help="Problem type (beams2d, photonics2d, thermoelastic2d)",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="full",
        choices=["full", "approximate", "natural", "workflow", "workflow-random", "workflow-conditional"],
        help="Prompt style used (default: full)",
    )
    parser.add_argument(
        "--rag-status",
        type=str,
        default="no_rag",
        choices=["rag", "no_rag"],
        help="RAG status (default: no_rag)",
    )
    parser.add_argument(
        "--input",
        help="Override input path (default: auto-construct from model/problem/prompt-style/rag-status)",
    )
    parser.add_argument(
        "--dataset",
        help="HuggingFace dataset name (default: auto-detect from problem config, e.g., IDEALLab/beams_2d_50_100_v0)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=10.0,
        help="Kernel bandwidth for MMD and DPP (default: 10.0)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--output",
        help="Output path for global metrics JSON (default: same dir as input)",
    )

    args = parser.parse_args()

    # Use config model if not specified
    model = args.model if args.model is not None else config.llm_model

    # Construct input path if not explicitly provided
    if args.input:
        input_path = args.input
    else:
        input_path = f"benchmarks/evaluations/results/models/{model.replace('/', '_').replace(':', '_')}/{args.problem}/{args.prompt_style}/{args.rag_status}/design_data.json"

    # Auto-detect dataset from problem config if not provided
    if args.dataset is None:
        try:
            problem_config = get_problem_config(args.problem)
            dataset_name = problem_config.dataset_name
            print(f"Auto-detected dataset from problem config: {dataset_name}")
        except Exception as e:
            print(f"❌ Failed to auto-detect dataset for problem '{args.problem}': {e}")
            print("Please specify --dataset explicitly")
            return
    else:
        dataset_name = args.dataset

    print("=" * 60)
    print("COMPUTING GLOBAL METRICS")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Input: {input_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Problem: {args.problem}")
    print(f"Prompt Style: {args.prompt_style}")
    print(f"RAG Status: {args.rag_status}")
    print(f"Sigma: {args.sigma}")
    print("=" * 60)

    # Load design data
    design_data = load_design_data(input_path)

    if not design_data:
        print("❌ No design data found")
        return

    # Extract model name from design data
    model_name = design_data[0].get("model_id", "unknown") if design_data else "unknown"
    print(f"Model: {model_name}")

    # Group designs by seed
    designs_by_seed: dict[int, list[dict]] = {}
    for design in design_data:
        seed = design.get("seed")
        if seed is not None:
            if seed not in designs_by_seed:
                designs_by_seed[seed] = []
            designs_by_seed[seed].append(design)

    print(
        f"Found {len(designs_by_seed)} unique seeds: {sorted(designs_by_seed.keys())}"
    )

    if not designs_by_seed:
        print("❌ No designs with seed information found")
        return

    # Compute metrics per seed
    per_seed_metrics = []

    for seed in sorted(designs_by_seed.keys()):
        print("\n" + "=" * 60)
        print(f"PROCESSING SEED {seed}")
        print("=" * 60)

        seed_designs = designs_by_seed[seed]

        # Extract valid designs for this seed
        (
            generated_designs,
            example_ids,
            conditions_list,
            optimization_histories,
            n_failed,
        ) = extract_valid_designs(seed_designs)

        if not generated_designs:
            print(f"⚠️  No valid designs for seed {seed}, skipping")
            continue

        # Compute MMD and DPP
        print("\n" + "-" * 60)
        print("MMD & DPP DIVERSITY")
        print("-" * 60)
        mmd_value, dpp_value = compute_mmd_and_dpp(
            generated_designs, dataset_name, args.sigma, args.split
        )

        # Compute optimality gaps
        print("\n" + "-" * 60)
        print("OPTIMALITY GAPS (IOG/COG/FOG)")
        print("-" * 60)
        iog, cog, fog = compute_optimality_gaps(
            optimization_histories, example_ids, dataset_name, args.problem, args.split
        )

        # Compute RVC
        print("\n" + "-" * 60)
        print("CONSTRAINT VIOLATIONS (RVC)")
        print("-" * 60)
        rvc_value, rvc_details = compute_rvc_metric(
            generated_designs, conditions_list, example_ids
        )

        # Store metrics for this seed
        seed_metrics = {
            "seed": seed,
            "n_designs": len(generated_designs),
            "n_failed": n_failed,
            "mmd": mmd_value,
            "dpp_diversity": dpp_value,
            "iog": iog,
            "cog": cog,
            "fog": fog,
            "rvc": rvc_value,
            "rvc_details": rvc_details,
        }
        per_seed_metrics.append(seed_metrics)

    # Aggregate results
    global_metrics = {
        "problem": args.problem,
        "dataset": dataset_name,
        "model": model_name,
        "sigma": args.sigma,
        "per_seed_metrics": per_seed_metrics,
    }

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / "global_metrics.json")

    # Save results
    save_global_metrics(global_metrics, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Total designs: {len(design_data)}")
    print(f"Seeds processed: {len(per_seed_metrics)}")
    print("\nPer-seed results:")
    for seed_metrics in per_seed_metrics:
        seed = seed_metrics["seed"]
        n_designs = seed_metrics["n_designs"]
        print(f"\n  Seed {seed} ({n_designs} designs):")
        if seed_metrics.get("mmd") is not None:
            print(f"    MMD: {seed_metrics['mmd']:.6f}")
        if seed_metrics.get("dpp_diversity") is not None:
            print(f"    DPP: {seed_metrics['dpp_diversity']:.6e}")
        if seed_metrics.get("iog") is not None:
            print(
                f"    IOG: {seed_metrics['iog']:.4f} | COG: {seed_metrics['cog']:.4f} | FOG: {seed_metrics['fog']:.4f}"
            )
        if seed_metrics.get("rvc") is not None:
            rvc_val = seed_metrics["rvc"]
            if isinstance(rvc_val, float):
                print(f"    RVC: {rvc_val:.4f} ({rvc_val * 100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
