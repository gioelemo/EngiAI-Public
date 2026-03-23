#!/usr/bin/env python3
"""Compare EngiBench optimization timing for beams2d vs photonics2d.

Runs topology optimization with default parameters on both problem types
and reports timing statistics.

Usage:
    python benchmarks/evaluations/compare_optimization_timing.py
    python benchmarks/evaluations/compare_optimization_timing.py --runs 5
    python benchmarks/evaluations/compare_optimization_timing.py --runs 10 --seed 42
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from engibench.problems.beams2d.v1 import Beams2D
from engibench.problems.photonics2d.v0 import Photonics2D


def run_optimization(problem_class, seed: int = 0, config: dict | None = None):
    """Run a single optimization and return timing + objective info."""
    try:
        problem = problem_class(seed=seed, config=config)
    except TypeError:
        problem = problem_class(seed=seed)

    design, _ = problem.random_design()

    start = time.perf_counter()
    optimized_design, opt_info = problem.optimize(
        starting_point=design, config=config or None
    )
    elapsed = time.perf_counter() - start

    # Get objectives
    objectives = problem.simulate(design=optimized_design, config=config or None)
    obj_dict = {}
    for i, (name, _direction) in enumerate(problem.objectives):
        if i < len(objectives):
            obj_dict[name] = float(objectives[i])

    n_iters = len(opt_info) if opt_info else 0

    return {
        "elapsed_s": elapsed,
        "objectives": obj_dict,
        "n_iterations": n_iters,
        "design_shape": optimized_design.shape,
    }


def run_benchmark(problem_name, problem_class, runs, base_seed, config=None):
    """Run multiple optimizations and collect statistics."""
    results = []
    for i in range(runs):
        seed = base_seed + i
        print(f"  Run {i + 1}/{runs} (seed={seed})...", end="", flush=True)
        result = run_optimization(problem_class, seed=seed, config=config)
        print(f" {result['elapsed_s']:.2f}s")
        results.append(result)

    times = [r["elapsed_s"] for r in results]
    return {
        "problem": problem_name,
        "runs": runs,
        "design_shape": results[0]["design_shape"],
        "times": times,
        "mean_s": np.mean(times),
        "std_s": np.std(times),
        "min_s": np.min(times),
        "max_s": np.max(times),
        "median_s": np.median(times),
        "n_iterations": [r["n_iterations"] for r in results],
        "objectives": [r["objectives"] for r in results],
    }


def print_results(results_list):
    """Print a comparison table."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION TIMING COMPARISON")
    print("=" * 70)

    header = f"{'Problem':<15} {'Shape':<15} {'Mean (s)':<12} {'Std (s)':<12} {'Min (s)':<12} {'Max (s)':<12} {'Runs':<6}"
    print(header)
    print("-" * 70)

    for r in results_list:
        shape_str = str(r["design_shape"])
        print(
            f"{r['problem']:<15} {shape_str:<15} {r['mean_s']:<12.2f} {r['std_s']:<12.2f} "
            f"{r['min_s']:<12.2f} {r['max_s']:<12.2f} {r['runs']:<6}"
        )

    print("-" * 70)

    # Speedup ratio
    _n_problems = 2
    if len(results_list) == _n_problems:
        r0, r1 = results_list
        if r1["mean_s"] > 0:
            ratio = r0["mean_s"] / r1["mean_s"]
            faster = r1["problem"] if ratio > 1 else r0["problem"]
            factor = ratio if ratio > 1 else 1 / ratio
            print(f"\n{faster} is {factor:.1f}x faster than the other problem.")

    # Per-run details
    for r in results_list:
        print(f"\n--- {r['problem']} ---")
        for i, (t, n_iter, obj) in enumerate(
            zip(r["times"], r["n_iterations"], r["objectives"], strict=True)
        ):
            obj_str = ", ".join(f"{k}={v:.4f}" for k, v in obj.items())
            print(f"  Run {i + 1}: {t:.2f}s, {n_iter} iterations, {obj_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare EngiBench optimization timing for beams2d vs photonics2d"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of optimization runs per problem (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base random seed (default: 0)"
    )
    args = parser.parse_args()

    results_list = []

    # Beams2D with defaults
    print(f"\nBeams2D (default parameters, {args.runs} runs):")
    beams_result = run_benchmark("beams2d", Beams2D, args.runs, args.seed)
    results_list.append(beams_result)

    # Photonics2D with defaults
    print(f"\nPhotonics2D (default parameters, {args.runs} runs):")
    photonics_result = run_benchmark("photonics2d", Photonics2D, args.runs, args.seed)
    results_list.append(photonics_result)

    print_results(results_list)


if __name__ == "__main__":
    main()
