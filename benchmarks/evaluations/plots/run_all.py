#!/usr/bin/env python3
"""
Generate All Figures

Run this script to generate all visualizations for a specific problem.
Individual plots can also be run separately.

Plots are organized by RAG status to avoid file overwriting:
    figures/{problem}/{prompt_style}/{rag_status}/

Usage:
    python run_all.py --problem beams2d                  # Generate for specific problem (required)
    python run_all.py --problem beams2d --prompt-style full  # Filter by prompt style
    python run_all.py --problem beams2d --rag-status rag     # Filter by RAG status
    python run_all.py --problem beams2d --prompt-style full --rag-status no_rag  # Combine filters
    python plot_dpp_vs_fog.py                            # Generate single figure
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.evaluations.plots.generate_summary_table import create_summary_table
from benchmarks.evaluations.plots.plot_combined_overall_score import (
    plot_combined_overall_score,
)
from benchmarks.evaluations.plots.plot_design_quality import plot_design_quality
from benchmarks.evaluations.plots.plot_dpp_vs_fog import plot_dpp_vs_fog
from benchmarks.evaluations.plots.plot_dpp_vs_mmd import plot_dpp_vs_mmd
from benchmarks.evaluations.plots.plot_hpc_training import (
    main as plot_hpc_training_main,
)
from benchmarks.evaluations.plots.plot_iou_vs_objective import plot_iou_vs_objective
from benchmarks.evaluations.plots.plot_metrics_comparison import plot_metrics_comparison
from benchmarks.evaluations.plots.plot_rag_evaluation import (
    main as plot_rag_evaluation_main,
)
from benchmarks.evaluations.plots.plot_tool_heatmap_counts import (
    plot_tool_heatmap_counts,
)
from benchmarks.evaluations.plots.plot_tool_usage import (
    plot_performance_distribution_by_tool_count,
    plot_tool_heatmap_by_model,
    plot_tool_heatmap_with_std,
    plot_tool_usage_by_model,
    plot_tool_usage_delta_heatmap,
    plot_tool_usage_frequency,
    plot_tool_usage_vs_performance,
)
from benchmarks.evaluations.plots.utils import (
    filter_by_problem,
    filter_by_prompt_style,
    filter_by_rag_status,
    get_combined_design_df,
    get_combined_global_df,
    get_problem_output_dir,
    get_problem_prompt_output_dir,
    load_data,
)
from benchmarks.shared.problem_registry import PROBLEMS


def _generate_global_plots(combined_global, output_dir, problem: str | None = None):
    """Generate global metric plots.

    Args:
        combined_global: DataFrame with global metrics
        output_dir: Directory to save figures
        problem: Optional problem name (if None, generates for all data)
    """
    label = f" [{problem}]" if problem else " [combined]"
    print("\n" + "-" * 40)
    print(f"Generating global metric plots{label}...")
    print("-" * 40)

    if combined_global is None:
        print("  No global metrics data available")
        return

    print(f"  Global metrics: {len(combined_global)} rows")

    # 1. DPP vs FOG
    print("\n[1/4] DPP vs FOG scatter...")
    plot_dpp_vs_fog(combined_global, "dpp_vs_fog.png", output_dir)

    # 2. DPP vs MMD
    print("\n[2/4] DPP vs MMD scatter...")
    plot_dpp_vs_mmd(combined_global, "dpp_vs_mmd.png", output_dir)

    # 3. Metrics comparison bars
    print("\n[3/4] Metrics comparison bars...")
    plot_metrics_comparison(combined_global, "metrics_comparison.png", output_dir)

    # 4. Summary table
    print("\n[4/4] Summary statistics table...")
    create_summary_table(combined_global, output_dir)


def _generate_design_plots(combined_design, output_dir, problem: str | None = None):
    """Generate design-level plots.

    Args:
        combined_design: DataFrame with design metrics
        output_dir: Directory to save figures
        problem: Optional problem name (if None, generates for all data)
    """
    label = f" [{problem}]" if problem else " [combined]"
    print("\n" + "-" * 40)
    print(f"Generating design-level plots{label}...")
    print("-" * 40)

    if combined_design is None:
        print("  No design metrics data available")
        return

    print(f"  Design metrics: {len(combined_design)} rows")

    # Combined overall score distribution
    print("\n[1/3] Combined overall score distribution...")
    plot_combined_overall_score(
        combined_design, "combined_overall_score_distribution.png", output_dir
    )

    # Design quality distribution
    print("\n[2/3] Design quality distribution...")
    plot_design_quality(combined_design, "design_quality_distribution.png", output_dir)

    # IoU vs Objective
    print("\n[3/3] IoU vs Objective score...")
    plot_iou_vs_objective(combined_design, "iou_vs_objective.png", output_dir)


def _generate_tool_usage_plots(
    combined_tools, combined_design, output_dir, problem: str | None = None
):
    """Generate tool usage plots.

    Args:
        combined_tools: DataFrame with tool usage data
        combined_design: DataFrame with design metrics (for correlation plots)
        output_dir: Directory to save figures
        problem: Optional problem name (if None, generates for all data)
    """
    label = f" [{problem}]" if problem else " [combined]"
    print("\n" + "-" * 40)
    print(f"Generating tool usage plots{label}...")
    print("-" * 40)

    if combined_tools is None or len(combined_tools) == 0:
        print("  ⚠️  No tool usage data found. Run extract_data.py first.")
        return

    print(f"  Tool usage: {len(combined_tools)} records")

    # Tool usage frequency
    print("\n[1/8] Tool usage frequency...")
    plot_tool_usage_frequency(combined_tools, output_dir)

    # Tool usage by model
    print("\n[2/8] Tool usage by model...")
    plot_tool_usage_by_model(combined_tools, output_dir)

    # Tool usage heatmap
    print("\n[3/8] Tool usage heatmap...")
    plot_tool_heatmap_by_model(combined_tools, output_dir)

    # Tool usage heatmap with std
    print("\n[4/8] Tool usage heatmap with std...")
    plot_tool_heatmap_with_std(combined_tools, output_dir)

    # Tool usage heatmap with raw counts
    print("\n[5/8] Tool usage heatmap (raw counts)...")
    plot_tool_heatmap_counts(combined_tools, output_dir)

    # Tool usage delta heatmap
    print("\n[6/8] Tool usage delta heatmap...")
    plot_tool_usage_delta_heatmap(combined_tools, output_dir)

    # Performance distribution by tool count
    if combined_design is not None:
        print("\n[7/8] Performance distribution by tool count...")
        plot_performance_distribution_by_tool_count(
            combined_tools, combined_design, output_dir
        )

    # Tool usage vs performance
    if combined_design is not None:
        print("\n[8/8] Tool usage vs performance...")
        plot_tool_usage_vs_performance(combined_tools, combined_design, output_dir)
    else:
        print("\n[7/8] Skipping performance plots (no design data)")


def _generate_plots_for_problem(  # noqa: PLR0913
    problem: str,
    combined_global,
    combined_design,
    combined_tools,
    prompt_style: str | None = None,
    rag_status: str | None = None,
):
    """Generate all plots for a specific problem.

    Args:
        problem: Problem name (e.g., "beams2d")
        combined_global: Full global metrics DataFrame
        combined_design: Full design metrics DataFrame
        combined_tools: Full tool usage DataFrame
        prompt_style: Optional prompt style for output directory
        rag_status: Optional RAG status for output directory (figures/{problem}/{prompt_style}/{rag_status}/)
    """
    print("\n" + "=" * 60)
    print(f"GENERATING PLOTS FOR: {problem.upper()}")
    if rag_status:
        print(f"RAG Status: {rag_status}")
    print("=" * 60)

    # Get problem-specific output directory
    if prompt_style is not None:
        output_dir = get_problem_prompt_output_dir(problem, prompt_style, rag_status)
    else:
        output_dir = get_problem_output_dir(problem)
    print(f"Output directory: {output_dir}")

    # Filter data for this problem
    problem_global = filter_by_problem(combined_global, problem)
    problem_design = filter_by_problem(combined_design, problem)
    problem_tools = filter_by_problem(combined_tools, problem)

    # Filter by RAG status if specified
    if rag_status is not None:
        problem_global = filter_by_rag_status(problem_global, rag_status)
        problem_design = filter_by_rag_status(problem_design, rag_status)
        problem_tools = filter_by_rag_status(problem_tools, rag_status)

    # RAG evaluation problems use a dedicated set of plots
    if problem == "rag_beams2d":
        print("\n" + "-" * 40)
        print("Generating RAG evaluation plots...")
        print("-" * 40)
        if problem_design is not None and not problem_design.empty:
            plot_rag_evaluation_main(problem_design, output_dir)
        else:
            print("  ⚠️  No design data — run extract_data.py first.")
        return

    # HPC training problems use a dedicated set of plots
    if problem == "hpc_train_beams2d":
        print("\n" + "-" * 40)
        print("Generating HPC training evaluation plots...")
        print("-" * 40)
        if problem_design is not None and not problem_design.empty:
            plot_hpc_training_main(
                problem_design, output_dir, prompt_style=prompt_style
            )
        else:
            print("  ⚠️  No design data — run extract_data.py first.")
        # Also generate tool usage plots (useful for HPC workflow analysis)
        _generate_tool_usage_plots(problem_tools, problem_design, output_dir, problem)
        return

    # Generate plots
    _generate_global_plots(problem_global, output_dir, problem)
    _generate_design_plots(problem_design, output_dir, problem)
    _generate_tool_usage_plots(problem_tools, problem_design, output_dir, problem)


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate benchmark visualizations")
    parser.add_argument(
        "--problem",
        type=str,
        choices=list(PROBLEMS.keys()),
        required=True,
        help="Problem to generate plots for (required)",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=[
            "full",
            "natural",
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
        help="Generate plots only for a specific prompt style (saves to figures/{problem}/{style}/)",
    )
    parser.add_argument(
        "--rag-status",
        type=str,
        choices=["rag", "no_rag", "empty_rag"],
        help="Filter by RAG status (rag, no_rag, or empty_rag)",
    )
    return parser.parse_args()


def _load_all_data():
    """Load and combine all data sources.

    Returns:
        Tuple of (combined_global, combined_design, combined_tools) or None if no data.
    """
    print("\nLoading data...")
    data = load_data()

    if not data:
        print("ERROR: No data found! Check file paths.")
        return None

    print("\nData loaded:")
    for key, df in data.items():
        print(f"  {key}: {len(df)} rows")

    combined_global = get_combined_global_df(data)
    combined_design = get_combined_design_df(data)

    # Tool usage data is now included in design data (JSON pipeline)
    combined_tools = combined_design  # Same DataFrame, includes tool metrics

    return combined_global, combined_design, combined_tools


def _apply_prompt_style_filter(
    combined_global, combined_design, combined_tools, prompt_style
):
    """Filter all dataframes by prompt style.

    Returns:
        Tuple of filtered (combined_global, combined_design, combined_tools) or None if empty.
    """
    print(f"\nFiltering by prompt_style: {prompt_style}")
    filtered_global = filter_by_prompt_style(combined_global, prompt_style)
    filtered_design = filter_by_prompt_style(combined_design, prompt_style)
    filtered_tools = filter_by_prompt_style(combined_tools, prompt_style)

    if filtered_global is None and filtered_design is None and filtered_tools is None:
        print(f"WARNING: No data found for prompt_style '{prompt_style}'")
        return None

    return filtered_global, filtered_design, filtered_tools


def _apply_rag_status_filter(
    combined_global, combined_design, combined_tools, rag_status
):
    """Filter all dataframes by RAG status.

    Returns:
        Tuple of filtered (combined_global, combined_design, combined_tools) or None if empty.
    """
    print(f"\nFiltering by rag_status: {rag_status}")
    filtered_global = filter_by_rag_status(combined_global, rag_status)
    filtered_design = filter_by_rag_status(combined_design, rag_status)
    filtered_tools = filter_by_rag_status(combined_tools, rag_status)

    if filtered_global is None and filtered_design is None and filtered_tools is None:
        print(f"WARNING: No data found for rag_status '{rag_status}'")
        return None

    return filtered_global, filtered_design, filtered_tools


def _detect_problems_with_data(combined_global, combined_design, combined_tools):
    """Detect which problems have data in any of the dataframes."""
    problems = set()
    for df in [combined_global, combined_design, combined_tools]:
        if df is not None and "problem" in df.columns:
            problems.update(df["problem"].unique())
    return problems


def _detect_rag_statuses_with_data(combined_global, combined_design, combined_tools):
    """Detect which RAG statuses have data in any of the dataframes."""
    rag_statuses = set()
    for df in [combined_global, combined_design, combined_tools]:
        if df is not None and "rag_status" in df.columns:
            # Filter out None/NaN values (from baselines)
            statuses = df["rag_status"].dropna().unique()
            rag_statuses.update(statuses)
    return rag_statuses


def main():  # noqa: PLR0912, PLR0915
    """Generate all visualizations."""
    args = _parse_args()

    print("=" * 60)
    print("Diversity vs Quality Analysis")
    print("=" * 60)

    # Load data
    data_result = _load_all_data()
    if data_result is None:
        return
    combined_global, combined_design, combined_tools = data_result

    # Apply prompt_style filter if specified
    if args.prompt_style:
        filter_result = _apply_prompt_style_filter(
            combined_global, combined_design, combined_tools, args.prompt_style
        )
        if filter_result is None:
            return
        combined_global, combined_design, combined_tools = filter_result

    # Apply rag_status filter if specified
    if args.rag_status:
        filter_result = _apply_rag_status_filter(
            combined_global, combined_design, combined_tools, args.rag_status
        )
        if filter_result is None:
            return
        combined_global, combined_design, combined_tools = filter_result

    # Detect problems and RAG statuses with data
    problems_with_data = _detect_problems_with_data(
        combined_global, combined_design, combined_tools
    )
    rag_statuses_with_data = _detect_rag_statuses_with_data(
        combined_global, combined_design, combined_tools
    )
    print(f"\nProblems with data: {sorted(problems_with_data)}")
    print(f"RAG statuses with data: {sorted(rag_statuses_with_data)}")

    # Require a specific problem to be specified
    if not args.problem:
        print("\nERROR: --problem argument is required")
        print("Please specify a problem: --problem beams2d")
        print(f"Available problems: {sorted(problems_with_data)}")
        return

    if args.problem not in problems_with_data:
        print(f"WARNING: No data found for problem '{args.problem}'")
        return

    # RAG evaluation problems: generate a single combined plot (both rag + no_rag in one figure)
    # No rag_status subfolder — output goes directly to figures/{problem}/{prompt_style}/
    if args.problem == "rag_beams2d":
        _generate_plots_for_problem(
            args.problem,
            combined_global,
            combined_design,
            combined_tools,
            prompt_style=args.prompt_style,
            rag_status=None,
        )
        if args.prompt_style:
            output_dir = get_problem_prompt_output_dir(args.problem, args.prompt_style)
        else:
            output_dir = get_problem_output_dir(args.problem)
        print("\n" + "=" * 60)
        print(f"DONE! Figures saved to: {output_dir}")
        print("=" * 60)
        return

    # Determine which RAG statuses to generate plots for
    if args.rag_status:
        # User explicitly specified RAG status via command line - use that
        rag_statuses_to_plot = [args.rag_status]
    elif rag_statuses_with_data:
        # Auto-detect RAG statuses from data
        rag_statuses_to_plot = sorted(rag_statuses_with_data)
    else:
        # No RAG data and no filter specified
        rag_statuses_to_plot = []

    # Generate plots
    if rag_statuses_to_plot:
        # Generate separate folders for each RAG status
        for rag_status in rag_statuses_to_plot:
            _generate_plots_for_problem(
                args.problem,
                combined_global,
                combined_design,
                combined_tools,
                prompt_style=args.prompt_style,
                rag_status=rag_status,
            )
            if args.prompt_style:
                output_path = (
                    f"figures/{args.problem}/{args.prompt_style}/{rag_status}/"
                )
            else:
                output_path = f"figures/{args.problem}/{rag_status}/"
            print("\n" + "=" * 60)
            print(f"DONE! Figures saved to: {output_path}")
            print("=" * 60)
    else:
        # No RAG data, generate plots without RAG subfolder
        _generate_plots_for_problem(
            args.problem,
            combined_global,
            combined_design,
            combined_tools,
            prompt_style=args.prompt_style,
            rag_status=None,
        )
        if args.prompt_style:
            output_dir = get_problem_prompt_output_dir(args.problem, args.prompt_style)
        else:
            output_dir = get_problem_output_dir(args.problem)
        print("\n" + "=" * 60)
        print(f"DONE! Figures saved to: {output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
