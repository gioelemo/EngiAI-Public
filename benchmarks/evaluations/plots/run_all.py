#!/usr/bin/env python3
"""
Generate All Figures

Run this script to generate all visualizations at once.
Individual plots can also be run separately.

Usage:
    python run_all.py                           # Generate all figures (all prompt styles combined)
    python run_all.py --prompt-style full       # Generate only for "full" prompt style
    python run_all.py --problem beams2d         # Generate only for beams2d
    python run_all.py --combined-only           # Generate only combined plots
    python run_all.py --prompt-style full --problem beams2d  # Combine filters
    python plot_dpp_vs_fog.py                   # Generate single figure
"""

import argparse

from generate_summary_table import create_summary_table
from plot_design_quality import plot_design_quality
from plot_dpp_vs_fog import plot_dpp_vs_fog
from plot_dpp_vs_mmd import plot_dpp_vs_mmd
from plot_iou_vs_objective import plot_iou_vs_objective
from plot_metrics_comparison import plot_metrics_comparison
from plot_token_latency import plot_latency, plot_token_usage
from plot_tool_usage import (
    plot_tool_heatmap_by_model,
    plot_tool_usage_by_model,
    plot_tool_usage_frequency,
    plot_tool_usage_vs_performance,
)

from utils import (
    filter_by_problem,
    filter_by_prompt_style,
    get_combined_design_df,
    get_combined_global_df,
    get_combined_tool_usage_df,
    get_output_dir,
    get_problem_output_dir,
    get_problem_prompt_output_dir,
    load_data,
    load_tool_usage_data,
)


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

    # Design quality distribution
    print("\n[1/2] Design quality distribution...")
    plot_design_quality(combined_design, "design_quality_distribution.png", output_dir)

    # IoU vs Objective
    print("\n[2/2] IoU vs Objective score...")
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
    print("\n[1/4] Tool usage frequency...")
    plot_tool_usage_frequency(combined_tools, output_dir)

    # Tool usage by model
    print("\n[2/4] Tool usage by model...")
    plot_tool_usage_by_model(combined_tools, output_dir)

    # Tool usage heatmap
    print("\n[3/4] Tool usage heatmap...")
    plot_tool_heatmap_by_model(combined_tools, output_dir)

    # Tool usage vs performance
    if combined_design is not None:
        print("\n[4/4] Tool usage vs performance...")
        plot_tool_usage_vs_performance(combined_tools, combined_design, output_dir)
    else:
        print("\n[4/4] Skipping tool usage vs performance (no design data)")


def _generate_token_latency_plots(
    combined_tools, output_dir, problem: str | None = None
):
    """Generate token usage and latency plots.

    Args:
        combined_tools: DataFrame with tool usage data (includes token/latency)
        output_dir: Directory to save figures
        problem: Optional problem name (if None, generates for all data)
    """
    label = f" [{problem}]" if problem else " [combined]"
    print("\n" + "-" * 40)
    print(f"Generating token usage and latency plots{label}...")
    print("-" * 40)

    if combined_tools is None or len(combined_tools) == 0:
        print("  ⚠️  No tool usage data found. Run extract_data.py first.")
        return

    print(f"  Tool usage: {len(combined_tools)} records")

    # Token usage plot
    print("\n[1/2] Token usage plot...")
    plot_token_usage(combined_tools, output_dir)

    # Latency plot
    print("\n[2/2] Latency plot...")
    plot_latency(combined_tools, output_dir)


def _generate_plots_for_problem(
    problem: str,
    combined_global,
    combined_design,
    combined_tools,
    prompt_style: str | None = None,
):
    """Generate all plots for a specific problem.

    Args:
        problem: Problem name (e.g., "beams2d")
        combined_global: Full global metrics DataFrame
        combined_design: Full design metrics DataFrame
        combined_tools: Full tool usage DataFrame
        prompt_style: Optional prompt style for output directory (figures/{problem}/{prompt_style}/)
    """
    print("\n" + "=" * 60)
    print(f"GENERATING PLOTS FOR: {problem.upper()}")
    print("=" * 60)

    # Get problem-specific output directory: figures/{problem}/{prompt_style}/ or figures/{problem}/
    if prompt_style is not None:
        output_dir = get_problem_prompt_output_dir(problem, prompt_style)
    else:
        output_dir = get_problem_output_dir(problem)
    print(f"Output directory: {output_dir}")

    # Filter data for this problem
    problem_global = filter_by_problem(combined_global, problem)
    problem_design = filter_by_problem(combined_design, problem)
    problem_tools = filter_by_problem(combined_tools, problem)

    # Generate plots
    _generate_global_plots(problem_global, output_dir, problem)
    _generate_design_plots(problem_design, output_dir, problem)
    _generate_tool_usage_plots(problem_tools, problem_design, output_dir, problem)
    _generate_token_latency_plots(problem_tools, output_dir, problem)


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate benchmark visualizations")
    parser.add_argument(
        "--problem",
        type=str,
        choices=["beams2d", "photonics2d", "thermoelastic2d"],
        help="Generate plots only for a specific problem",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=["full", "approximate", "natural", "workflow"],
        help="Generate plots only for a specific prompt style (saves to figures/{problem}/{style}/)",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Generate only combined plots (skip per-problem)",
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

    tool_data = load_tool_usage_data()
    combined_tools = get_combined_tool_usage_df(tool_data)

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


def _detect_problems_with_data(combined_global, combined_design, combined_tools):
    """Detect which problems have data in any of the dataframes."""
    problems = set()
    for df in [combined_global, combined_design, combined_tools]:
        if df is not None and "problem" in df.columns:
            problems.update(df["problem"].unique())
    return problems


def _generate_all_plots(
    args, combined_global, combined_design, combined_tools, problems_with_data
):
    """Generate all requested plots based on args."""
    base_output_dir = get_output_dir()

    # Generate per-problem plots if not combined-only
    if not args.combined_only:
        for problem in sorted(problems_with_data):
            _generate_plots_for_problem(
                problem,
                combined_global,
                combined_design,
                combined_tools,
                prompt_style=args.prompt_style,
            )

    # Generate combined plots
    label = (
        f"prompt_style: {args.prompt_style}"
        if args.prompt_style
        else "all prompt styles"
    )
    print("\n" + "=" * 60)
    print(f"GENERATING COMBINED PLOTS ({label})")
    print("=" * 60)
    print(f"Output directory: {base_output_dir}")

    _generate_global_plots(combined_global, base_output_dir)
    _generate_design_plots(combined_design, base_output_dir)
    _generate_tool_usage_plots(combined_tools, combined_design, base_output_dir)
    _generate_token_latency_plots(combined_tools, base_output_dir)

    print("\n" + "=" * 60)
    print(f"DONE! All figures saved to: {base_output_dir}")
    if not args.combined_only and problems_with_data:
        problems_list = ", ".join(sorted(problems_with_data))
        if args.prompt_style:
            print(
                f"Problem-specific figures in: {base_output_dir}/<problem>/{args.prompt_style}/"
            )
        else:
            print(f"Problem-specific figures in: {base_output_dir}/<problem>/")
        print(f"  Problems: {problems_list}")
    print("=" * 60)


def main():
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

    # Detect problems with data
    problems_with_data = _detect_problems_with_data(
        combined_global, combined_design, combined_tools
    )
    print(f"\nProblems with data: {sorted(problems_with_data)}")

    # Handle single problem case
    if args.problem:
        if args.problem not in problems_with_data:
            print(f"WARNING: No data found for problem '{args.problem}'")
            return
        _generate_plots_for_problem(
            args.problem,
            combined_global,
            combined_design,
            combined_tools,
            prompt_style=args.prompt_style,
        )
        if args.prompt_style:
            output_dir = get_problem_prompt_output_dir(args.problem, args.prompt_style)
        else:
            output_dir = get_problem_output_dir(args.problem)
        print("\n" + "=" * 60)
        print(f"DONE! Figures saved to: {output_dir}")
        print("=" * 60)
        return

    # Generate all plots
    _generate_all_plots(
        args, combined_global, combined_design, combined_tools, problems_with_data
    )


if __name__ == "__main__":
    main()
