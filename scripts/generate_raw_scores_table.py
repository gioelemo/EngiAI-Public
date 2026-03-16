"""
Generate a LaTeX table of raw per-metric scores for all models and prompt styles.

This table reports individual sub-scores (IoU, PA, Obj, Constr, Conn, WT,
Tool Eff., Task Compl.) so that readers can reweight as they see fit.

Usage:
    python scripts/generate_raw_scores_table.py [--problem beams2d|photonics2d] [--output path.tex]
"""

import argparse
import json
from pathlib import Path

import numpy as np

# --- Configuration ---
RESULTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "evaluations"
    / "results"
    / "models"
)

MODEL_DISPLAY = {
    "openai_gpt-5-mini": "GPT-5-mini",
    "google_genai_gemini-3-flash-preview": "Gemini-3-Flash",
    "ollama_qwen3_4b-instruct-2507-q8_0": "Qwen3-4B",
    "ollama_qwen3.5_4b-q8_0": "Qwen3.5-4B",
}

MODEL_ORDER = list(MODEL_DISPLAY.keys())

STYLE_DISPLAY = {
    "full": r"\textsc{Full}",
    "natural": r"\textsc{Natural}",
    "workflow-random": r"\textsc{W-Rand}",
    "workflow-derived-params": r"\textsc{W-Derived}",
    "workflow-distractor": r"\textsc{W-Distract}",
    "workflow-conditional": r"\textsc{W-Cond}",
    "workflow-multi-export": r"\textsc{W-Multi}",
}

STYLE_ORDER = list(STYLE_DISPLAY.keys())

# Metrics to extract and their display names
METRICS = [
    ("iou", "IoU"),
    ("pixel_accuracy", "PA"),
    ("objective_score", "Obj"),
    ("constraint_score", "Constr"),
    ("connected_design", "Conn"),  # Boolean -> fraction
    ("is_watertight", "WT"),  # Boolean/None -> fraction
    ("tool_efficiency_score", "Tool Eff."),
    ("task_completion_score", "TC"),
    ("design_quality_score", "DQ"),
    ("combined_overall_score", "CO"),
]


def load_design_data(model_dir: str, problem: str, style: str) -> list[dict] | None:
    """Load design_data.json for a given model/problem/style."""
    # Try no_rag first, then direct path
    for rag_status in ["no_rag", ""]:
        if rag_status:
            path = (
                RESULTS_DIR
                / model_dir
                / problem
                / style
                / rag_status
                / "design_data.json"
            )
        else:
            path = RESULTS_DIR / model_dir / problem / style / "design_data.json"
        if path.exists():
            with path.open() as f:
                return json.load(f)
    return None


def compute_metric_stats(data: list[dict], metric_key: str) -> tuple[float, float, int]:
    """Compute mean and std for a metric, handling booleans and None values.

    Returns (mean, std, count).
    """
    values = []
    for d in data:
        v = d.get(metric_key)
        if v is None:
            continue
        if isinstance(v, bool):
            values.append(1.0 if v else 0.0)
        elif isinstance(v, (int, float)):
            values.append(float(v))
    if not values:
        return float("nan"), float("nan"), 0
    return float(np.mean(values)), float(np.std(values)), len(values)


def format_cell(mean: float, std: float, count: int) -> str:
    """Format a table cell as mean ± std, or --- if no data."""
    if np.isnan(mean) or count == 0:
        return "---"
    return f"{mean:.2f}\\scriptsize{{$\\pm${std:.2f}}}"


def auto_detect_styles(problem: str) -> list[str]:
    """Detect which styles have data for at least one model."""
    available = set()
    for model_dir in MODEL_ORDER:
        problem_dir = RESULTS_DIR / model_dir / problem
        if problem_dir.exists():
            for style_dir in problem_dir.iterdir():
                if style_dir.is_dir() and style_dir.name in STYLE_DISPLAY:
                    available.add(style_dir.name)
    return [s for s in STYLE_ORDER if s in available]


def generate_table(problem: str) -> str:
    """Generate full LaTeX table."""
    styles = auto_detect_styles(problem)
    lines = []

    # Header
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    problem_display = problem.capitalize().replace("2d", "2D")
    lines.append(
        r"\caption[Raw per-metric scores for all models and prompt styles on "
        + problem_display
        + r".]{Raw per-metric scores (mean $\pm$ std) for all models and prompt styles on "
        + problem_display
        + r". Most cells aggregate 15 runs (3 seeds $\times$ 5 samples); a few Qwen3.5-4B configurations have 13--14 runs due to JSON parsing errors. Metric abbreviations are defined in Section~\ref{subsubsec:scoring_methodology}.}"
    )
    lines.append(r"\label{tab:raw_scores_" + problem + "}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    n_metrics = len(METRICS)
    col_spec = "@{}ll" + "r" * n_metrics + "@{}"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Column headers
    metric_headers = " & ".join(f"\\textbf{{{m[1]}}}" for m in METRICS)
    lines.append(r"\textbf{Style} & \textbf{Model} & " + metric_headers + r" \\")
    lines.append(r"\midrule")

    for si, style in enumerate(styles):
        style_label = STYLE_DISPLAY[style]

        for mi, model_dir in enumerate(MODEL_ORDER):
            model_label = MODEL_DISPLAY[model_dir]
            data = load_design_data(model_dir, problem, style)

            if data is None:
                cells = ["---"] * n_metrics
            else:
                cells = []
                for metric_key, _ in METRICS:
                    mean, std, count = compute_metric_stats(data, metric_key)
                    cells.append(format_cell(mean, std, count))

            # Only show style name on first model row
            row_style = style_label if mi == 0 else ""
            row = f"{row_style} & {model_label} & " + " & ".join(cells) + r" \\"
            lines.append(row)

        # Add midrule between styles (but not after last)
        if si < len(styles) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate raw scores LaTeX table")
    parser.add_argument(
        "--problem", default="beams2d", choices=["beams2d", "photonics2d"]
    )
    parser.add_argument("--output", default=None, help="Output .tex file path")
    args = parser.parse_args()

    table = generate_table(args.problem)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table)
        print(f"Written to {output_path}")
    else:
        print(table)


if __name__ == "__main__":
    main()
