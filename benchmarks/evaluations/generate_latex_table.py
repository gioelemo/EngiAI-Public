#!/usr/bin/env python
"""Generate a LaTeX summary table from extracted design_data.json files.

Reads all design_data.json files under results/models/ and produces a compact
LaTeX table with Task Completion (TC) and Combined Overall (CO) scores as
mean ± std, grouped by workflow style and model.

Usage:
    python benchmarks/evaluations/generate_latex_table.py --problem beams2d
    python benchmarks/evaluations/generate_latex_table.py --problem beams2d --rag-status rag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────

# Display order (top = easiest)
STYLE_ORDER = [
    "full",
    "natural",
    "workflow",
    "workflow-random",
    "workflow-derived-params",
    "workflow-distractor",
    "workflow-conditional",
    "workflow-multi-export",
]

STYLE_LABELS = {
    "full": r"\textsc{Full}",
    "natural": r"\textsc{Natural}",
    "workflow": r"\textsc{W-Base}",
    "workflow-random": r"\textsc{W-Rand}",
    "workflow-derived-params": r"\textsc{W-Derived}",
    "workflow-distractor": r"\textsc{W-Distract}",
    "workflow-conditional": r"\textsc{W-Cond}",
    "workflow-multi-export": r"\textsc{W-Multi}",
}

# Display order for model columns (left to right)
MODEL_ORDER = [
    "openai_gpt-5-mini",
    "google_genai_gemini-3-flash-preview",
    "ollama_qwen3_4b-instruct-2507-q8_0",
]

# Short model labels for column headers
MODEL_LABELS = {
    "openai_gpt-5-mini": "GPT-5-mini",
    "google_genai_gemini-3-flash-preview": "Gemini-3-Flash",
    "ollama_qwen3_4b-instruct-2507-q8_0": "Qwen3-4B",
}

RESULTS_DIR = Path(__file__).parent / "results" / "models"

# Scores to include in the table
SCORES = [
    ("task_completion_score", "TC"),
    ("combined_overall_score", "CO"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_scores(
    problem: str, rag_status: str
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Load scores from all design_data.json files.

    Returns:
        Nested dict: model_dir -> style -> score_name -> list[float]
    """
    data: dict[str, dict[str, dict[str, list[float]]]] = {}

    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}", file=sys.stderr)
        return data

    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_key = model_dir.name
        data[model_key] = {}

        for style in STYLE_ORDER:
            json_path = model_dir / problem / style / rag_status / "design_data.json"
            if not json_path.exists():
                continue

            designs = json.loads(json_path.read_text())
            if not designs:
                continue

            data[model_key][style] = {}
            for score_key, _ in SCORES:
                vals = [
                    float(d[score_key])
                    for d in designs
                    if d.get(score_key) is not None
                ]
                data[model_key][style][score_key] = vals

    return data


def _fmt(vals: list[float], bold: bool = False) -> str:
    """Format mean ± std for LaTeX. Always shows ±std for consistent alignment."""
    if not vals:
        return "---"
    mean = np.mean(vals)
    std = np.std(vals)
    text = f"{mean:.2f}\\tiny{{$\\pm${std:.2f}}}"
    if bold:
        return f"\\textbf{{{text}}}"
    return text


def generate_table(problem: str, rag_status: str) -> str:
    """Generate the full LaTeX table string."""
    data = _load_scores(problem, rag_status)

    if not data:
        return "% No data found."

    # Discover models that have data, respecting MODEL_ORDER
    ordered = [m for m in MODEL_ORDER if m in data and any(data[m].get(s) for s in STYLE_ORDER)]
    # Append any models not in MODEL_ORDER (future-proof)
    extra = [m for m in data if m not in MODEL_ORDER and any(data[m].get(s) for s in STYLE_ORDER)]
    models = ordered + extra
    if not models:
        return "% No models with data found."

    # Discover styles that have data for at least one model
    styles = [s for s in STYLE_ORDER if any(data[m].get(s) for m in models)]

    n_score_cols = len(SCORES)
    n_models = len(models)

    # Total data columns = n_models * n_score_cols
    total_data_cols = n_models * n_score_cols

    lines: list[str] = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Workflow evaluation results (mean $\pm$ std). "
        r"TC = Task Completion rate, CO = Combined Overall score. "
        r"\textbf{Bold} = best model per row.}"
    )
    lines.append(r"\label{tab:workflow_results}")
    lines.append(r"\small")
    # Use tabular* with full \textwidth; @{\extracolsep{\fill}} spreads columns
    lines.append(
        r"\begin{tabular*}{\textwidth}"
        r"{@{\extracolsep{\fill}}l"
        + "c" * total_data_cols
        + r"@{}}"
    )
    lines.append(r"\toprule")

    # Header row 1: model names (spanning TC+CO columns each)
    header1_parts = [""]
    for m in models:
        label = MODEL_LABELS.get(m, m.replace("_", r"\_"))
        header1_parts.append(
            f"\\multicolumn{{{n_score_cols}}}{{c}}{{{label}}}"
        )
    lines.append(" & ".join(header1_parts) + r" \\")

    # cmidrule under each model group
    col_pos = 2  # first data column (1-indexed, after style label)
    cmidrule_parts = []
    for _ in models:
        end = col_pos + n_score_cols - 1
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_pos}-{end}}}")
        col_pos = end + 1
    lines.append(" ".join(cmidrule_parts))

    # Header row 2: score abbreviations under each model
    header2_parts = [r"\textbf{Style}"]
    for _ in models:
        for _, abbrev in SCORES:
            header2_parts.append(f"\\textbf{{{abbrev}}}")
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    # Separator after Natural (index 1) to split non-STL from STL styles
    separator_after = {"natural"}

    # Column averages for final row
    col_sums: list[list[float]] = [[] for _ in range(n_models * n_score_cols)]

    for style in styles:
        label = STYLE_LABELS.get(style, style.replace("-", r"\text{-}"))
        row_parts = [label]

        # Collect CO values across models to find the best
        co_means: dict[str, float] = {}
        for m in models:
            vals = data[m].get(style, {}).get("combined_overall_score", [])
            co_means[m] = float(np.mean(vals)) if vals else -1.0
        best_co_model = max(co_means, key=lambda k: co_means[k])

        col_idx = 0
        for m in models:
            is_best = m == best_co_model and co_means[m] > 0
            style_data = data[m].get(style, {})
            for score_key, _ in SCORES:
                vals = style_data.get(score_key, [])
                bold = is_best and score_key == "combined_overall_score"
                row_parts.append(_fmt(vals, bold=bold))
                if vals:
                    col_sums[col_idx].append(float(np.mean(vals)))
                col_idx += 1

        lines.append(" & ".join(row_parts) + r" \\")
        if style in separator_after:
            lines.append(r"\midrule")

    # Average row
    lines.append(r"\midrule")
    avg_parts = [r"\textit{Average}"]
    for col_vals in col_sums:
        if col_vals:
            avg_parts.append(f"\\textit{{{np.mean(col_vals):.2f}}}")
        else:
            avg_parts.append("---")
    lines.append(" & ".join(avg_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX summary table from evaluation results"
    )
    parser.add_argument("--problem", required=True, help="Problem (beams2d)")
    parser.add_argument(
        "--rag-status",
        default="no_rag",
        choices=["rag", "no_rag"],
        help="RAG status (default: no_rag)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .tex file (default: print to stdout)",
    )
    args = parser.parse_args()

    table = generate_table(args.problem, args.rag_status)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(table)
        print(f"Saved to {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()
