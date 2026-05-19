"""
Latency Analysis Plots

Publication-ready box plots showing end-to-end agent latency
across prompt styles, models, and problems.

Usage:
    python benchmarks/evaluations/plots/plot_latency.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.evaluations.plots.utils import (
    FULL_WIDTH,
    save_figure,
    setup_style,
)

# ── Constants ────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "results" / "models"

STYLE_MAP = {
    "full": "Full",
    "natural": "Nat.",
    "workflow-random": "W-Rand",
    "workflow-derived-params": "W-Deriv",
    "workflow-distractor": "W-Dist",
    "workflow-conditional": "W-Cond",
    "workflow-multi-export": "W-Multi",
}

BEAMS_STYLE_ORDER = ["Full", "Nat.", "W-Rand", "W-Deriv", "W-Dist", "W-Cond", "W-Multi"]
PHOTONICS_STYLE_ORDER = ["W-Rand", "W-Dist", "W-Cond"]

# Brighter palette for latency plots (sky blue for Gemini visibility)
LATENCY_PALETTE = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]

# Outlier exclusions: (model_label, prompt_style, problem, seed)
# Qwen3.5-4B W-Distract seed 2 on Beams2D: machine sleep artifact (~4300s)
EXCLUDED_RUNS = [
    ("Qwen3.5-4B-Q8", "workflow-distractor", "beams2d", 2),
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_model_label(model_name: str) -> str:
    """Convert model directory name to display label."""
    if "_" in model_name:
        parts = model_name.split("_", 1)
        if parts[0] in ["openai", "anthropic", "google", "ollama"]:
            model_name = parts[1]
    if "gemini" in model_name.lower():
        model_name = model_name.replace("genai_", "").replace("-preview", "")
        parts = model_name.split("-")
        model_name = "-".join(p.capitalize() if p != "genai" else p for p in parts)
        model_name = model_name.replace("Genai_", "")
    model_name = model_name.replace("gpt-", "GPT-")
    if model_name.lower().startswith("qwen"):
        model_name = model_name.replace("_", "-")
        model_name = "Qwen" + model_name[4:]
        model_name = re.sub(
            r"-(\d+)b", lambda m: f"-{m.group(1)}B", model_name, flags=re.IGNORECASE
        )
        model_name = re.sub(r"-instruct-\d+", "", model_name, flags=re.IGNORECASE)
        model_name = re.sub(r"-q(\d+)[_-]0", r"-Q\\1", model_name, flags=re.IGNORECASE)
    return model_name


def _is_excluded(model_label: str, prompt_style: str, problem: str, seed: int) -> bool:
    """Check if a run should be excluded (e.g., sleep artifacts)."""
    return (model_label, prompt_style, problem, seed) in EXCLUDED_RUNS


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_latency_data(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Load model_latency from all design_data.json files.

    Returns:
        DataFrame with columns: model, problem, prompt_style, category, latency_s
    """
    rows = []
    for f in results_dir.rglob("design_data.json"):
        with f.open() as fh:
            data = json.load(fh)
        rel = f.relative_to(results_dir)
        parts = list(rel.parts)
        model_raw, problem, prompt_style = parts[0], parts[1], parts[2]
        model_label = _get_model_label(model_raw)

        for entry in data:
            lat = entry.get("model_latency")
            if lat is None:
                continue
            seed = entry.get("seed")
            if _is_excluded(model_label, prompt_style, problem, seed):
                continue
            if problem.startswith(("hpc", "rag")):
                continue

            category = (
                problem.replace("2d", "2D")
                .replace("beams", "Beams")
                .replace("photonics", "Photonics")
            )
            rows.append(
                {
                    "model": model_label,
                    "problem": problem,
                    "prompt_style": prompt_style,
                    "category": category,
                    "latency_s": lat,
                }
            )

    return pd.DataFrame(rows)


# ── Plot ─────────────────────────────────────────────────────────────────────


def plot_latency_workflow(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    filename: str = "latency_workflow",
) -> Figure:
    """Box plots of agent latency by prompt style, model, and problem.

    Args:
        df: DataFrame from load_latency_data()
        output_dir: Directory to save figures (None = don't save)
        filename: Base filename (without extension)

    Returns:
        Matplotlib Figure
    """
    workflow_df = df[df["category"].isin(["Beams2D", "Photonics2D"])].copy()
    workflow_df["style_label"] = workflow_df["prompt_style"].map(STYLE_MAP)

    model_order = sorted(workflow_df["model"].unique())
    workflow_df["model"] = pd.Categorical(
        workflow_df["model"], categories=model_order, ordered=True
    )

    boxplot_kw = {
        "palette": LATENCY_PALETTE[: len(model_order)],
        "fliersize": 2,
        "linewidth": 0.8,
        "width": 0.7,
        "medianprops": {"color": "black", "linewidth": 1.2},
        "boxprops": {"edgecolor": "black", "linewidth": 0.5},
        "whiskerprops": {"color": "black", "linewidth": 0.5},
        "capprops": {"color": "black", "linewidth": 0.5},
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_WIDTH, 2.8),
        sharey=False,
        gridspec_kw={"width_ratios": [7, 3]},
    )

    # Beams2D
    beams_df = workflow_df[workflow_df["category"] == "Beams2D"].dropna(
        subset=["style_label", "model"]
    )
    beams_df["style_label"] = pd.Categorical(
        beams_df["style_label"], categories=BEAMS_STYLE_ORDER, ordered=True
    )
    sns.boxplot(
        data=beams_df,
        x="style_label",
        y="latency_s",
        hue="model",
        ax=axes[0],
        **boxplot_kw,
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Latency (s)")
    axes[0].set_title("Beams2D")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(title="", loc="upper left", ncol=2, fontsize=6)

    # Photonics2D
    photonics_df = workflow_df[workflow_df["category"] == "Photonics2D"].dropna(
        subset=["style_label", "model"]
    )
    photonics_df["style_label"] = pd.Categorical(
        photonics_df["style_label"], categories=PHOTONICS_STYLE_ORDER, ordered=True
    )
    sns.boxplot(
        data=photonics_df,
        x="style_label",
        y="latency_s",
        hue="model",
        ax=axes[1],
        **boxplot_kw,
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Latency (s)")
    axes[1].set_title("Photonics2D")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].get_legend().remove()

    fig.tight_layout()

    if output_dir is not None:
        save_figure(fig, f"{filename}.png", output_dir)

    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_style()

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    print("Loading latency data...")
    df = load_latency_data()
    print(f"  {len(df)} data points across {df['model'].nunique()} models")

    print("Generating latency plot...")
    fig = plot_latency_workflow(df, output_dir=output_dir)

    # Also save to paper figures
    paper_dir = (
        Path(_PROJECT_ROOT) / "paper-revision" / "asmeconf" / "figures" / "benchmarks"
    )
    if paper_dir.exists():
        save_figure(fig, "latency_workflow.png", paper_dir)
        print(f"  Also saved to {paper_dir}")

    plt.close("all")
    print("Done!")
