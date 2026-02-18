"""
HPC Training Evaluation Plots

Four publication-ready figures for hpc_train_beams2d evaluations:
  1. plot_step_completion_heatmap  -- heatmap: models x 6 workflow steps (completed/not)
  2. plot_workflow_score_bars      -- grouped bars: hpc_workflow_score per model x config
  3. plot_step_completion_rate     -- horizontal bars: step_completion_rate per model
  4. plot_compliance_comparison    -- box/strip plot: compliance values per model x condition

Usage:
    # 1. Extract results from Weave:
    #    python benchmarks/evaluations/extract_data.py --problem hpc_train_beams2d --prompt-style hpc-train --rag-status no_rag
    # 2. Generate plots:
    python run_all.py --problem hpc_train_beams2d --prompt-style hpc-train
    # Or standalone:
    python plot_hpc_training.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.evaluations.plots.utils import (  # noqa: E402
    COLOR_PALETTE,
    PLOT_STYLE,
    get_problem_prompt_output_dir,
    save_figure,
    setup_style,
)

# ── Step labels (display-friendly) ────────────────────────────────────────────
WORKFLOW_STEPS = [
    "generate_training_command",
    "submit_slurm_job",
    "monitor_job_until_complete",
    "download_wandb_model",
    "sample_designs_from_model",
    "simulate_design",
]

STEP_SHORT_LABELS = {
    "generate_training_command": "Generate cmd",
    "submit_slurm_job": "Submit job",
    "monitor_job_until_complete": "Monitor job",
    "download_wandb_model": "Download model",
    "sample_designs_from_model": "Sample designs",
    "simulate_design": "Simulate",
}

# Config labels from example_id
_CONFIG_LABELS = {
    0: "seed=1, ep=20",
    1: "seed=2, ep=50",
    2: "seed=3, ep=100",
}


def _short_model(model_id: str) -> str:
    """Return a short display name for a model."""
    return model_id.rsplit(":", maxsplit=1)[-1] if ":" in model_id else model_id


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and add derived fields."""
    df = df.copy()
    if "model_id" in df.columns and "model" not in df.columns:
        df["model"] = df["model_id"]
    if "model" in df.columns:
        df["model_short"] = df["model"].apply(_short_model)
    if "example_id" in df.columns:
        df["config_label"] = df["example_id"].map(_CONFIG_LABELS).fillna("Unknown")
    return df


# ── Plot 1: Step Completion Heatmap ──────────────────────────────────────────


def plot_step_completion_heatmap(
    df: pd.DataFrame,
    filename: str = "step_completion_heatmap.png",
    output_dir: Path | None = None,
) -> None:
    """Heatmap showing which workflow steps each model completed per config.

    Rows = model x config, Columns = 6 workflow steps.
    Green = completed, Red = not completed.
    """
    setup_style()
    df = _prepare_data(df)

    step_cols = [f"step_{s}" for s in WORKFLOW_STEPS]
    available_cols = [c for c in step_cols if c in df.columns]
    if not available_cols:
        print("  No step completion data found, skipping heatmap")
        return

    # Build matrix: rows = (model, config), cols = steps
    group_cols = ["model_short"]
    if "config_label" in df.columns:
        group_cols.append("config_label")

    grouped = df.groupby(group_cols, sort=False)[available_cols].mean()

    # Create row labels
    if len(group_cols) > 1:
        row_labels = [f"{m} | {c}" for m, c in grouped.index]
    else:
        row_labels = list(grouped.index)

    col_labels = [
        STEP_SHORT_LABELS.get(c.replace("step_", ""), c) for c in available_cols
    ]

    n_rows = len(row_labels)
    fig_height = max(1.8, 0.35 * n_rows + 0.8)
    fig, ax = plt.subplots(figsize=(PLOT_STYLE["figsize_full_width"][0], fig_height))

    # Custom colormap: red (0) → green (1)
    cmap = sns.color_palette(["#d9534f", "#f0ad4e", "#5cb85c"], as_cmap=True)
    sns.heatmap(
        grouped.values,
        annot=True,
        fmt=".0%",
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": "Completion rate", "shrink": 0.6},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.close(fig)


# ── Plot 2: Workflow Score Bars ──────────────────────────────────────────────


def plot_workflow_score_bars(
    df: pd.DataFrame,
    filename: str = "workflow_score_bars.png",
    output_dir: Path | None = None,
) -> None:
    """Grouped bar chart: hpc_workflow_score per model, grouped by config.

    One group per training config (example_id), bars = models.
    """
    setup_style()
    df = _prepare_data(df)

    if "hpc_workflow_score" not in df.columns:
        print("  No hpc_workflow_score data found, skipping bars")
        return

    models = sorted(df["model_short"].unique())
    configs = sorted(df["example_id"].unique()) if "example_id" in df.columns else [0]
    n_models = len(models)
    n_configs = len(configs)

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_full_width"])

    bar_width = 0.7 / max(n_models, 1)
    x = np.arange(n_configs)

    for i, model in enumerate(models):
        model_df = df[df["model_short"] == model]
        means = []
        for cfg in configs:
            subset = (
                model_df[model_df["example_id"] == cfg]
                if "example_id" in model_df.columns
                else model_df
            )
            means.append(
                subset["hpc_workflow_score"].mean() if len(subset) > 0 else 0.0
            )

        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            means,
            bar_width * 0.9,
            label=model,
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            alpha=0.85,
        )
        # Add value labels on bars
        _min_label = 0.02
        for bar, val in zip(bars, means, strict=False):
            if val > _min_label:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=PLOT_STYLE["font_sizes"]["annotation"],
                )

    config_labels = [_CONFIG_LABELS.get(c, f"Config {c}") for c in configs]
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.set_ylabel("HPC Workflow Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.close(fig)


# ── Plot 3: Step Completion Rate ─────────────────────────────────────────────


def plot_step_completion_rate(
    df: pd.DataFrame,
    filename: str = "step_completion_rate.png",
    output_dir: Path | None = None,
) -> None:
    """Horizontal bar chart: mean step_completion_rate per model.

    Simple overview showing what fraction of the 6 workflow steps each
    model completes on average across all configs.
    """
    setup_style()
    df = _prepare_data(df)

    if "step_completion_rate" not in df.columns:
        print("  No step_completion_rate data found, skipping")
        return

    model_means = (
        df.groupby("model_short")["step_completion_rate"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=True)
    )

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_single_col_tall"])

    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(model_means))]
    ax.barh(
        range(len(model_means)),
        model_means["mean"],
        xerr=model_means["std"],
        color=colors,
        alpha=0.85,
        capsize=3,
    )
    ax.set_yticks(range(len(model_means)))
    ax.set_yticklabels(model_means.index)
    ax.set_xlabel("Step Completion Rate")
    ax.set_xlim(0, 1.05)

    # Add value labels
    for i, (val, std) in enumerate(
        zip(model_means["mean"], model_means["std"], strict=False)
    ):
        label = f"{val:.0%}"
        if not np.isnan(std) and std > 0:
            label += f" ({std:.0%})"
        ax.text(
            min(val + 0.02, 1.0),
            i,
            label,
            va="center",
            fontsize=PLOT_STYLE["font_sizes"]["annotation"],
        )

    fig.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.close(fig)


# ── Plot 4: Compliance Comparison ────────────────────────────────────────────


def plot_compliance_comparison(
    df: pd.DataFrame,
    filename: str = "compliance_comparison.png",
    output_dir: Path | None = None,
) -> None:
    """Box/strip plot of compliance values per model.

    Each design's compliance value is a point; boxes show the distribution.
    Only generated if compliance_values are present and non-empty.
    """
    setup_style()
    df = _prepare_data(df)

    if "compliance_values" not in df.columns:
        print("  No compliance_values data found, skipping")
        return

    # Explode compliance_values (lists) into individual rows
    rows = []
    for _, row in df.iterrows():
        vals = row.get("compliance_values", [])
        if not isinstance(vals, list) or len(vals) == 0:
            continue
        model = row.get("model_short", "unknown")
        config = row.get("config_label", "unknown")
        rows.extend(
            {"model": model, "config": config, "compliance": float(v)} for v in vals
        )

    if not rows:
        print("  No compliance values extracted, skipping")
        return

    comp_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_full_width"])

    models = sorted(comp_df["model"].unique())
    palette = {m: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, m in enumerate(models)}

    sns.boxplot(
        data=comp_df,
        x="model",
        y="compliance",
        hue="model",
        palette=palette,
        width=0.5,
        linewidth=0.8,
        fliersize=3,
        ax=ax,
        legend=False,
    )
    sns.stripplot(
        data=comp_df,
        x="model",
        y="compliance",
        hue="model",
        palette=palette,
        size=3,
        alpha=0.5,
        jitter=True,
        ax=ax,
        legend=False,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Compliance")
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.close(fig)


# ── Main entrypoint ──────────────────────────────────────────────────────────


def main(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Generate all HPC training evaluation plots.

    Args:
        df: Design-level DataFrame from extract_data.py (with HPC workflow fields).
        output_dir: Directory to save figures.
    """
    if df is None or df.empty:
        print("  No data for HPC training plots")
        return

    print(f"  HPC training data: {len(df)} rows")

    print("\n[1/4] Step completion heatmap...")
    plot_step_completion_heatmap(df, output_dir=output_dir)

    print("\n[2/4] Workflow score bars...")
    plot_workflow_score_bars(df, output_dir=output_dir)

    print("\n[3/4] Step completion rate...")
    plot_step_completion_rate(df, output_dir=output_dir)

    print("\n[4/4] Compliance comparison...")
    plot_compliance_comparison(df, output_dir=output_dir)


if __name__ == "__main__":
    # Standalone usage: load data and generate plots
    from benchmarks.evaluations.plots.utils import (
        filter_by_problem,
        get_combined_design_df,
        load_data,
    )

    setup_style()
    data = load_data()
    combined_design = get_combined_design_df(data)
    hpc_design = filter_by_problem(combined_design, "hpc_train_beams2d")

    if hpc_design is not None and not hpc_design.empty:
        out = get_problem_prompt_output_dir("hpc_train_beams2d", "hpc-train")
        main(hpc_design, out)
    else:
        print("No HPC training data found. Run extract_data.py first.")
