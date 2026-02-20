"""
HPC Training Evaluation Plots

Five publication-ready figures for hpc_train_beams2d evaluations:
  1. plot_step_completion_heatmap  -- heatmap: models x workflow steps (completed/not)
  2. plot_workflow_score_bars      -- grouped bars: hpc_workflow_score per model x config
  3. plot_step_completion_rate     -- horizontal bars: step_completion_rate per model
  4. plot_evaluation_metrics       -- grouped bars: IOG/COG/FOG/MMD/DPP/viol per model x config
  5. plot_baseline_comparison      -- agent vs official baseline side-by-side per metric

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
from typing import Any

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
from benchmarks.problems.hpc_train_beams2d.generate_prompts import (  # noqa: E402
    TRAINING_CONFIGS,
)

# ── Step labels (display-friendly) ────────────────────────────────────────────
WORKFLOW_STEPS = [
    "generate_training_command",
    "submit_slurm_job",
    "monitor_job_until_complete",
    "evaluate_model",
]

STEP_SHORT_LABELS = {
    "generate_training_command": "Generate cmd",
    "submit_slurm_job": "Submit job",
    "monitor_job_until_complete": "Monitor job",
    "evaluate_model": "Evaluate",
}

# ── Dynamic config mappings (derived from TRAINING_CONFIGS) ───────────────────
_ALGO_SHORT: dict[str, str] = {"cgan_cnn_2d": "cGAN", "diffusion_2d_cond": "Diff"}

# Config labels from example_id (e.g. "s1 e20 cGAN")
_CONFIG_LABELS: dict[int, str] = {
    i: f"s{c['seed']} e{c['epochs']} {_ALGO_SHORT.get(c['algorithm'], c['algorithm'][:4])}"
    for i, c in enumerate(TRAINING_CONFIGS)
}

# Map example_id → full training config dict
_EXAMPLE_ID_TO_CONFIG: dict[int, dict] = dict(enumerate(TRAINING_CONFIGS))

# Map example_id → actual training seed
_EXAMPLE_ID_TO_TRAINING_SEED: dict[int, int] = {
    i: c["seed"] for i, c in enumerate(TRAINING_CONFIGS)
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

    Rows = model x config, Columns = 4 workflow steps.
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

    Simple overview showing what fraction of the 4 workflow steps each
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


# ── Plot 4: Evaluation Metrics ─────────────────────────────────────────────


EVAL_METRIC_NAMES = [
    "eval_IOG",
    "eval_COG",
    "eval_FOG",
    "eval_MMD",
    "eval_DPP",
    "eval_viol",
]
EVAL_METRIC_LABELS = {
    "eval_IOG": "IOG",
    "eval_COG": "COG",
    "eval_FOG": "FOG",
    "eval_MMD": "MMD",
    "eval_DPP": "DPP",
    "eval_viol": "Viol. Rate",
}


def plot_evaluation_metrics(
    df: pd.DataFrame,
    filename: str = "evaluation_metrics.png",
    output_dir: Path | None = None,
) -> None:
    """Grouped bar chart of EngiOpt evaluation metrics per model x config.

    Shows IOG, COG, FOG, MMD, DPP, violation rate extracted from
    evaluate_cgan_2d.py output.  Uses a 2x3 grid to fit within
    publication full-width dimensions.
    """
    setup_style()
    df = _prepare_data(df)

    available = [m for m in EVAL_METRIC_NAMES if m in df.columns]
    if not available:
        print("  No evaluation metrics data found, skipping")
        return

    models = sorted(df["model_short"].unique())
    configs = sorted(df["example_id"].unique()) if "example_id" in df.columns else [0]
    font_sizes = PLOT_STYLE["font_sizes"]

    n_cols = min(len(available), 3)
    n_rows = (len(available) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=PLOT_STYLE["figsize_full_width_tall"], squeeze=False
    )

    for idx, metric in enumerate(available):
        ax = axes[idx // n_cols, idx % n_cols]
        n_models = len(models)
        n_configs = len(configs)
        bar_width = 0.7 / max(n_models, 1)
        x = np.arange(n_configs)

        for i, model in enumerate(models):
            model_df = df[df["model_short"] == model]
            vals = []
            for cfg in configs:
                subset = (
                    model_df[model_df["example_id"] == cfg]
                    if "example_id" in model_df.columns
                    else model_df
                )
                vals.append(subset[metric].mean() if len(subset) > 0 else 0.0)

            offset = (i - (n_models - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                vals,
                bar_width * 0.9,
                label=model,
                color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                alpha=PLOT_STYLE["alpha"],
            )

        config_labels = [_CONFIG_LABELS.get(c, f"Config {c}") for c in configs]
        ax.set_xticks(x)
        ax.set_xticklabels(
            config_labels, fontsize=font_sizes["tick_label"], rotation=30, ha="right"
        )
        ax.set_ylabel(EVAL_METRIC_LABELS.get(metric, metric))
        ax.grid(True, axis="y", alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=font_sizes["legend"], loc="upper right")

    # Hide unused subplots
    for idx in range(len(available), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.close(fig)


# ── Plot 5: Agent vs Baseline Comparison ─────────────────────────────────────

# Baseline CSV directory
_BASELINE_DIR = (
    PROJECT_ROOT / "benchmarks" / "problems" / "hpc_train_beams2d" / "data" / "baseline"
)

# Metrics to compare (column names in the CSV files)
_COMPARE_METRICS = ["IOG", "COG", "FOG", "MMD", "DPP", "viol"]

# All seeds have official baselines (cgan_cnn_2d at this epoch count).
_BASELINE_EPOCHS = 100
_BASELINE_ALGORITHM = "cgan_cnn_2d"
_BASELINE_SEEDS = {1, 2, 3}


def _load_baseline_csvs(baseline_dir: Path) -> pd.DataFrame | None:
    """Load all baseline metric CSVs from the given directory."""
    patterns = ["seed*_metrics.csv", "cgan_*_metrics.csv"]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(baseline_dir.rglob(pattern))
    found = sorted(set(found))
    if not found:
        return None
    dfs = []
    for p in found:
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else None


def _extract_agent_metrics(df: pd.DataFrame) -> pd.DataFrame | None:
    """Extract per-config agent metrics from the eval_* columns.

    Uses TRAINING_CONFIGS to resolve (seed, epochs, algorithm) from example_id.
    """
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        eid = row.get("example_id", 0)
        cfg = _EXAMPLE_ID_TO_CONFIG.get(eid, {})
        entry: dict[str, Any] = {
            "seed": cfg.get("seed", eid),
            "epochs": cfg.get("epochs", 0),
            "algorithm": cfg.get("algorithm", "unknown"),
            "example_id": eid,
            "model": row.get("model_short", row.get("model_id", "unknown")),
        }
        for m in _COMPARE_METRICS:
            val = row.get(f"eval_{m}")
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                entry[m] = float(val)
        _min_fields = 5  # seed + epochs + algorithm + example_id + model
        if len(entry) > _min_fields:
            rows.append(entry)
    return pd.DataFrame(rows) if rows else None


def _extract_baseline_metrics(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-seed baseline metrics from CSV data."""
    rows: list[dict[str, Any]] = []
    seeds = (
        sorted(baseline_df["seed"].unique()) if "seed" in baseline_df.columns else [0]
    )
    for seed in seeds:
        seed_df = (
            baseline_df[baseline_df["seed"] == seed]
            if "seed" in baseline_df.columns
            else baseline_df
        )
        entry: dict[str, Any] = {"seed": int(seed)}
        for m in _COMPARE_METRICS:
            col = (
                m
                if m in seed_df.columns
                else (m.lower() if m.lower() in seed_df.columns else None)
            )
            if col and not seed_df[col].dropna().empty:
                entry[m] = float(seed_df[col].mean())
        rows.append(entry)
    return pd.DataFrame(rows)


def _get_seed_val(frame: pd.DataFrame, seed: int, metric: str) -> float:
    """Get metric value for a seed, returning 0.0 if missing."""
    row = frame[frame["seed"] == seed]
    if len(row) == 0 or metric not in row.columns or row[metric].isna().all():
        return 0.0
    return float(row[metric].iloc[0])


def _plot_baseline_group(
    metrics: list[str],
    agent_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    filename: str,
    output_dir: Path | None,
) -> None:
    """Plot a 1x3 baseline comparison panel for the given metrics.

    Filters agent results to 100-epoch cgan_cnn_2d configs (comparable to
    the baseline) and groups by seed for a fair side-by-side comparison.
    """
    font_sizes = PLOT_STYLE["font_sizes"]

    # Filter agent data to comparable configs (100 epochs, cgan_cnn_2d)
    comparable = agent_metrics
    if "epochs" in comparable.columns:
        comparable = comparable[comparable["epochs"] == _BASELINE_EPOCHS]
    if "algorithm" in comparable.columns:
        comparable = comparable[comparable["algorithm"] == _BASELINE_ALGORITHM]

    if comparable.empty:
        print("  No 100-epoch cgan_cnn_2d agent data for baseline comparison")
        return

    models = sorted(comparable["model"].unique().tolist())
    all_seeds = sorted(comparable["seed"].unique().tolist())
    n_bars = len(models) + 1  # +1 for baseline
    bar_w = 0.7 / max(n_bars, 1)

    fig, axes = plt.subplots(
        1, len(metrics), figsize=PLOT_STYLE["figsize_full_width"], squeeze=False
    )

    for col_idx, metric in enumerate(metrics):
        ax = axes[0, col_idx]
        x = np.arange(len(all_seeds))

        for i, model in enumerate(models):
            model_df = comparable[comparable["model"] == model]
            vals = [_get_seed_val(model_df, s, metric) for s in all_seeds]
            offset = (i - (n_bars - 1) / 2) * bar_w
            ax.bar(
                x + offset,
                vals,
                bar_w * 0.9,
                label=model,
                color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                alpha=PLOT_STYLE["alpha"],
            )

        # Baseline bar (only for seeds with a matching official model)
        b_vals = [
            _get_seed_val(baseline_metrics, s, metric) if s in _BASELINE_SEEDS else 0.0
            for s in all_seeds
        ]
        b_offset = (len(models) - (n_bars - 1) / 2) * bar_w
        ax.bar(
            x + b_offset,
            b_vals,
            bar_w * 0.9,
            label="Baseline (100 ep)",
            color=COLOR_PALETTE[len(models) % len(COLOR_PALETTE)],
            alpha=0.50,
            hatch="//",
        )

        seed_labels = [f"seed {s}" for s in all_seeds]
        ax.set_xticks(x)
        ax.set_xticklabels(
            seed_labels, fontsize=font_sizes["tick_label"], rotation=30, ha="right"
        )
        ax.set_ylabel(EVAL_METRIC_LABELS.get(f"eval_{metric}", metric))
        ax.grid(True, axis="y", alpha=0.3)
        if col_idx == 0:
            ax.legend(fontsize=font_sizes["legend"], loc="upper right")

    fig.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.close(fig)


def plot_baseline_comparison(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    baseline_dir: Path | None = None,
) -> None:
    """Side-by-side grouped bars: agent-trained model vs official baseline.

    Produces two separate figures to fit publication column width:
      - baseline_comparison_quality.png  (IOG, COG, FOG)
      - baseline_comparison_stats.png    (MMD, DPP, viol)
    """
    setup_style()
    df = _prepare_data(df)

    bdir = baseline_dir or _BASELINE_DIR
    baseline_df = _load_baseline_csvs(bdir)
    if baseline_df is None or baseline_df.empty:
        print(f"  No baseline CSVs found in {bdir}, skipping baseline comparison")
        return

    agent_metrics = _extract_agent_metrics(df)
    if agent_metrics is None:
        print(
            "  No agent evaluation metrics found in data, skipping baseline comparison"
        )
        return

    baseline_metrics = _extract_baseline_metrics(baseline_df)

    available = [
        m
        for m in _COMPARE_METRICS
        if m in agent_metrics.columns or m in baseline_metrics.columns
    ]
    if not available:
        print("  No overlapping metrics between agent and baseline, skipping")
        return

    # Split into quality metrics (IOG, COG, FOG) and statistical metrics (MMD, DPP, viol)
    quality = [m for m in ["IOG", "COG", "FOG"] if m in available]
    stats = [m for m in ["MMD", "DPP", "viol"] if m in available]

    if quality:
        _plot_baseline_group(
            quality,
            agent_metrics,
            baseline_metrics,
            "baseline_comparison_quality.png",
            output_dir,
        )
    if stats:
        _plot_baseline_group(
            stats,
            agent_metrics,
            baseline_metrics,
            "baseline_comparison_stats.png",
            output_dir,
        )


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

    print("\n[1/6] Step completion heatmap...")
    plot_step_completion_heatmap(df, output_dir=output_dir)

    print("\n[2/6] Workflow score bars...")
    plot_workflow_score_bars(df, output_dir=output_dir)

    print("\n[3/6] Step completion rate...")
    plot_step_completion_rate(df, output_dir=output_dir)

    print("\n[4/6] Evaluation metrics...")
    plot_evaluation_metrics(df, output_dir=output_dir)

    print("\n[5-6/6] Baseline comparison (quality + stats)...")
    plot_baseline_comparison(df, output_dir=output_dir)


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
