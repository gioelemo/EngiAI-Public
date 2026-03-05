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
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.evaluations.plots.utils import (
    COLOR_PALETTE,
    FULL_WIDTH,
    PLOT_STYLE,
    get_problem_prompt_output_dir,
    save_figure,
    setup_style,
)
from benchmarks.problems.hpc_train_beams2d.generate_prompts import (
    ALGORITHMS,
    PROMPT_STYLES,
    SEEDS,
    TrainingConfig,
    get_training_configs,
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

# ── Config mapping helpers ────────────────────────────────────────────────────
_ALGO_SHORT: dict[str, str] = {"cgan_cnn_2d": "cGAN", "diffusion_2d_cond": "Diff"}


def _build_config_mappings(
    algorithm: str = "cgan_cnn_2d",
) -> tuple[dict[int, str], dict[int, TrainingConfig], dict[int, int]]:
    """Build example_id -> config mappings for the given algorithm.

    Returns (config_labels, example_id_to_config, example_id_to_seed).
    """
    configs = get_training_configs(algorithm)
    config_labels = {i: f"seed {c['seed']}" for i, c in enumerate(configs)}
    example_id_to_config = dict(enumerate(configs))
    example_id_to_seed = {i: c["seed"] for i, c in enumerate(configs)}
    return config_labels, example_id_to_config, example_id_to_seed


def _algorithm_from_style(prompt_style: str | None) -> str:
    """Look up the algorithm encoded in a prompt style, defaulting to cgan."""
    if prompt_style and prompt_style in PROMPT_STYLES:
        return PROMPT_STYLES[prompt_style]["algorithm"]
    return ALGORITHMS[0]


# Default mappings (cgan) — used when no prompt_style is specified
_CONFIG_LABELS, _EXAMPLE_ID_TO_CONFIG, _EXAMPLE_ID_TO_TRAINING_SEED = (
    _build_config_mappings()
)


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

    # Coerce step columns to numeric (they arrive as bool/object from JSON)
    for col in available_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

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

    # Custom colormap: red (0) -> green (1)
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


# ── Plot 1b: Step Completion Heatmap (averaged per model) ─────────────────────


def plot_step_completion_heatmap_avg(
    df: pd.DataFrame,
    filename: str = "step_completion_heatmap_avg.png",
    output_dir: Path | None = None,
) -> None:
    """Heatmap showing average workflow step completion per model.

    Rows = models (averaged across all seeds/configs), Columns = 4 workflow steps.
    Cell values are completion percentages (0-100%).
    """
    setup_style()
    df = _prepare_data(df)

    step_cols = [f"step_{s}" for s in WORKFLOW_STEPS]
    available_cols = [c for c in step_cols if c in df.columns]
    if not available_cols:
        print("  No step completion data found, skipping averaged heatmap")
        return

    for col in available_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Average and std across all seeds per model
    grouped_mean = df.groupby("model_short", sort=False)[available_cols].mean()
    grouped_std = df.groupby("model_short", sort=False)[available_cols].std(ddof=1)

    row_labels = list(grouped_mean.index)
    col_labels = [
        STEP_SHORT_LABELS.get(c.replace("step_", ""), c) for c in available_cols
    ]

    n_rows = len(row_labels)
    fig_height = max(1.8, 0.8 * n_rows + 0.8)
    fig, ax = plt.subplots(figsize=(PLOT_STYLE["figsize_full_width"][0], fig_height))

    # Display as 0-100% with explicit "%" in cell annotations
    mean_vals = grouped_mean.to_numpy() * 100
    std_vals = grouped_std.to_numpy() * 100
    sns.heatmap(
        mean_vals,
        annot=False,
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar=False,
        ax=ax,
    )
    # Manually place mean +/- std annotations (escape % for LaTeX)
    for i in range(mean_vals.shape[0]):
        for j in range(mean_vals.shape[1]):
            m = mean_vals[i, j]
            s = std_vals[i, j]
            ax.text(
                j + 0.5,
                i + 0.38,
                f"{m:.0f}\\%",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
            )
            ax.text(
                j + 0.5,
                i + 0.62,
                f"$\\pm${s:.1f}\\%",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)

    # Place colorbar manually so it spans the full heatmap height.
    # tight_layout first, then read the axes position and create cbar axes.
    fig.tight_layout()
    pos = ax.get_position()
    cax = fig.add_axes((pos.x1 + 0.01, pos.y0, 0.015, pos.height))
    cbar = fig.colorbar(ax.collections[0], cax=cax)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}\\%"))
    cbar.set_label("Avg. completion rate")
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

    # Scale width with number of groups
    fig_w = max(FULL_WIDTH, 0.55 * n_configs)
    fig, ax = plt.subplots(figsize=(fig_w, 3.0))

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
                subset["hpc_workflow_score"].mean() if len(subset) > 0 else float("nan")
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
        # Add value labels on bars (skip when too many models → bars too narrow)
        _max_models_for_labels = 3
        if n_models <= _max_models_for_labels:
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

    # Short labels: just seed number
    config_labels = [str(_EXAMPLE_ID_TO_TRAINING_SEED.get(c, c)) for c in configs]
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.set_xlabel("Seed")
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


def _draw_eval_subplot(ax: plt.Axes, metric: str, ctx: dict) -> None:
    """Draw a single subplot in the evaluation metrics grid."""
    df = ctx["df"]
    models = ctx["models"]
    configs = ctx["configs"]
    n_bars = ctx["n_bars"]
    bar_width = ctx["bar_width"]
    baseline_metrics = ctx["baseline_metrics"]
    algo_short = ctx["algo_short"]
    eid_to_seed = ctx["eid_to_seed"]
    font_sizes = PLOT_STYLE["font_sizes"]
    x = np.arange(len(configs))
    base_metric = metric.replace("eval_", "")

    for i, model in enumerate(models):
        model_df = df[df["model_short"] == model]
        vals = []
        for cfg in configs:
            subset = (
                model_df[model_df["example_id"] == cfg]
                if "example_id" in model_df.columns
                else model_df
            )
            vals.append(subset[metric].mean() if len(subset) > 0 else float("nan"))

        offset = (i - (n_bars - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            vals,
            bar_width * 0.9,
            label=model,
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            alpha=PLOT_STYLE["alpha"],
        )

    # Baseline bars
    if baseline_metrics is not None and base_metric in baseline_metrics.columns:
        b_vals = []
        for cfg in configs:
            seed = eid_to_seed.get(cfg, cfg)
            b_vals.append(_get_seed_val(baseline_metrics, seed, base_metric))
        b_offset = (len(models) - (n_bars - 1) / 2) * bar_width
        ax.bar(
            x + b_offset,
            b_vals,
            bar_width * 0.9,
            label=f"Baseline ({algo_short})",
            color=COLOR_PALETTE[len(models) % len(COLOR_PALETTE)],
            alpha=0.50,
            hatch="//",
        )

    # Short labels: just seed number
    config_labels = [str(eid_to_seed.get(c, c)) for c in configs]
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=font_sizes["tick_label"])
    ax.set_xlabel("Seed", fontsize=font_sizes["axes_label"])
    ax.set_ylabel(EVAL_METRIC_LABELS.get(metric, metric))

    # Use log scale when values span many orders of magnitude
    all_plotted = [
        v
        for model in models
        for cfg in configs
        for v in [
            df[(df["model_short"] == model) & (df["example_id"] == cfg)][metric].mean()
            if "example_id" in df.columns
            else df[df["model_short"] == model][metric].mean()
        ]
        if not np.isnan(v) and v > 0
    ]
    _log_scale_ratio = 1000
    if (
        len(all_plotted) > 1
        and min(all_plotted) > 0
        and max(all_plotted) / min(all_plotted) > _log_scale_ratio
    ):
        ax.set_yscale("log")
        ax.set_ylim(bottom=min(all_plotted) * 0.3)

    ax.grid(True, axis="y", alpha=0.3)


def plot_evaluation_metrics(
    df: pd.DataFrame,
    filename: str = "evaluation_metrics.png",
    output_dir: Path | None = None,
    prompt_style: str | None = None,
) -> None:
    """Grouped bar chart of EngiOpt evaluation metrics per model x config.

    Shows IOG, COG, FOG, MMD, DPP, violation rate extracted from
    the evaluation output, using a dynamically sized grid based on available
    metrics. Includes baselines when available.
    """
    setup_style()
    df = _prepare_data(df)

    available = [m for m in EVAL_METRIC_NAMES if m in df.columns]
    if not available:
        print("  No evaluation metrics data found, skipping")
        return

    # Load baseline data
    algorithm = _algorithm_from_style(prompt_style)
    algo_short = _ALGO_SHORT.get(algorithm, algorithm[:4])
    _, _eid_to_config, eid_to_seed = _build_config_mappings(algorithm)
    baseline_df = _load_baseline_csvs(_BASELINE_DIR)
    baseline_metrics = None
    if baseline_df is not None and not baseline_df.empty:
        baseline_metrics = _extract_baseline_metrics(baseline_df)
        if "algorithm" in baseline_metrics.columns:
            baseline_metrics = baseline_metrics[
                baseline_metrics["algorithm"] == algorithm
            ]
        if baseline_metrics.empty:
            baseline_metrics = None

    models = sorted(df["model_short"].unique())
    configs = sorted(df["example_id"].unique()) if "example_id" in df.columns else [0]

    n_cols = min(len(available), 3)
    n_rows = (len(available) + n_cols - 1) // n_cols
    n_configs = len(configs)
    n_bars = len(models) + (1 if baseline_metrics is not None else 0)
    bar_width = 0.7 / max(n_bars, 1)

    # Scale width with number of groups; extra height for shared legend
    fig_w = max(FULL_WIDTH, 0.55 * n_configs * n_cols / 3)
    fig_h = max(4.5, 2.4 * n_rows + 0.4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    ctx = {
        "df": df,
        "models": models,
        "configs": configs,
        "n_bars": n_bars,
        "bar_width": bar_width,
        "baseline_metrics": baseline_metrics,
        "algo_short": algo_short,
        "eid_to_seed": eid_to_seed,
    }
    for idx, metric in enumerate(available):
        ax = axes[idx // n_cols, idx % n_cols]
        _draw_eval_subplot(ax, metric, ctx)

    # Hide unused subplots
    for idx in range(len(available), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    # Leave 5% at bottom for shared legend
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=PLOT_STYLE["font_sizes"]["legend"],
        loc="lower center",
        ncol=n_bars,
        frameon=True,
    )
    save_figure(fig, filename, output_dir)
    plt.close(fig)


# ── Plot 5: Agent vs Baseline Comparison ─────────────────────────────────────

# Baseline CSV directory
_BASELINE_DIR = (
    Path(__file__).parent.parent.parent
    / "problems"
    / "hpc_train_beams2d"
    / "data"
    / "baseline"
)

# Metrics to compare (column names in the CSV files)
_COMPARE_METRICS = ["IOG", "COG", "FOG", "MMD", "DPP", "viol"]

# Baselines exist for these algorithms across all configured seeds.
_BASELINE_ALGORITHMS = ALGORITHMS
_BASELINE_SEEDS = set(SEEDS)


def _load_baseline_csvs(baseline_dir: Path) -> pd.DataFrame | None:
    """Load all baseline metric CSVs from the given directory."""
    patterns = ["cgan_*_metrics.csv", "diffusion_*_metrics.csv"]
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


def _extract_agent_metrics(
    df: pd.DataFrame,
    example_id_to_config: dict[int, TrainingConfig] | None = None,
) -> pd.DataFrame | None:
    """Extract per-config agent metrics from the eval_* columns.

    Uses example_id_to_config to resolve (seed, epochs, algorithm) from example_id.
    Falls back to the module-level default mappings (cgan) if not provided.
    """
    src = example_id_to_config or _EXAMPLE_ID_TO_CONFIG
    eid_map: dict[int, dict[str, Any]] = {k: dict(v) for k, v in src.items()}
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        eid = row.get("example_id", 0)
        cfg = eid_map.get(eid, {})
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
    """Extract per-(seed, algorithm) baseline metrics from CSV data."""
    rows: list[dict[str, Any]] = []

    # Group by (model_id, seed) to keep algorithms separate
    has_model = "model_id" in baseline_df.columns
    has_seed = "seed" in baseline_df.columns
    algorithms = sorted(baseline_df["model_id"].unique()) if has_model else ["unknown"]
    seeds = sorted(baseline_df["seed"].unique()) if has_seed else [0]

    for algo in algorithms:
        for seed in seeds:
            subset = baseline_df
            if has_model:
                subset = subset[subset["model_id"] == algo]
            if has_seed:
                subset = subset[subset["seed"] == seed]
            if subset.empty:
                continue

            entry: dict[str, Any] = {"seed": int(seed), "algorithm": algo}
            for m in _COMPARE_METRICS:
                col = (
                    m
                    if m in subset.columns
                    else (m.lower() if m.lower() in subset.columns else None)
                )
                if col and not subset[col].dropna().empty:
                    entry[m] = float(subset[col].mean())
            rows.append(entry)
    return pd.DataFrame(rows)


def _get_seed_val(frame: pd.DataFrame, seed: int, metric: str) -> float:
    """Get metric value for a seed, returning NaN if missing."""
    row = frame[frame["seed"] == seed]
    if len(row) == 0 or metric not in row.columns or row[metric].isna().all():
        return float("nan")
    return float(row[metric].iloc[0])


def _draw_missing_markers(
    ax: plt.Axes,
    x: np.ndarray,
    model_series: list[tuple[list[float], float, str]],
) -> None:
    """Draw 'x' markers at the axes bottom for missing/unplottable values.

    Args:
        ax: The matplotlib axes.
        x: Group x-positions.
        model_series: List of (values, bar_offset, color) tuples.
    """
    is_log = ax.get_yscale() == "log"
    transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    for vals, offset, color in model_series:
        for j, v in enumerate(vals):
            if np.isnan(v) or (is_log and v <= 0):
                ax.plot(
                    x[j] + offset,
                    0,
                    marker="x",
                    color=color,
                    markersize=5,
                    markeredgewidth=1.5,
                    clip_on=False,
                    transform=transform,
                    zorder=5,
                )


def _plot_baseline_metric(  # noqa: PLR0913
    metric: str,
    agent_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    algorithm: str,
    filename: str,
    output_dir: Path | None,
) -> None:
    """Plot a single-metric baseline comparison figure.

    One figure per metric with grouped bars (models + baseline) per seed + Avg.
    """
    algo_short = _ALGO_SHORT.get(algorithm, algorithm[:4])

    # Filter agent data to matching algorithm
    comparable = agent_metrics
    if "algorithm" in comparable.columns:
        comparable = comparable[comparable["algorithm"] == algorithm]

    if comparable.empty:
        print(f"  No {algo_short} agent data for {metric}")
        return

    # Filter baseline to matching algorithm
    algo_baseline = baseline_metrics
    if "algorithm" in algo_baseline.columns:
        algo_baseline = algo_baseline[algo_baseline["algorithm"] == algorithm]
    if algo_baseline.empty:
        print(f"  No {algo_short} baseline data for {metric}")
        return

    models = sorted(comparable["model"].unique().tolist())
    all_seeds = sorted(comparable["seed"].unique().tolist())
    n_bars = len(models) + 1  # +1 for baseline
    bar_w = 0.7 / max(n_bars, 1)
    n_groups = len(all_seeds) + 1  # +1 for average group

    # Scale figure width with number of groups
    fig_w = max(FULL_WIDTH, 0.45 * n_groups)
    fig, ax = plt.subplots(figsize=(fig_w, 2.8))

    x = np.arange(n_groups)

    for i, model in enumerate(models):
        model_df = comparable[comparable["model"] == model]
        vals = [_get_seed_val(model_df, s, metric) for s in all_seeds]
        avg = float(np.nanmean(vals)) if not all(np.isnan(vals)) else 0.0
        vals.append(avg)
        offset = (i - (n_bars - 1) / 2) * bar_w
        ax.bar(
            x + offset,
            vals,
            bar_w * 0.9,
            label=model,
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            alpha=PLOT_STYLE["alpha"],
        )
    # Baseline bar
    b_vals = [
        _get_seed_val(algo_baseline, s, metric)
        if s in _BASELINE_SEEDS
        else float("nan")
        for s in all_seeds
    ]
    b_avg = float(np.nanmean(b_vals)) if not all(np.isnan(b_vals)) else 0.0
    b_vals.append(b_avg)
    b_offset = (len(models) - (n_bars - 1) / 2) * bar_w
    ax.bar(
        x + b_offset,
        b_vals,
        bar_w * 0.9,
        label=f"Baseline ({algo_short})",
        color=COLOR_PALETTE[len(models) % len(COLOR_PALETTE)],
        alpha=0.50,
        hatch="//",
    )

    _format_baseline_axes(
        ax,
        x,
        all_seeds,
        models,
        comparable,
        metric,
        b_vals,
        b_offset,
        bar_w,
    )

    fig.subplots_adjust(bottom=0.22)
    save_figure(fig, filename, output_dir)
    plt.close(fig)


def _format_baseline_axes(  # noqa: PLR0913
    ax: plt.Axes,
    x: np.ndarray,
    all_seeds: list[int],
    models: list[str],
    comparable: pd.DataFrame,
    metric: str,
    b_vals: list[float],
    b_offset: float,
    bar_w: float,
) -> None:
    """Format axes: labels, log scale, missing markers, legend."""
    font_sizes = PLOT_STYLE["font_sizes"]

    # Vertical separator line before Avg group
    ax.axvline(x=len(all_seeds) - 0.5, color="grey", linewidth=0.5, linestyle="--")

    # Short labels: just seed number + "Avg"
    group_labels = [str(s) for s in all_seeds] + ["Avg"]
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=font_sizes["tick_label"])
    ax.set_xlabel("Seed", fontsize=font_sizes["axes_label"])
    ax.set_ylabel(EVAL_METRIC_LABELS.get(f"eval_{metric}", metric))
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

    # Use log scale when values span many orders of magnitude
    all_plotted = [v for v in b_vals if not np.isnan(v) and v > 0]
    for model in models:
        model_df = comparable[comparable["model"] == model]
        all_plotted.extend(
            v
            for s in all_seeds
            for v in [_get_seed_val(model_df, s, metric)]
            if not np.isnan(v) and v > 0
        )
    _log_scale_ratio = 1000
    if (
        len(all_plotted) > 1
        and min(all_plotted) > 0
        and max(all_plotted) / min(all_plotted) > _log_scale_ratio
    ):
        ax.set_yscale("log")
        ax.set_ylim(bottom=min(all_plotted) * 0.3)

    # Mark missing data with "x" on the axes bottom edge (must be after yscale)
    n_bars = len(models) + 1
    marker_series: list[tuple[list[float], float, str]] = []
    for i, model in enumerate(models):
        model_df = comparable[comparable["model"] == model]
        model_vals = [_get_seed_val(model_df, s, metric) for s in all_seeds]
        model_avg = (
            float(np.nanmean(model_vals)) if not all(np.isnan(model_vals)) else 0.0
        )
        model_vals.append(model_avg)
        offset = (i - (n_bars - 1) / 2) * bar_w
        marker_series.append(
            (model_vals, offset, COLOR_PALETTE[i % len(COLOR_PALETTE)])
        )
    marker_series.append(
        (b_vals, b_offset, COLOR_PALETTE[len(models) % len(COLOR_PALETTE)])
    )
    _draw_missing_markers(ax, x, marker_series)

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        fontsize=font_sizes["legend"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(models) + 1,
        frameon=True,
    )


def plot_baseline_comparison(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    baseline_dir: Path | None = None,
    prompt_style: str | None = None,
) -> None:
    """One figure per metric: agent-trained model vs official baseline.

    Produces per-(algorithm, metric) figures:
      - baseline_{metric}_{algo}.png  (e.g. baseline_IOG_cgan.png)

    When prompt_style is given, only the algorithm encoded in that style is
    plotted and the correct example_id -> config mapping is used.
    """
    setup_style()
    df = _prepare_data(df)

    bdir = baseline_dir or _BASELINE_DIR
    baseline_df = _load_baseline_csvs(bdir)
    if baseline_df is None or baseline_df.empty:
        print(f"  No baseline CSVs found in {bdir}, skipping baseline comparison")
        return

    # Build config mappings for the correct algorithm
    algorithm = _algorithm_from_style(prompt_style)
    _, eid_to_config, _ = _build_config_mappings(algorithm)

    agent_metrics = _extract_agent_metrics(df, example_id_to_config=eid_to_config)
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

    # When prompt_style is given, only plot the matching algorithm;
    # otherwise iterate over all baseline algorithms.
    algos_to_plot = [algorithm] if prompt_style else _BASELINE_ALGORITHMS

    for algo in algos_to_plot:
        algo_tag = _ALGO_SHORT.get(algo, algo[:4]).lower()
        for metric in available:
            _plot_baseline_metric(
                metric,
                agent_metrics,
                baseline_metrics,
                algo,
                f"baseline_{metric}_{algo_tag}.png",
                output_dir,
            )


# ── Plot 6: Baseline Grid (2x2) ──────────────────────────────────────────────

# Metrics for the grid: (column_name, display_label, direction)
_GRID_METRICS = [
    ("COG", "COG", "down"),
    ("viol", "RVC", "down"),
    ("MMD", "MMD", "down"),
    ("DPP", "DPP", "up"),
]


def _draw_grid_subplot(
    ax: plt.Axes,
    metric: str,
    label: str,
    direction: str,
    ctx: dict,
) -> None:
    """Draw a single subplot in the baseline grid."""
    comparable = ctx["comparable"]
    baseline_metrics = ctx["baseline_metrics"]
    models = ctx["models"]
    all_seeds = ctx["all_seeds"]
    n_bars = ctx["n_bars"]
    bar_w = ctx["bar_w"]
    algo_short = ctx["algo_short"]
    font_sizes = PLOT_STYLE["font_sizes"]
    x = np.arange(1)  # single group (Avg only)

    # ── Model bars (average across seeds) ──
    avg_vals = []
    for i, model in enumerate(models):
        model_df = comparable[comparable["model"] == model]
        per_seed = [_get_seed_val(model_df, s, metric) for s in all_seeds]
        avg = float(np.nanmean(per_seed)) if not all(np.isnan(per_seed)) else 0.0
        avg_vals.append(avg)
        offset = (i - (n_bars - 1) / 2) * bar_w
        ax.bar(
            x + offset,
            [avg],
            bar_w * 0.9,
            label=model,
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            alpha=PLOT_STYLE["alpha"],
        )

    # ── Baseline bar (average across seeds) ──
    b_per_seed = [
        _get_seed_val(baseline_metrics, s, metric)
        if s in _BASELINE_SEEDS
        else float("nan")
        for s in sorted(baseline_metrics["seed"].unique())
    ]
    b_avg = float(np.nanmean(b_per_seed)) if not all(np.isnan(b_per_seed)) else 0.0
    b_offset = (len(models) - (n_bars - 1) / 2) * bar_w
    ax.bar(
        x + b_offset,
        [b_avg],
        bar_w * 0.9,
        label=f"Baseline ({algo_short})",
        color=COLOR_PALETTE[len(models) % len(COLOR_PALETTE)],
        alpha=0.50,
        hatch="//",
    )

    # ── Axis formatting ──
    arrow = r"$\downarrow$" if direction == "down" else r"$\uparrow$"
    ax.set_ylabel(f"{label} {arrow}")
    ax.set_xticks(x)
    ax.set_xticklabels(["Avg"], fontsize=font_sizes["tick_label"])
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

    # ── Log scale when values span many orders of magnitude ──
    all_plotted = [v for v in [*avg_vals, b_avg] if not np.isnan(v) and v > 0]
    _log_scale_ratio = 1000
    if (
        len(all_plotted) > 1
        and min(all_plotted) > 0
        and max(all_plotted) / min(all_plotted) > _log_scale_ratio
    ):
        ax.set_yscale("log")
        ax.set_ylim(bottom=min(all_plotted) * 0.3)

    ax.grid(True, axis="y", alpha=0.3)


def plot_baseline_grid(
    df: pd.DataFrame,
    filename: str = "baseline_grid.png",
    output_dir: Path | None = None,
    prompt_style: str | None = None,
) -> None:
    """2x2 grid of baseline comparison plots for COG, RVC, MMD, DPP.

    Each subplot shows average bars (models + baseline),
    matching the style of individual baseline_*.png figures.
    Arrows indicate desired direction for each metric.
    """
    setup_style()
    df = _prepare_data(df)

    # Load data (same pipeline as plot_baseline_comparison)
    algorithm = _algorithm_from_style(prompt_style)
    algo_short = _ALGO_SHORT.get(algorithm, algorithm[:4])
    _, eid_to_config, _ = _build_config_mappings(algorithm)

    agent_metrics = _extract_agent_metrics(df, example_id_to_config=eid_to_config)
    if agent_metrics is None:
        print("  No agent metrics found, skipping baseline grid")
        return

    baseline_df = _load_baseline_csvs(_BASELINE_DIR)
    if baseline_df is None or baseline_df.empty:
        print("  No baseline CSVs found, skipping baseline grid")
        return

    baseline_metrics = _extract_baseline_metrics(baseline_df)
    if "algorithm" in baseline_metrics.columns:
        baseline_metrics = baseline_metrics[baseline_metrics["algorithm"] == algorithm]
    if baseline_metrics.empty:
        print("  No matching baseline data, skipping baseline grid")
        return

    # Filter agent data to matching algorithm
    comparable = agent_metrics
    if "algorithm" in comparable.columns:
        comparable = comparable[comparable["algorithm"] == algorithm]
    if comparable.empty:
        print("  No matching agent data, skipping baseline grid")
        return

    models = sorted(comparable["model"].unique().tolist())
    all_seeds = sorted(comparable["seed"].unique().tolist())
    n_bars = len(models) + 1  # +1 for baseline
    bar_w = 0.7 / max(n_bars, 1)
    font_sizes = PLOT_STYLE["font_sizes"]

    ctx = {
        "comparable": comparable,
        "baseline_metrics": baseline_metrics,
        "models": models,
        "all_seeds": all_seeds,
        "n_bars": n_bars,
        "bar_w": bar_w,
        "algo_short": algo_short,
    }

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.2))

    for idx, (metric, label, direction) in enumerate(_GRID_METRICS):
        _draw_grid_subplot(axes[idx // 2, idx % 2], metric, label, direction, ctx)

    # ── Shared legend at bottom ──
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=font_sizes["legend"],
        loc="lower center",
        ncol=n_bars,
        frameon=True,
    )

    save_figure(fig, filename, output_dir)
    plt.close(fig)


# ── Plot 7: Baseline Grid (violin) ────────────────────────────────────────────


def plot_baseline_grid_violin(  # noqa: PLR0912, PLR0915
    df: pd.DataFrame,
    filename: str = "baseline_grid.png",
    output_dir: Path | None = None,
    prompt_style: str | None = None,
) -> None:
    """2x2 grid of violin plots for COG, RVC, MMD, DPP.

    Each subplot shows per-seed distributions as violins (models + baseline)
    with individual data points overlaid.
    Arrows indicate desired direction for each metric.
    """
    setup_style()
    df = _prepare_data(df)

    # Load data (same pipeline as plot_baseline_grid)
    algorithm = _algorithm_from_style(prompt_style)
    algo_short = _ALGO_SHORT.get(algorithm, algorithm[:4])
    _, eid_to_config, _ = _build_config_mappings(algorithm)

    agent_metrics = _extract_agent_metrics(df, example_id_to_config=eid_to_config)
    if agent_metrics is None:
        print("  No agent metrics found, skipping baseline grid violin")
        return

    baseline_df = _load_baseline_csvs(_BASELINE_DIR)
    if baseline_df is None or baseline_df.empty:
        print("  No baseline CSVs found, skipping baseline grid violin")
        return

    baseline_metrics = _extract_baseline_metrics(baseline_df)
    if "algorithm" in baseline_metrics.columns:
        baseline_metrics = baseline_metrics[baseline_metrics["algorithm"] == algorithm]
    if baseline_metrics.empty:
        print("  No matching baseline data, skipping baseline grid violin")
        return

    # Filter agent data to matching algorithm
    comparable = agent_metrics
    if "algorithm" in comparable.columns:
        comparable = comparable[comparable["algorithm"] == algorithm]
    if comparable.empty:
        print("  No matching agent data, skipping baseline grid violin")
        return

    models = sorted(comparable["model"].unique().tolist())
    all_seeds = sorted(comparable["seed"].unique().tolist())
    font_sizes = PLOT_STYLE["font_sizes"]

    # Assign consistent colours: models + baseline
    group_names = [*models, f"Baseline ({algo_short})"]
    palette = {
        name: COLOR_PALETTE[i % len(COLOR_PALETTE)]
        for i, name in enumerate(group_names)
    }

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.2))

    for idx, (metric, label, direction) in enumerate(_GRID_METRICS):
        ax = axes[idx // 2, idx % 2]

        # Build long-form DataFrame for this metric
        rows: list[dict[str, Any]] = []
        for model in models:
            model_df = comparable[comparable["model"] == model]
            for seed in all_seeds:
                val = _get_seed_val(model_df, seed, metric)
                if not np.isnan(val):
                    rows.append({"group": model, "value": val})

        baseline_label = f"Baseline ({algo_short})"
        for seed in sorted(baseline_metrics["seed"].unique()):
            val = _get_seed_val(baseline_metrics, int(seed), metric)
            if not np.isnan(val):
                rows.append({"group": baseline_label, "value": val})

        if not rows:
            ax.set_visible(False)
            continue

        long_df = pd.DataFrame(rows)

        # Violin plot with inner box (median + IQR)
        sns.violinplot(
            data=long_df,
            x="group",
            y="value",
            order=group_names,
            palette=palette,
            inner="box",
            linewidth=0.5,
            saturation=0.8,
            cut=0,
            ax=ax,
        )

        # Mean annotations at bottom of each violin
        for i, name in enumerate(group_names):
            subset = long_df[long_df["group"] == name]["value"]
            if not subset.empty:
                mean_val = subset.mean()
                # Use scientific notation for large/tiny values
                _sci_upper = 1000
                _sci_lower = 0.01
                if abs(mean_val) >= _sci_upper or (
                    mean_val != 0 and abs(mean_val) < _sci_lower
                ):
                    mean_str = f"$\\mu$={mean_val:.1e}"
                else:
                    mean_str = f"$\\mu$={mean_val:.2f}"
                ax.annotate(
                    mean_str,
                    xy=(i, ax.get_ylim()[0]),
                    ha="center",
                    va="bottom",
                    fontsize=font_sizes["annotation"],
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "alpha": 0.8,
                        "edgecolor": "0.8",
                    },
                )

        arrow = r"$\downarrow$" if direction == "down" else r"$\uparrow$"
        ax.set_ylabel(f"{label} {arrow}")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=font_sizes["tick_label"])
        # Rotate x-labels if many groups
        _max_unrotated_groups = 3
        if len(group_names) > _max_unrotated_groups:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.close(fig)


# ── Main entrypoint ──────────────────────────────────────────────────────────


def main(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    prompt_style: str | None = None,
) -> None:
    """Generate all HPC training evaluation plots.

    Args:
        df: Design-level DataFrame from extract_data.py (with HPC workflow fields).
        output_dir: Directory to save figures.
        prompt_style: Prompt style (e.g. "hpc-train-cgan") — used to resolve
            the correct algorithm for config mappings and baseline comparison.
    """
    if df is None or df.empty:
        print("  No data for HPC training plots")
        return

    print(f"  HPC training data: {len(df)} rows")

    print("\n[1/7] Step completion heatmap...")
    plot_step_completion_heatmap(df, output_dir=output_dir)

    print("\n[2/7] Step completion heatmap (averaged per model)...")
    plot_step_completion_heatmap_avg(df, output_dir=output_dir)

    print("\n[3/7] Workflow score bars...")
    plot_workflow_score_bars(df, output_dir=output_dir)

    print("\n[4/7] Step completion rate...")
    plot_step_completion_rate(df, output_dir=output_dir)

    print("\n[5/7] Evaluation metrics...")
    plot_evaluation_metrics(df, output_dir=output_dir, prompt_style=prompt_style)

    print("\n[6/7] Baseline comparison (one figure per metric)...")
    plot_baseline_comparison(df, output_dir=output_dir, prompt_style=prompt_style)

    print("\n[7/7] Baseline grid violin (2x2: COG, RVC, MMD, DPP)...")
    plot_baseline_grid_violin(df, output_dir=output_dir, prompt_style=prompt_style)


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
