"""
RAG Evaluation Plots

Five publication-ready figures comparing RAG-on vs RAG-off performance across models:
  1. plot_rag_benefit_score        -- grouped bars: rag_benefit_score per model x rag_status
  2. plot_rag_score_components     -- stacked bars: weighted gated score decomposition
  3. plot_rag_accuracy_comparison  -- stacked bars: raw (ungated) accuracy + RAG usage
  4. plot_rag_score_combined       -- single axes: all conditions (P0/P1 x on/off)
  5. plot_rag_uplift               -- horizontal bars: delta-score (rag_on - rag_off) per model

Usage:
    # 1. Extract results from Weave (run twice, once per RAG status):
    #    python benchmarks/evaluations/extract_data.py --problem rag_beams2d --prompt-style rag-eval --rag-status rag
    #    python benchmarks/evaluations/extract_data.py --problem rag_beams2d --prompt-style rag-eval --rag-status no_rag
    # 2. Generate plots:
    python run_all.py --problem rag_beams2d --prompt-style rag-eval
    # Or standalone:
    python plot_rag_evaluation.py
"""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.evaluations.plots.utils import (
    COLOR_PALETTE,
    PLOT_STYLE,
    get_problem_prompt_output_dir,
    save_figure,
    setup_style,
)
from benchmarks.shared.scorers.rag_scorer import (
    COMPONENT_WEIGHTS_DOUBLE,
    COMPONENT_WEIGHTS_RMIN,
    COMPONENT_WEIGHTS_SINGLE,
    COMPONENT_WEIGHTS_TRIPLE,
)

# Minimum bar value to display an annotation label
_MIN_LABEL_VALUE = 0.05

# ── Prompt labels ──────────────────────────────────────────────────────────────
_PROMPT_SHORT_LABELS = {0: "P0", 1: "P1", 2: "P2", 3: "P3"}
_PROMPT_LABELS = {
    0: "P0 — Easy (volfrac)",
    1: "P1 — Hard (volfrac + forcedist)",
    2: "P2 — Post-cutoff (volfrac + rmin)",
    3: "P3 — Mixed sources (all three)",
}

# ── RAG status display labels and styles ───────────────────────────────────────
_RAG_DISPLAY = {
    "rag": {"label": "RAG on", "hatch": "", "alpha": 0.85},
    "empty_rag": {"label": "Empty RAG", "hatch": "..", "alpha": 0.70},
    "no_rag": {"label": "RAG off", "hatch": "//", "alpha": 0.55},
}

# Canonical order for RAG statuses when iterating
_RAG_STATUS_ORDER = list(_RAG_DISPLAY.keys())

# Colors for stacked bar components
_COMPONENT_COLORS = {
    "eff_volfrac": COLOR_PALETTE[0],  # blue
    "eff_forcedist": COLOR_PALETTE[1],  # orange
    "eff_rmin": COLOR_PALETTE[4],  # red/pink
    "rag_called": COLOR_PALETTE[2],  # green
}
_COMPONENT_LABELS = {
    "eff_volfrac": r"\texttt{volfrac} accuracy",
    "eff_forcedist": r"\texttt{forcedist} accuracy",
    "eff_rmin": r"\texttt{rmin} accuracy",
    "rag_called": "RAG tool called",
}


def _short_model(model_id: str) -> str:
    """Return a short display name for a model."""
    return model_id.rsplit(":", maxsplit=1)[-1] if ":" in model_id else model_id


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and add derived fields."""
    df = df.copy()
    # model column can be model_id or model
    if "model_id" in df.columns and "model" not in df.columns:
        df["model"] = df["model_id"]
    # rag_status may come from directory path (mmore_on / mmore_off)
    if "rag_status" not in df.columns:
        df["rag_status"] = "unknown"
    # short label
    df["model_short"] = df["model"].apply(_short_model)
    return df


def _apply_weights(
    df: pd.DataFrame,
    present: dict[str, str],
    mode_weights: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Add ``_wt_{col}`` weighted-contribution columns to a copy of *df*.

    Selects weights per row based on ``forcedist_tested`` / ``rmin_tested`` flags.
    Four modes: single, double (forcedist), rmin, triple (both).

    Args:
        mode_weights: ``{"single": …, "double": …, "rmin": …, "triple": …}``.
    """
    df = df.copy()
    forcedist_mode = (
        df["forcedist_tested"].astype(bool)
        if "forcedist_tested" in df.columns
        else pd.Series(False, index=df.index, dtype=bool)
    )
    rmin_mode = (
        df["rmin_tested"].astype(bool)
        if "rmin_tested" in df.columns
        else pd.Series(False, index=df.index, dtype=bool)
    )
    ws, wd = mode_weights["single"], mode_weights["double"]
    wr, wt = mode_weights["rmin"], mode_weights["triple"]
    for col in present:
        w = pd.Series(ws.get(col, 0.0), index=df.index)
        w[rmin_mode & ~forcedist_mode] = wr.get(col, 0.0)
        w[forcedist_mode & ~rmin_mode] = wd.get(col, 0.0)
        w[forcedist_mode & rmin_mode] = wt.get(col, 0.0)
        df[f"_wt_{col}"] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) * w
    return df


# ── Plot 1: Grouped bar — rag_benefit_score ───────────────────────────────────


def plot_rag_benefit_score(
    df: pd.DataFrame,
    filename: str = "rag_benefit_score.png",
    output_dir: Path | None = None,
) -> None:
    """Grouped bar chart: mean rag_benefit_score per model x RAG status.

    Two subplots side-by-side: easy prompt (example_id=0) and hard prompt
    (example_id=1).  Each model has two bars: RAG-on (solid) and RAG-off
    (hatched lighter).

    Args:
        df: Design data DataFrame with rag_benefit_score, model, rag_status,
            example_id columns.
        filename: Output filename.
        output_dir: Directory to save figures (created if needed).
    """
    setup_style()
    df = _prepare_data(df)

    if "rag_benefit_score" not in df.columns:
        print("  ⚠️  No rag_benefit_score column — skipping plot_rag_benefit_score")
        return

    fs = PLOT_STYLE["font_sizes"]
    example_ids = sorted(df["example_id"].dropna().unique())
    n_prompts = len(example_ids)

    fig, axes = plt.subplots(
        1,
        n_prompts,
        figsize=PLOT_STYLE["figsize_full_width"]
        if n_prompts > 1
        else PLOT_STYLE["figsize_single_col"],
        sharey=True,
    )
    if n_prompts == 1:
        axes = [axes]

    rag_statuses = [s for s in _RAG_STATUS_ORDER if s in df["rag_status"].unique()]
    models = sorted(df["model_short"].unique())
    x = np.arange(len(models))
    n_rs = len(rag_statuses)
    width = 0.7 / max(n_rs, 1)

    for ax, eid in zip(axes, example_ids, strict=False):
        sub = df[df["example_id"] == eid]
        for i, rs in enumerate(rag_statuses):
            style = _RAG_DISPLAY.get(rs, {"label": rs, "hatch": "", "alpha": 0.8})
            means = []
            for m in models:
                mask = (sub["model_short"] == m) & (sub["rag_status"] == rs)
                means.append(
                    sub.loc[mask, "rag_benefit_score"].mean() if mask.any() else 0.0
                )

            offset = (i - (n_rs - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                means,
                width,
                color=COLOR_PALETTE[i],
                alpha=style["alpha"],
                hatch=style["hatch"],
                label=style["label"],
                edgecolor="white",
                linewidth=0.5,
            )
            # Value labels
            for bar, val in zip(bars, means, strict=False):
                if not np.isnan(val) and val > _MIN_LABEL_VALUE:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=fs["annotation"],
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=fs["tick_label"])
        ax.set_ylabel("RAG benefit score", fontsize=fs["axes_label"])
        ax.set_ylim(0, 1.15)
        ax.set_title(
            _PROMPT_LABELS.get(int(eid), f"Prompt {int(eid)}"),
            fontsize=fs["axes_title"],
        )
        ax.tick_params(axis="y", labelsize=fs["tick_label"])
        ax.grid(axis="y", linewidth=PLOT_STYLE.get("grid.linewidth", 0.3), alpha=0.4)

    axes[0].legend(fontsize=fs["legend"], loc="upper right")
    fig.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.close(fig)
    print(f"  ✅ {filename}")


# ── Plot 2: Stacked bar — score decomposition ─────────────────────────────────


def plot_rag_score_components(
    df: pd.DataFrame,
    filename: str = "rag_score_components.png",
    output_dir: Path | None = None,
) -> None:
    """Stacked bar chart: score component breakdown per model x RAG status x prompt.

    Two subplots side-by-side (RAG on / RAG off).  Each model has two grouped
    stacked bars — one per prompt level (easy / hard) — distinguished by hatch.
    Bar heights are weighted contributions so the total equals rag_benefit_score.

    Args:
        df: Design data DataFrame.
        filename: Output filename.
        output_dir: Directory to save figures.
    """
    setup_style()
    df = _prepare_data(df)

    # Maps scorer output column → short key used in _COMPONENT_COLORS/_COMPONENT_LABELS
    components = {
        "effective_volfrac_accuracy": "eff_volfrac",
        "effective_forcedist_accuracy": "eff_forcedist",
        "effective_rmin_accuracy": "eff_rmin",
        "rag_tool_called": "rag_called",
    }
    present = {k: v for k, v in components.items() if k in df.columns}
    if not present:
        print("  ⚠️  No RAG component columns — skipping plot_rag_score_components")
        return

    # Weights imported from rag_scorer.py — single source of truth.
    # Applied per-row via mode flags; stacked total equals rag_benefit_score.
    df = _apply_weights(
        df,
        present,
        {
            "single": COMPONENT_WEIGHTS_SINGLE,
            "double": COMPONENT_WEIGHTS_DOUBLE,
            "rmin": COMPONENT_WEIGHTS_RMIN,
            "triple": COMPONENT_WEIGHTS_TRIPLE,
        },
    )

    fs = PLOT_STYLE["font_sizes"]
    rag_statuses = [s for s in _RAG_STATUS_ORDER if s in df["rag_status"].unique()]
    models = sorted(df["model_short"].unique())
    example_ids = sorted(df["example_id"].dropna().unique())

    # One bar per prompt level per model group; bar_width shrinks to avoid overlap.
    n_eids = len(example_ids)
    bar_width = 0.7 / max(n_eids, 1)
    offsets = (
        np.linspace(-(n_eids - 1) * bar_width / 2, (n_eids - 1) * bar_width / 2, n_eids)
        if n_eids > 1
        else [0.0]
    )
    prompt_hatches = dict(zip(example_ids, ["", "//", "..", "xx"], strict=False))

    x = np.arange(len(models))

    n_panels = len(rag_statuses)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(
            PLOT_STYLE["figsize_full_width"][0],
            PLOT_STYLE["figsize_full_width"][1],
        ),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, rs in zip(axes, rag_statuses, strict=False):
        sub = df[df["rag_status"] == rs]
        for eid, offset in zip(example_ids, offsets, strict=False):
            sub_eid = sub[sub["example_id"] == eid]
            hatch = prompt_hatches.get(eid, "")
            bottoms = np.zeros(len(models))
            for col, key in present.items():
                vals = []
                for m in models:
                    mask = sub_eid["model_short"] == m
                    v = sub_eid.loc[mask, f"_wt_{col}"].mean() if mask.any() else 0.0
                    vals.append(0.0 if np.isnan(v) else float(v))
                bar_vals = np.array(vals)
                ax.bar(
                    x + offset,
                    bar_vals,
                    bar_width,
                    bottom=bottoms,
                    color=_COMPONENT_COLORS.get(key, COLOR_PALETTE[0]),
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=0.5,
                )
                bottoms += bar_vals

        # P0/P1 sub-labels under each bar
        xaxis_tr = ax.get_xaxis_transform()
        for eid2, off2 in zip(example_ids, offsets, strict=False):
            plabel = _PROMPT_SHORT_LABELS.get(int(eid2), f"P{int(eid2)}")
            for xi in x:
                ax.text(
                    xi + off2,
                    -0.03,
                    plabel,
                    ha="center",
                    va="top",
                    fontsize=fs["annotation"],
                    transform=xaxis_tr,
                    clip_on=False,
                )

        style = _RAG_DISPLAY.get(rs, {"label": rs})
        ax.set_title(style["label"], fontsize=fs["axes_title"])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=fs["tick_label"])
        ax.tick_params(axis="x", which="major", pad=10)
        ax.tick_params(axis="y", labelsize=fs["tick_label"])
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    axes[0].set_ylabel("Weighted score contribution", fontsize=fs["axes_label"])

    # Legend: colour patches only (prompt level shown via P0/P1 x-axis labels)
    component_patches = [
        mpatches.Patch(color=_COMPONENT_COLORS[key], label=label)
        for key, label in _COMPONENT_LABELS.items()
        if key in set(present.values())
    ]
    axes[-1].legend(
        handles=component_patches,
        fontsize=fs["legend"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
    )
    fig.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.close(fig)
    print(f"  ✅ {filename}")


# ── Shared helpers for raw/ungated accuracy plots ─────────────────────────────

# Maps effective_* scorer keys → raw accuracy column names for the ungated plot.
_EFFECTIVE_TO_RAW = {
    "effective_volfrac_accuracy": "volfrac_accuracy",
    "effective_forcedist_accuracy": "forcedist_accuracy",
    "effective_rmin_accuracy": "rmin_accuracy",
    "rag_tool_called": "rag_tool_called",
}


def _renormalize_weights(src: dict[str, float]) -> dict[str, float]:
    """Remap effective_* keys to raw column names and renormalise to sum to 1.0."""
    remapped = {}
    for orig_key, raw_key in _EFFECTIVE_TO_RAW.items():
        if orig_key in src:
            remapped[raw_key] = src[orig_key]
    total = sum(remapped.values()) or 1.0
    return {k: v / total for k, v in remapped.items()}


# ── Plot 2b: Score components grouped by prompt ──────────────────────────────


_MODEL_ABBREVIATIONS = {
    "GPT-5-mini": "GPT-5m",
    "Gemini-3-flash": "Gem-3f",
    "Gemini-3-flash-preview": "Gem-3f",
    "Qwen3-4B-Q8": "Qw3-4B",
}


def _abbreviate_model(name: str) -> str:
    """Return a compact display name for a model (for dense bar charts)."""
    return _MODEL_ABBREVIATIONS.get(name, name[:8])


def _draw_by_prompt_axes(ax: plt.Axes, df_sub: pd.DataFrame, cfg: dict) -> None:
    """Draw stacked bars and axis labels for one RAG-status panel."""
    models, example_ids = cfg["models"], cfg["example_ids"]
    offsets, bar_width = cfg["offsets"], cfg["bar_width"]
    model_hatches, abbrev = cfg["model_hatches"], cfg["abbrev"]
    present, x, fs = cfg["present"], cfg["x"], cfg["fs"]

    for model, offset in zip(models, offsets, strict=False):
        sub_model = df_sub[df_sub["model_short"] == model]
        hatch = model_hatches.get(model, "")
        bottoms = np.zeros(len(example_ids))
        for col, key in present.items():
            vals = []
            for eid in example_ids:
                series = sub_model.loc[sub_model["example_id"] == eid, f"_wt_{col}"]
                v = series.mean() if len(series) else 0.0
                vals.append(0.0 if np.isnan(v) else float(v))
            vals_arr = np.array(vals)
            ax.bar(
                x + offset,
                vals_arr,
                bar_width,
                bottom=bottoms,
                color=_COMPONENT_COLORS.get(key, COLOR_PALETTE[0]),
                hatch=hatch,
                edgecolor="white",
                linewidth=0.5,
            )
            bottoms += vals_arr

    # Per-bar ticks: one tick per model under each bar
    tick_pos = [xi + off for xi in x for _, off in zip(models, offsets, strict=False)]
    tick_labels = [abbrev[m] for _ in x for m in models]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=fs["annotation"], rotation=45, ha="right")

    # Group labels (P0, P1, …) centered below each prompt cluster
    xaxis_tr = ax.get_xaxis_transform()
    for eid, xi in zip(example_ids, x, strict=False):
        plabel = _PROMPT_SHORT_LABELS.get(int(eid), f"P{int(eid)}")
        ax.text(
            xi,
            -0.22,
            plabel,
            ha="center",
            va="top",
            fontsize=fs["tick_label"],
            fontweight="bold",
            transform=xaxis_tr,
            clip_on=False,
        )

    style = _RAG_DISPLAY.get(cfg["rag_status"], {"label": cfg["rag_status"]})
    ax.set_title(style["label"], fontsize=fs["axes_title"])
    ax.tick_params(axis="y", labelsize=fs["tick_label"])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)


def plot_rag_score_components_by_prompt(
    df: pd.DataFrame,
    filename: str = "rag_score_components_by_prompt.png",
    output_dir: Path | None = None,
    exclude_rag_statuses: list[str] | None = None,
) -> None:
    """Stacked bar chart: score component breakdown grouped by **prompt**.

    Same data as ``plot_rag_score_components`` but transposed:
    x-axis = prompt (P0, P1, …) and within each prompt group there is one
    bar per model, distinguished by hatch pattern.  One subplot per
    RAG status.

    Args:
        df: Design data DataFrame.
        filename: Output filename.
        output_dir: Directory to save figures.
        exclude_rag_statuses: RAG statuses to exclude (default: ``["no_rag"]``).
    """
    setup_style()
    df = _prepare_data(df)

    # Use raw (ungated) accuracy so RAG-off bars are non-zero when a model
    # guesses parameters correctly — same data as plot_rag_accuracy_comparison.
    components = {
        "volfrac_accuracy": "eff_volfrac",
        "forcedist_accuracy": "eff_forcedist",
        "rmin_accuracy": "eff_rmin",
        "rag_tool_called": "rag_called",
    }
    present = {k: v for k, v in components.items() if k in df.columns}
    if not present:
        print("  ⚠️  No accuracy columns — skipping plot_rag_score_components_by_prompt")
        return

    raw_weights = {
        "single": _renormalize_weights(COMPONENT_WEIGHTS_SINGLE),
        "double": _renormalize_weights(COMPONENT_WEIGHTS_DOUBLE),
        "rmin": _renormalize_weights(COMPONENT_WEIGHTS_RMIN),
        "triple": _renormalize_weights(COMPONENT_WEIGHTS_TRIPLE),
    }
    df = _apply_weights(df, present, raw_weights)

    fs = PLOT_STYLE["font_sizes"]
    models = sorted(df["model_short"].unique())
    example_ids = sorted(df["example_id"].dropna().unique())
    abbrev = {m: _abbreviate_model(m) for m in models}

    n_models = len(models)
    bar_width = 0.7 / max(n_models, 1)
    offsets = (
        np.linspace(
            -(n_models - 1) * bar_width / 2,
            (n_models - 1) * bar_width / 2,
            n_models,
        )
        if n_models > 1
        else [0.0]
    )
    model_hatches = dict(zip(models, ["", "//", "..", "xx", "++", "oo"], strict=False))
    x = np.arange(len(example_ids))

    if exclude_rag_statuses is None:
        exclude_rag_statuses = ["no_rag"]
    rag_statuses = [
        s
        for s in _RAG_STATUS_ORDER
        if s in df["rag_status"].unique() and s not in exclude_rag_statuses
    ]
    n_panels = len(rag_statuses)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(
            PLOT_STYLE["figsize_full_width"][0],
            PLOT_STYLE["figsize_full_width"][1],
        ),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    cfg = {
        "models": models,
        "example_ids": example_ids,
        "offsets": offsets,
        "bar_width": bar_width,
        "model_hatches": model_hatches,
        "abbrev": abbrev,
        "present": present,
        "x": x,
        "fs": fs,
    }
    for ax, rs in zip(axes, rag_statuses, strict=False):
        _draw_by_prompt_axes(ax, df[df["rag_status"] == rs], {**cfg, "rag_status": rs})

    axes[0].set_ylabel("Weighted score contribution", fontsize=fs["axes_label"])

    # Legend: colour patches only (models identified by sub-labels)
    component_patches = [
        mpatches.Patch(color=_COMPONENT_COLORS[key], label=label)
        for key, label in _COMPONENT_LABELS.items()
        if key in set(present.values())
    ]
    axes[-1].legend(
        handles=component_patches,
        fontsize=fs["legend"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
    )
    fig.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.close(fig)
    print(f"  ✅ {filename}")


# ── Plot 3: Raw accuracy comparison — ungated accuracy + RAG usage ────────────


def plot_rag_accuracy_comparison(  # noqa: PLR0915
    df: pd.DataFrame,
    filename: str = "rag_accuracy_comparison.png",
    output_dir: Path | None = None,
) -> None:
    """Stacked bar chart: raw accuracy + RAG usage per model x RAG status x prompt.

    Unlike plot_rag_score_components (which uses effective accuracy gated on
    rag_tool_called), this plot uses **raw** volfrac/forcedist accuracy so that
    no-RAG bars are non-zero when a model guesses parameters correctly from
    prior knowledge.  The green "RAG tool called" segment is the visual
    discriminator between RAG-on and RAG-off conditions.

    Weights are derived from the canonical scorer weights (renormalised to sum to 1.0).

    Args:
        df: Design data DataFrame.
        filename: Output filename.
        output_dir: Directory to save figures.
    """
    setup_style()
    df = _prepare_data(df)

    # Raw accuracy columns → same display keys as the gated version
    components = {
        "volfrac_accuracy": "eff_volfrac",
        "forcedist_accuracy": "eff_forcedist",
        "rmin_accuracy": "eff_rmin",
        "rag_tool_called": "rag_called",
    }
    present = {k: v for k, v in components.items() if k in df.columns}
    if not present:
        print("  ⚠️  No accuracy columns — skipping plot_rag_accuracy_comparison")
        return

    # Derive weights: map effective_* → raw column names, renormalise to sum to 1.0.
    raw_weights = {
        "single": _renormalize_weights(COMPONENT_WEIGHTS_SINGLE),
        "double": _renormalize_weights(COMPONENT_WEIGHTS_DOUBLE),
        "rmin": _renormalize_weights(COMPONENT_WEIGHTS_RMIN),
        "triple": _renormalize_weights(COMPONENT_WEIGHTS_TRIPLE),
    }
    df = _apply_weights(df, present, raw_weights)

    fs = PLOT_STYLE["font_sizes"]
    rag_statuses = [s for s in _RAG_STATUS_ORDER if s in df["rag_status"].unique()]
    models = sorted(df["model_short"].unique())
    example_ids = sorted(df["example_id"].dropna().unique())

    n_eids = len(example_ids)
    bar_width = 0.7 / max(n_eids, 1)
    offsets = (
        np.linspace(-(n_eids - 1) * bar_width / 2, (n_eids - 1) * bar_width / 2, n_eids)
        if n_eids > 1
        else [0.0]
    )
    prompt_hatches = dict(zip(example_ids, ["", "//", "..", "xx"], strict=False))

    x = np.arange(len(models))

    n_panels = len(rag_statuses)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(
            PLOT_STYLE["figsize_full_width"][0],
            PLOT_STYLE["figsize_full_width"][1],
        ),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, rs in zip(axes, rag_statuses, strict=False):
        sub = df[df["rag_status"] == rs]
        for eid, offset in zip(example_ids, offsets, strict=False):
            sub_eid = sub[sub["example_id"] == eid]
            hatch = prompt_hatches.get(eid, "")
            bottoms = np.zeros(len(models))
            for col, key in present.items():
                vals = []
                for m in models:
                    mask = sub_eid["model_short"] == m
                    v = sub_eid.loc[mask, f"_wt_{col}"].mean() if mask.any() else 0.0
                    vals.append(0.0 if np.isnan(v) else float(v))
                vals_arr = np.array(vals)
                ax.bar(
                    x + offset,
                    vals_arr,
                    bar_width,
                    bottom=bottoms,
                    color=_COMPONENT_COLORS.get(key, COLOR_PALETTE[0]),
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=0.5,
                )
                bottoms += vals_arr

        # P0/P1 sub-labels under each bar
        xaxis_tr = ax.get_xaxis_transform()
        for eid2, off2 in zip(example_ids, offsets, strict=False):
            plabel = _PROMPT_SHORT_LABELS.get(int(eid2), f"P{int(eid2)}")
            for xi in x:
                ax.text(
                    xi + off2,
                    -0.03,
                    plabel,
                    ha="center",
                    va="top",
                    fontsize=fs["annotation"],
                    transform=xaxis_tr,
                    clip_on=False,
                )

        style = _RAG_DISPLAY.get(rs, {"label": rs})
        ax.set_title(style["label"], fontsize=fs["axes_title"])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=fs["tick_label"])
        ax.tick_params(axis="x", which="major", pad=10)
        ax.tick_params(axis="y", labelsize=fs["tick_label"])
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    axes[0].set_ylabel("Weighted accuracy contribution", fontsize=fs["axes_label"])

    # Legend: colour patches only (prompt level shown via P0/P1 x-axis labels)
    component_patches = [
        mpatches.Patch(color=_COMPONENT_COLORS[key], label=_COMPONENT_LABELS[key])
        for key in present.values()
        if key in _COMPONENT_COLORS
    ]
    axes[-1].legend(
        handles=component_patches,
        fontsize=fs["legend"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
    )
    fig.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.close(fig)
    print(f"  ✅ {filename}")


# ── Shared helper for stacked bar drawing ─────────────────────────────────────


def _draw_stacked_bars(  # noqa: PLR0913
    ax: plt.Axes,
    df: pd.DataFrame,
    models: list[str],
    x: np.ndarray,
    example_ids: list,
    offsets: list[float],
    bar_width: float,
    prompt_hatches: dict,
    present: dict[str, str],
    fs: dict,
) -> None:
    """Draw stacked component bars with P0/P1 sub-labels on *ax*."""
    for eid, offset in zip(example_ids, offsets, strict=False):
        sub_eid = df[df["example_id"] == eid]
        hatch = prompt_hatches.get(eid, "")
        bottoms = np.zeros(len(models))
        for col, key in present.items():
            vals = []
            for m in models:
                mask = sub_eid["model_short"] == m
                v = sub_eid.loc[mask, f"_wt_{col}"].mean() if mask.any() else 0.0
                vals.append(0.0 if np.isnan(v) else float(v))
            vals_arr = np.array(vals)
            ax.bar(
                x + offset,
                vals_arr,
                bar_width,
                bottom=bottoms,
                color=_COMPONENT_COLORS.get(key, COLOR_PALETTE[0]),
                hatch=hatch,
                edgecolor="white",
                linewidth=0.5,
            )
            bottoms += vals_arr

    xaxis_tr = ax.get_xaxis_transform()
    for eid2, off2 in zip(example_ids, offsets, strict=False):
        plabel = _PROMPT_SHORT_LABELS.get(int(eid2), f"P{int(eid2)}")
        for xi in x:
            ax.text(
                xi + off2,
                -0.03,
                plabel,
                ha="center",
                va="top",
                fontsize=fs["annotation"],
                transform=xaxis_tr,
                clip_on=False,
            )


# ── Plot 4: RAG uplift — delta-score per model ────────────────────────────────


def plot_rag_uplift(
    df: pd.DataFrame,
    filename: str = "rag_uplift.png",
    output_dir: Path | None = None,
) -> None:
    """Horizontal bar chart: delta-score = rag_on - rag_off per model x prompt.

    Positive delta (RAG helps) shown in green, negative in orange.
    Sorted by overall uplift descending.

    Args:
        df: Design data DataFrame.
        filename: Output filename.
        output_dir: Directory to save figures.
    """
    setup_style()
    df = _prepare_data(df)

    if "rag_benefit_score" not in df.columns:
        print("  ⚠️  No rag_benefit_score column — skipping plot_rag_uplift")
        return

    fs = PLOT_STYLE["font_sizes"]
    example_ids = sorted(df["example_id"].dropna().unique())
    models = sorted(df["model_short"].unique())

    # Compute delta-score per model x example
    rows = []
    for m in models:
        for eid in example_ids:
            on_mask = (
                (df["model_short"] == m)
                & (df["rag_status"] == "rag")
                & (df["example_id"] == eid)
            )
            off_mask = (
                (df["model_short"] == m)
                & (df["rag_status"] == "no_rag")
                & (df["example_id"] == eid)
            )
            s_on = (
                df.loc[on_mask, "rag_benefit_score"].mean() if on_mask.any() else np.nan
            )
            s_off = (
                df.loc[off_mask, "rag_benefit_score"].mean()
                if off_mask.any()
                else np.nan
            )
            if not (np.isnan(s_on) and np.isnan(s_off)):
                rows.append(
                    {
                        "model": m,
                        "example_id": int(eid),
                        "delta": float(np.nan_to_num(s_on) - np.nan_to_num(s_off)),
                        "label": f"{m} ({_PROMPT_SHORT_LABELS.get(int(eid), f'P{int(eid)}')})",
                    }
                )

    if not rows:
        print("  ⚠️  Insufficient data for RAG uplift plot (need both rag and no_rag)")
        return

    delta_df = pd.DataFrame(rows).sort_values("delta", ascending=True)

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize_single_col_tall"])
    colors = [
        COLOR_PALETTE[2] if d >= 0 else COLOR_PALETTE[5] for d in delta_df["delta"]
    ]
    y = np.arange(len(delta_df))
    ax.barh(y, delta_df["delta"], color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(delta_df["label"], fontsize=fs["tick_label"])
    ax.set_xlabel(
        r"RAG uplift  ($\Delta$score = on $-$ off)", fontsize=fs["axes_label"]
    )
    ax.tick_params(axis="x", labelsize=fs["tick_label"])
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.close(fig)
    print(f"  ✅ {filename}")


# ── CLI entry point ────────────────────────────────────────────────────────────


def main(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Generate all RAG evaluation plots.

    Args:
        df: Design data DataFrame containing RAG metrics.
        output_dir: Directory to save figures.
    """
    print("\n[1/5] RAG benefit score grouped bar...")
    plot_rag_benefit_score(df, output_dir=output_dir)

    print("\n[2/5] Score component breakdown (weighted, gated)...")
    plot_rag_score_components(df, output_dir=output_dir)

    print("\n[3/5] Score component breakdown grouped by prompt...")
    plot_rag_score_components_by_prompt(df, output_dir=output_dir)

    print("\n[4/5] Raw accuracy comparison (ungated)...")
    plot_rag_accuracy_comparison(df, output_dir=output_dir)

    print("\n[5/5] RAG uplift delta bars...")
    plot_rag_uplift(df, output_dir=output_dir)


if __name__ == "__main__":
    import argparse

    from benchmarks.evaluations.plots.utils import (
        filter_by_problem,
        get_combined_design_df,
        load_data,
    )

    parser = argparse.ArgumentParser(description="Generate RAG evaluation plots")
    parser.add_argument("--problem", default="rag_beams2d")
    parser.add_argument("--prompt-style", default="rag-eval")
    parser.add_argument("--rag-status", default=None)
    args = parser.parse_args()

    data = load_data()
    df = get_combined_design_df(data)
    if df is not None:
        df = filter_by_problem(df, args.problem)
    if df is None or df.empty:
        print("No data found — run extract_data.py first.")
        sys.exit(1)

    out = get_problem_prompt_output_dir(
        args.problem, args.prompt_style, args.rag_status
    )
    main(df, out)
