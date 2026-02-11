"""
RAG Evaluation Plots

Four publication-ready figures comparing RAG-on vs RAG-off performance across models:
  1. plot_rag_benefit_score        -- grouped bars: rag_benefit_score per model x rag_status
  2. plot_rag_score_components     -- stacked bars: weighted gated score decomposition
  3. plot_rag_accuracy_comparison  -- stacked bars: raw (ungated) accuracy + RAG usage
  4. plot_rag_uplift               -- horizontal bars: delta-score (rag_on - rag_off) per model

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

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.evaluations.plots.utils import (  # noqa: E402
    COLOR_PALETTE,
    PLOT_STYLE,
    get_problem_prompt_output_dir,
    save_figure,
    setup_style,
)
from benchmarks.shared.scorers.rag_scorer import (  # noqa: E402
    COMPONENT_WEIGHTS_DOUBLE,
    COMPONENT_WEIGHTS_SINGLE,
)

# Minimum bar value to display an annotation label
_MIN_LABEL_VALUE = 0.05

# ── Prompt labels ──────────────────────────────────────────────────────────────
_PROMPT_LABELS = {
    0: "Easy (volfrac)",
    1: "Hard (volfrac + forcedist)",
}

# ── RAG status display labels and styles ───────────────────────────────────────
_RAG_DISPLAY = {
    "rag": {"label": "RAG on", "hatch": "", "alpha": 0.85},
    "no_rag": {"label": "RAG off", "hatch": "//", "alpha": 0.55},
}

# Colors for stacked bar components
_COMPONENT_COLORS = {
    "eff_volfrac": COLOR_PALETTE[0],  # blue
    "eff_forcedist": COLOR_PALETTE[1],  # orange
    "rag_called": COLOR_PALETTE[2],  # green
    "cited": COLOR_PALETTE[3],  # purple
}
_COMPONENT_LABELS = {
    "eff_volfrac": "Volfrac accuracy",
    "eff_forcedist": "Forcedist accuracy",
    "rag_called": "RAG tool called",
    "cited": "Source cited",
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

    rag_statuses = ["rag", "no_rag"]
    models = sorted(df["model_short"].unique())
    x = np.arange(len(models))
    width = 0.35

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

            offset = (i - 0.5) * width
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
        "rag_tool_called": "rag_called",
        "source_cited": "cited",
    }
    present = {k: v for k, v in components.items() if k in df.columns}
    if not present:
        print("  ⚠️  No RAG component columns — skipping plot_rag_score_components")
        return

    # Weights imported from rag_scorer.py — single source of truth.
    # Applied per-row via forcedist_tested; stacked total equals rag_benefit_score.
    df = df.copy()
    two_param = (
        df.get("forcedist_tested", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
    )
    for col in present:
        w = two_param.map(
            {
                True: COMPONENT_WEIGHTS_DOUBLE.get(col, 0.0),
                False: COMPONENT_WEIGHTS_SINGLE.get(col, 0.0),
            }
        )
        df[f"_wt_{col}"] = df[col].fillna(0.0) * w

    fs = PLOT_STYLE["font_sizes"]
    rag_statuses = ["rag", "no_rag"]
    models = sorted(df["model_short"].unique())
    example_ids = sorted(df["example_id"].dropna().unique())

    # Two bars per model group, one per prompt level
    bar_width = 0.35
    offsets = (
        np.linspace(-bar_width / 2, bar_width / 2, len(example_ids))
        if len(example_ids) > 1
        else [0.0]
    )
    prompt_hatches = dict(zip(example_ids, ["", "//"], strict=False))

    x = np.arange(len(models))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=PLOT_STYLE["figsize_full_width"],
        sharey=True,
    )

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
                vals = np.array(vals)
                ax.bar(
                    x + offset,
                    vals,
                    bar_width,
                    bottom=bottoms,
                    color=_COMPONENT_COLORS.get(key, COLOR_PALETTE[0]),
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=0.5,
                )
                bottoms += vals

        style = _RAG_DISPLAY.get(rs, {"label": rs})
        ax.set_title(style["label"], fontsize=fs["axes_title"])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=fs["tick_label"])
        ax.tick_params(axis="y", labelsize=fs["tick_label"])
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    axes[0].set_ylabel("Weighted score contribution", fontsize=fs["axes_label"])

    # Legend: colour patches for components + hatch patches for prompt levels
    component_patches = [
        mpatches.Patch(color=_COMPONENT_COLORS[key], label=label)
        for key, label in _COMPONENT_LABELS.items()
        if key in set(present.values())
    ]
    prompt_patches = [
        mpatches.Patch(
            facecolor="grey",
            hatch=prompt_hatches.get(eid, ""),
            edgecolor="white",
            label=_PROMPT_LABELS.get(int(eid), f"Prompt {int(eid)}"),
        )
        for eid in example_ids
    ]
    axes[1].legend(
        handles=component_patches + prompt_patches,
        fontsize=fs["legend"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
    )
    fig.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.close(fig)
    print(f"  ✅ {filename}")


# ── Plot 3: Raw accuracy comparison — ungated accuracy + RAG usage ────────────

# Maps effective_* scorer keys → raw accuracy column names for the ungated plot.
_EFFECTIVE_TO_RAW = {
    "effective_volfrac_accuracy": "volfrac_accuracy",
    "effective_forcedist_accuracy": "forcedist_accuracy",
    "rag_tool_called": "rag_tool_called",
}


def _renormalize_weights(src: dict[str, float]) -> dict[str, float]:
    """Remap effective_* keys to raw column names, drop source_cited, renormalise."""
    remapped = {}
    for orig_key, raw_key in _EFFECTIVE_TO_RAW.items():
        if orig_key in src:
            remapped[raw_key] = src[orig_key]
    total = sum(remapped.values()) or 1.0
    return {k: v / total for k, v in remapped.items()}


def plot_rag_accuracy_comparison(
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

    Weights are derived from the canonical scorer weights (source_cited
    excluded, renormalised to sum to 1.0).

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
        "rag_tool_called": "rag_called",
    }
    present = {k: v for k, v in components.items() if k in df.columns}
    if not present:
        print("  ⚠️  No accuracy columns — skipping plot_rag_accuracy_comparison")
        return

    # Derive weights: map effective_* → raw column, drop source_cited, renormalise.
    w_single = _renormalize_weights(COMPONENT_WEIGHTS_SINGLE)
    w_double = _renormalize_weights(COMPONENT_WEIGHTS_DOUBLE)

    # Apply per-row weights
    df = df.copy()
    two_param = (
        df.get("forcedist_tested", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
    )
    for col in present:
        w = two_param.map({True: w_double.get(col, 0.0), False: w_single.get(col, 0.0)})
        df[f"_wt_{col}"] = df[col].fillna(0.0) * w

    fs = PLOT_STYLE["font_sizes"]
    rag_statuses = ["rag", "no_rag"]
    models = sorted(df["model_short"].unique())
    example_ids = sorted(df["example_id"].dropna().unique())

    bar_width = 0.35
    offsets = (
        np.linspace(-bar_width / 2, bar_width / 2, len(example_ids))
        if len(example_ids) > 1
        else [0.0]
    )
    prompt_hatches = dict(zip(example_ids, ["", "//"], strict=False))

    x = np.arange(len(models))

    fig, axes = plt.subplots(
        1, 2, figsize=PLOT_STYLE["figsize_full_width"], sharey=True
    )

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

        style = _RAG_DISPLAY.get(rs, {"label": rs})
        ax.set_title(style["label"], fontsize=fs["axes_title"])
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=fs["tick_label"])
        ax.tick_params(axis="y", labelsize=fs["tick_label"])
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    axes[0].set_ylabel("Weighted accuracy contribution", fontsize=fs["axes_label"])

    # Legend: colour patches for components + hatch patches for prompt levels
    component_patches = [
        mpatches.Patch(color=_COMPONENT_COLORS[key], label=_COMPONENT_LABELS[key])
        for key in present.values()
        if key in _COMPONENT_COLORS
    ]
    prompt_patches = [
        mpatches.Patch(
            facecolor="grey",
            hatch=prompt_hatches.get(eid, ""),
            edgecolor="white",
            label=_PROMPT_LABELS.get(int(eid), f"Prompt {int(eid)}"),
        )
        for eid in example_ids
    ]
    axes[1].legend(
        handles=component_patches + prompt_patches,
        fontsize=fs["legend"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
    )
    fig.tight_layout()
    save_figure(fig, filename, output_dir)
    plt.close(fig)
    print(f"  ✅ {filename}")


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
                        "label": f"{m}\n{_PROMPT_LABELS.get(int(eid), str(eid))}",
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
    print("\n[1/4] RAG benefit score grouped bar...")
    plot_rag_benefit_score(df, output_dir=output_dir)

    print("\n[2/4] Score component breakdown (weighted, gated)...")
    plot_rag_score_components(df, output_dir=output_dir)

    print("\n[3/4] Raw accuracy comparison (ungated)...")
    plot_rag_accuracy_comparison(df, output_dir=output_dir)

    print("\n[4/4] RAG uplift delta bars...")
    plot_rag_uplift(df, output_dir=output_dir)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from utils import filter_by_problem, get_combined_design_df, load_data

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
