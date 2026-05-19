"""Generate a connected design comparison figure for the paper.

Two modes:
  --find-best   Scan all models/styles to find the best disagreeing pair
  (default)     Generate the 2-row x 3-col comparison figure

Usage:
    python plot_design_comparison_paper.py --find-best
    python plot_design_comparison_paper.py \
        --model-a openai_gpt-5-mini --model-b ollama_qwen3_4b-instruct-2507-q8_0 \
        --style workflow-conditional --seed 1 --example-id 4 \
        --output paper-revision/asmeconf/figures/benchmarks/comparison_connected.pdf
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# Add project root to path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.evaluations.plots.utils import (
    FULL_WIDTH,
    KNOWN_PROMPT_STYLES,
    MODELS_DIR,
    _get_model_label,
    setup_style,
)
from benchmarks.shared.utils import get_hf_dataset


def load_all_design_data() -> dict[tuple, dict]:
    """Load all design_data.json files from the results directory.

    Returns:
        Dict mapping (model_dir_name, style, example_id, seed) to design entry.
    """
    all_data: dict[tuple, dict] = {}
    if not MODELS_DIR.exists():
        return all_data

    for model_dir in sorted(MODELS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        beams_dir = model_dir / "beams2d"
        if not beams_dir.exists():
            continue
        for style_dir in beams_dir.iterdir():
            if not style_dir.is_dir() or style_dir.name not in KNOWN_PROMPT_STYLES:
                continue
            design_path = style_dir / "no_rag" / "design_data.json"
            if not design_path.exists():
                continue
            with design_path.open() as f:
                data = json.load(f)
            # Index by (model, style, example_id, seed)
            for entry in data:
                key = (
                    model_dir.name,
                    style_dir.name,
                    entry.get("example_id"),
                    entry.get("seed"),
                )
                all_data[key] = entry

    return all_data


def find_best(top_n: int = 10):
    """Find the best connected examples with maximum model disagreement."""
    all_data = load_all_design_data()

    # Group by (style, example_id, seed) -> {model: entry}
    groups: dict[tuple, dict[str, dict]] = {}
    for (model, style, eid, seed), entry in all_data.items():
        group_key = (style, eid, seed)
        if group_key not in groups:
            groups[group_key] = {}
        groups[group_key][model] = entry

    # Score all model pairs for each group
    candidates = []
    for (style, eid, seed), model_entries in groups.items():
        models_with_design = {
            m: e
            for m, e in model_entries.items()
            if e.get("design_found") and e.get("design") is not None
        }
        min_models_for_pair = 2
        if len(models_with_design) < min_models_for_pair:
            continue

        for model_a, model_b in combinations(sorted(models_with_design.keys()), 2):
            ea, eb = models_with_design[model_a], models_with_design[model_b]
            tc_a = ea.get("task_completion_score", 0) or 0
            tc_b = eb.get("task_completion_score", 0) or 0
            iou_a = ea.get("iou", 0) or 0
            iou_b = eb.get("iou", 0) or 0

            tc_diff = abs(tc_a - tc_b)
            iou_diff = abs(iou_a - iou_b)

            # Primary: TC disagreement (one passes, one fails)
            # Secondary: IoU difference (visually interesting)
            score = tc_diff * 10 + iou_diff

            candidates.append(
                {
                    "style": style,
                    "example_id": eid,
                    "seed": seed,
                    "model_a": model_a,
                    "model_b": model_b,
                    "tc_a": tc_a,
                    "tc_b": tc_b,
                    "iou_a": iou_a,
                    "iou_b": iou_b,
                    "tc_diff": tc_diff,
                    "iou_diff": iou_diff,
                    "score": score,
                }
            )

    candidates.sort(key=lambda c: c["score"], reverse=True)

    print(f"\nTop {top_n} candidates (highest disagreement):\n")
    print(
        f"{'Rank':<5} {'Style':<25} {'Seed':<5} {'ExID':<5} "
        f"{'Model A':<35} {'Model B':<35} "
        f"{'TC_A':<6} {'TC_B':<6} {'IoU_A':<7} {'IoU_B':<7} {'Score':<7}"
    )
    print("-" * 150)

    for i, c in enumerate(candidates[:top_n]):
        label_a = _get_model_label(c["model_a"])
        label_b = _get_model_label(c["model_b"])
        print(
            f"{i + 1:<5} {c['style']:<25} {c['seed']:<5} {c['example_id']:<5} "
            f"{label_a + ' (' + c['model_a'] + ')':<35} "
            f"{label_b + ' (' + c['model_b'] + ')':<35} "
            f"{c['tc_a']:<6.2f} {c['tc_b']:<6.2f} "
            f"{c['iou_a']:<7.3f} {c['iou_b']:<7.3f} {c['score']:<7.3f}"
        )

    return candidates


def generate_figure(  # noqa: PLR0913, PLR0915
    model_a: str,
    model_b: str,
    style: str,
    seed: int,
    example_id: int,
    output: str,
):
    """Generate the 2-row x 3-col comparison figure."""
    all_data = load_all_design_data()

    entry_a = all_data.get((model_a, style, example_id, seed))
    entry_b = all_data.get((model_b, style, example_id, seed))

    if entry_a is None or entry_b is None:
        print("ERROR: Could not find data for the specified parameters.")
        if entry_a is None:
            print(
                f"  Missing: model={model_a}, style={style}, seed={seed}, eid={example_id}"
            )
        if entry_b is None:
            print(
                f"  Missing: model={model_b}, style={style}, seed={seed}, eid={example_id}"
            )
        sys.exit(1)

    design_a = np.array(entry_a["design"])
    design_b = np.array(entry_b["design"])

    # Load ground truth: prefer stored, fall back to HuggingFace dataset
    gt_raw = entry_a.get("gt_design") or entry_b.get("gt_design")
    if gt_raw is not None:
        gt = np.array(gt_raw)
    else:
        print(f"Loading ground truth from HuggingFace for example_id={example_id}...")
        dataset = get_hf_dataset("IDEALLab/beams_2d_50_100_v0", split="test")
        gt = np.array(dataset[example_id]["optimal_design"])

    tc_a = entry_a.get("task_completion_score", 0) or 0
    tc_b = entry_b.get("task_completion_score", 0) or 0
    iou_a = entry_a.get("iou", 0) or 0
    iou_b = entry_b.get("iou", 0) or 0

    label_a = _get_model_label(model_a)
    label_b = _get_model_label(model_b)

    # Setup publication style
    setup_style()

    # Create figure: 2 rows x 3 cols + space for colorbars
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(3.25, 2.0),
        gridspec_kw={"wspace": 0.08, "hspace": 0.35, "left": 0.18},
    )

    # Compute differences
    diff_a = np.abs(design_a - gt)
    diff_b = np.abs(design_b - gt)

    # Use a perceptually stronger colormap for differences
    diff_cmap = "hot_r"

    # Font sizes
    title_fs = 7
    label_fs = 5

    # Plot row A
    axes[0, 0].imshow(design_a, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
    axes[0, 1].imshow(gt, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
    axes[0, 2].imshow(diff_a, cmap=diff_cmap, vmin=0, vmax=1, aspect="equal")

    # Plot row B
    axes[1, 0].imshow(design_b, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
    axes[1, 1].imshow(gt, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
    axes[1, 2].imshow(diff_b, cmap=diff_cmap, vmin=0, vmax=1, aspect="equal")

    # MAE annotations on difference panels
    for ax, diff in [(axes[0, 2], diff_a), (axes[1, 2], diff_b)]:
        mae = diff.mean()
        ax.text(
            0.97,
            0.03,
            f"MAE={mae:.3f}",
            transform=ax.transAxes,
            fontsize=5,
            fontweight="bold",
            ha="right",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "0.5",
            },
        )

    # Column titles (top row only)
    axes[0, 0].set_title("Agent Design", fontsize=title_fs, fontweight="bold")
    axes[0, 1].set_title("Ground Truth", fontsize=title_fs, fontweight="bold")
    axes[0, 2].set_title("Difference", fontsize=title_fs, fontweight="bold")

    # Row labels as ylabels
    row_label_a = f"{label_a}\n(TC={tc_a:.1f}, IoU={iou_a:.2f})"
    row_label_b = f"{label_b}\n(TC={tc_b:.1f}, IoU={iou_b:.2f})"
    axes[0, 0].set_ylabel(row_label_a, fontsize=label_fs, labelpad=3)
    axes[1, 0].set_ylabel(row_label_b, fontsize=label_fs, labelpad=3)

    # Turn off all ticks
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    # Shared colorbars
    # Gray colorbar spanning the left 2 columns
    cbar_gray_ax = fig.add_axes((0.125, 0.02, 0.42, 0.025))
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="gray_r", norm=Normalize(0, 1)),
        cax=cbar_gray_ax,
        orientation="horizontal",
        label="Material Density",
    )
    cbar_gray_ax.tick_params(labelsize=5)
    cbar_gray_ax.set_xlabel("Material Density", fontsize=5)

    # Red colorbar for difference column
    cbar_red_ax = fig.add_axes((0.60, 0.02, 0.25, 0.025))
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=diff_cmap, norm=Normalize(0, 1)),
        cax=cbar_red_ax,
        orientation="horizontal",
    )
    cbar_red_ax.tick_params(labelsize=5)
    cbar_red_ax.set_xlabel("Absolute Difference", fontsize=5)

    # Save
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")

    # Also save the other format
    if output_path.suffix == ".pdf":
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {png_path}")
    else:
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {pdf_path}")

    plt.close(fig)
    print("\nFigure info:")
    print(f"  Style: {style}, Seed: {seed}, Example ID: {example_id}")
    print(
        f"  Model A: {label_a} (TC={tc_a:.2f}, IoU={iou_a:.3f}, MAE={diff_a.mean():.3f})"
    )
    print(
        f"  Model B: {label_b} (TC={tc_b:.2f}, IoU={iou_b:.3f}, MAE={diff_b.mean():.3f})"
    )


def _load_example_data(model_a, model_b, style, seed, example_id):
    """Load data for both models and ground truth. Shared by both layout functions."""
    all_data = load_all_design_data()

    entry_a = all_data.get((model_a, style, example_id, seed))
    entry_b = all_data.get((model_b, style, example_id, seed))

    if entry_a is None or entry_b is None:
        print("ERROR: Could not find data for the specified parameters.")
        if entry_a is None:
            print(
                f"  Missing: model={model_a}, style={style}, seed={seed}, eid={example_id}"
            )
        if entry_b is None:
            print(
                f"  Missing: model={model_b}, style={style}, seed={seed}, eid={example_id}"
            )
        sys.exit(1)

    design_a = np.array(entry_a["design"])
    design_b = np.array(entry_b["design"])

    gt_raw = entry_a.get("gt_design") or entry_b.get("gt_design")
    if gt_raw is not None:
        gt = np.array(gt_raw)
    else:
        print(f"Loading ground truth from HuggingFace for example_id={example_id}...")
        dataset = get_hf_dataset("IDEALLab/beams_2d_50_100_v0", split="test")
        gt = np.array(dataset[example_id]["optimal_design"])

    tc_a = entry_a.get("task_completion_score", 0) or 0
    tc_b = entry_b.get("task_completion_score", 0) or 0
    iou_a = entry_a.get("iou", 0) or 0
    iou_b = entry_b.get("iou", 0) or 0

    label_a = _get_model_label(model_a)
    label_b = _get_model_label(model_b)

    return design_a, design_b, gt, tc_a, tc_b, iou_a, iou_b, label_a, label_b


def _save_figure(  # noqa: PLR0913
    fig,
    output,
    label_a,
    label_b,
    style,
    seed,
    example_id,
    tc_a,
    tc_b,
    iou_a,
    iou_b,
    diff_a,
    diff_b,
):
    """Save figure in PDF+PNG and print info. Shared by both layout functions."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")

    if output_path.suffix == ".pdf":
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {png_path}")
    else:
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {pdf_path}")

    plt.close(fig)
    print("\nFigure info:")
    print(f"  Style: {style}, Seed: {seed}, Example ID: {example_id}")
    print(
        f"  Model A: {label_a} (TC={tc_a:.2f}, IoU={iou_a:.3f}, MAE={diff_a.mean():.3f})"
    )
    print(
        f"  Model B: {label_b} (TC={tc_b:.2f}, IoU={iou_b:.3f}, MAE={diff_b.mean():.3f})"
    )


def generate_figure_2x3v2(  # noqa: PLR0913
    model_a: str,
    model_b: str,
    style: str,
    seed: int,
    example_id: int,
    output: str,
):
    """Generate a 2x3-style layout with ground truth shown once between the rows.

    Layout (5 rows x 2 cols via GridSpec):
        Row 0: Model A design  | Difference A
        Row 1: (ground truth centered, spanning both cols)
        Row 2: Model B design  | Difference B
    """
    (design_a, design_b, gt, tc_a, tc_b, iou_a, iou_b, label_a, label_b) = (
        _load_example_data(model_a, model_b, style, seed, example_id)
    )

    diff_a = np.abs(design_a - gt)
    diff_b = np.abs(design_b - gt)

    setup_style()

    diff_cmap = "hot_r"
    title_fs = 11
    label_fs = 10

    fig = plt.figure(figsize=(6.75, 4.8))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1, 1, 1],
        hspace=0.25,
        wspace=0.08,
    )

    # Row 0: Model A
    ax_a = fig.add_subplot(gs[0, 0])
    ax_da = fig.add_subplot(gs[0, 1])
    ax_a.imshow(design_a, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
    ax_da.imshow(diff_a, cmap=diff_cmap, vmin=0, vmax=1, aspect="equal")

    row_label_a = f"{label_a}\n(TC={tc_a:.1f}, IoU={iou_a:.2f})"
    ax_a.set_title("Agent Design", fontsize=title_fs, fontweight="bold")
    ax_da.set_title("Difference", fontsize=title_fs, fontweight="bold")
    ax_a.set_ylabel(row_label_a, fontsize=label_fs, fontweight="bold")

    # Row 1: Ground truth centered
    ax_gt = fig.add_subplot(gs[1, :])
    ax_gt.imshow(gt, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
    ax_gt.set_ylabel("Ground Truth", fontsize=label_fs, fontweight="bold")

    # Row 2: Model B
    ax_b = fig.add_subplot(gs[2, 0])
    ax_db = fig.add_subplot(gs[2, 1])
    ax_b.imshow(design_b, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
    ax_db.imshow(diff_b, cmap=diff_cmap, vmin=0, vmax=1, aspect="equal")

    row_label_b = f"{label_b}\n(TC={tc_b:.1f}, IoU={iou_b:.2f})"
    ax_b.set_ylabel(row_label_b, fontsize=label_fs, fontweight="bold")

    # MAE annotations
    for ax, diff in [(ax_da, diff_a), (ax_db, diff_b)]:
        mae = diff.mean()
        ax.text(
            0.97,
            0.03,
            f"MAE={mae:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="right",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "0.5",
            },
        )

    # Turn off ticks
    for ax in [ax_a, ax_da, ax_gt, ax_b, ax_db]:
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colorbars
    cbar_gray_ax = fig.add_axes((0.125, 0.02, 0.38, 0.02))
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="gray_r", norm=Normalize(0, 1)),
        cax=cbar_gray_ax,
        orientation="horizontal",
    )
    cbar_gray_ax.tick_params(labelsize=7)
    cbar_gray_ax.set_xlabel("Material Density", fontsize=8)

    cbar_diff_ax = fig.add_axes((0.57, 0.02, 0.33, 0.02))
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=diff_cmap, norm=Normalize(0, 1)),
        cax=cbar_diff_ax,
        orientation="horizontal",
    )
    cbar_diff_ax.tick_params(labelsize=7)
    cbar_diff_ax.set_xlabel("Absolute Difference", fontsize=8)

    _save_figure(
        fig,
        output,
        label_a,
        label_b,
        style,
        seed,
        example_id,
        tc_a,
        tc_b,
        iou_a,
        iou_b,
        diff_a,
        diff_b,
    )


def generate_figure_1x6(  # noqa: PLR0913
    model_a: str,
    model_b: str,
    style: str,
    seed: int,
    example_id: int,
    output: str,
):
    """Generate a single-row layout with all 6 panels: A | GT | Diff A | B | GT | Diff B."""
    (design_a, design_b, gt, tc_a, tc_b, iou_a, iou_b, label_a, label_b) = (
        _load_example_data(model_a, model_b, style, seed, example_id)
    )

    diff_a = np.abs(design_a - gt)
    diff_b = np.abs(design_b - gt)

    setup_style()

    diff_cmap = "hot_r"
    title_fs = 8
    suptitle_fs = 8

    # Use FULL_WIDTH from utils; 7 gridspec columns (3 + separator + 3)
    fig_h = FULL_WIDTH / 6 * 0.5 + 0.55  # image height + space for titles/colorbars
    fig = plt.figure(figsize=(FULL_WIDTH, fig_h))
    gs = fig.add_gridspec(
        1,
        7,
        width_ratios=[1, 1, 1, 0.08, 1, 1, 1],
        wspace=0.04,
        left=0.01,
        right=0.99,
        top=0.78,
        bottom=0.22,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in [0, 1, 2, 4, 5, 6]]
    # Draw vertical separator line between the two groups
    sep_ax = fig.add_subplot(gs[0, 3])
    sep_ax.set_xlim(0, 1)
    sep_ax.set_ylim(0, 1)
    sep_ax.axvline(0.5, color="0.4", linewidth=0.8, linestyle="-")
    sep_ax.set_axis_off()

    # Model A group
    axes[0].imshow(design_a, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
    axes[1].imshow(gt, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
    axes[2].imshow(diff_a, cmap=diff_cmap, vmin=0, vmax=1, aspect="equal")

    # Model B group
    axes[3].imshow(design_b, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
    axes[4].imshow(gt, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
    axes[5].imshow(diff_b, cmap=diff_cmap, vmin=0, vmax=1, aspect="equal")

    # Column titles
    for i, t in enumerate(["Agent", "GT", "Diff", "Agent", "GT", "Diff"]):
        axes[i].set_title(t, fontsize=title_fs, fontweight="bold", pad=2)

    # Group supertitles
    fig.text(
        0.25,
        0.97,
        f"{label_a}  (TC={tc_a:.1f}, IoU={iou_a:.2f})",
        ha="center",
        va="top",
        fontsize=suptitle_fs,
        fontweight="bold",
    )
    fig.text(
        0.75,
        0.97,
        f"{label_b}  (TC={tc_b:.1f}, IoU={iou_b:.2f})",
        ha="center",
        va="top",
        fontsize=suptitle_fs,
        fontweight="bold",
    )

    # MAE annotations
    for ax, diff in [(axes[2], diff_a), (axes[5], diff_b)]:
        mae = diff.mean()
        ax.text(
            0.95,
            0.05,
            f"MAE={mae:.3f}",
            transform=ax.transAxes,
            fontsize=5.5,
            fontweight="bold",
            ha="right",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.1",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "0.5",
                "linewidth": 0.5,
            },
        )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Colorbars — tight at bottom
    cbar_gray_ax = fig.add_axes((0.01, 0.06, 0.45, 0.04))
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="gray_r", norm=Normalize(0, 1)),
        cax=cbar_gray_ax,
        orientation="horizontal",
    )
    cbar_gray_ax.tick_params(labelsize=6)
    cbar_gray_ax.set_xlabel("Material Density", fontsize=6.5, labelpad=1)

    cbar_diff_ax = fig.add_axes((0.54, 0.06, 0.45, 0.04))
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=diff_cmap, norm=Normalize(0, 1)),
        cax=cbar_diff_ax,
        orientation="horizontal",
    )
    cbar_diff_ax.tick_params(labelsize=6)
    cbar_diff_ax.set_xlabel("Absolute Difference", fontsize=6.5, labelpad=1)

    _save_figure(
        fig,
        output,
        label_a,
        label_b,
        style,
        seed,
        example_id,
        tc_a,
        tc_b,
        iou_a,
        iou_b,
        diff_a,
        diff_b,
    )


def generate_figure_1x2(  # noqa: PLR0913
    model_a: str,
    model_b: str,
    style: str,
    seed: int,
    example_id: int,
    output: str,
):
    """Generate a single-row layout: Model A | Ground Truth | Model B."""
    (design_a, design_b, gt, tc_a, tc_b, iou_a, iou_b, label_a, label_b) = (
        _load_example_data(model_a, model_b, style, seed, example_id)
    )

    diff_a = np.abs(design_a - gt)
    diff_b = np.abs(design_b - gt)

    setup_style()

    # 1 row x 3 cols: Model A | Ground Truth | Model B
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6.75, 2.4),
        gridspec_kw={"wspace": 0.08},
    )

    title_fs = 11
    label_fs = 9

    axes[0].imshow(design_a, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
    axes[1].imshow(gt, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
    axes[2].imshow(design_b, cmap="gray_r", vmin=0, vmax=1, aspect="equal")

    # Titles with model name + metrics
    caption_a = f"{label_a}\n(TC={tc_a:.1f}, IoU={iou_a:.2f})"
    caption_b = f"{label_b}\n(TC={tc_b:.1f}, IoU={iou_b:.2f})"
    axes[0].set_title(caption_a, fontsize=label_fs, fontweight="bold")
    axes[1].set_title("Ground Truth", fontsize=title_fs, fontweight="bold")
    axes[2].set_title(caption_b, fontsize=label_fs, fontweight="bold")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Single shared colorbar
    cbar_ax = fig.add_axes((0.25, 0.02, 0.50, 0.03))
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="gray_r", norm=Normalize(0, 1)),
        cax=cbar_ax,
        orientation="horizontal",
    )
    cbar_ax.tick_params(labelsize=7)
    cbar_ax.set_xlabel("Material Density", fontsize=8)

    _save_figure(
        fig,
        output,
        label_a,
        label_b,
        style,
        seed,
        example_id,
        tc_a,
        tc_b,
        iou_a,
        iou_b,
        diff_a,
        diff_b,
    )


def main():
    parser = argparse.ArgumentParser(description="Design comparison figure for paper")
    parser.add_argument(
        "--find-best", action="store_true", help="Find best disagreeing examples"
    )
    parser.add_argument(
        "--top-n", type=int, default=10, help="Number of candidates to show"
    )
    parser.add_argument("--model-a", type=str, help="Model A directory name")
    parser.add_argument("--model-b", type=str, help="Model B directory name")
    parser.add_argument("--style", type=str, help="Prompt style")
    parser.add_argument("--seed", type=int, help="Seed")
    parser.add_argument("--example-id", type=int, help="Example ID")
    parser.add_argument(
        "--layout",
        type=str,
        choices=["2x3", "2x3v2", "1x6", "1x2"],
        default="2x3",
        help="Figure layout: 2x3, 2x3v2 (shared GT row), 1x6 (all 6 in one row), or 1x2 (designs only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper-revision/asmeconf/figures/benchmarks/comparison_connected.pdf",
        help="Output path",
    )
    args = parser.parse_args()

    if args.find_best:
        find_best(top_n=args.top_n)
    elif (
        args.model_a
        and args.model_b
        and args.style
        and args.seed is not None
        and args.example_id is not None
    ):
        layout_map = {
            "2x3": generate_figure,
            "2x3v2": generate_figure_2x3v2,
            "1x6": generate_figure_1x6,
            "1x2": generate_figure_1x2,
        }
        gen_fn = layout_map[args.layout]
        gen_fn(
            model_a=args.model_a,
            model_b=args.model_b,
            style=args.style,
            seed=args.seed,
            example_id=args.example_id,
            output=args.output,
        )
    else:
        parser.print_help()
        print(
            "\nEither use --find-best or provide all of: --model-a, --model-b, --style, --seed, --example-id"
        )


if __name__ == "__main__":
    main()
