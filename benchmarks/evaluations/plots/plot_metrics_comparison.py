#!/usr/bin/env python3
"""
Metrics Comparison Bar Chart

Compares DPP, MMD, FOG, RVC across models and problems.
Shows mean ± std for each metric.
"""

import matplotlib.pyplot as plt
import numpy as np

from utils import (
    PLOT_STYLE,
    get_combined_global_df,
    load_data,
    save_figure,
    setup_style,
)


def plot_metrics_comparison(combined_df, output_path=None):
    """Create bar chart comparing metrics across models."""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=PLOT_STYLE["figsize_bars"])

    metrics = [
        ("dpp", "DPP Diversity", "higher is better"),
        ("mmd", "MMD Distribution Match", "lower is better"),
        ("fog", "Final Optimality Gap", "lower is better"),
        ("rvc", "Constraint Violation Rate", "lower is better"),
    ]

    groups = combined_df.groupby(["model", "problem"])

    for idx, (metric, title, direction) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        means = []
        stds = []
        labels = []

        for (model, problem), group in groups:
            values = group[metric].dropna()
            if len(values) > 0:
                means.append(values.mean())
                stds.append(values.std())
                labels.append(f"{model}\n({problem})")

        if means:
            x = np.arange(len(labels))
            colors = [
                PLOT_STYLE["colors"]["GPT-4.1"]
                if "GPT" in label
                else PLOT_STYLE["colors"]["cGAN-CNN"]
                for label in labels
            ]

            bars = ax.bar(
                x,
                means,
                yerr=stds,
                capsize=5,
                color=colors,
                alpha=0.8,
                edgecolor="white",
                linewidth=1,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel(metric.upper(), fontsize=11)
            ax.set_title(f"{title}\n({direction})", fontsize=12, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)

            # Value annotations
            for bar, mean, _std in zip(bars, means, stds, strict=False):
                ax.annotate(
                    f"{mean:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def main():
    """Generate metrics comparison plot."""
    print("Loading data...")
    data = load_data()
    combined_df = get_combined_global_df(data)

    if combined_df is not None:
        plot_metrics_comparison(combined_df, "metrics_comparison.png")
    else:
        print("No data available")


if __name__ == "__main__":
    main()
