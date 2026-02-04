"""
Token and Latency Distribution Plots (NeurIPS format)
"""

from pathlib import Path

import matplotlib.pyplot as plt

from utils import PLOT_STYLE, save_figure, setup_style


def plot_token_usage(combined_tools, output_dir):
    """Plot token usage from the combined tools data (NeurIPS format)."""
    # Check if token usage data is available
    if "total_tokens" not in combined_tools.columns:
        print("⚠️  Token usage data (total_tokens) not available, skipping plot")
        return

    setup_style()
    token_usage = combined_tools["total_tokens"]

    # Skip if all values are null
    if token_usage.isna().all():
        print("⚠️  Token usage data is all null, skipping plot")
        return

    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col"], constrained_layout=True
    )
    ax.hist(
        token_usage,
        bins=20,
        color="#0072B2",  # Blue from colorblind-friendly palette
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)

    save_figure(fig, "token_usage.png", Path(output_dir))
    plt.close()


def plot_latency(combined_tools, output_dir):
    """Plot latency from the combined tools data (NeurIPS format)."""
    # Check if latency data is available
    if "latency_ms" not in combined_tools.columns:
        print("⚠️  Latency data (latency_ms) not available, skipping plot")
        return

    setup_style()
    latency = combined_tools["latency_ms"]

    # Skip if all values are null
    if latency.isna().all():
        print("⚠️  Latency data is all null, skipping plot")
        return

    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize_single_col"], constrained_layout=True
    )
    ax.hist(
        latency,
        bins=20,
        color="#E69F00",  # Orange from colorblind-friendly palette
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)

    save_figure(fig, "latency.png", Path(output_dir))
    plt.close()
