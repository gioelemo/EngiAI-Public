from pathlib import Path

import matplotlib.pyplot as plt


def plot_token_usage(combined_tools, output_dir):
    """Plot token usage from the combined tools data."""
    token_usage = combined_tools["total_tokens"]

    plt.figure(figsize=(10, 6))
    plt.hist(token_usage, bins=30, color="blue", alpha=0.7)
    plt.title("Token Usage Distribution")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.grid(True)

    output_path = Path(output_dir) / "token_usage.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Token usage plot saved to {output_path}")


def plot_latency(combined_tools, output_dir):
    """Plot latency from the combined tools data."""
    latency = combined_tools["latency_ms"]

    plt.figure(figsize=(10, 6))
    plt.hist(latency, bins=30, color="green", alpha=0.7)
    plt.title("Latency Distribution")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Frequency")
    plt.grid(True)

    output_path = Path(output_dir) / "latency.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Latency plot saved to {output_path}")
