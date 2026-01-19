"""
Shared utilities for plots.

- Data loading and cleaning
- Path configuration
- Plot styling
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Constants
DEFAULT_N_SAMPLES = 10
MIN_CORRELATION_SAMPLES = 2

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Data paths
DATA_PATHS = {
    "gpt_beams_global": RESULTS_DIR
    / "openai_gpt-4.1/beams2d/output_quality_global_metrics.csv",
    "gpt_beams_design": RESULTS_DIR
    / "openai_gpt-4.1/beams2d/output_quality_design_metrics.csv",
    "gpt_photonics_global": RESULTS_DIR
    / "openai_gpt-4.1/photonics2d/output_quality_global_metrics.csv",
    "gpt_photonics_design": RESULTS_DIR
    / "openai_gpt-4.1/photonics2d/output_quality_design_metrics.csv",
    "cgan_beams": Path(__file__).parent.parent.parent.parent
    / "cgan_cnn_2d_beams2d_metrics.csv",
}

# Plot style configuration
PLOT_STYLE = {
    "figsize_scatter": (8, 6),
    "figsize_bars": (12, 10),
    "figsize_violin": (10, 6),
    "dpi": 300,
    "markers": {"GPT-4.1": "o", "cGAN-CNN": "s"},
    "colors": {
        "beams2d": "#2196F3",
        "photonics2d": "#4CAF50",
        "GPT-4.1": "#2196F3",
        "cGAN-CNN": "#FF9800",
    },
    "alpha": 0.7,
    "marker_size": 80,
}


def _check_latex_available():
    """Check if LaTeX is available on the system."""
    return shutil.which("latex") is not None


def setup_style(use_latex=None):
    """Configure matplotlib/seaborn for publication-quality figures with LaTeX fonts.

    Args:
        use_latex: If True, force LaTeX. If False, disable LaTeX.
                   If None (default), auto-detect LaTeX availability.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.2)

    # Auto-detect LaTeX if not specified
    if use_latex is None:
        use_latex = _check_latex_available()

    if use_latex:
        try:
            plt.rcParams.update(
                {
                    # Use LaTeX for text rendering
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                    # LaTeX preamble for math support
                    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
                    # Consistent font sizes
                    "axes.labelsize": 12,
                    "axes.titlesize": 14,
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "legend.fontsize": 10,
                    "figure.titlesize": 14,
                }
            )
            print("Using LaTeX fonts for plots")
        except Exception:
            # Fall back if LaTeX setup fails
            use_latex = False

    if not use_latex:
        # Clean serif style without LaTeX (looks similar)
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "mathtext.fontset": "cm",  # Computer Modern math fonts
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.titlesize": 14,
            }
        )
        print("Using serif fonts (LaTeX not available)")


def get_output_dir():
    """Return output directory for figures."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def load_data():
    """
    Load all metrics CSVs, clean and deduplicate.

    Returns:
        dict: Dictionary with loaded DataFrames
    """
    data = {}

    # GPT-4.1 beams2d global
    if DATA_PATHS["gpt_beams_global"].exists():
        df = pd.read_csv(DATA_PATHS["gpt_beams_global"])
        df = df.drop_duplicates(subset=["seed", "n_samples"], keep="first")
        df = df[df["n_samples"] == DEFAULT_N_SAMPLES]
        df["model"] = "GPT-4.1"
        df["problem"] = "beams2d"
        data["gpt_beams_global"] = df

    # GPT-4.1 photonics2d global
    if DATA_PATHS["gpt_photonics_global"].exists():
        df = pd.read_csv(DATA_PATHS["gpt_photonics_global"])
        df = df.drop_duplicates(subset=["seed", "n_samples"], keep="first")
        df = df[df["n_samples"] == DEFAULT_N_SAMPLES]
        df["model"] = "GPT-4.1"
        df["problem"] = "photonics2d"
        data["gpt_photonics_global"] = df

    # cGAN baseline
    if DATA_PATHS["cgan_beams"].exists():
        df = pd.read_csv(DATA_PATHS["cgan_beams"])
        df = df.rename(columns={"viol": "rvc"})
        df["model"] = "cGAN-CNN"
        df["problem"] = "beams2d"
        data["cgan_beams_global"] = df

    # Design-level metrics
    if DATA_PATHS["gpt_beams_design"].exists():
        df = pd.read_csv(DATA_PATHS["gpt_beams_design"])
        df = df.drop_duplicates(subset=["seed", "example_id"], keep="first")
        df["model"] = "GPT-4.1"
        data["gpt_beams_design"] = df

    if DATA_PATHS["gpt_photonics_design"].exists():
        df = pd.read_csv(DATA_PATHS["gpt_photonics_design"])
        df = df.drop_duplicates(subset=["seed", "example_id"], keep="first")
        df["model"] = "GPT-4.1"
        data["gpt_photonics_design"] = df

    return data


def get_combined_global_df(data):
    """Combine all global metrics into a single DataFrame."""
    keys = ["gpt_beams_global", "gpt_photonics_global", "cgan_beams_global"]
    dfs = [data[key] for key in keys if key in data]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def get_combined_design_df(data):
    """Combine all design-level metrics into a single DataFrame."""
    design_dfs = []

    if "gpt_beams_design" in data:
        df = data["gpt_beams_design"].copy()
        df["source"] = "GPT-4.1 (beams2d)"
        df["problem"] = "beams2d"
        design_dfs.append(df)

    if "gpt_photonics_design" in data:
        df = data["gpt_photonics_design"].copy()
        df["source"] = "GPT-4.1 (photonics2d)"
        df["problem"] = "photonics2d"
        design_dfs.append(df)

    return pd.concat(design_dfs, ignore_index=True) if design_dfs else None


def save_figure(fig, filename, output_dir=None):
    """Save figure to output directory."""
    if output_dir is None:
        output_dir = get_output_dir()
    path = output_dir / filename
    fig.savefig(path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    print(f"Saved: {path}")
    return path
