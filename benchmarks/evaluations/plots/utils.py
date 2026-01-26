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
    "gpt_beams_tools": RESULTS_DIR / "openai_gpt-4.1/beams2d/data.csv",
    "gpt_photonics_global": RESULTS_DIR
    / "openai_gpt-4.1/photonics2d/output_quality_global_metrics.csv",
    "gpt_photonics_design": RESULTS_DIR
    / "openai_gpt-4.1/photonics2d/output_quality_design_metrics.csv",
    "gpt_photonics_tools": RESULTS_DIR / "openai_gpt-4.1/photonics2d/data.csv",
    "gpt5_beams_global": RESULTS_DIR
    / "openai_gpt-5.1/beams2d/output_quality_global_metrics.csv",
    "gpt5_beams_design": RESULTS_DIR
    / "openai_gpt-5.1/beams2d/output_quality_design_metrics.csv",
    "gpt5_beams_tools": RESULTS_DIR / "openai_gpt-5.1/beams2d/data.csv",
    "gpt5_photonics_global": RESULTS_DIR
    / "openai_gpt-5.1/photonics2d/output_quality_global_metrics.csv",
    "gpt5_photonics_design": RESULTS_DIR
    / "openai_gpt-5.1/photonics2d/output_quality_design_metrics.csv",
    "gpt5_photonics_tools": RESULTS_DIR / "openai_gpt-5.1/photonics2d/data.csv",
    "cgan_beams": Path(__file__).parent.parent.parent.parent
    / "cgan_cnn_2d_beams2d_metrics.csv",
}

# NeurIPS 2-column format dimensions (inches)
# Single column: ~3.25", Full width: ~6.75"
NEURIPS_COLUMN_WIDTH = 3.25
NEURIPS_FULL_WIDTH = 6.75

# Colorblind-friendly palette (Okabe-Ito)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky_blue": "#56B4E9",
    "vermillion": "#D55E00",
    "yellow": "#F0E442",
    "black": "#000000",
}

# Plot style configuration for NeurIPS publication
PLOT_STYLE = {
    # Figure sizes for 2-column conference format
    "figsize_single_col": (NEURIPS_COLUMN_WIDTH, 2.4),
    "figsize_single_col_tall": (NEURIPS_COLUMN_WIDTH, 3.0),
    "figsize_full_width": (NEURIPS_FULL_WIDTH, 2.8),
    "figsize_full_width_tall": (NEURIPS_FULL_WIDTH, 4.0),
    # Legacy sizes (for reference, prefer new sizes)
    "figsize_scatter": (NEURIPS_COLUMN_WIDTH, 2.4),
    "figsize_bars": (NEURIPS_FULL_WIDTH, 4.0),
    "figsize_violin": (NEURIPS_COLUMN_WIDTH, 2.4),
    "dpi": 300,
    "markers": {"GPT-4.1": "o", "GPT-5.1": "^", "cGAN-CNN": "s"},
    # Colorblind-friendly colors mapped to models/problems
    "colors": {
        "beams2d": COLORS["blue"],
        "photonics2d": COLORS["orange"],
        "GPT-4.1": COLORS["blue"],
        "GPT-5.1": COLORS["purple"],
        "cGAN-CNN": COLORS["vermillion"],
    },
    "alpha": 0.7,
    "marker_size": 40,  # Smaller for publication
    # Font sizes for print readability
    "font_sizes": {
        "axes_label": 8,
        "axes_title": 9,  # Not used (no titles)
        "tick_label": 7,
        "legend": 7,
        "annotation": 6,
    },
}


def setup_style(use_latex=None):
    """Configure matplotlib/seaborn for NeurIPS publication-quality figures.

    Args:
        use_latex: If True, force LaTeX. If False, disable LaTeX.
                   If None (default), auto-detect LaTeX availability.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=0.9)

    font_sizes = PLOT_STYLE["font_sizes"]

    # Auto-detect LaTeX if not specified
    if use_latex is None:
        use_latex = shutil.which("latex") is not None

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
                    # NeurIPS publication font sizes
                    "axes.labelsize": font_sizes["axes_label"],
                    "axes.titlesize": font_sizes["axes_title"],
                    "xtick.labelsize": font_sizes["tick_label"],
                    "ytick.labelsize": font_sizes["tick_label"],
                    "legend.fontsize": font_sizes["legend"],
                    "figure.titlesize": font_sizes["axes_title"],
                    # Line and marker settings
                    "lines.linewidth": 1.0,
                    "lines.markersize": 4,
                    "axes.linewidth": 0.5,
                    "grid.linewidth": 0.3,
                    "grid.alpha": 0.4,
                    # Legend settings
                    "legend.framealpha": 0.9,
                    "legend.edgecolor": "0.8",
                    "legend.borderpad": 0.3,
                    "legend.handlelength": 1.5,
                    # Tick settings
                    "xtick.major.width": 0.5,
                    "ytick.major.width": 0.5,
                    "xtick.major.size": 3,
                    "ytick.major.size": 3,
                }
            )
        except Exception:
            # Fall back if LaTeX setup fails
            pass
        else:
            return  # Successfully configured LaTeX

    # Clean serif style without LaTeX (looks similar)
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern math fonts
            # NeurIPS publication font sizes
            "axes.labelsize": font_sizes["axes_label"],
            "axes.titlesize": font_sizes["axes_title"],
            "xtick.labelsize": font_sizes["tick_label"],
            "ytick.labelsize": font_sizes["tick_label"],
            "legend.fontsize": font_sizes["legend"],
            "figure.titlesize": font_sizes["axes_title"],
            # Line and marker settings
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.3,
            "grid.alpha": 0.4,
            # Legend settings
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "legend.borderpad": 0.3,
            "legend.handlelength": 1.5,
            # Tick settings
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )


def get_output_dir():
    """Return output directory for figures."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def _load_global_metrics(path, model, problem):
    """Load and clean global metrics for a specific model/problem."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None
    # Filter to rows with valid n_samples (numeric)
    df = df[pd.to_numeric(df["n_samples"], errors="coerce").notna()]
    df["n_samples"] = pd.to_numeric(df["n_samples"])
    df = df.drop_duplicates(subset=["seed", "n_samples"], keep="first")
    df = df[df["n_samples"] == DEFAULT_N_SAMPLES]
    df["model"] = model
    df["problem"] = problem
    return df


def _load_design_metrics(path, model):
    """Load and clean design-level metrics for a specific model."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None
    df = df.drop_duplicates(subset=["seed", "example_id"], keep="first")
    df["model"] = model
    return df


def _load_cgan_metrics(path):
    """Load and clean cGAN baseline metrics."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None
    df = df.rename(columns={"viol": "rvc"})
    df["model"] = "cGAN-CNN"
    df["problem"] = "beams2d"
    return df


def load_data():
    """
    Load all metrics CSVs, clean and deduplicate.

    Returns:
        dict: Dictionary with loaded DataFrames
    """
    data = {}

    # Global metrics
    global_configs = [
        ("gpt_beams_global", "GPT-4.1", "beams2d"),
        ("gpt_photonics_global", "GPT-4.1", "photonics2d"),
        ("gpt5_beams_global", "GPT-5.1", "beams2d"),
        ("gpt5_photonics_global", "GPT-5.1", "photonics2d"),
    ]
    for key, model, problem in global_configs:
        df = _load_global_metrics(DATA_PATHS[key], model, problem)
        if df is not None:
            data[key] = df

    # cGAN baseline
    df = _load_cgan_metrics(DATA_PATHS["cgan_beams"])
    if df is not None:
        data["cgan_beams_global"] = df

    # Design-level metrics
    design_configs = [
        ("gpt_beams_design", "GPT-4.1"),
        ("gpt_photonics_design", "GPT-4.1"),
        ("gpt5_beams_design", "GPT-5.1"),
        ("gpt5_photonics_design", "GPT-5.1"),
    ]
    for key, model in design_configs:
        df = _load_design_metrics(DATA_PATHS[key], model)
        if df is not None:
            data[key] = df

    return data


def load_tool_usage_data():
    """Load tool usage data from CSV files.

    Returns:
        dict: Dictionary with loaded tool usage DataFrames
    """
    data = {}

    # Tool usage configs
    tool_configs = [
        ("gpt_beams_tools", "GPT-4.1", "beams2d"),
        ("gpt_photonics_tools", "GPT-4.1", "photonics2d"),
        ("gpt5_beams_tools", "GPT-5.1", "beams2d"),
        ("gpt5_photonics_tools", "GPT-5.1", "photonics2d"),
    ]

    for key, model, problem in tool_configs:
        path = DATA_PATHS[key]
        if path.exists():
            df = pd.read_csv(path)
            df["model"] = model
            df["problem"] = problem
            data[key] = df

    return data


def get_combined_tool_usage_df(data):
    """Combine all tool usage data into a single DataFrame.

    Args:
        data: Dictionary with tool usage DataFrames

    Returns:
        Combined DataFrame or None if no data available
    """
    keys = [
        "gpt_beams_tools",
        "gpt_photonics_tools",
        "gpt5_beams_tools",
        "gpt5_photonics_tools",
    ]
    dfs = [data[key] for key in keys if key in data]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def get_combined_global_df(data):
    """Combine all global metrics into a single DataFrame."""
    keys = [
        "gpt_beams_global",
        "gpt_photonics_global",
        "gpt5_beams_global",
        "gpt5_photonics_global",
        "cgan_beams_global",
    ]
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

    if "gpt5_beams_design" in data:
        df = data["gpt5_beams_design"].copy()
        df["source"] = "GPT-5.1 (beams2d)"
        df["problem"] = "beams2d"
        design_dfs.append(df)

    if "gpt5_photonics_design" in data:
        df = data["gpt5_photonics_design"].copy()
        df["source"] = "GPT-5.1 (photonics2d)"
        df["problem"] = "photonics2d"
        design_dfs.append(df)

    return pd.concat(design_dfs, ignore_index=True) if design_dfs else None


def save_figure(fig, filename, output_dir=None, save_pdf=True):
    """Save figure to output directory in PNG and optionally PDF format.

    Args:
        fig: Matplotlib figure to save
        filename: Output filename (e.g., "plot.png")
        output_dir: Output directory (default: figures/)
        save_pdf: If True, also save PDF version for LaTeX inclusion

    Returns:
        Path to saved PNG file
    """
    if output_dir is None:
        output_dir = get_output_dir()
    path = output_dir / filename
    fig.savefig(path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")

    # Also save PDF for LaTeX
    if save_pdf:
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {pdf_path}")

    return path
