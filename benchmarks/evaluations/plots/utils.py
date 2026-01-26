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

# Known problem types
KNOWN_PROBLEMS = ["beams2d", "photonics2d", "thermoelastic2d"]

# CGAN baseline directory
CGAN_DIR = RESULTS_DIR / "cgan_cnn_2d"

# Minimum parts when parsing keys like "{model}_{problem}_{type}"
MIN_KEY_PARTS = 3


def discover_data_paths() -> dict[str, Path]:
    """Auto-discover available result files in the results directory.

    Returns:
        Dictionary mapping keys to file paths
    """
    paths = {}

    # Discover agent model directories (everything except cgan_cnn_2d)
    if RESULTS_DIR.exists():
        for model_dir in RESULTS_DIR.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            # Skip CGAN directory (handled separately)
            if model_name == "cgan_cnn_2d":
                continue

            for problem in KNOWN_PROBLEMS:
                problem_dir = model_dir / problem

                # Global metrics
                global_path = problem_dir / "output_quality_global_metrics.csv"
                if global_path.exists():
                    key = f"{model_name}_{problem}_global"
                    paths[key] = global_path

                # Design metrics
                design_path = problem_dir / "output_quality_design_metrics.csv"
                if design_path.exists():
                    key = f"{model_name}_{problem}_design"
                    paths[key] = design_path

                # Tool usage data
                tools_path = problem_dir / "data.csv"
                if tools_path.exists():
                    key = f"{model_name}_{problem}_tools"
                    paths[key] = tools_path

    # Discover CGAN results
    if CGAN_DIR.exists():
        for problem in KNOWN_PROBLEMS:
            cgan_path = CGAN_DIR / problem / "output_quality_global_metrics.csv"
            if cgan_path.exists():
                paths[f"cgan_{problem}_global"] = cgan_path

    return paths


def _get_model_label(model_name: str) -> str:
    """Convert model directory name to display label.

    Examples:
        openai_gpt-4o -> GPT-4o
        openai_gpt-4.1 -> GPT-4.1
        anthropic_claude-3-5-sonnet -> Claude-3.5-Sonnet
    """
    # Remove provider prefix
    if "_" in model_name:
        parts = model_name.split("_", 1)
        if parts[0] in ["openai", "anthropic", "google"]:
            model_name = parts[1]

    # Clean up common patterns
    model_name = model_name.replace("gpt-", "GPT-")
    model_name = model_name.replace("claude-", "Claude-")

    return model_name


# Legacy DATA_PATHS for backwards compatibility (auto-discovered)
DATA_PATHS = discover_data_paths()

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

    # Rename columns for consistency
    rename_map = {
        "viol": "rvc",  # Old column name
        "model_id": "model",  # From new benchmark script
        "problem_id": "problem",  # From new benchmark script
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure model column exists with correct label
    if "model" not in df.columns or df["model"].isna().all():
        df["model"] = "cGAN-CNN"
    else:
        # Standardize model name
        df["model"] = df["model"].replace({"cgan_cnn_2d": "cGAN-CNN"})

    return df


def load_data():
    """
    Load all metrics CSVs, clean and deduplicate.

    Auto-discovers available results in the results directory.

    Returns:
        dict: Dictionary with loaded DataFrames
    """
    data = {}

    # Refresh discovered paths
    discovered_paths = discover_data_paths()

    # Load all discovered global metrics
    for key, path in discovered_paths.items():
        if not key.endswith("_global"):
            continue

        # Parse key to extract model and problem
        # Format: {model_name}_{problem}_global or cgan_{problem}_global
        parts = key.rsplit("_", 2)  # Split from right: [model, problem, "global"]
        if len(parts) < MIN_KEY_PARTS:
            continue

        problem = parts[-2]

        if key.startswith("cgan_"):
            # CGAN baseline
            df = _load_cgan_metrics(path)
            if df is not None:
                df["problem"] = problem
                data[key] = df
        else:
            # Agent model - extract model name
            model_dir_name = "_".join(parts[:-2])  # Everything before problem
            model_label = _get_model_label(model_dir_name)
            df = _load_global_metrics(path, model_label, problem)
            if df is not None:
                data[key] = df

    # Load all discovered design metrics
    for key, path in discovered_paths.items():
        if not key.endswith("_design"):
            continue

        parts = key.rsplit("_", 2)
        if len(parts) < MIN_KEY_PARTS:
            continue

        model_dir_name = "_".join(parts[:-2])
        model_label = _get_model_label(model_dir_name)
        df = _load_design_metrics(path, model_label)
        if df is not None:
            data[key] = df

    return data


def load_tool_usage_data():
    """Load tool usage data from CSV files.

    Auto-discovers available tool usage files in the results directory.

    Returns:
        dict: Dictionary with loaded tool usage DataFrames
    """
    data = {}

    # Refresh discovered paths
    discovered_paths = discover_data_paths()

    for key, path in discovered_paths.items():
        if not key.endswith("_tools"):
            continue

        # Parse key: {model_name}_{problem}_tools
        parts = key.rsplit("_", 2)
        if len(parts) < MIN_KEY_PARTS:
            continue

        problem = parts[-2]
        model_dir_name = "_".join(parts[:-2])
        model_label = _get_model_label(model_dir_name)

        try:
            df = pd.read_csv(path)
            df["model"] = model_label
            df["problem"] = problem
            data[key] = df
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")

    return data


def get_combined_tool_usage_df(data):
    """Combine all tool usage data into a single DataFrame.

    Args:
        data: Dictionary with tool usage DataFrames

    Returns:
        Combined DataFrame or None if no data available
    """
    # Get all keys ending with _tools
    keys = [k for k in data if k.endswith("_tools")]
    dfs = [data[key] for key in keys]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def get_combined_global_df(data):
    """Combine all global metrics into a single DataFrame."""
    # Get all keys ending with _global
    keys = [k for k in data if k.endswith("_global")]
    dfs = [data[key] for key in keys]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def get_combined_design_df(data):
    """Combine all design-level metrics into a single DataFrame."""
    design_dfs = []

    # Get all keys ending with _design
    for key in data:
        if not key.endswith("_design"):
            continue

        df = data[key].copy()

        # Extract model and problem from key
        parts = key.rsplit("_", 2)
        if len(parts) >= MIN_KEY_PARTS:
            problem = parts[-2]
            model_dir_name = "_".join(parts[:-2])
            model_label = _get_model_label(model_dir_name)

            df["source"] = f"{model_label} ({problem})"
            if "problem" not in df.columns:
                df["problem"] = problem

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
