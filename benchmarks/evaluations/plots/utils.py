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

# Directory structure:
# results/baselines/{baseline_type}/{model_id}/{problem}/  - for baselines (CGAN, CNN, etc.)
# results/models/{model_name}/{problem}/                   - for LLM agent models
# Legacy: results/cgan_cnn_2d/{problem}/                   - old structure (still supported)
# Legacy: results/{model_name}/{problem}/                  - old structure (still supported)
BASELINES_DIR = RESULTS_DIR / "baselines"
MODELS_DIR = RESULTS_DIR / "models"

# Known baseline types
KNOWN_BASELINE_TYPES = ["cgan_cnn_2d"]

# CGAN baseline directory (new structure is preferred)
# New: baselines/cgan_cnn_2d/{model_id}/{problem}/
CGAN_DIR = BASELINES_DIR / "cgan_cnn_2d"  # New structure (preferred)
CGAN_DIR_LEGACY = RESULTS_DIR / "cgan_cnn_2d"  # Legacy (for backwards compat)

# Minimum parts when parsing keys like "{model}_{problem}_{type}"
MIN_KEY_PARTS = 3


def _discover_model_paths(model_dir: Path, model_name: str) -> dict[str, Path]:
    """Discover result files for a single model directory.

    Args:
        model_dir: Path to the model directory
        model_name: Name of the model (for key generation)

    Returns:
        Dictionary mapping keys to file paths
    """
    paths = {}

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

    return paths


def _discover_baseline_paths() -> dict[str, Path]:
    """Discover baseline result files from baselines directory.

    Structure: results/baselines/{baseline_type}/{problem}/

    Returns:
        Dictionary mapping keys to file paths
    """
    paths = {}
    if not BASELINES_DIR.exists():
        return paths

    for baseline_type_dir in BASELINES_DIR.iterdir():
        if not baseline_type_dir.is_dir():
            continue

        baseline_type = baseline_type_dir.name  # e.g., "cgan_cnn_2d"

        for problem in KNOWN_PROBLEMS:
            global_path = baseline_type_dir / problem / "output_quality_global_metrics.csv"
            if global_path.exists():
                paths[f"{baseline_type}_{problem}_global"] = global_path

    return paths


def _discover_legacy_cgan_paths() -> dict[str, Path]:
    """Discover CGAN results from legacy location only.

    Legacy structure: results/cgan_cnn_2d/{problem}/
    (NOT baselines/cgan_cnn_2d/ - that's handled by _discover_baseline_paths)

    Returns:
        Dictionary mapping keys to file paths
    """
    paths = {}
    # Only check legacy location (results/cgan_cnn_2d/)
    # New location (baselines/cgan_cnn_2d/) is handled by _discover_baseline_paths
    if not (CGAN_DIR_LEGACY.exists() and CGAN_DIR_LEGACY.is_dir()):
        return paths

    for problem in KNOWN_PROBLEMS:
        cgan_path = CGAN_DIR_LEGACY / problem / "output_quality_global_metrics.csv"
        if cgan_path.exists():
            paths[f"cgan_legacy_{problem}_global"] = cgan_path

    return paths


def _discover_legacy_model_paths() -> dict[str, Path]:
    """Discover model results from legacy root location.

    Legacy structure: results/{model_name}/{problem}/

    Returns:
        Dictionary mapping keys to file paths
    """
    paths = {}
    if not RESULTS_DIR.exists():
        return paths

    skip_dirs = {"baselines", "models", *KNOWN_BASELINE_TYPES}
    for model_dir in RESULTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        if model_dir.name in skip_dirs:
            continue
        paths.update(_discover_model_paths(model_dir, model_dir.name))

    return paths


def discover_data_paths() -> dict[str, Path]:
    """Auto-discover available result files in the results directory.

    Supports both new structure (baselines/, models/) and legacy structure.

    Directory structure:
        New:
            results/baselines/{baseline_type}/{model_id}/{problem}/
            results/models/{model_name}/{problem}/
        Legacy:
            results/cgan_cnn_2d/{problem}/
            results/{model_name}/{problem}/

    Returns:
        Dictionary mapping keys to file paths
    """
    paths = {}

    # 1. Discover from new structure: results/models/
    if MODELS_DIR.exists():
        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir():
                paths.update(_discover_model_paths(model_dir, model_dir.name))

    # 2. Discover from new structure: results/baselines/
    paths.update(_discover_baseline_paths())

    # 3. Legacy: Discover agent model directories at root
    paths.update(_discover_legacy_model_paths())

    # 4. Legacy: Discover CGAN results
    paths.update(_discover_legacy_cgan_paths())

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
# These can be used as a list for dynamic assignment
COLOR_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]

# Marker styles to cycle through
MARKER_PALETTE = ["o", "s", "^", "D", "v", "p", "*", "h"]

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
    # Color palette for dynamic assignment (cycle through for models)
    "color_palette": COLOR_PALETTE,
    "marker_palette": MARKER_PALETTE,
    # Fixed colors for problems only
    "colors": {
        "beams2d": COLOR_PALETTE[0],
        "photonics2d": COLOR_PALETTE[1],
        "thermoelastic2d": COLOR_PALETTE[2],
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


def get_model_style(models: list[str]) -> dict[str, dict]:
    """Get colors and markers for a list of models.

    Dynamically assigns colors and markers from palettes.

    Args:
        models: List of model names

    Returns:
        Dict mapping model name to {"color": ..., "marker": ...}
    """
    styles = {}
    for i, model in enumerate(models):
        styles[model] = {
            "color": COLOR_PALETTE[i % len(COLOR_PALETTE)],
            "marker": MARKER_PALETTE[i % len(MARKER_PALETTE)],
        }
    return styles


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

    # Rename columns for consistency (same as CGAN)
    rename_map = {
        "model_id": "model_id_orig",  # Preserve original
        "problem_id": "problem",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Filter to rows with valid n_samples (numeric)
    if "n_samples" in df.columns:
        df = df[pd.to_numeric(df["n_samples"], errors="coerce").notna()]
        df["n_samples"] = pd.to_numeric(df["n_samples"])
        df = df.drop_duplicates(subset=["seed", "n_samples"], keep="first")

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


def get_problem_output_dir(problem: str) -> Path:
    """Return output directory for problem-specific figures.

    Args:
        problem: Problem name (e.g., "beams2d")

    Returns:
        Path to the problem-specific output directory
    """
    problem_dir = OUTPUT_DIR / problem
    problem_dir.mkdir(exist_ok=True)
    return problem_dir


def filter_by_problem(df, problem: str):
    """Filter a DataFrame to only include data for a specific problem.

    Args:
        df: DataFrame with a 'problem' column
        problem: Problem name to filter by

    Returns:
        Filtered DataFrame or None if empty
    """
    if df is None:
        return None
    if "problem" not in df.columns:
        return df
    filtered = df[df["problem"] == problem]
    return filtered if len(filtered) > 0 else None


def is_baseline(model_name: str) -> bool:
    """Check if a model name refers to a baseline (not an LLM agent).

    Args:
        model_name: Model name to check

    Returns:
        True if the model is a baseline
    """
    baseline_patterns = ["cGAN", "cgan", "CNN", "baseline"]
    return any(pattern.lower() in model_name.lower() for pattern in baseline_patterns)


def filter_models_only(df):
    """Filter a DataFrame to only include LLM agent models (exclude baselines).

    Args:
        df: DataFrame with a 'model' column

    Returns:
        Filtered DataFrame or None if empty
    """
    if df is None:
        return None
    if "model" not in df.columns:
        return df
    filtered = df[~df["model"].apply(is_baseline)]
    return filtered if len(filtered) > 0 else None


def filter_baselines_only(df):
    """Filter a DataFrame to only include baselines (exclude LLM agents).

    Args:
        df: DataFrame with a 'model' column

    Returns:
        Filtered DataFrame or None if empty
    """
    if df is None:
        return None
    if "model" not in df.columns:
        return df
    filtered = df[df["model"].apply(is_baseline)]
    return filtered if len(filtered) > 0 else None


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
