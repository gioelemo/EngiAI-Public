"""
Shared utilities for plots.

- Data loading and cleaning
- Path configuration
- Plot styling
"""

import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

# Known prompt styles
KNOWN_PROMPT_STYLES = ["full", "approximate", "natural", "workflow"]

# Directory structure:
# results/baselines/{baseline_type}/{problem}/                              - for baselines (CGAN, CNN, etc.)
# results/models/{model_name}/{problem}/{prompt_style}/{rag_status}/       - for LLM agent models
BASELINES_DIR = RESULTS_DIR / "baselines"
MODELS_DIR = RESULTS_DIR / "models"

# Known RAG statuses (order matters: check longer patterns first!)
KNOWN_RAG_STATUSES = ["no_rag", "rag"]

# Minimum parts when parsing keys like "{model}_{prompt_style}_{problem}_{type}"
MIN_KEY_PARTS = 3


def _discover_baseline_paths() -> dict[str, Path]:
    """Discover baseline result files from baselines directory.

    Structure: results/baselines/{baseline_type}/{problem}/

    Returns:
        Dictionary mapping keys to file paths
    """
    paths: dict[str, Path] = {}
    if not BASELINES_DIR.exists():
        return paths

    for baseline_type_dir in BASELINES_DIR.iterdir():
        if not baseline_type_dir.is_dir():
            continue

        baseline_type = baseline_type_dir.name  # e.g., "cgan_cnn_2d"

        for problem in KNOWN_PROBLEMS:
            global_path = (
                baseline_type_dir / problem / "output_quality_global_metrics.csv"
            )
            if global_path.exists():
                paths[f"{baseline_type}_{problem}_global"] = global_path

    return paths


def _discover_models_dir_paths() -> dict[str, Path]:
    """Discover model result files from the models directory.

    Structure: results/models/{model}/{problem}/{prompt_style}/{rag_status}/

    Returns:
        Dictionary mapping keys to file paths
    """
    paths: dict[str, Path] = {}
    if not MODELS_DIR.exists():
        return paths

    for model_dir in MODELS_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        for subdir in model_dir.iterdir():
            if not subdir.is_dir():
                continue

            if subdir.name in KNOWN_PROBLEMS:
                paths.update(_discover_problem_subdir(model_dir, subdir))

    return paths


def _discover_problem_subdir(model_dir: Path, problem_dir: Path) -> dict[str, Path]:
    """Discover results under a model/problem/ directory.

    Structure:
        - model/problem/prompt_style/rag_status/ (CSVs in rag_status dir)
        - model/problem/data.csv (tool usage from extract_data.py)

    Returns:
        Dictionary mapping keys to file paths
    """
    paths: dict[str, Path] = {}
    problem = problem_dir.name
    model_name = model_dir.name

    # Check for data.csv directly in problem directory (from extract_data.py)
    direct_tools_path = problem_dir / "data.csv"
    if direct_tools_path.exists():
        key_prefix = f"{model_name}_full_{problem}"
        paths[f"{key_prefix}_tools"] = direct_tools_path

    for ps_dir in problem_dir.iterdir():
        if not ps_dir.is_dir() or ps_dir.name not in KNOWN_PROMPT_STYLES:
            continue
        prompt_style = ps_dir.name
        for rag_dir in ps_dir.iterdir():
            if not rag_dir.is_dir() or rag_dir.name not in KNOWN_RAG_STATUSES:
                continue

            rag_status = rag_dir.name  # "rag" or "no_rag"
            key_prefix = f"{model_name}_{prompt_style}_{rag_status}_{problem}"

            global_path = rag_dir / "output_quality_global_metrics.csv"
            if global_path.exists():
                paths[f"{key_prefix}_global"] = global_path

            design_path = rag_dir / "output_quality_design_metrics.csv"
            if design_path.exists():
                paths[f"{key_prefix}_design"] = design_path

            tools_path = rag_dir / "data.csv"
            if tools_path.exists():
                paths[f"{key_prefix}_tools"] = tools_path

    return paths


def discover_data_paths() -> dict[str, Path]:
    """Auto-discover available result files in the results directory.

    Directory structure:
        results/baselines/{baseline_type}/{problem}/
        results/models/{model_name}/{problem}/{prompt_style}/{rag_status}/

    Returns:
        Dictionary mapping keys to file paths
    """
    paths: dict[str, Path] = {}

    # 1. Discover from results/models/
    paths.update(_discover_models_dir_paths())

    # 2. Discover from results/baselines/
    paths.update(_discover_baseline_paths())

    return paths


def _get_model_label(model_name: str) -> str:
    """Convert model directory name to display label.

    Examples:
        openai_gpt-4o -> GPT-4o
        openai_gpt-4.1 -> GPT-4.1
        anthropic_claude-3-5-sonnet -> Claude-3.5-Sonnet
        google_genai_gemini-3-flash-preview -> Gemini-3-Flash
        ollama_qwen3_8b-q8_0 -> Qwen3-8B-Q8_0
        ollama_qwen3_4b -> Qwen3-4B
        ollama_qwen3_4b-instruct-2507-q8_0 -> Qwen3-4B-Instruct-2507-Q8_0
    """
    # Remove provider prefix
    if "_" in model_name:
        parts = model_name.split("_", 1)
        if parts[0] in ["openai", "anthropic", "google", "ollama"]:
            model_name = parts[1]

    # Clean up Google models (e.g., genai_gemini-3-flash-preview -> Gemini-3-Flash)
    if "gemini" in model_name.lower():
        # Remove "genai_" prefix if present
        model_name = model_name.replace("genai_", "")
        # Remove "-preview" suffix if present
        model_name = model_name.replace("-preview", "")
        # Capitalize Gemini
        model_name = model_name.replace("gemini", "Gemini")

    # Clean up common patterns
    model_name = model_name.replace("gpt-", "GPT-")
    model_name = model_name.replace("claude-", "Claude-")

    # Clean up Ollama-style model names (e.g., qwen3_8b-q8_0 -> Qwen3-8B-Q8)
    if model_name.lower().startswith("qwen"):
        # Replace underscores with hyphens for readability
        model_name = model_name.replace("_", "-")
        # Capitalize "qwen" prefix
        model_name = "Qwen" + model_name[4:]
        # Uppercase size suffixes like 8b, 4b
        model_name = re.sub(
            r"-(\d+)b", lambda m: f"-{m.group(1)}B", model_name, flags=re.IGNORECASE
        )
        # Remove verbose parts like "-instruct-2507" to shorten the name
        model_name = re.sub(r"-instruct-\d+", "", model_name, flags=re.IGNORECASE)
        # Shorten quantization format: -q8_0 or -q8-0 -> -Q8
        model_name = re.sub(r"-q(\d+)[_-]0", r"-Q\1", model_name, flags=re.IGNORECASE)

    return model_name


def _parse_data_key(key: str) -> dict[str, str | bool | None] | None:
    """Parse a data key to extract model, prompt_style, rag_status, problem, and type.

    Key formats:
        Model: {model}_{prompt_style}_{rag_status}_{problem}_{type}
        CGAN:  cgan_cnn_2d_{problem}_{type}

    Returns:
        Dictionary with 'model', 'prompt_style', 'rag_status', 'problem', 'type' or None if invalid.
    """
    parts = key.rsplit("_", 2)  # Split from right: [model_part, problem, type]
    if len(parts) < MIN_KEY_PARTS:
        return None

    data_type = parts[-1]  # "global", "design", "tools"
    problem = parts[-2]

    if problem not in KNOWN_PROBLEMS:
        return None

    model_part = "_".join(parts[:-2])

    # Check for CGAN baseline
    if key.startswith("cgan_"):
        return {
            "model": model_part,
            "prompt_style": None,
            "rag_status": None,
            "problem": problem,
            "type": data_type,
            "is_baseline": True,
        }

    # Format: {model_name}_{prompt_style}_{rag_status}
    # Try to extract rag_status first
    rag_status = None
    for status in KNOWN_RAG_STATUSES:
        if model_part.endswith(f"_{status}"):
            rag_status = status
            model_part = model_part[: -(len(status) + 1)]  # Remove _status suffix
            break

    # Now extract prompt_style
    for style in KNOWN_PROMPT_STYLES:
        if model_part.endswith(f"_{style}"):
            model_name = model_part[: -(len(style) + 1)]  # Remove _style suffix
            return {
                "model": model_name,
                "prompt_style": style,
                "rag_status": rag_status,
                "problem": problem,
                "type": data_type,
                "is_baseline": False,
            }

    return None


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


def make_label(model: str, problem: str, single_problem: bool = False) -> str:
    """Create a display label for a model/problem combination.

    When only one problem is present, omits the redundant problem suffix.

    Args:
        model: Model display name
        problem: Problem name (e.g., "beams2d")
        single_problem: If True, omit problem from label

    Returns:
        Display label string
    """
    if single_problem:
        return model
    return f"{model} ({problem})"


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


def _load_global_metrics(path, model, problem, prompt_style="full", rag_status=None):
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
    df["prompt_style"] = prompt_style
    df["rag_status"] = rag_status
    return df


def _load_design_metrics(path, model, prompt_style="full", rag_status=None):
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
    df["prompt_style"] = prompt_style
    df["rag_status"] = rag_status
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

        # Parse key to extract model, prompt_style, and problem
        parsed = _parse_data_key(key)
        if parsed is None:
            continue

        problem = parsed["problem"]

        if parsed.get("is_baseline"):
            # CGAN baseline
            df = _load_cgan_metrics(path)
            if df is not None:
                df["problem"] = problem
                df["prompt_style"] = None  # Baselines don't have prompt_style
                df["rag_status"] = None  # Baselines don't have rag_status
                data[key] = df
        else:
            # Agent model - get display label
            model_label = _get_model_label(parsed["model"])
            prompt_style = parsed.get("prompt_style", "full")
            rag_status = parsed.get("rag_status")
            df = _load_global_metrics(
                path, model_label, problem, prompt_style, rag_status
            )
            if df is not None:
                data[key] = df

    # Load all discovered design metrics
    for key, path in discovered_paths.items():
        if not key.endswith("_design"):
            continue

        parsed = _parse_data_key(key)
        if parsed is None:
            continue

        model_label = _get_model_label(parsed["model"])
        prompt_style = parsed.get("prompt_style", "full")
        rag_status = parsed.get("rag_status")
        df = _load_design_metrics(path, model_label, prompt_style, rag_status)
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

        # Parse key: {model_name}_{prompt_style}_{rag_status}_{problem}_tools
        parsed = _parse_data_key(key)
        if parsed is None:
            continue

        problem = parsed["problem"]
        model_label = _get_model_label(parsed["model"])
        prompt_style = parsed.get("prompt_style", "full")
        rag_status = parsed.get("rag_status")

        try:
            df = pd.read_csv(path)
            df["model"] = model_label
            df["problem"] = problem
            df["prompt_style"] = prompt_style
            df["rag_status"] = rag_status
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

        parsed = _parse_data_key(key)
        if parsed is not None:
            problem = parsed["problem"]
            model_label = _get_model_label(parsed["model"])

            df["model_display"] = model_label
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


def filter_by_prompt_style(df, prompt_style: str, include_baselines: bool = True):
    """Filter a DataFrame to only include data for a specific prompt style.

    Args:
        df: DataFrame with a 'prompt_style' column
        prompt_style: Prompt style to filter by (full, approximate, natural, workflow)
        include_baselines: If True, include baselines (which have no prompt_style)

    Returns:
        Filtered DataFrame or None if empty
    """
    if df is None:
        return None
    if "prompt_style" not in df.columns:
        return df

    if include_baselines:
        # Include rows matching prompt_style OR baselines (prompt_style is None)
        filtered = df[
            (df["prompt_style"] == prompt_style) | (df["prompt_style"].isna())
        ]
    else:
        filtered = df[df["prompt_style"] == prompt_style]

    return filtered if len(filtered) > 0 else None


def filter_by_rag_status(df, rag_status: str, include_baselines: bool = True):
    """Filter a DataFrame to only include data for a specific RAG status.

    Args:
        df: DataFrame with a 'rag_status' column
        rag_status: RAG status to filter by ("rag", "no_rag")
        include_baselines: If True, include baselines (which have no rag_status)

    Returns:
        Filtered DataFrame or None if empty
    """
    if df is None:
        return None
    if "rag_status" not in df.columns:
        return df

    if include_baselines:
        # Include rows matching rag_status OR baselines (rag_status is None)
        filtered = df[(df["rag_status"] == rag_status) | (df["rag_status"].isna())]
    else:
        filtered = df[df["rag_status"] == rag_status]

    return filtered if len(filtered) > 0 else None


def get_prompt_style_output_dir(prompt_style: str):
    """Return output directory for a specific prompt style.

    Args:
        prompt_style: Prompt style (full, approximate, natural, workflow)

    Returns:
        Path to the output directory for this prompt style
    """
    output_dir = OUTPUT_DIR / prompt_style
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_problem_prompt_output_dir(
    problem: str, prompt_style: str, rag_status: str | None = None
) -> Path:
    """Return output directory for a specific problem, prompt style, and RAG status.

    Structure: figures/{problem}/{prompt_style}/{rag_status}/

    Args:
        problem: Problem name (e.g., "beams2d")
        prompt_style: Prompt style (e.g., "full", "approximate")
        rag_status: RAG status (e.g., "rag", "no_rag"), optional for backwards compatibility

    Returns:
        Path to the output directory
    """
    if rag_status is not None:
        output_dir = OUTPUT_DIR / problem / prompt_style / rag_status
    else:
        output_dir = OUTPUT_DIR / problem / prompt_style
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def is_baseline(model_name: str) -> bool:
    """Check if a model name refers to a baseline (not an LLM agent).

    Args:
        model_name: Model name to check

    Returns:
        True if the model is a baseline
    """
    baseline_patterns = ["cGAN", "cgan", "CNN", "baseline"]
    return any(pattern.lower() in model_name.lower() for pattern in baseline_patterns)


def filter_by_model_type(df, model_type: str):
    """Filter a DataFrame by model type (baseline or agent).

    Args:
        df: DataFrame with a 'model' column
        model_type: Type of model to include ("baseline" or "agent")

    Returns:
        Filtered DataFrame or None if empty
    """
    if df is None:
        return None
    if "model" not in df.columns:
        return df

    if model_type == "baseline":
        filtered = df[df["model"].apply(is_baseline)]
    elif model_type == "agent":
        filtered = df[~df["model"].apply(is_baseline)]
    else:
        raise ValueError(
            f"model_type must be 'baseline' or 'agent', got '{model_type}'"
        )

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


def identify_pareto_front(scores, minimize_x=False, minimize_y=True):
    """
    Finds the pareto-efficient points.
    :param scores: An (n_points, 2) array [x_values, y_values]
    :return: A boolean array indicating if each point is on the Pareto front.
    """
    is_efficient = np.ones(scores.shape[0], dtype=bool)
    for i, c in enumerate(scores):
        if is_efficient[i]:
            # Keep any point with a lower cost (if minimizing) or higher (if maximizing)
            # For DPP (X) we usually maximize, for FOG/MMD (Y) we minimize
            if minimize_x and minimize_y:
                is_efficient[is_efficient] = np.any(scores[is_efficient] < c, axis=1)
            elif not minimize_x and minimize_y:
                # Maximize X, Minimize Y
                is_efficient[is_efficient] = np.any(
                    [scores[is_efficient, 0] > c[0], scores[is_efficient, 1] < c[1]],
                    axis=0,
                )
            is_efficient[i] = True  # Keep self
    return is_efficient
