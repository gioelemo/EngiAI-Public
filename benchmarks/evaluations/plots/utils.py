"""
Shared utilities for plots.

- Data loading and cleaning
- Path configuration
- Plot styling
"""

import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from benchmarks.evaluations.extract_data import HPC_WORKFLOW_OUTPUT_FIELDS  # noqa: E402
from benchmarks.shared.problem_registry import PROBLEMS as _PROBLEMS  # noqa: E402
from benchmarks.shared.scorers.rag_scorer import RAG_OUTPUT_FIELDS  # noqa: E402

# Constants
DEFAULT_N_SAMPLES = 10
MIN_CORRELATION_SAMPLES = 2

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Known problem types — derived from the problem registry to stay in sync
KNOWN_PROBLEMS = list(_PROBLEMS.keys())

# Known prompt styles
KNOWN_PROMPT_STYLES = [
    "full",
    "natural",
    "workflow",
    "workflow-random",
    "workflow-derived-params",
    "workflow-distractor",
    "workflow-conditional",
    "workflow-multi-export",
    "rag-eval",
    "hpc-train-cgan",
    "hpc-train-diff",
    "hpc-train-natural-cgan",
    "hpc-train-natural-diff",
]

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
        - model/problem/prompt_style/rag_status/ (JSON files in rag_status dir)

    Returns:
        Dictionary mapping keys to file paths
    """
    paths: dict[str, Path] = {}
    problem = problem_dir.name
    model_name = model_dir.name

    for ps_dir in problem_dir.iterdir():
        if not ps_dir.is_dir() or ps_dir.name not in KNOWN_PROMPT_STYLES:
            continue
        prompt_style = ps_dir.name
        for rag_dir in ps_dir.iterdir():
            if not rag_dir.is_dir() or rag_dir.name not in KNOWN_RAG_STATUSES:
                continue

            rag_status = rag_dir.name  # "rag" or "no_rag"
            key_prefix = f"{model_name}_{prompt_style}_{rag_status}_{problem}"

            # Check for global metrics JSON
            global_json_path = rag_dir / "global_metrics.json"
            if global_json_path.exists():
                paths[f"{key_prefix}_global"] = global_json_path

            # Check for design data JSON
            design_json_path = rag_dir / "design_data.json"
            if design_json_path.exists():
                paths[f"{key_prefix}_design"] = design_json_path

    return paths


def discover_data_paths() -> dict[str, Path]:
    """Auto-discover available result files in the results directory.

    Directory structure:
        results/baselines/{baseline_type}/{problem}/                          - CSV files
        results/models/{model_name}/{problem}/{prompt_style}/{rag_status}/   - JSON files

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
    # Check for CGAN baseline first
    if key.startswith("cgan_"):
        parts = key.rsplit("_", 2)
        if len(parts) < MIN_KEY_PARTS:
            return None
        return {
            "model": "_".join(parts[:-2]),
            "prompt_style": None,
            "rag_status": None,
            "problem": parts[-2],
            "type": parts[-1],
            "is_baseline": True,
        }

    # 1. Extract data type from the end of the key
    data_type = None
    key_without_type = key
    for suffix in ("_global", "_design"):
        if key.endswith(suffix):
            data_type = suffix[1:]  # "global" or "design"
            key_without_type = key[: -len(suffix)]
            break
    if data_type is None:
        return None

    # 2-4. Try each known problem (longest first) with backtracking.
    #    For each candidate problem, attempt to parse rag_status + prompt_style
    #    from the remainder. If that fails, try the next shorter problem.
    #    This avoids false matches like "no_rag_beams2d" → "rag_beams2d".
    for known_problem in sorted(KNOWN_PROBLEMS, key=len, reverse=True):
        if not key_without_type.endswith(f"_{known_problem}"):
            continue

        candidate_model = key_without_type[: -(len(known_problem) + 1)]

        # 3. Extract rag_status (check "no_rag" before "rag" to avoid partial match)
        candidate_rag = None
        for status in KNOWN_RAG_STATUSES:
            if candidate_model.endswith(f"_{status}"):
                candidate_rag = status
                candidate_model = candidate_model[: -(len(status) + 1)]
                break

        # 4. Extract prompt_style (longest styles first to avoid partial matches)
        for style in sorted(KNOWN_PROMPT_STYLES, key=len, reverse=True):
            if candidate_model.endswith(f"_{style}"):
                model_name = candidate_model[: -(len(style) + 1)]
                return {
                    "model": model_name,
                    "prompt_style": style,
                    "rag_status": candidate_rag,
                    "problem": known_problem,
                    "type": data_type,
                    "is_baseline": False,
                }

    return None


# 2-column format dimensions (inches)
# Single column: ~3.25", Full width: ~6.75"
COLUMN_WIDTH = 3.25
FULL_WIDTH = 6.75

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

# Plot style configuration for publication
PLOT_STYLE = {
    # Figure sizes for 2-column format
    "figsize_single_col": (COLUMN_WIDTH, 2.4),
    "figsize_single_col_tall": (COLUMN_WIDTH, 3.0),
    "figsize_full_width": (FULL_WIDTH, 2.8),
    "figsize_full_width_tall": (FULL_WIDTH, 4.0),
    "dpi": 300,
    # Color palette for dynamic assignment (cycle through for models)
    "color_palette": COLOR_PALETTE,
    "marker_palette": MARKER_PALETTE,
    # Fixed colors for problems only
    "colors": {
        "beams2d": COLOR_PALETTE[0],
        "photonics2d": COLOR_PALETTE[1],
        "thermoelastic2d": COLOR_PALETTE[2],
        "rag_beams2d": COLOR_PALETTE[3],
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
    """Configure matplotlib/seaborn for publication-quality figures.

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
                    # Publication font sizes
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
            # Publication font sizes
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
    """Load and clean global metrics for a specific model/problem.

    Expects JSON format from compute_global_metrics.py.

    Args:
        path: Path to the JSON metrics file.
        model: Model name for the DataFrame column.
        problem: Problem name for the DataFrame column.
        prompt_style: Prompt style identifier.
        rag_status: RAG status identifier (e.g. "rag" or "no_rag").
    """
    if not path.exists():
        return None

    if path.suffix != ".json":
        print(f"Warning: Expected JSON file, got {path.suffix} for {path}")
        return None

    try:
        with path.open() as f:
            data = json.load(f)

        # Convert per_seed_metrics to DataFrame rows
        if "per_seed_metrics" not in data:
            print(f"Warning: No per_seed_metrics in {path}")
            return None

        rows = []
        for seed_metrics in data["per_seed_metrics"]:
            row = {
                "seed": seed_metrics["seed"],
                "n_samples": seed_metrics.get("n_designs", 0)
                + seed_metrics.get("n_failed", 0),
                "mmd": seed_metrics.get("mmd"),
                "dpp": seed_metrics.get("dpp_diversity"),
                "dpp_diversity": seed_metrics.get("dpp_diversity"),
                "iog": seed_metrics.get("iog"),
                "cog": seed_metrics.get("cog"),
                "fog": seed_metrics.get("fog"),
                "rvc": seed_metrics.get("rvc"),
                # Remove rvc_details to avoid large nested objects
            }
            rows.append(row)

        df = pd.DataFrame(rows)

    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None

    df["model"] = model
    df["problem"] = problem
    df["prompt_style"] = prompt_style
    df["rag_status"] = rag_status
    return df


def _load_design_metrics(path, model, prompt_style="full", rag_status=None):
    """Load and clean design-level metrics for a specific model.

    Expects JSON format from extract_data.py.

    Args:
        path: Path to the JSON design metrics file.
        model: Model name for the DataFrame column.
        prompt_style: Prompt style identifier.
        rag_status: RAG status identifier (e.g. "rag" or "no_rag").
    """
    if not path.exists():
        return None

    if path.suffix != ".json":
        print(f"Warning: Expected JSON file, got {path.suffix} for {path}")
        return None

    try:
        with path.open() as f:
            data = json.load(f)

        # Convert list of design dicts to DataFrame
        # Extract only the metric fields, not the full design arrays
        rows = []
        for design in data:
            row = {
                "seed": design.get("seed"),
                "example_id": design.get("example_id"),
                "problem_id": design.get("problem_id"),
                "model_id": design.get("model_id"),
                "design_found": design.get("design_found", False),
                "combined_overall_score": design.get("combined_overall_score"),
                "design_quality_score": design.get("design_quality_score"),
                "tool_efficiency_score": design.get("tool_efficiency_score"),
                "task_completion_score": design.get("task_completion_score"),
                "iou": design.get("iou"),
                "pixel_accuracy": design.get("pixel_accuracy"),
                "mse": design.get("mse"),
                "constraint_score": design.get("constraint_score"),
                "objective_score": design.get("objective_score"),
                "efficiency_ratio": design.get("efficiency_ratio"),
                "total_tools": design.get("total_tools"),
                "unique_tools": design.get("unique_tools"),
                "success_rate": design.get("success_rate"),
                "connected_design": design.get("connected_design"),
                "num_components": design.get("num_components"),
                "is_watertight": design.get("is_watertight"),
                "volume_mm3": design.get("volume_mm3"),
                "surface_area_mm2": design.get("surface_area_mm2"),
            }

            # Also include any individual tool usage fields (tool_*)
            row.update({k: v for k, v in design.items() if k.startswith("tool_")})

            # Include RAG evaluation fields (rag_beams2d problems)
            row.update({f: design[f] for f in RAG_OUTPUT_FIELDS if f in design})

            # Include HPC workflow fields (hpc_train_beams2d problems)
            row.update(
                {f: design[f] for f in HPC_WORKFLOW_OUTPUT_FIELDS if f in design}
            )

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["seed", "example_id"], keep="first")

    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None

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
        "viol": "rvc",  # metrics.py outputs "viol", plots use "rvc"
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
    Load all metrics from JSON files, clean and deduplicate.

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
        prompt_style: Prompt style to filter by (full, natural, workflow)
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
        prompt_style: Prompt style (full, natural, workflow)

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
        prompt_style: Prompt style (e.g., "full", "natural")
        rag_status: RAG status (e.g., "rag", "no_rag"), optional

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
    """Find the pareto-efficient points.

    Args:
        scores: An (n_points, 2) array [x_values, y_values].
        minimize_x: Whether to minimize the x-axis metric.
        minimize_y: Whether to minimize the y-axis metric.

    Returns:
        A boolean array indicating if each point is on the Pareto front.
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
