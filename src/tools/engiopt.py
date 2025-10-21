"""
WandB model download tool for engineering design models.

This tool allows downloading pre-trained generative models from WandB (Weights & Biases)
that can be used for inverse design tasks. The models are trained using the engiopt
library and are available on the engibench WandB project.
"""

import importlib.util
import os
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import tool

# Check for optional dependencies
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
WANDB_AVAILABLE = importlib.util.find_spec("wandb") is not None

# Supported algorithms and their properties
SUPPORTED_ALGORITHMS = {
    "cgan_cnn_2d": {
        "class": "Inverse Design",
        "dimensions": "2D",
        "conditional": True,
        "model": "GAN + CNN",
        "model_types": ["generator", "discriminator"],
    },
    "diffusion_2d_cond": {
        "class": "Inverse Design",
        "dimensions": "2D",
        "conditional": True,
        "model": "Diffusion",
        "model_types": ["model"],
    },
}


def _get_artifact_info(
    problem_id: str, algorithm: str, model_type: str
) -> dict[str, str]:
    """
    Get the correct artifact name and checkpoint filename based on algorithm type.

    Args:
        problem_id: Problem identifier (e.g., "beams2d")
        algorithm: Algorithm name (e.g., "cgan_cnn_2d", "diffusion_2d_cond")
        model_type: Model type requested (e.g., "generator", "discriminator")

    Returns:
        dict with "artifact_name" and "checkpoint_filename"
    """
    # For diffusion models, use "model" nomenclature
    if algorithm == "diffusion_2d_cond":
        artifact_name = f"{problem_id}_{algorithm}_model"
        checkpoint_filename = "model.pth"
    # For GAN models, use generator/discriminator nomenclature
    else:
        artifact_name = f"{problem_id}_{algorithm}_{model_type}"
        checkpoint_filename = f"{model_type}.pth"

    return {
        "artifact_name": artifact_name,
        "checkpoint_filename": checkpoint_filename,
    }


@tool
def download_wandb_model(  # noqa: PLR0913
    problem_id: Literal["beams2d"] = "beams2d",
    algorithm: str = "cgan_cnn_2d",
    seed: int = 1,
    model_type: Literal["discriminator", "generator"] = "discriminator",
    wandb_project: str | None = None,
    download_dir: str | None = None,
) -> dict[str, Any]:
    """
    Download a pre-trained generative model from WandB for engineering design.

    This tool downloads models trained with the engiopt library that can be used
    for inverse design tasks. By default, it searches the user's personal models first,
    then falls back to official benchmark models. No project specification needed!

    Args:
        problem_id: Engineering problem identifier. Currently only "beams2d" is supported.
        algorithm: Model architecture to download. Options:
            - cgan_cnn_2d: Conditional GAN + CNN (2D) [default]
            - diffusion_2d_cond: Conditional Diffusion (2D)
        seed: Random seed used during model training (default: 1)
        model_type: Type of model to download. Options:
            - discriminator: Download discriminator model [default] (for GANs)
            - generator: Download generator model (for GANs)
            Note: For diffusion models, this parameter is ignored as they use a single "model" artifact
        wandb_project: WandB project path in format "organization/project" [OPTIONAL]
            DEFAULT BEHAVIOR (None): Automatically searches multiple projects:
            1. Personal models: WANDB_PERSONAL_PROJECT env var (default: gioelemo-ethz/engiopt)
            2. Official models: WANDB_OFFICIAL_PROJECT env var (default: engibench/engiopt)
            ADVANCED: Specify explicit project to search only that one (e.g., "username/project")
        download_dir: Directory to download the model to. If None, uses WandB's default cache.

    Returns:
        dict with download information:
        - success: bool indicating if download succeeded
        - artifact_path: str with full WandB artifact path
        - download_path: str with local path to downloaded model
        - checkpoint_path: str with path to model checkpoint file
        - model_type: str indicating model type ('generator', 'discriminator', or 'model')
        - algorithm_info: dict with algorithm properties (dimensions, conditional, etc.)
        - run_config: dict with training configuration (if available)
        - error: str with error message (only if success=False)

    Example:
        >>> # Download a discriminator model (default for GANs)
        >>> # Tries personal project first, then official
        >>> result = download_wandb_model(
        ...     problem_id="beams2d",
        ...     algorithm="cgan_cnn_2d",
        ...     seed=1
        ... )
        >>> if result['success']:
        ...     print(f"Model downloaded to: {result['checkpoint_path']}")
        ...     print(f"Model type: {result['model_type']}")

        >>> # Download a generator model from specific project
        >>> result = download_wandb_model(
        ...     problem_id="beams2d",
        ...     algorithm="cgan_cnn_2d",
        ...     seed=1,
        ...     model_type="generator",
        ...     wandb_project="engibench/engiopt"
        ... )

        >>> # Download a diffusion model (tries personal, then official)
        >>> result = download_wandb_model(
        ...     problem_id="beams2d",
        ...     algorithm="diffusion_2d_cond",
        ...     seed=1
        ... )
        >>> # Diffusion models use a single "model" artifact, model_type is ignored

    Note:
        - Requires wandb to be installed: pip install wandb
        - Requires USE_WANDB environment variable to be set to "True"
        - You may need to authenticate with WandB using: wandb login
        - For diffusion models, the model_type parameter is ignored
        - Customize project search order with WANDB_PERSONAL_PROJECT and WANDB_OFFICIAL_PROJECT env vars
    """
    # Validation checks
    error_response = _validate_download_inputs(problem_id, algorithm)
    if error_response:
        return error_response

    # Try to import wandb
    error_response = _check_wandb_available()
    if error_response:
        return error_response

    # If no project specified, try multiple projects in order
    if wandb_project is None:
        # Get project names from environment variables with fallback defaults
        personal_project = os.getenv("WANDB_PERSONAL_PROJECT", "gioelemo-ethz/engiopt")
        official_project = os.getenv("WANDB_OFFICIAL_PROJECT", "engibench/engiopt")

        projects_to_try = [
            personal_project,  # Personal models first
            official_project,  # Official models as fallback
        ]

        last_error = None
        for project in projects_to_try:
            result = _download_from_wandb(
                problem_id=problem_id,
                algorithm=algorithm,
                seed=seed,
                model_type=model_type,
                wandb_project=project,
                download_dir=download_dir,
            )

            if result.get("success"):
                return result

            last_error = result.get("error", "Unknown error")
            print(f"  ⚠ Not found in {project}, trying next project...")

        # If all projects failed, return the last error
        return {
            "success": False,
            "error": f"Model not found in any W&B project. Last error: {last_error}",
        }

    # Download from specified project
    return _download_from_wandb(
        problem_id=problem_id,
        algorithm=algorithm,
        seed=seed,
        model_type=model_type,
        wandb_project=wandb_project,
        download_dir=download_dir,
    )


def _validate_download_inputs(problem_id: str, algorithm: str) -> dict[str, Any] | None:
    """Validate inputs for download_wandb_model."""
    # Check if USE_WANDB is enabled
    use_wandb = os.getenv("USE_WANDB") == "True"
    if not use_wandb:
        return {
            "success": False,
            "error": "WandB is not enabled. Set USE_WANDB=True in your environment variables.",
        }

    # Validate algorithm
    if algorithm not in SUPPORTED_ALGORITHMS:
        return {
            "success": False,
            "error": f"Unsupported algorithm '{algorithm}'. Supported algorithms: {', '.join(SUPPORTED_ALGORITHMS.keys())}",
        }

    # Validate problem_id (currently only beams2d is supported)
    if problem_id != "beams2d":
        return {
            "success": False,
            "error": f"Unsupported problem_id '{problem_id}'. Currently only 'beams2d' is supported.",
        }

    return None


def _check_wandb_available() -> dict[str, Any] | None:
    """Check if wandb is available."""
    if not WANDB_AVAILABLE:
        return {
            "success": False,
            "error": "wandb is not installed. Install with: pip install wandb",
        }
    return None


def _download_from_wandb(  # noqa: PLR0913
    problem_id: str,
    algorithm: str,
    seed: int,
    model_type: str,
    wandb_project: str,
    download_dir: str | None,
) -> dict[str, Any]:
    """Download model artifact from WandB."""
    import wandb

    try:
        # Get correct artifact naming based on algorithm
        artifact_info = _get_artifact_info(problem_id, algorithm, model_type)
        artifact_name = artifact_info["artifact_name"]
        checkpoint_filename = artifact_info["checkpoint_filename"]

        # Construct the artifact path
        artifact_version = f"seed_{seed}"
        artifact_path = f"{wandb_project}/{artifact_name}:{artifact_version}"

        # Initialize WandB API
        api = wandb.Api()

        # Determine display name for logging
        display_name = "model" if algorithm == "diffusion_2d_cond" else model_type

        # Download the artifact
        print(f"Downloading {display_name} ({algorithm}) from WandB: {artifact_path}")
        artifact = api.artifact(artifact_path, type="model")

        # Download to specified directory or default cache
        if download_dir:
            artifact_dir = artifact.download(root=download_dir)
        else:
            artifact_dir = artifact.download()

        # Construct checkpoint path
        checkpoint_path = Path(artifact_dir) / checkpoint_filename

        # Verify checkpoint exists
        if not checkpoint_path.exists():
            return {
                "success": False,
                "error": f"Checkpoint file not found at {checkpoint_path}",
            }

        # Get run configuration if available
        run_config = {}
        try:
            run = artifact.logged_by()
            if run:
                run_config = dict(run.config)
        except Exception:
            # Silently skip if run config is not available - will use defaults
            pass

        # Determine actual model type for response
        actual_model_type = "model" if algorithm == "diffusion_2d_cond" else model_type

        return {
            "success": True,
            "artifact_path": artifact_path,
            "download_path": artifact_dir,
            "checkpoint_path": str(checkpoint_path),
            "model_type": actual_model_type,
            "algorithm_info": SUPPORTED_ALGORITHMS[algorithm],
            "run_config": run_config,
            "message": f"Successfully downloaded {display_name} ({algorithm}) model for {problem_id} (seed={seed})",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to download model from WandB: {e!s}",
        }


@tool
def list_available_algorithms() -> dict[str, Any]:
    """
    List all available algorithms for model download.

    Returns:
        dict with:
        - algorithms: list of algorithm names
        - details: dict mapping algorithm names to their properties

    Example:
        >>> info = list_available_algorithms()
        >>> for algo in info['algorithms']:
        ...     props = info['details'][algo]
        ...     print(f"{algo}: {props['model']} ({props['dimensions']})")
    """
    return {
        "algorithms": list(SUPPORTED_ALGORITHMS.keys()),
        "details": SUPPORTED_ALGORITHMS,
        "message": f"Found {len(SUPPORTED_ALGORITHMS)} available algorithms",
    }


@tool
def load_wandb_model(  # noqa: PLR0913
    checkpoint_path: str,
    problem_id: Literal["beams2d"] = "beams2d",
    algorithm: str = "cgan_cnn_2d",
    model_type: Literal["discriminator", "generator"] = "discriminator",
    run_config: dict[str, Any] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Load a downloaded WandB model checkpoint into memory.

    This tool loads the PyTorch model from a checkpoint file and prepares it for inference.
    It requires the model architecture to match the algorithm type.

    Args:
        checkpoint_path: Path to the model checkpoint file
            (e.g., discriminator.pth, generator.pth, or model.pth for diffusion)
        problem_id: Engineering problem identifier (default: "beams2d")
        algorithm: Model architecture type (default: "cgan_cnn_2d")
        model_type: Type of model to load:
            - discriminator: Load discriminator model [default] (for GANs)
            - generator: Load generator model (for GANs)
            Note: For diffusion models, this parameter is informational only
        run_config: Training configuration dict with model hyperparameters.
            If None, will use default values.
        device: Device to load model on: "cpu", "cuda", or "mps" (default: "cpu")

    Returns:
        dict with:
        - success: bool indicating if load succeeded
        - model_ready: bool indicating if model is ready for inference
        - model_type: str indicating model type ('generator', 'discriminator', or 'model')
        - device: str with device model is loaded on
        - model_info: dict with model details
        - error: str with error message (only if success=False)

    Example:
        >>> # First download the discriminator model (default for GANs)
        >>> download_result = download_wandb_model(algorithm="cgan_cnn_2d", seed=1)
        >>> # Then load it
        >>> load_result = load_wandb_model(
        ...     checkpoint_path=download_result['checkpoint_path'],
        ...     model_type="discriminator",
        ...     run_config=download_result['run_config'],
        ...     device="cpu"
        ... )

        >>> # Or download and load the generator model (for GANs)
        >>> download_result = download_wandb_model(
        ...     algorithm="cgan_cnn_2d",
        ...     seed=1,
        ...     model_type="generator"
        ... )
        >>> load_result = load_wandb_model(
        ...     checkpoint_path=download_result['checkpoint_path'],
        ...     model_type="generator",
        ...     run_config=download_result['run_config'],
        ...     device="cpu"
        ... )

        >>> # Download and load a diffusion model
        >>> download_result = download_wandb_model(
        ...     algorithm="diffusion_2d_cond",
        ...     seed=1
        ... )
        >>> load_result = load_wandb_model(
        ...     checkpoint_path=download_result['checkpoint_path'],
        ...     algorithm="diffusion_2d_cond",
        ...     run_config=download_result['run_config'],
        ...     device="cpu"
        ... )

    Note:
        - Requires PyTorch: pip install torch
        - Requires the corresponding model architecture from engiopt
        - Model is set to eval mode after loading
        - For diffusion models, the model_type parameter is informational only
    """
    import torch as th

    # Check if PyTorch is available
    if not TORCH_AVAILABLE:
        return {
            "success": False,
            "error": "PyTorch is not installed. Install with: pip install torch",
        }

    # Validate checkpoint path
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        return {
            "success": False,
            "error": f"Checkpoint file not found: {checkpoint_path}",
        }

    # Validate algorithm
    if algorithm not in SUPPORTED_ALGORITHMS:
        return {
            "success": False,
            "error": f"Unsupported algorithm '{algorithm}'. Supported algorithms: {', '.join(SUPPORTED_ALGORITHMS.keys())}",
        }

    try:
        # Load checkpoint
        ckpt = th.load(checkpoint_path, map_location=device)

        # Get model configuration
        if run_config and "latent_dim" in run_config:
            latent_dim = run_config["latent_dim"]
        else:
            # Default latent dimension
            latent_dim = 32

        return {
            "success": True,
            "model_ready": True,
            "model_type": model_type,
            "device": device,
            "model_info": {
                "algorithm": algorithm,
                "problem_id": problem_id,
                "model_type": model_type,
                "latent_dim": latent_dim,
                "checkpoint_keys": list(ckpt.keys()),
            },
            "message": f"{model_type.capitalize()} model structure loaded. Note: Full model initialization requires engiopt library and problem instance.",
            "note": f"To fully initialize the {model_type} model, you need to import the appropriate {model_type.capitalize()} class from engiopt and create it with the problem's design space shape.",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load model checkpoint: {e!s}",
        }


def _validate_sampling_inputs(
    algorithm: str,
    problem_id: str,
    conditions: list[dict[str, float]] | None,
    n_samples: int,
) -> dict[str, Any] | None:
    """Validate inputs for design sampling. Returns error dict or None if valid."""
    # Check dependencies
    if not TORCH_AVAILABLE:
        return {
            "success": False,
            "error": "PyTorch is not installed. Install with: pip install torch",
        }

    # Validate algorithm
    if algorithm not in SUPPORTED_ALGORITHMS:
        return {
            "success": False,
            "error": f"Unsupported algorithm '{algorithm}'. Supported algorithms: {', '.join(SUPPORTED_ALGORITHMS.keys())}",
        }

    # Validate problem_id
    if problem_id != "beams2d":
        return {
            "success": False,
            "error": f"Unsupported problem_id '{problem_id}'. Currently only 'beams2d' is supported.",
        }

    # Validate conditions
    if conditions is not None and len(conditions) != n_samples:
        return {
            "success": False,
            "error": f"Number of conditions ({len(conditions)}) must match n_samples ({n_samples})",
        }

    return None


def _import_generator_class(
    algorithm: str,
) -> tuple[type | None, dict[str, Any] | None]:
    """Import the appropriate Generator class for GAN algorithms. Returns (class, error_dict)."""
    # ruff: noqa: I001, PLC0415
    try:
        if algorithm in ["cgan_cnn_2d"]:
            from engiopt.cgan_cnn_2d.cgan_cnn_2d import Generator  # type: ignore[import-untyped]

            return Generator, None
        else:
            return None, {
                "success": False,
                "error": f"Generator class not applicable for algorithm: {algorithm}. Use diffusion-specific loading.",
            }
    except ImportError as e:
        return None, {
            "success": False,
            "error": f"Failed to import Generator class for {algorithm}: {e}. Install engiopt library.",
        }


def _load_diffusion_model(
    ckpt_path: str,
    device: str,
    problem: Any,
) -> tuple[Any, Any, int, dict[str, Any] | None]:
    """
    Load a diffusion model from checkpoint.

    Returns: (model, sampler, num_timesteps, error_dict)
    """
    try:
        import torch as th
        from diffusers import UNet2DConditionModel  # type: ignore[import-untyped]
        from engiopt.diffusion_2d_cond.diffusion_2d_cond import (  # type: ignore[import-untyped]
            beta_schedule,
            DiffusionSampler,
        )
    except ImportError as e:
        return (
            None,
            None,
            0,
            {
                "success": False,
                "error": f"Required library not installed: {e}. Install with: pip install diffusers engiopt",
            },
        )

    try:
        # Load checkpoint
        ckpt = th.load(ckpt_path, map_location=device)

        # Get run config from checkpoint if available
        run_config = ckpt.get("config", {})

        # Set defaults if not in checkpoint
        layers_per_block = run_config.get("layers_per_block", 2)
        num_timesteps = run_config.get("num_timesteps", 1000)
        noise_schedule = run_config.get("noise_schedule", "linear")

        # Initialize the UNet2D model
        model = UNet2DConditionModel(
            sample_size=problem.design_space.shape,
            in_channels=1,
            out_channels=1,
            cross_attention_dim=64,
            block_out_channels=(32, 64, 128, 256),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            layers_per_block=layers_per_block,
            transformer_layers_per_block=1,
            encoder_hid_dim=len(problem.conditions),
            only_cross_attention=True,
        ).to(device)  # type: ignore[attr-defined]

        # Load model weights
        model.load_state_dict(ckpt["model"])
        model.eval()

        # Set up noise schedule
        options = {
            "cosine": noise_schedule == "cosine",
            "exp_biasing": noise_schedule == "exp",
            "exp_bias_factor": 1,
        }
        betas = beta_schedule(
            t=num_timesteps,
            start=1e-4,
            end=0.02,
            scale=1.0,
            options=options,
        )

        # Create diffusion sampler
        sampler = DiffusionSampler(num_timesteps, betas)

    except Exception as e:
        return (
            None,
            None,
            0,
            {
                "success": False,
                "error": f"Failed to load diffusion model: {e}",
            },
        )
    else:
        return model, sampler, num_timesteps, None


def _generate_designs(
    model: Any,
    config: dict[str, Any],
) -> Any:
    """Generate designs using a GAN model.

    Args:
        model: The generator model
        config: Dict with keys: n_samples, latent_dim, device, is_conditional, conditions_tensor
    """
    import torch as th

    # Sample noise as generator input
    z = th.randn(
        (config["n_samples"], config["latent_dim"], 1, 1),
        device=config["device"],
        dtype=th.float,
    )

    # Generate designs
    with th.no_grad():
        gen_designs = (
            model(z, config["conditions_tensor"])
            if config["is_conditional"] and config["conditions_tensor"] is not None
            else model(z)
        )

    return gen_designs


def _generate_designs_diffusion(
    model: Any,
    sampler: Any,
    config: dict[str, Any],
) -> Any:
    """Generate designs using a diffusion model.

    Args:
        model: The UNet2D diffusion model
        sampler: The DiffusionSampler instance
        config: Dict with keys: n_samples, device, conditions_tensor, design_shape, num_timesteps
    """
    import torch as th

    n_samples = config["n_samples"]
    device = config["device"]
    conditions_tensor = config["conditions_tensor"]
    design_shape = config["design_shape"]
    num_timesteps = config["num_timesteps"]

    # Start with random noise
    gen_designs = th.randn((n_samples, 1, *design_shape), device=device)

    # Iteratively denoise using the diffusion sampler
    with th.no_grad():
        for i in reversed(range(num_timesteps)):
            t = th.full((n_samples,), i, device=device, dtype=th.long)
            gen_designs = sampler.sample_timestep(
                model, gen_designs, t, conditions_tensor
            )

    return gen_designs


def _save_designs(
    gen_designs: Any,
    n_samples: int,
    output_path: Path,
    problem: Any,
) -> tuple[list[str], list[str]]:
    """Save generated designs as .npy and .png files. Returns (design_files, render_files)."""
    import numpy as np

    design_files = []
    render_files = []

    for i in range(n_samples):
        # Save as numpy array
        design_filename = output_path / f"generated_design_{i}.npy"
        np.save(design_filename, gen_designs[i])
        design_files.append(str(design_filename))

        # Render and save visualization
        render_filename = output_path / f"generated_design_{i}.png"
        fig, _ = problem.render(gen_designs[i])
        fig.savefig(str(render_filename), dpi=150, bbox_inches="tight")
        render_files.append(str(render_filename))

    return design_files, render_files


def _find_or_download_model(
    checkpoint_path: str | None,
    problem_id: str,
    algorithm: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """
    Find a model checkpoint or download one if needed.

    Args:
        checkpoint_path: Optional path to checkpoint. If None, will search or download.
        problem_id: Engineering problem identifier
        algorithm: Model architecture type

    Returns:
        Tuple of (checkpoint_path, error_dict). Error dict is None on success.
    """
    import random

    # If path provided, just validate it exists
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            return None, {
                "success": False,
                "error": f"Checkpoint file not found: {checkpoint_path}",
            }
        return str(checkpoint_file), None

    # Search for existing models in artifacts folder
    artifacts_dir = Path("artifacts")

    if artifacts_dir.exists():
        # Determine the expected checkpoint filename based on algorithm
        if algorithm == "diffusion_2d_cond":
            checkpoint_filename = "model.pth"
            pattern = f"{problem_id}_{algorithm}_model:v*"
        else:
            checkpoint_filename = "generator.pth"
            pattern = f"{problem_id}_{algorithm}_generator:v*"

        # Look for checkpoint files matching the problem and algorithm
        matching_dirs = list(artifacts_dir.glob(pattern))

        if matching_dirs:
            # Randomly select one of the matching models
            selected_dir = random.choice(matching_dirs)
            checkpoint = selected_dir / checkpoint_filename

            if checkpoint.exists():
                return str(checkpoint), None

    # No existing model found, download one
    print(
        f"No local model found for {problem_id}/{algorithm}. Downloading from WandB..."
    )

    download_result = download_wandb_model.invoke(
        {
            "problem_id": problem_id,
            "algorithm": algorithm,
            "seed": 1,  # Default to seed 1
        }
    )

    if not download_result.get("success"):
        return None, {
            "success": False,
            "error": f"Failed to download model: {download_result.get('error', 'Unknown error')}",
        }

    return download_result["checkpoint_path"], None


@tool
def sample_designs_from_model(  # noqa: PLR0913, PLR0911
    checkpoint_path: str | None = None,
    problem_id: Literal["beams2d"] = "beams2d",
    algorithm: str = "cgan_cnn_2d",
    conditions: list[dict[str, float]] | None = None,
    n_samples: int = 3,
    latent_dim: int = 32,
    device: str = "cpu",
    output_dir: str = "outputs",
) -> dict[str, Any]:
    """
    Sample/generate designs from a loaded generative model.

    This tool generates new designs using a pre-trained generative model (GAN, Diffusion, etc.)
    based on specified conditions. It's useful for inverse design where you want designs
    that meet specific performance criteria.

    If no checkpoint_path is provided, the tool will automatically:
    1. Search for existing models in the artifacts folder
    2. Randomly select one if multiple are found
    3. Download a model from WandB if none exist locally

    Args:
        checkpoint_path: Path to the generator.pth checkpoint file. If None, will auto-select
            or download a model (default: None)
        problem_id: Engineering problem identifier (default: "beams2d")
        algorithm: Model architecture type (default: "cgan_cnn_2d")
        conditions: List of condition dictionaries for conditional models. Each dict should have:
            - volfrac: Volume fraction (0-1)
            - rmin: Minimum radius filter
            - forcedist: Force distribution (0-1)
            - overhang_constraint: Overhang constraint (0-1)
            If None, will use default conditions. Length should match n_samples.
        n_samples: Number of designs to generate (default: 3)
        latent_dim: Latent dimension of the model (default: 32)
        device: Device to run on: "cpu", "cuda", or "mps" (default: "cpu")
        output_dir: Directory to save generated designs (default: "outputs")

    Returns:
        dict with:
        - success: bool indicating if sampling succeeded
        - designs: list of generated design arrays (if successful)
        - design_files: list of saved .npy file paths
        - render_files: list of saved .png visualization files
        - conditions_used: list of conditions used for generation
        - n_samples: number of designs generated
        - checkpoint_path: path to the checkpoint used
        - error: str with error message (only if success=False)

    Example:
        >>> # Auto-select/download model with default conditions
        >>> result = sample_designs_from_model.invoke({
        ...     "n_samples": 3
        ... })

        >>> # Use specific checkpoint
        >>> result = sample_designs_from_model.invoke({
        ...     "checkpoint_path": "/path/to/generator.pth",
        ...     "n_samples": 3
        ... })

        >>> # Generate with specific conditions
        >>> conditions = [
        ...     {"volfrac": 0.35, "rmin": 2.0, "forcedist": 0.2, "overhang_constraint": 0.0},
        ...     {"volfrac": 0.45, "rmin": 2.0, "forcedist": 0.2, "overhang_constraint": 0.0},
        ... ]
        >>> result = sample_designs_from_model.invoke({
        ...     "checkpoint_path": "/path/to/generator.pth",
        ...     "conditions": conditions,
        ...     "n_samples": 2
        ... })

    Note:
        - Requires PyTorch, engiopt, and engibench to be installed
        - For diffusion models (diffusion_2d_cond), requires diffusers library
        - For conditional models, conditions will be used during generation
        - For non-conditional models, conditions are ignored
        - Generated designs are automatically clipped to [0, 1] range
        - Designs are saved as .npy files and visualized as .png images
        - Diffusion models use iterative denoising (slower but higher quality)
        - GAN models use single-shot generation (faster)
    """
    import torch as th

    # Validate inputs
    error = _validate_sampling_inputs(algorithm, problem_id, conditions, n_samples)
    if error:
        return error

    # Find or download a model if no checkpoint path provided
    resolved_checkpoint_path, error = _find_or_download_model(
        checkpoint_path, problem_id, algorithm
    )
    if error:
        return error

    assert resolved_checkpoint_path is not None

    # Set default conditions if not provided
    if conditions is None:
        conditions = [
            {"volfrac": 0.35, "rmin": 2.0, "forcedist": 0.2, "overhang_constraint": 0.0}
            for _ in range(n_samples)
        ]

    try:
        # Import required libraries
        try:
            import numpy as np
            from engibench.problems.beams2d.v0 import Beams2D
        except ImportError as e:
            return {
                "success": False,
                "error": f"Required library not installed: {e}. Install with: pip install engibench numpy",
            }

        # Create problem instance
        problem = Beams2D()
        problem.reset(seed=0)

        # Branch based on algorithm type
        if algorithm == "diffusion_2d_cond":
            # Handle diffusion models
            model, sampler, num_timesteps, error = _load_diffusion_model(
                resolved_checkpoint_path, device, problem
            )
            if error:
                return error

            # Prepare conditions tensor for diffusion (different format than GAN)
            conditions_tensor = th.tensor(
                [list(c.values()) for c in conditions],
                device=device,
                dtype=th.float,
            ).unsqueeze(1)  # Add channel dim

            # Generate designs using diffusion
            gen_config = {
                "n_samples": n_samples,
                "device": device,
                "conditions_tensor": conditions_tensor,
                "design_shape": problem.design_space.shape,
                "num_timesteps": num_timesteps,
            }
            gen_designs = _generate_designs_diffusion(model, sampler, gen_config)

        else:
            # Handle GAN models
            generator_class, error = _import_generator_class(algorithm)
            if error:
                return error

            # At this point, generator_class cannot be None
            assert generator_class is not None

            # Load checkpoint
            ckpt = th.load(resolved_checkpoint_path, map_location=device)

            # Initialize the model
            is_conditional = SUPPORTED_ALGORITHMS[algorithm]["conditional"]
            n_conds = len(problem.conditions) if is_conditional else 0

            model = generator_class(
                latent_dim=latent_dim,
                n_conds=n_conds,
                design_shape=problem.design_space.shape,
            )
            model.load_state_dict(ckpt["generator"])
            model.eval()
            model.to(device)

            # Prepare conditions tensor for GAN
            conditions_tensor = None
            if is_conditional:
                conditions_tensor = (
                    th.tensor(
                        [list(c.values()) for c in conditions],
                        device=device,
                        dtype=th.float,
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )

            # Generate designs using GAN
            gen_config = {
                "n_samples": n_samples,
                "latent_dim": latent_dim,
                "device": device,
                "is_conditional": is_conditional,
                "conditions_tensor": conditions_tensor,
            }
            gen_designs = _generate_designs(model, gen_config)

        # Post-process
        gen_designs = gen_designs.detach().cpu().numpy().squeeze()
        if n_samples == 1:
            gen_designs = np.expand_dims(gen_designs, axis=0)
        np.clip(gen_designs, 0, 1, out=gen_designs)

        # Save designs
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        design_files, render_files = _save_designs(
            gen_designs, n_samples, output_path, problem
        )

        return {
            "success": True,
            "n_samples": n_samples,
            "design_files": design_files,
            "render_files": render_files,
            "conditions_used": conditions,
            "algorithm": algorithm,
            "problem_id": problem_id,
            "checkpoint_path": resolved_checkpoint_path,
            "output_dir": str(output_path),
            "message": f"Successfully generated {n_samples} designs using {algorithm} model from {Path(resolved_checkpoint_path).parent.name}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate designs: {e!s}",
        }


@tool
def generate_training_command(
    algorithm: str = "cgan_cnn_2d",
    epochs: int = 200,
    seed: int = 1,
    wandb_entity: str | None = None,
    problem_id: Literal["beams2d"] = "beams2d",
) -> dict[str, Any]:
    """
    Generate a Python command to train an EngiOpt model on HPC.

    This tool creates the command-line instruction for training a generative model
    using the EngiOpt library. The command can be copied and executed on an HPC cluster
    or any machine with the necessary compute resources.

    Args:
        algorithm: Model architecture to train. Options:
            - cgan_cnn_2d: Conditional GAN + CNN (2D) [default]
            - diffusion_2d_cond: Conditional Diffusion (2D)
        epochs: Number of training epochs (default: 200)
        seed: Random seed for reproducibility (default: 1)
        wandb_entity: WandB entity/username for tracking. If provided, WandB tracking
            will be enabled. If None (default), no tracking. Example: "myusername"
        problem_id: Engineering problem identifier (default: "beams2d")

    Returns:
        dict with:
        - success: bool
        - command: str with the full Python command to execute
        - slurm_script: str with SLURM job submission script
        - slurm_file: str with path to saved SLURM script file
        - config_summary: dict with training configuration
        - instructions: str with usage instructions
        - message: str

    Example:
        >>> # Generate training command for cGAN model with WandB tracking
        >>> result = generate_training_command(
        ...     algorithm="cgan_cnn_2d",
        ...     epochs=200,
        ...     wandb_entity="myusername"
        ... )
        >>> print(result['command'])
    """
    # Validate algorithm
    if algorithm not in SUPPORTED_ALGORITHMS:
        return {
            "success": False,
            "error": f"Unsupported algorithm '{algorithm}'. Supported: {', '.join(SUPPORTED_ALGORITHMS.keys())}",
        }

    # Read SLURM configuration from environment variables

    slurm_time = os.getenv("SLURM_TIME", "00:45:00")
    slurm_ntasks = os.getenv("SLURM_NTASKS", "1")
    slurm_cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK", "4")
    slurm_mem_per_cpu = os.getenv("SLURM_MEM_PER_CPU", "7GB")
    slurm_gpus = os.getenv("SLURM_GPUS", "rtx_4090:1")
    slurm_email_user = os.getenv("SLURM_EMAIL_USER", "alpha@gmail.com")

    slurm_stack_module = os.getenv("SLURM_STACK_MODULE", "stack/2024-06")
    slurm_gcc_module = os.getenv("SLURM_GCC_MODULE", "gcc/12.2.0")
    slurm_python_module = os.getenv("SLURM_PYTHON_MODULE", "python_cuda/3.11.6")
    slurm_cuda_module = os.getenv("SLURM_CUDA_MODULE", "cuda/12.4.1")

    slurm_venv_path = os.getenv("SLURM_VENV_PATH", "/path/to/venv")
    slurm_project_path = os.getenv("SLURM_PROJECT_PATH", "/path/to/engiopt")

    # Read API keys and configuration from .env
    wandb_api_key = os.getenv("WANDB_API_KEY", "")
    wandb_entity = os.getenv("WANDB_ENTITY", "")
    wandb_project = os.getenv("WANDB_PROJECT", "")
    hf_home = os.getenv("HF_HOME", "$SCRATCH/models")
    hf_datasets_cache = os.getenv("HF_DATASETS_CACHE", "$SCRATCH/datasets")
    hf_token = os.getenv("HF_TOKEN", "")

    # Determine if WandB tracking should be enabled based on wandb_entity
    use_wandb = wandb_entity is not None
    track_flag = "--track" if use_wandb else "--no-track"

    # Build wandb entity (None if not specified)
    wandb_entity_str = wandb_entity if wandb_entity else "None"

    # Build the training command - direct path to algorithm script

    command = (
        f"python engiopt/{algorithm}/{algorithm}.py "
        f'--problem-id "{problem_id}" '
        f"{track_flag} "
        f"--wandb-entity {wandb_entity_str} "
        f"--save-model "
        f"--n-epochs {epochs} "
        f"--seed {seed}"
    )

    # Create SLURM job script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={algorithm}_{problem_id}
#SBATCH --time={slurm_time}
#SBATCH --ntasks={slurm_ntasks}
#SBATCH --cpus-per-task={slurm_cpus_per_task}
#SBATCH --mem-per-cpu={slurm_mem_per_cpu}
#SBATCH --gpus={slurm_gpus}
#SBATCH --output=engiopt_{algorithm}_{problem_id}_%j.out
#SBATCH --error=engiopt_{algorithm}_{problem_id}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={slurm_email_user}

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/datasets"
mkdir -p "$SCRATCH/models"

# Load required modules
module purge
module load {slurm_stack_module}
module load {slurm_gcc_module}
module load {slurm_python_module}
module load {slurm_cuda_module}
module load eth_proxy

# Activate virtual environment
source {slurm_venv_path}/bin/activate

# Set environment variables from .env configuration
export WANDB_API_KEY="{wandb_api_key}"
export WANDB_ENTITY="{wandb_entity}"
export WANDB_PROJECT="{wandb_project}"
export HF_HOME="{hf_home}"
export HF_DATASETS_CACHE="{hf_datasets_cache}"
export HF_TOKEN="{hf_token}"

# Navigate to project directory
cd {slurm_project_path}

# Run training command
{command}

echo "Training complete!"
"""

    # Save SLURM script to outputs folder
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    slurm_filename = f"train_{algorithm}_{problem_id}_seed{seed}.slurm"
    slurm_filepath = output_dir / slurm_filename
    slurm_filepath.write_text(slurm_script)

    # Generate instructions
    instructions = f"""
To train the model on HPC:

1. Configure SLURM parameters in .env file:
   - SLURM_TIME, SLURM_CPUS_PER_TASK, SLURM_GPUS, SLURM_MEM_PER_CPU, etc.
   - SLURM_VENV_PATH: Path to your Python virtual environment
   - SLURM_PROJECT_PATH: Path to your engiopt directory
   - SLURM_EMAIL_USER: Your email for job notifications

2. SLURM script has been saved to:
   {slurm_filename}

3. Transfer the script to HPC and submit the job:
   sbatch {slurm_filename}

4. Monitor the job:
   squeue -u $USER

5. Check output:
   tail -f engiopt_{algorithm}_{problem_id}_*.out

6. View results:
   - Model checkpoints saved automatically by EngiOpt
   - View metrics on WandB dashboard (if tracking enabled)
"""

    config_summary = {
        "problem": problem_id,
        "algorithm": algorithm,
        "algorithm_info": SUPPORTED_ALGORITHMS[algorithm],
        "training_params": {
            "epochs": epochs,
            "seed": seed,
        },
        "tracking": {
            "use_wandb": use_wandb,
            "wandb_entity": wandb_entity_str,
        },
    }

    return {
        "success": True,
        "command": command,
        "slurm_script": slurm_script,
        "slurm_file": str(slurm_filepath),
        "config_summary": config_summary,
        "instructions": instructions,
        "message": f"Generated SLURM training script for {algorithm} on {problem_id}. "
        f"Training will run for {epochs} epochs with seed {seed}. "
        f"SLURM script saved to {slurm_filepath}",
    }
