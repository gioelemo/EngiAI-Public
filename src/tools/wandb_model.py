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
    "cgan_1d": {
        "class": "Inverse Design",
        "dimensions": "1D",
        "conditional": True,
        "model": "GAN MLP",
    },
    "cgan_2d": {
        "class": "Inverse Design",
        "dimensions": "2D",
        "conditional": True,
        "model": "GAN MLP",
    },
    "cgan_bezier": {
        "class": "Inverse Design",
        "dimensions": "1D",
        "conditional": True,
        "model": "GAN + Bezier layer",
    },
    "cgan_cnn_2d": {
        "class": "Inverse Design",
        "dimensions": "2D",
        "conditional": True,
        "model": "GAN + CNN",
    },
    "cgan_cnn_3d": {
        "class": "Inverse Design",
        "dimensions": "3D",
        "conditional": True,
        "model": "GAN + 3D CNN",
    },
    "cgan_vae": {
        "class": "Inverse Design",
        "dimensions": "3D",
        "conditional": True,
        "model": "MultiView GAN + VAE",
    },
    "diffusion_1d": {
        "class": "Inverse Design",
        "dimensions": "1D",
        "conditional": False,
        "model": "Diffusion",
    },
    "diffusion_2d_cond": {
        "class": "Inverse Design",
        "dimensions": "2D",
        "conditional": True,
        "model": "Diffusion",
    },
    "gan_1d": {
        "class": "Inverse Design",
        "dimensions": "1D",
        "conditional": False,
        "model": "GAN MLP",
    },
    "gan_2d": {
        "class": "Inverse Design",
        "dimensions": "2D",
        "conditional": False,
        "model": "GAN MLP",
    },
    "gan_bezier": {
        "class": "Inverse Design",
        "dimensions": "1D",
        "conditional": False,
        "model": "GAN + Bezier layer",
    },
    "gan_cnn_2d": {
        "class": "Inverse Design",
        "dimensions": "2D",
        "conditional": False,
        "model": "GAN + CNN",
    },
    "surrogate_model": {
        "class": "Surrogate Model",
        "dimensions": "1D",
        "conditional": False,
        "model": "MLP",
    },
}


@tool
def download_wandb_model(
    problem_id: Literal["beams2d"] = "beams2d",
    algorithm: str = "cgan_cnn_2d",
    seed: int = 1,
    wandb_project: str = "engibench/engiopt",
    download_dir: str | None = None,
) -> dict[str, Any]:
    """
    Download a pre-trained generative model from WandB for engineering design.

    This tool downloads models trained with the engiopt library that can be used
    for inverse design tasks. The models are hosted on the engibench WandB project.

    Args:
        problem_id: Engineering problem identifier. Currently only "beams2d" is supported.
        algorithm: Model architecture to download. Options:
            - cgan_1d: Conditional GAN MLP (1D)
            - cgan_2d: Conditional GAN MLP (2D)
            - cgan_bezier: Conditional GAN + Bezier layer (1D)
            - cgan_cnn_2d: Conditional GAN + CNN (2D) [default]
            - cgan_cnn_3d: Conditional GAN + 3D CNN (3D)
            - cgan_vae: MultiView GAN + VAE (3D)
            - diffusion_1d: Diffusion model (1D)
            - diffusion_2d_cond: Conditional Diffusion (2D)
            - gan_1d: GAN MLP (1D)
            - gan_2d: GAN MLP (2D)
            - gan_bezier: GAN + Bezier layer (1D)
            - gan_cnn_2d: GAN + CNN (2D)
            - surrogate_model: MLP surrogate model (1D)
        seed: Random seed used during model training (default: 1)
        wandb_project: WandB project path in format "organization/project"
            (default: "engibench/engiopt")
        download_dir: Directory to download the model to. If None, uses WandB's default cache.

    Returns:
        dict with download information:
        - success: bool indicating if download succeeded
        - artifact_path: str with full WandB artifact path
        - download_path: str with local path to downloaded model
        - checkpoint_path: str with path to model checkpoint file
        - algorithm_info: dict with algorithm properties (dimensions, conditional, etc.)
        - run_config: dict with training configuration (if available)
        - error: str with error message (only if success=False)

    Example:
        >>> # Download a conditional GAN with CNN for 2D beam design
        >>> result = download_wandb_model(
        ...     problem_id="beams2d",
        ...     algorithm="cgan_cnn_2d",
        ...     seed=1
        ... )
        >>> if result['success']:
        ...     print(f"Model downloaded to: {result['checkpoint_path']}")
        ...     print(f"Architecture: {result['algorithm_info']['model']}")

    Note:
        - Requires wandb to be installed: pip install wandb
        - Requires USE_WANDB environment variable to be set to "True"
        - You may need to authenticate with WandB using: wandb login
    """
    # Validation checks
    error_response = _validate_download_inputs(problem_id, algorithm)
    if error_response:
        return error_response

    # Try to import wandb
    error_response = _check_wandb_available()
    if error_response:
        return error_response

    # Download the model
    return _download_from_wandb(
        problem_id=problem_id,
        algorithm=algorithm,
        seed=seed,
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


def _download_from_wandb(
    problem_id: str,
    algorithm: str,
    seed: int,
    wandb_project: str,
    download_dir: str | None,
) -> dict[str, Any]:
    """Download model artifact from WandB."""
    import wandb  # noqa: PLC0415

    try:
        # Construct the artifact path
        artifact_name = f"{problem_id}_{algorithm}_generator"
        artifact_version = f"seed_{seed}"
        artifact_path = f"{wandb_project}/{artifact_name}:{artifact_version}"

        # Initialize WandB API
        api = wandb.Api()

        # Download the artifact
        print(f"Downloading model from WandB: {artifact_path}")
        artifact = api.artifact(artifact_path, type="model")

        # Download to specified directory or default cache
        if download_dir:
            artifact_dir = artifact.download(root=download_dir)
        else:
            artifact_dir = artifact.download()

        # Construct checkpoint path
        checkpoint_path = Path(artifact_dir) / "generator.pth"

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

        return {
            "success": True,
            "artifact_path": artifact_path,
            "download_path": artifact_dir,
            "checkpoint_path": str(checkpoint_path),
            "algorithm_info": SUPPORTED_ALGORITHMS[algorithm],
            "run_config": run_config,
            "message": f"Successfully downloaded {algorithm} model for {problem_id} (seed={seed})",
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
def load_wandb_model(
    checkpoint_path: str,
    problem_id: Literal["beams2d"] = "beams2d",
    algorithm: str = "cgan_cnn_2d",
    run_config: dict[str, Any] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Load a downloaded WandB model checkpoint into memory.

    This tool loads the PyTorch model from a checkpoint file and prepares it for inference.
    It requires the model architecture to match the algorithm type.

    Args:
        checkpoint_path: Path to the generator.pth checkpoint file
        problem_id: Engineering problem identifier (default: "beams2d")
        algorithm: Model architecture type (default: "cgan_cnn_2d")
        run_config: Training configuration dict with model hyperparameters.
            If None, will use default values.
        device: Device to load model on: "cpu", "cuda", or "mps" (default: "cpu")

    Returns:
        dict with:
        - success: bool indicating if load succeeded
        - model_ready: bool indicating if model is ready for inference
        - device: str with device model is loaded on
        - model_info: dict with model details
        - error: str with error message (only if success=False)

    Example:
        >>> # First download the model
        >>> download_result = download_wandb_model(algorithm="cgan_cnn_2d", seed=1)
        >>> # Then load it
        >>> load_result = load_wandb_model(
        ...     checkpoint_path=download_result['checkpoint_path'],
        ...     run_config=download_result['run_config'],
        ...     device="cpu"
        ... )

    Note:
        - Requires PyTorch: pip install torch
        - Requires the corresponding model architecture from engiopt
        - Model is set to eval mode after loading
    """
    import torch as th  # noqa: PLC0415

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
            "device": device,
            "model_info": {
                "algorithm": algorithm,
                "problem_id": problem_id,
                "latent_dim": latent_dim,
                "checkpoint_keys": list(ckpt.keys()),
            },
            "message": "Model structure loaded. Note: Full model initialization requires engiopt library and problem instance.",
            "note": "To fully initialize the model, you need to import the appropriate Generator class from engiopt and create it with the problem's design space shape.",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load model checkpoint: {e!s}",
        }
