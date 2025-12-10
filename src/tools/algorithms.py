"""Algorithm type definitions for EngiOpt integration.

This is the SINGLE FILE to update when adding a new algorithm type:
1. Add to AlgorithmId Literal type
2. Add to SUPPORTED_ALGORITHMS mapping with properties

Everything else (algorithm lists, registries) is derived automatically.
"""

from typing import Literal, get_args

# Type alias for algorithm IDs - single source of truth for supported algorithms
AlgorithmId = Literal["cgan_cnn_2d", "diffusion_2d_cond"]

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

# List of supported algorithm IDs - automatically derived from AlgorithmId
ALGORITHM_IDS = list(get_args(AlgorithmId))

# Export for use in other modules
__all__ = [
    "ALGORITHM_IDS",
    "SUPPORTED_ALGORITHMS",
    "AlgorithmId",
]
