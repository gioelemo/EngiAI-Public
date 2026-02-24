"""Problem type definitions and registry for EngiBench integration.

This is the SINGLE FILE to update when adding a new problem type:
1. Add import for the problem class
2. Add to ProblemId Literal type
3. Add to PROBLEM_CLASSES mapping

Everything else (SUPPORTED_PROBLEMS, registries) is derived automatically.
"""

from typing import Literal, get_args

from engibench.problems.beams2d.v1 import Beams2D  # type: ignore[import-untyped]
from engibench.problems.photonics2d.v0 import (
    Photonics2D,  # type: ignore[import-untyped]
)

# Type alias for problem IDs - this is the single source of truth for supported problems
ProblemId = Literal["beams2d", "photonics2d"]

# Map problem IDs to their class implementations
PROBLEM_CLASSES = {
    "beams2d": Beams2D,
    "photonics2d": Photonics2D,
}

# Supported problem types - automatically derived from ProblemId
SUPPORTED_PROBLEMS = list(get_args(ProblemId))

# Export problem classes for type hints
__all__ = [
    "PROBLEM_CLASSES",
    "SUPPORTED_PROBLEMS",
    "Beams2D",
    "Photonics2D",
    "ProblemId",
]
