"""Constants used across the engineering assistant tools."""

from typing import Literal

# Supported problem types for EngiOpt/EngiBench
SUPPORTED_PROBLEMS = ["beams2d", "thermoelastic2d"]

# Type alias for problem IDs
ProblemId = Literal["beams2d", "thermoelastic2d"]
