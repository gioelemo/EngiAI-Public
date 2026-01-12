"""Central registry for all benchmark problems.

This module maintains the single source of truth for problem configurations.
To add a new problem, simply add its configuration to the PROBLEMS dictionary.
"""

from benchmarks.shared.problem_config import (
    ConditionConfig,
    ObjectiveConfig,
    ProblemConfig,
)

# Registry of all available problems
PROBLEMS: dict[str, ProblemConfig] = {
    "beams2d": ProblemConfig(
        name="beams2d",
        dataset_name="IDEALLab/beams_2d_50_100_v0",
        design_field="optimal_design",
        tool_name="optimize_design",
        objectives=[
            ObjectiveConfig(
                name="compliance",
                field_name="final_compliance",
                target_field="c",
                direction="minimize",
                relative_error_threshold=0.2,
                aliases=["final_c", "c"],
            )
        ],
        conditions=[
            ConditionConfig(
                name="volume_fraction",
                field_name="volfrac",
                constraint_type="equality",
                tolerance=0.01,
                aliases=["volume"],
            )
        ],
        design_metrics_weights={
            "iou": 0.4,
            "pixel_accuracy": 0.25,
            "constraint_match": 0.15,
            "objective_match": 0.2,
        },
        prompt_file_template="beams2d_prompts_50_samples_{split}.json",
    ),
    "photonics2d": ProblemConfig(
        name="photonics2d",
        dataset_name="IDEALLab/photonics_2d_120_120_v0",
        design_field="optimal_design",
        tool_name="optimize_design",
        objectives=[
            ObjectiveConfig(
                name="total_overlap",
                field_name="total_overlap",
                target_field="total_overlap",
                direction="maximize",
                relative_error_threshold=0.2,
                aliases=["overlap"],
            )
        ],
        conditions=[
            ConditionConfig(
                name="lambda1",
                field_name="lambda1",
                constraint_type="none",  # Not a constraint, just a condition parameter
            ),
            ConditionConfig(
                name="lambda2",
                field_name="lambda2",
                constraint_type="none",
            ),
            ConditionConfig(
                name="blur_radius",
                field_name="blur_radius",
                constraint_type="none",
            ),
        ],
        design_metrics_weights={
            "iou": 0.4,
            "pixel_accuracy": 0.25,
            "constraint_match": 0.15,  # No constraints, but keep structure
            "objective_match": 0.2,
        },
        prompt_file_template="photonics2d_prompts_50_samples_{split}.json",
    ),
}


def get_problem_config(problem_name: str) -> ProblemConfig:
    """Get problem configuration by name.

    Args:
        problem_name: Name of the problem (e.g., 'beams2d', 'photonics2d')

    Returns:
        ProblemConfig for the requested problem

    Raises:
        ValueError: If problem_name is not in the registry
    """
    if problem_name not in PROBLEMS:
        available = ", ".join(sorted(PROBLEMS.keys()))
        raise ValueError(
            f"Unknown problem: '{problem_name}'. Available problems: {available}"
        )
    return PROBLEMS[problem_name]


def list_problems() -> list[str]:
    """Get list of all registered problem names."""
    return sorted(PROBLEMS.keys())


def register_problem(config: ProblemConfig) -> None:
    """Register a new problem configuration.

    This allows dynamic registration of problems at runtime.

    Args:
        config: ProblemConfig to register

    Raises:
        ValueError: If a problem with this name already exists
    """
    if config.name in PROBLEMS:
        raise ValueError(
            f"Problem '{config.name}' already registered. "
            f"Use update_problem() to modify existing configurations."
        )
    PROBLEMS[config.name] = config


def update_problem(config: ProblemConfig) -> None:
    """Update an existing problem configuration.

    Args:
        config: Updated ProblemConfig

    Raises:
        ValueError: If problem doesn't exist
    """
    if config.name not in PROBLEMS:
        raise ValueError(
            f"Problem '{config.name}' not found. "
            f"Use register_problem() to add new problems."
        )
    PROBLEMS[config.name] = config
