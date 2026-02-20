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
        tool_name="simulate_design",
        objectives=[
            ObjectiveConfig(
                name="compliance",
                field_name="compliance",
                target_field="compliance",
                direction="minimize",
                relative_error_threshold=0.2,
                aliases=["c", "final_compliance", "final_c"],
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
        score_categories={
            "design_quality": {
                "weight": 0.65,  # Merged with printability (0.50 + 0.15)
                "iou": 0.31,
                "pixel_accuracy": 0.19,
                "constraint_match": 0.12,
                "objective_match": 0.15,
                "connectivity": 0.12,
                "watertightness": 0.11,
            },
            "tool_efficiency": {
                "weight": 0.20,
                "efficiency_ratio": 1.0,  # Tool ordering not scored
            },
            "task_completion": {
                "weight": 0.15,
                "success_rate": 1.0,
            },
        },
        prompt_file_template="beams2d_prompts_{n_samples}_samples_{split}_{style}_seed{seed}.json",
    ),
    "photonics2d": ProblemConfig(
        name="photonics2d",
        dataset_name="IDEALLab/photonics_2d_120_120_v0",
        design_field="optimal_design",
        tool_name="simulate_design",
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
        score_categories={
            "design_quality": {
                "weight": 0.65,  # Merged with printability (0.50 + 0.15)
                "iou": 0.31,
                "pixel_accuracy": 0.19,
                "constraint_match": 0.12,  # No constraints, but keep structure
                "objective_match": 0.15,
                "connectivity": 0.12,
                "watertightness": 0.11,
            },
            "tool_efficiency": {
                "weight": 0.20,
                "efficiency_ratio": 1.0,  # Tool ordering not scored
            },
            "task_completion": {
                "weight": 0.15,
                "success_rate": 1.0,
            },
        },
        prompt_file_template="photonics2d_prompts_{n_samples}_samples_{split}_{style}_seed{seed}.json",
    ),
    "thermoelastic2d": ProblemConfig(
        name="thermoelastic2d",
        dataset_name="IDEALLab/thermoelastic_2d_v0",
        design_field="optimal_design",
        tool_name="simulate_design",
        objectives=[
            ObjectiveConfig(
                name="structural_compliance",
                field_name="structural_compliance",
                target_field="structural_compliance",
                direction="minimize",
                relative_error_threshold=0.2,
                weight=0.4,  # Primary structural objective
                aliases=["struct_c", "sc", "final_structural_compliance"],
            ),
            ObjectiveConfig(
                name="thermal_compliance",
                field_name="thermal_compliance",
                target_field="thermal_compliance",
                direction="minimize",
                relative_error_threshold=0.2,
                weight=0.4,  # Primary thermal objective
                aliases=["thermal_c", "tc", "final_thermal_compliance"],
            ),
            ObjectiveConfig(
                name="volume_fraction",
                field_name="material_usage",
                target_field="volume_fraction",
                direction="minimize",
                relative_error_threshold=0.1,
                weight=0.2,  # Secondary objective
                aliases=["volfrac", "vf", "volume_fraction"],
            ),
        ],
        conditions=[
            ConditionConfig(
                name="volfrac",
                field_name="volfrac",
                constraint_type="none",  # Used as initial condition, not constraint
            ),
            ConditionConfig(
                name="rmin",
                field_name="rmin",
                constraint_type="none",
            ),
            ConditionConfig(
                name="weight",
                field_name="weight",
                constraint_type="none",  # Weighting between objectives
            ),
            ConditionConfig(
                name="fixed_elements",
                field_name="fixed_elements",
                constraint_type="none",
            ),
            ConditionConfig(
                name="force_elements_x",
                field_name="force_elements_x",
                constraint_type="none",
            ),
            ConditionConfig(
                name="force_elements_y",
                field_name="force_elements_y",
                constraint_type="none",
            ),
            ConditionConfig(
                name="heatsink_elements",
                field_name="heatsink_elements",
                constraint_type="none",
            ),
        ],
        score_categories={
            "design_quality": {
                "weight": 0.65,  # Merged with printability (0.50 + 0.15)
                "iou": 0.31,
                "pixel_accuracy": 0.19,
                "constraint_match": 0.0,  # No constraints to check
                "objective_match": 0.27,  # Higher weight for multi-objective matching
                "connectivity": 0.12,
                "watertightness": 0.11,
            },
            "tool_efficiency": {
                "weight": 0.20,
                "efficiency_ratio": 1.0,  # Tool ordering not scored
            },
            "task_completion": {
                "weight": 0.15,
                "success_rate": 1.0,
            },
        },
        prompt_file_template="thermoelastic2d_prompts_{n_samples}_samples_{split}_{style}_seed{seed}.json",
    ),
    "rag_beams2d": ProblemConfig(
        name="rag_beams2d",
        dataset_name="",  # No HuggingFace dataset — handcrafted prompts only
        design_field="optimal_design",
        tool_name="optimize_design",
        objectives=[],  # No ground-truth design comparison
        conditions=[
            ConditionConfig(
                name="expected_volfrac",
                field_name="expected_volfrac",
                constraint_type="none",  # Used for accuracy scoring, not a hard constraint
                tolerance=0.05,
                aliases=["volfrac"],
            )
        ],
        score_categories={
            "rag_accuracy": {
                "weight": 1.0,
                "rag_benefit_score": 1.0,
            },
        },
        prompt_file_template="rag_beams2d_prompts_4_samples_{split}_{style}.json",
    ),
    "hpc_train_beams2d": ProblemConfig(
        name="hpc_train_beams2d",
        dataset_name="",  # Handcrafted prompts — no HuggingFace sampling
        design_field="optimal_design",
        tool_name="generate_training_command",  # Primary tool for this benchmark
        objectives=[],  # Design quality scored offline via compare_hpc_designs.py
        conditions=[
            ConditionConfig(
                name="seed",
                field_name="seed",
                constraint_type="none",
                aliases=["training_seed"],
            ),
            ConditionConfig(
                name="epochs",
                field_name="epochs",
                constraint_type="none",
                aliases=["n_epochs"],
            ),
            ConditionConfig(
                name="algorithm",
                field_name="algorithm",
                constraint_type="none",
                aliases=["model_algorithm"],
            ),
        ],
        score_categories={
            "workflow_completion": {
                "weight": 0.60,
                "step_completion_rate": 1.0,
            },
            "task_completion": {
                "weight": 0.20,
                "success_rate": 1.0,
            },
            "tool_efficiency": {
                "weight": 0.20,
                "efficiency_ratio": 1.0,
            },
        },
        prompt_file_template="hpc_train_beams2d_prompts_3_samples_{split}_{style}.json",
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
