"""Problem configuration dataclasses for generic benchmark evaluation.

This module defines the structure for configuring benchmark problems in a declarative way,
enabling easy addition of new problems without writing problem-specific scorer code.
"""

from dataclasses import dataclass, field

# Constants for validation
WEIGHT_SUM_MIN = 0.99
WEIGHT_SUM_MAX = 1.01


@dataclass
class ObjectiveConfig:
    """Configuration for a single objective metric.

    Objectives are quantities that the optimizer tries to minimize or maximize
    (e.g., compliance in beams2d, total_overlap in photonics2d).
    """

    name: str
    """Human-readable name for the objective (e.g., 'compliance', 'total_overlap')"""

    field_name: str
    """Primary field name in tool output (e.g., 'final_compliance', 'total_overlap')"""

    target_field: str
    """Field name in the dataset for ground truth value (e.g., 'c', 'total_overlap')"""

    direction: str = "minimize"
    """Optimization direction: 'minimize' or 'maximize'"""

    relative_error_threshold: float = 0.2
    """Threshold for computing objective match score (relative error tolerance)"""

    weight: float = 1.0
    """Weight for this objective if multiple objectives exist"""

    aliases: list[str] = field(default_factory=list)
    """Alternative field names to try when extracting from tool output"""

    def __post_init__(self):
        """Validate configuration."""
        if self.direction not in ["minimize", "maximize"]:
            raise ValueError(
                f"direction must be 'minimize' or 'maximize', got '{self.direction}'"
            )
        if self.relative_error_threshold <= 0:
            raise ValueError(
                f"relative_error_threshold must be positive, got {self.relative_error_threshold}"
            )
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")


@dataclass
class ConditionConfig:
    """Configuration for problem conditions/constraints.

    Conditions are user-specified parameters that alter the problem definition
    (e.g., volume_fraction in beams2d, wavelengths in photonics2d).
    Some conditions are constraints (must be satisfied), others are just parameters.
    """

    name: str
    """Human-readable name for the condition (e.g., 'volume_fraction', 'lambda1')"""

    field_name: str
    """Field name in the dataset (e.g., 'volfrac', 'lambda1')"""

    constraint_type: str = "none"
    """Constraint type: 'equality', 'inequality', or 'none' (just a parameter)"""

    tolerance: float = 0.01
    """Tolerance for constraint satisfaction (used for equality constraints)"""

    aliases: list[str] = field(default_factory=list)
    """Alternative field names in the dataset"""

    def __post_init__(self):
        """Validate configuration."""
        if self.constraint_type not in ["equality", "inequality", "none"]:
            raise ValueError(
                f"constraint_type must be 'equality', 'inequality', or 'none', "
                f"got '{self.constraint_type}'"
            )
        if self.tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {self.tolerance}")


@dataclass
class ProblemConfig:
    """Complete configuration for a benchmark problem.

    This dataclass contains all information needed to:
    - Load datasets
    - Extract objectives from tool outputs
    - Check constraints
    - Compute scores
    """

    name: str
    """Problem identifier (e.g., 'beams2d', 'photonics2d')"""

    dataset_name: str
    """HuggingFace dataset identifier (e.g., 'IDEALLab/beams_2d_50_100_v0')"""

    design_field: str = "optimal_design"
    """Field name for design array in the dataset"""

    tool_name: str = "optimize_design"
    """Name of the tool that returns optimization results"""

    objectives: list[ObjectiveConfig] = field(default_factory=list)
    """List of objectives to track and evaluate"""

    conditions: list[ConditionConfig] = field(default_factory=list)
    """List of conditions/constraints to check"""

    score_categories: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
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
        }
    )
    """Hierarchical score configuration with category weights and metric weights within categories"""

    prompt_file_template: str = "{problem}_prompts_{samples}_samples_{split}.json"
    """Template for prompt file naming"""

    def __post_init__(self):
        """Validate configuration and compute derived properties."""
        # Validate hierarchical score categories
        category_weights_sum = 0.0
        for category_name, category_config in self.score_categories.items():
            # Extract category weight
            if "weight" not in category_config:
                raise ValueError(f"Category '{category_name}' must have a 'weight' key")
            category_weight = category_config["weight"]
            category_weights_sum += category_weight

            # Validate metric weights within category sum to 1.0
            metric_weights = {k: v for k, v in category_config.items() if k != "weight"}
            if metric_weights:  # Only validate if there are metrics
                metric_sum = sum(metric_weights.values())
                if not (WEIGHT_SUM_MIN <= metric_sum <= WEIGHT_SUM_MAX):
                    raise ValueError(
                        f"Metric weights in category '{category_name}' should sum to ~1.0, "
                        f"got {metric_sum}. Weights: {metric_weights}"
                    )

        # Validate category weights sum to 1.0
        if not (WEIGHT_SUM_MIN <= category_weights_sum <= WEIGHT_SUM_MAX):
            raise ValueError(
                f"Category weights should sum to ~1.0, got {category_weights_sum}. "
                f"Categories: {list(self.score_categories.keys())}"
            )

        # Validate objectives
        if not self.objectives:
            raise ValueError(f"Problem '{self.name}' must have at least one objective")

        # Check for duplicate objective names
        obj_names = [obj.name for obj in self.objectives]
        if len(obj_names) != len(set(obj_names)):
            raise ValueError(f"Duplicate objective names found: {obj_names}")

        # Check for duplicate condition names
        cond_names = [cond.name for cond in self.conditions]
        if len(cond_names) != len(set(cond_names)):
            raise ValueError(f"Duplicate condition names found: {cond_names}")

    def get_objective_by_name(self, name: str) -> ObjectiveConfig | None:
        """Get objective configuration by name."""
        for obj in self.objectives:
            if obj.name == name:
                return obj
        return None

    def get_condition_by_name(self, name: str) -> ConditionConfig | None:
        """Get condition configuration by name."""
        for cond in self.conditions:
            if cond.name == name:
                return cond
        return None

    def get_constraint_conditions(self) -> list[ConditionConfig]:
        """Get only conditions that are actual constraints."""
        return [cond for cond in self.conditions if cond.constraint_type != "none"]

    def get_category_weight(self, category_name: str) -> float:
        """Get the weight for a specific score category."""
        if category_name not in self.score_categories:
            return 0.0
        return self.score_categories[category_name].get("weight", 0.0)

    def get_metric_weights(self, category_name: str) -> dict[str, float]:
        """Get metric weights for a specific category (excluding the category weight)."""
        if category_name not in self.score_categories:
            return {}
        return {
            k: v
            for k, v in self.score_categories[category_name].items()
            if k != "weight"
        }
