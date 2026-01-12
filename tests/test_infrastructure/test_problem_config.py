"""Tests for problem configuration and registry."""

import pytest

from benchmarks.shared.problem_config import (
    ConditionConfig,
    ObjectiveConfig,
    ProblemConfig,
)
from benchmarks.shared.problem_registry import (
    get_problem_config,
    list_problems,
    register_problem,
)


class TestObjectiveConfig:
    """Tests for ObjectiveConfig dataclass."""

    def test_valid_config(self):
        """Test creating a valid objective config."""
        obj = ObjectiveConfig(
            name="compliance",
            field_name="final_compliance",
            target_field="c",
            direction="minimize",
        )
        assert obj.name == "compliance"
        assert obj.direction == "minimize"
        assert obj.weight == 1.0

    def test_invalid_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="direction must be"):
            ObjectiveConfig(
                name="test",
                field_name="test",
                target_field="test",
                direction="invalid",
            )

    def test_negative_threshold(self):
        """Test that negative threshold raises error."""
        with pytest.raises(
            ValueError, match="relative_error_threshold must be positive"
        ):
            ObjectiveConfig(
                name="test",
                field_name="test",
                target_field="test",
                relative_error_threshold=-0.1,
            )


class TestConditionConfig:
    """Tests for ConditionConfig dataclass."""

    def test_valid_config(self):
        """Test creating a valid condition config."""
        cond = ConditionConfig(
            name="volume_fraction",
            field_name="volfrac",
            constraint_type="equality",
            tolerance=0.01,
        )
        assert cond.name == "volume_fraction"
        assert cond.constraint_type == "equality"

    def test_invalid_constraint_type(self):
        """Test that invalid constraint type raises error."""
        with pytest.raises(ValueError, match="constraint_type must be"):
            ConditionConfig(
                name="test",
                field_name="test",
                constraint_type="invalid",
            )


class TestProblemConfig:
    """Tests for ProblemConfig dataclass."""

    def test_beams2d_config(self):
        """Test that beams2d config is valid."""
        config = get_problem_config("beams2d")
        assert config.name == "beams2d"
        assert len(config.objectives) == 1
        assert config.objectives[0].name == "compliance"
        assert len(config.conditions) == 1
        assert config.conditions[0].name == "volume_fraction"

    def test_photonics2d_config(self):
        """Test that photonics2d config is valid."""
        config = get_problem_config("photonics2d")
        assert config.name == "photonics2d"
        assert len(config.objectives) == 1
        assert config.objectives[0].name == "total_overlap"
        assert len(config.conditions) == 3

    def test_thermoelastic2d_config(self):
        """Test that thermoelastic2d config is valid."""
        config = get_problem_config("thermoelastic2d")
        assert config.name == "thermoelastic2d"
        assert len(config.objectives) == 3  # Multi-objective problem
        assert config.objectives[0].name == "structural_compliance"
        assert config.objectives[1].name == "thermal_compliance"
        assert config.objectives[2].name == "volume_fraction"
        assert len(config.conditions) == 7  # volfrac, rmin, weight, + 4 element arrays

        # Verify multi-objective weights sum to 1.0
        objective_weights = sum(obj.weight for obj in config.objectives)
        assert 0.99 <= objective_weights <= 1.01

    def test_weights_sum_to_one(self):
        """Test that design metric weights sum to approximately 1.0."""
        config = get_problem_config("beams2d")
        total = sum(config.design_metrics_weights.values())
        assert 0.99 <= total <= 1.01

    def test_get_objective_by_name(self):
        """Test retrieving objective by name."""
        config = get_problem_config("beams2d")
        obj = config.get_objective_by_name("compliance")
        assert obj is not None
        assert obj.name == "compliance"

        # Non-existent objective
        assert config.get_objective_by_name("nonexistent") is None

    def test_get_condition_by_name(self):
        """Test retrieving condition by name."""
        config = get_problem_config("beams2d")
        cond = config.get_condition_by_name("volume_fraction")
        assert cond is not None
        assert cond.name == "volume_fraction"

    def test_get_constraint_conditions(self):
        """Test filtering constraint conditions."""
        beams_config = get_problem_config("beams2d")
        constraints = beams_config.get_constraint_conditions()
        assert len(constraints) == 1
        assert constraints[0].constraint_type == "equality"

        photonics_config = get_problem_config("photonics2d")
        constraints = photonics_config.get_constraint_conditions()
        assert len(constraints) == 0  # No actual constraints


class TestProblemRegistry:
    """Tests for problem registry functions."""

    def test_list_problems(self):
        """Test listing all problems."""
        problems = list_problems()
        assert "beams2d" in problems
        assert "photonics2d" in problems
        assert len(problems) >= 2

    def test_get_unknown_problem(self):
        """Test that getting unknown problem raises error."""
        with pytest.raises(ValueError, match="Unknown problem"):
            get_problem_config("nonexistent_problem")

    def test_register_new_problem(self):
        """Test registering a new problem."""
        new_config = ProblemConfig(
            name="test_problem",
            dataset_name="test/dataset",
            objectives=[
                ObjectiveConfig(
                    name="test_obj",
                    field_name="test",
                    target_field="test",
                )
            ],
        )

        register_problem(new_config)
        assert "test_problem" in list_problems()

        retrieved = get_problem_config("test_problem")
        assert retrieved.name == "test_problem"

    def test_duplicate_registration_fails(self):
        """Test that registering duplicate problem raises error."""
        config = get_problem_config("beams2d")
        with pytest.raises(ValueError, match="already registered"):
            register_problem(config)
