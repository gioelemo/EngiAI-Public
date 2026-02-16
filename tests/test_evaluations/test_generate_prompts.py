"""Tests for workflow-random and workflow-conditional prompt generation functionality.

Tests for the _generate_random_stl_params() and _generate_random_conditional_params()
functions and corresponding prompt creation in benchmarks/problems/beams2d/generate_prompts.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path to import benchmark modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.problems.beams2d.generate_prompts import (  # noqa: E402
    _create_workflow_conditional_prompt,
    _create_workflow_random_prompt,
    _generate_random_conditional_params,
    _generate_random_stl_params,
    create_prompt_from_conditions,
)

# ============================================================================
# RANDOM PARAMETER GENERATION TESTS
# ============================================================================


@pytest.mark.unit
def test_generate_random_stl_params_deterministic():
    """Test that same seed produces same random parameters."""
    seed = 42
    params1 = _generate_random_stl_params(seed)
    params2 = _generate_random_stl_params(seed)

    assert params1 == params2
    assert params1["mirror_y"] == params2["mirror_y"]
    assert params1["scale_xy"] == params2["scale_xy"]
    assert params1["scale_z"] == params2["scale_z"]
    assert params1["threshold"] == params2["threshold"]


@pytest.mark.unit
def test_generate_random_stl_params_different_seeds():
    """Test that different seeds produce different parameters."""
    params1 = _generate_random_stl_params(seed=1)
    params2 = _generate_random_stl_params(seed=2)

    # At least one parameter should be different
    # (extremely unlikely to be all identical with different seeds)
    assert (
        params1["mirror_y"] != params2["mirror_y"]
        or params1["scale_xy"] != params2["scale_xy"]
        or params1["scale_z"] != params2["scale_z"]
        or params1["threshold"] != params2["threshold"]
    )


@pytest.mark.unit
def test_generate_random_stl_params_structure():
    """Test that generated parameters have correct structure and types."""
    params = _generate_random_stl_params(seed=42)

    # Check all expected keys exist
    assert "mirror_y" in params
    assert "scale_xy" in params
    assert "scale_z" in params
    assert "threshold" in params

    # Check types
    assert isinstance(params["mirror_y"], bool)
    assert isinstance(params["scale_xy"], float)
    assert isinstance(params["scale_z"], float)
    assert isinstance(params["threshold"], float)


@pytest.mark.unit
def test_generate_random_stl_params_ranges():
    """Test that generated parameters are within expected ranges."""
    # Test multiple seeds to ensure ranges are respected
    for seed in range(10):
        params = _generate_random_stl_params(seed)

        # mirror_y is boolean (no range check needed)
        assert isinstance(params["mirror_y"], bool)

        # scale_xy should be in [0.5, 5.0]
        assert 0.5 <= params["scale_xy"] <= 5.0

        # scale_z should be in [5.0, 20.0]
        assert 5.0 <= params["scale_z"] <= 20.0

        # threshold should be in [0.3, 0.7]
        assert 0.3 <= params["threshold"] <= 0.7


@pytest.mark.unit
def test_generate_random_stl_params_no_seed():
    """Test that function works without explicit seed."""
    params = _generate_random_stl_params()

    # Should still return valid parameters
    assert "mirror_y" in params
    assert "scale_xy" in params
    assert "scale_z" in params
    assert "threshold" in params

    # Check types
    assert isinstance(params["mirror_y"], bool)
    assert isinstance(params["scale_xy"], float)
    assert isinstance(params["scale_z"], float)
    assert isinstance(params["threshold"], float)


@pytest.mark.unit
def test_generate_random_stl_params_mirror_boolean_distribution():
    """Test that mirror_y generates both True and False values across seeds."""
    mirror_values = set()
    for seed in range(20):
        params = _generate_random_stl_params(seed)
        mirror_values.add(params["mirror_y"])

    # Should see both True and False across 20 seeds
    assert True in mirror_values
    assert False in mirror_values


# ============================================================================
# WORKFLOW-RANDOM PROMPT CREATION TESTS
# ============================================================================


@pytest.mark.unit
def test_create_workflow_random_prompt_structure():
    """Test that workflow-random prompt returns correct structure."""
    volfrac = 0.3
    forcedist = 0.5
    rmin = 2.0
    example_id = 0
    seed = 42

    prompt, stl_params = _create_workflow_random_prompt(
        volfrac, forcedist, rmin, example_id, seed
    )

    # Check prompt is a string
    assert isinstance(prompt, str)
    assert len(prompt) > 0

    # Check stl_params has expected structure
    assert isinstance(stl_params, dict)
    assert "mirror_y" in stl_params
    assert "scale_xy" in stl_params
    assert "scale_z" in stl_params
    assert "threshold" in stl_params


@pytest.mark.unit
def test_create_workflow_random_prompt_contains_parameters():
    """Test that prompt contains the specified parameters."""
    volfrac = 0.225
    forcedist = 0.65
    rmin = 3.0
    example_id = 1
    seed = 42

    prompt, stl_params = _create_workflow_random_prompt(
        volfrac, forcedist, rmin, example_id, seed
    )

    # Check that optimization parameters are in prompt
    assert str(volfrac) in prompt
    assert str(forcedist) in prompt
    assert str(rmin) in prompt

    # Check that STL parameters are in prompt
    assert f"{stl_params['threshold']:.2f}" in prompt
    assert f"{stl_params['scale_xy']:.2f}" in prompt
    assert f"{stl_params['scale_z']:.1f}" in prompt

    # Check mirror instruction
    if stl_params["mirror_y"]:
        assert "Mirror the design across the y-axis" in prompt
    else:
        assert "Do NOT mirror the design" in prompt


@pytest.mark.unit
def test_create_workflow_random_prompt_unique_per_example():
    """Test that same seed but different example_id produces different params."""
    seed = 42
    volfrac = 0.3
    forcedist = 0.5
    rmin = 2.0

    _, params1 = _create_workflow_random_prompt(volfrac, forcedist, rmin, 0, seed)
    _, params2 = _create_workflow_random_prompt(volfrac, forcedist, rmin, 1, seed)

    # Different example_id should produce different params
    assert (
        params1["mirror_y"] != params2["mirror_y"]
        or params1["scale_xy"] != params2["scale_xy"]
        or params1["scale_z"] != params2["scale_z"]
        or params1["threshold"] != params2["threshold"]
    )


@pytest.mark.unit
def test_create_workflow_random_prompt_deterministic():
    """Test that same inputs produce same prompt and params."""
    volfrac = 0.3
    forcedist = 0.5
    rmin = 2.0
    example_id = 5
    seed = 42

    prompt1, params1 = _create_workflow_random_prompt(
        volfrac, forcedist, rmin, example_id, seed
    )
    prompt2, params2 = _create_workflow_random_prompt(
        volfrac, forcedist, rmin, example_id, seed
    )

    assert prompt1 == prompt2
    assert params1 == params2


@pytest.mark.unit
def test_create_workflow_random_prompt_mirror_instructions():
    """Test that mirror instructions are clear and unambiguous."""
    volfrac = 0.3
    forcedist = 0.5
    rmin = 2.0

    # Test with seed that produces mirror_y=True
    for seed in range(100):
        _, params = _create_workflow_random_prompt(volfrac, forcedist, rmin, 0, seed)
        prompt, _ = _create_workflow_random_prompt(volfrac, forcedist, rmin, 0, seed)

        if params["mirror_y"]:
            assert "Mirror the design across the y-axis" in prompt
            assert "Do NOT mirror" not in prompt
        else:
            assert "Do NOT mirror the design" in prompt
            assert (
                "Mirror the design across the y-axis" not in prompt
                or "Do NOT mirror" in prompt
            )


# ============================================================================
# INTEGRATION WITH create_prompt_from_conditions TESTS
# ============================================================================


@pytest.mark.unit
def test_create_prompt_workflow_random_integration():
    """Test that workflow-random is properly integrated in create_prompt_from_conditions."""
    example = {
        "volfrac": 0.3,
        "forcedist": 0.5,
        "rmin": 2.0,
        "c": 100.0,
        "optimal_design": np.zeros((10, 10)),
        "example_id": 0,
    }

    prompt_data = create_prompt_from_conditions(
        example, include_target=True, prompt_style="workflow-random", seed=42
    )

    # Check structure
    assert "prompt" in prompt_data
    assert "prompt_style" in prompt_data
    assert prompt_data["prompt_style"] == "workflow-random"
    assert "stl_expected_params" in prompt_data

    # Check STL params
    stl_params = prompt_data["stl_expected_params"]
    assert "mirror_y" in stl_params
    assert "scale_xy" in stl_params
    assert "scale_z" in stl_params
    assert "threshold" in stl_params

    # Check metadata
    assert "metadata" in prompt_data
    assert prompt_data["metadata"]["success_criteria"] == "stl_export_with_params"

    # Check optimal tool calls
    assert "optimal_tool_calls" in prompt_data
    tool_names = [tool["name"] for tool in prompt_data["optimal_tool_calls"]]
    assert "optimize_design" in tool_names
    assert "simulate_design" in tool_names
    assert "convert_design_to_stl" in tool_names


@pytest.mark.unit
def test_create_prompt_workflow_random_deterministic_with_seed():
    """Test that create_prompt_from_conditions is deterministic with seed."""
    example = {
        "volfrac": 0.3,
        "forcedist": 0.5,
        "rmin": 2.0,
        "c": 100.0,
        "optimal_design": np.zeros((10, 10)),
        "example_id": 0,
    }
    seed = 42

    data1 = create_prompt_from_conditions(
        example, include_target=False, prompt_style="workflow-random", seed=seed
    )
    data2 = create_prompt_from_conditions(
        example, include_target=False, prompt_style="workflow-random", seed=seed
    )

    assert data1["prompt"] == data2["prompt"]
    assert data1["stl_expected_params"] == data2["stl_expected_params"]


@pytest.mark.unit
def test_create_prompt_workflow_style_unchanged():
    """Test that regular workflow style still works and has no stl_expected_params."""
    example = {
        "volfrac": 0.3,
        "forcedist": 0.5,
        "rmin": 2.0,
        "c": 100.0,
        "optimal_design": np.zeros((10, 10)),
        "example_id": 0,
    }

    # Test regular workflow (should not have randomized params)
    prompt_data = create_prompt_from_conditions(
        example, include_target=False, prompt_style="workflow", seed=42
    )

    assert prompt_data["prompt_style"] == "workflow"
    assert "stl_expected_params" not in prompt_data
    assert prompt_data["metadata"]["success_criteria"] == "stl_export"

    # Prompt should have hardcoded values
    assert "0.5 density threshold" in prompt_data["prompt"]
    assert "5 units in the Z-axis" in prompt_data["prompt"]


@pytest.mark.unit
def test_create_prompt_full_style_unchanged():
    """Test that full style still works and has no stl_expected_params."""
    example = {
        "volfrac": 0.3,
        "forcedist": 0.5,
        "rmin": 2.0,
        "c": 100.0,
        "optimal_design": np.zeros((10, 10)),
        "example_id": 0,
    }

    prompt_data = create_prompt_from_conditions(
        example, include_target=False, prompt_style="full", seed=42
    )

    assert prompt_data["prompt_style"] == "full"
    assert "stl_expected_params" not in prompt_data
    assert prompt_data["metadata"]["success_criteria"] == "render_design"


@pytest.mark.unit
def test_workflow_random_different_examples_different_params():
    """Test that different examples get different random parameters."""
    base_example = {
        "volfrac": 0.3,
        "forcedist": 0.5,
        "rmin": 2.0,
        "c": 100.0,
        "optimal_design": np.zeros((10, 10)),
    }
    seed = 42

    example1 = {**base_example, "example_id": 0}
    example2 = {**base_example, "example_id": 1}

    data1 = create_prompt_from_conditions(
        example1, include_target=False, prompt_style="workflow-random", seed=seed
    )
    data2 = create_prompt_from_conditions(
        example2, include_target=False, prompt_style="workflow-random", seed=seed
    )

    # Same seed but different example_id should give different params
    assert data1["stl_expected_params"] != data2["stl_expected_params"]


# ============================================================================
# WORKFLOW-CONDITIONAL RANDOM PARAMETER GENERATION TESTS
# ============================================================================


@pytest.mark.unit
def test_generate_random_conditional_params_deterministic():
    """Test that same seed produces same conditional parameters."""
    params1 = _generate_random_conditional_params(seed=42)
    params2 = _generate_random_conditional_params(seed=42)
    assert params1 == params2


@pytest.mark.unit
def test_generate_random_conditional_params_structure():
    """Test conditional params have correct structure and value ranges."""
    params = _generate_random_conditional_params(seed=42)

    assert params["conditional"] is True
    assert "compliance_threshold" in params
    assert "branch_high" in params
    assert "branch_low" in params
    assert "common" in params

    # Check ranges
    assert 100.0 <= params["compliance_threshold"] <= 300.0
    assert 0.3 <= params["branch_high"]["threshold"] <= 0.7
    assert 0.3 <= params["branch_low"]["threshold"] <= 0.7
    assert isinstance(params["branch_high"]["mirror_y"], bool)
    assert isinstance(params["branch_low"]["mirror_y"], bool)
    assert 0.5 <= params["common"]["scale_xy"] <= 5.0
    assert 5.0 <= params["common"]["scale_z"] <= 20.0


@pytest.mark.unit
def test_generate_random_conditional_params_branches_distinct():
    """Test that branches are guaranteed distinct across many seeds."""
    for seed in range(20):
        params = _generate_random_conditional_params(seed=seed)
        bh = params["branch_high"]
        bl = params["branch_low"]

        # mirror_y must be opposite
        assert bh["mirror_y"] != bl["mirror_y"]

        # thresholds must be at least 0.1 apart
        assert abs(bh["threshold"] - bl["threshold"]) >= 0.1


# ============================================================================
# WORKFLOW-CONDITIONAL PROMPT CREATION TESTS
# ============================================================================


@pytest.mark.unit
def test_create_workflow_conditional_prompt_contains_if_then():
    """Test that prompt text contains conditional instructions."""
    prompt, params = _create_workflow_conditional_prompt(0.3, 0.5, 2.0, 0, seed=42)

    ct = params["compliance_threshold"]
    assert f"If compliance > {ct:.1f}" in prompt
    assert f"If compliance <= {ct:.1f}" in prompt
    assert f"{params['branch_high']['threshold']:.2f}" in prompt
    assert f"{params['branch_low']['threshold']:.2f}" in prompt
    assert f"{params['common']['scale_xy']:.2f}" in prompt
    assert f"{params['common']['scale_z']:.1f}" in prompt
    assert "In both cases" in prompt


@pytest.mark.unit
def test_create_workflow_conditional_prompt_deterministic():
    """Test that same inputs produce same prompt and params."""
    prompt1, params1 = _create_workflow_conditional_prompt(0.3, 0.5, 2.0, 5, seed=42)
    prompt2, params2 = _create_workflow_conditional_prompt(0.3, 0.5, 2.0, 5, seed=42)

    assert prompt1 == prompt2
    assert params1 == params2


@pytest.mark.unit
def test_create_workflow_conditional_prompt_unique_per_example():
    """Test that different example_ids produce different params."""
    _, params1 = _create_workflow_conditional_prompt(0.3, 0.5, 2.0, 0, seed=42)
    _, params2 = _create_workflow_conditional_prompt(0.3, 0.5, 2.0, 1, seed=42)

    assert params1 != params2


# ============================================================================
# WORKFLOW-CONDITIONAL INTEGRATION WITH create_prompt_from_conditions
# ============================================================================


@pytest.mark.unit
def test_create_prompt_workflow_conditional_integration():
    """Test workflow-conditional integration in create_prompt_from_conditions."""
    example = {
        "volfrac": 0.3,
        "forcedist": 0.5,
        "rmin": 2.0,
        "c": 250.0,
        "optimal_design": np.zeros((10, 10)),
        "example_id": 0,
    }

    prompt_data = create_prompt_from_conditions(
        example, include_target=True, prompt_style="workflow-conditional", seed=42
    )

    assert prompt_data["prompt_style"] == "workflow-conditional"
    assert "stl_expected_params" in prompt_data
    assert prompt_data["stl_expected_params"]["conditional"] is True
    assert prompt_data["metadata"]["success_criteria"] == "stl_export_with_params"

    # Target should include compliance
    assert "target" in prompt_data
    assert "compliance" in prompt_data["target"]

    # Optimal tool calls should include optimize, simulate, convert_to_stl
    tool_names = [tool["name"] for tool in prompt_data["optimal_tool_calls"]]
    assert "optimize_design" in tool_names
    assert "simulate_design" in tool_names
    assert "convert_design_to_stl" in tool_names


@pytest.mark.unit
def test_workflow_conditional_no_stl_params_for_other_styles():
    """Test that existing styles are not affected by conditional additions."""
    example = {
        "volfrac": 0.3,
        "forcedist": 0.5,
        "rmin": 2.0,
        "c": 100.0,
        "optimal_design": np.zeros((10, 10)),
        "example_id": 0,
    }

    for style in ["full", "workflow"]:
        prompt_data = create_prompt_from_conditions(
            example, include_target=False, prompt_style=style, seed=42
        )
        assert "stl_expected_params" not in prompt_data
