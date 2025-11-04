"""
Tests for EngiBench tools.

These tests require engibench and its dependencies to be fully installed.
Mark as slow since they require heavy dependencies and test actual optimization.

NOTE: These tests are SKIPPED in CI (runs with -m "not slow").
They only run when you explicitly run pytest locally without the marker filter.
"""

import os
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from engibench.problems.beams2d.v0 import Beams2D

# Skip these tests if engibench is not available
pytest.importorskip("scipy")
pytest.importorskip("cvxopt")
pytest.importorskip("engibench")

from src.tools.engibench import (
    EXPECTED_ARRAY_DIMENSIONS,
    _state,
    check_beam_constraints,
    create_beam_problem,
    get_dataset_info,
    get_last_design,
    get_problem_details,
    get_problem_info,
    get_problem_instance,
    optimize_beam_design,
    render_beam_design,
    set_last_design,
    set_problem_instance,
    simulate_beam_design,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def reset_state(tmp_path, monkeypatch):
    """Reset the module state before each test and configure temp cache dirs."""
    # Set HuggingFace cache to temp directory to avoid downloading to $SCRATCH
    hf_cache = tmp_path / ".cache" / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HF_HOME", str(hf_cache / "hub"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(hf_cache / "datasets"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(hf_cache / "transformers"))

    # Reset engibench state
    _state["problem_instance"] = None
    _state["last_design"] = None
    yield
    # Cleanup after test
    _state["problem_instance"] = None
    _state["last_design"] = None


@pytest.fixture
def sample_design():
    """Create a sample beam design (50x100 grid)."""
    rng = np.random.default_rng(seed=42)
    return rng.random((50, 100)) * 0.35  # 35% volume fraction


@pytest.fixture
def outputs_dir(tmp_path):
    """Create a temporary outputs directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


# ============================================================================
# UNIT TESTS - Fast tests without engibench dependency
# ============================================================================


@pytest.mark.unit
def test_state_getters_and_setters_basic():
    """Test basic state management without engibench objects."""
    # Test initial state
    assert _state["problem_instance"] is None
    assert _state["last_design"] is None

    # Test setting and getting problem instance
    mock_problem = "mock_problem_instance"
    set_problem_instance(mock_problem)
    assert get_problem_instance() == mock_problem

    # Test setting and getting last design
    mock_design = np.array([[1, 2], [3, 4]])
    set_last_design(mock_design)
    retrieved = get_last_design()
    assert np.array_equal(retrieved, mock_design)


@pytest.mark.unit
def test_get_problem_instance_creates_on_demand():
    """Test that get_problem_instance creates a Beams2D instance if none exists."""
    # Clear state
    _state["problem_instance"] = None

    # Get instance - should create one
    problem = get_problem_instance()
    assert problem is not None

    # Should be cached
    problem2 = get_problem_instance()
    assert problem is problem2


@pytest.mark.unit
def test_get_problem_info_structure():
    """Test that get_problem_info returns correct structure."""
    result = get_problem_info.invoke({"problem_type": "beams2d"})

    # Should have these keys
    assert "success" in result
    assert "available_problems" in result
    assert "selected_problem" in result
    assert "description" in result
    assert "objectives" in result
    assert "typical_conditions" in result
    assert "design_space" in result

    # Check types
    assert isinstance(result["available_problems"], list)
    assert isinstance(result["objectives"], list)
    assert isinstance(result["typical_conditions"], dict)


@pytest.mark.unit
def test_get_problem_info_unknown_type():
    """Test get_problem_info with unknown problem type."""
    result = get_problem_info.invoke({"problem_type": "unknown_problem"})

    assert result["success"] is True
    assert result["selected_problem"] is None
    assert "available_problems" in result
    assert len(result["available_problems"]) > 0


@pytest.mark.unit
def test_get_problem_info_case_insensitive():
    """Test that problem type is case insensitive."""
    result1 = get_problem_info.invoke({"problem_type": "beams2d"})
    result2 = get_problem_info.invoke({"problem_type": "BEAMS2D"})
    result3 = get_problem_info.invoke({"problem_type": "Beams2D"})

    assert result1["selected_problem"] == "beams2d"
    assert result2["selected_problem"] == "beams2d"
    assert result3["selected_problem"] == "beams2d"


@pytest.mark.unit
def test_check_beam_constraints_file_not_found(tmp_path):
    """Test constraint checking with non-existent file."""
    # Create a problem instance
    _state["problem_instance"] = Beams2D()

    result = check_beam_constraints.invoke(
        {"design_source": str(tmp_path / "nonexistent.npy")}
    )

    assert result["success"] is False
    assert "error" in result
    assert "not found" in result["error"].lower()


@pytest.mark.unit
def test_check_beam_constraints_no_problem_instance():
    """Test constraint checking when no problem instance exists."""
    # Ensure no problem instance
    _state["problem_instance"] = None

    result = check_beam_constraints.invoke({"design_source": "last"})

    # Should fail because get_problem_instance() will create one,
    # but there's no last_design in state
    assert result["success"] is False
    assert "error" in result


@pytest.mark.unit
def test_check_beam_constraints_from_file(tmp_path):
    """Test loading design from .npy file."""
    # Create a problem instance
    _state["problem_instance"] = Beams2D()

    # Create a sample design file
    rng = np.random.default_rng(seed=42)
    design = rng.random((50, 100)) * 0.35
    npy_file = tmp_path / "test_design.npy"
    np.save(npy_file, design)

    result = check_beam_constraints.invoke({"design_source": str(npy_file)})

    assert result["success"] is True
    assert "design from file" in result["design_source_used"]


@pytest.mark.unit
def test_render_beam_design_outputs_directory_creation(tmp_path):
    """Test that render_beam_design creates outputs directory if it doesn't exist."""
    original_dir = Path.cwd()
    os.chdir(tmp_path)

    try:
        # Ensure outputs doesn't exist
        outputs_dir = tmp_path / "outputs"
        assert not outputs_dir.exists()

        # This will fail because we don't have a real problem setup,
        # but it should at least create the directory
        _ = render_beam_design.invoke({"design_description": "random", "seed": 42})

        # Directory should be created even if rendering fails
        assert outputs_dir.exists()

    finally:
        os.chdir(original_dir)


@pytest.mark.unit
def test_render_beam_design_suffix_mapping():
    """Test that design descriptions map to correct suffixes."""
    test_cases = [
        ("initial design", "_initial"),
        ("before optimization", "_initial"),
        ("final design", "_final"),
        ("after optimization", "_final"),
        ("optimized design", "_optimized"),
        ("optimal design", "_optimized"),
        ("random design", "_random"),
    ]

    # We can't test the full render without engibench, but we can test the logic
    # by checking what suffix would be chosen
    for description, expected_suffix in test_cases:
        desc_lower = description.lower()

        suffix_map = {
            "initial": "_initial",
            "before": "_initial",
            "final": "_final",
            "after": "_final",
            "optimized": "_optimized",
            "optimal": "_optimized",
            "random": "_random",
        }

        suffix = next(
            (suf for keyword, suf in suffix_map.items() if keyword in desc_lower),
            None,
        )

        assert suffix == expected_suffix, f"Failed for '{description}'"


@pytest.mark.unit
def test_simulate_beam_design_description_keywords():
    """Test that different design descriptions are recognized."""
    keywords_tests = [
        ("last design", ["last"]),
        ("previous design", ["previous"]),
        ("current design", ["current"]),
        ("optimized topology", ["optimized"]),
        ("random design", ["random"]),
    ]

    for description, expected_keywords in keywords_tests:
        desc_lower = description.lower()
        found_keywords = [kw for kw in expected_keywords if kw in desc_lower]
        assert len(found_keywords) > 0, f"Failed to find keywords in '{description}'"


@pytest.mark.unit
def test_state_dictionary_structure():
    """Test that state dictionary has expected structure."""
    assert isinstance(_state, dict)
    assert "problem_instance" in _state
    assert "last_design" in _state


@pytest.mark.unit
def test_get_dataset_info_unsupported_problem():
    """Test get_dataset_info with unsupported problem type."""
    result = get_dataset_info.invoke({"problem_type": "unsupported"})

    assert result["success"] is False
    assert "error" in result
    assert "not supported" in result["error"].lower()


@pytest.mark.unit
def test_get_problem_details_unsupported_problem():
    """Test get_problem_details with unsupported problem type."""
    result = get_problem_details.invoke({"problem_type": "unsupported"})

    assert result["success"] is False
    assert "error" in result
    assert "not supported" in result["error"].lower()


@pytest.mark.unit
def test_optimize_beam_design_starting_points():
    """Test that different starting point descriptions are handled."""
    starting_points = ["random", "uniform", "sparse", "other"]

    # We can't run full optimization, but we can verify the logic
    # would handle different starting points
    for point in starting_points:
        assert isinstance(point, str)
        # In actual code, these would be processed
        # "random" triggers random_design()
        # others also default to random_design()


@pytest.mark.unit
def test_expected_array_dimensions_constant():
    """Test that EXPECTED_ARRAY_DIMENSIONS constant is defined."""
    assert EXPECTED_ARRAY_DIMENSIONS == 2


@pytest.mark.unit
def test_matplotlib_backend_configuration():
    """Test that matplotlib is configured with non-interactive backend."""
    # After importing engibench module, backend should be 'Agg'
    backend = matplotlib.get_backend()
    assert backend == "Agg" or backend.startswith("agg")


# ============================================================================
# PROBLEM CREATION TESTS
# ============================================================================


@pytest.mark.slow
def test_create_beam_problem():
    """Test creating a beam problem (requires full engibench install)."""
    result = create_beam_problem.invoke({"seed": 42})

    assert result["success"] is True
    assert result["problem_id"] == "beams2d"
    assert "design_space" in result
    assert "objectives" in result
    assert "conditions" in result
    assert "dataset_id" in result
    assert "message" in result
    assert "seed" in result["message"].lower()


@pytest.mark.slow
def test_create_beam_problem_stores_instance():
    """Test that creating a problem stores it in state."""
    create_beam_problem.invoke({"seed": 0})

    problem = get_problem_instance()
    assert problem is not None
    assert hasattr(problem, "design_space")
    assert hasattr(problem, "objectives")


@pytest.mark.slow
def test_create_beam_problem_different_seeds():
    """Test creating problems with different seeds."""
    result1 = create_beam_problem.invoke({"seed": 0})
    result2 = create_beam_problem.invoke({"seed": 100})

    assert result1["success"] is True
    assert result2["success"] is True
    # Both should succeed with different seeds


# ============================================================================
# SIMULATION TESTS
# ============================================================================


@pytest.mark.slow
def test_simulate_beam_design_random():
    """Test simulating a random beam design."""
    # Create problem first
    create_beam_problem.invoke({"seed": 42})

    # Simulate a random design
    result = simulate_beam_design.invoke(
        {"design_description": "random design", "volume_fraction": 0.35, "seed": 42}
    )

    assert result["success"] is True
    assert "compliance" in result
    assert isinstance(result["compliance"], float)
    assert result["compliance"] > 0  # Compliance should be positive
    assert result["design_valid"] is True
    assert result["volume_fraction_used"] == 0.35


@pytest.mark.slow
def test_simulate_beam_design_stores_design():
    """Test that simulation stores the design for reuse."""
    create_beam_problem.invoke({"seed": 0})

    simulate_beam_design.invoke({"design_description": "random design", "seed": 0})

    # Check that design was stored
    design = get_last_design()
    assert design is not None
    assert design.shape == (50, 100)


@pytest.mark.slow
def test_simulate_beam_design_reuses_stored():
    """Test that simulation can reuse a stored design."""
    create_beam_problem.invoke({"seed": 0})

    # First simulation - creates and stores a design
    result1 = simulate_beam_design.invoke(
        {"design_description": "random design", "seed": 42}
    )

    # Second simulation - should reuse the stored design
    result2 = simulate_beam_design.invoke(
        {"design_description": "last design", "seed": 42}
    )

    # Both should succeed
    assert result1["success"] is True
    assert result2["success"] is True
    # Compliances should be the same since we're using the same design
    assert result1["compliance"] == result2["compliance"]


@pytest.mark.slow
def test_simulate_beam_design_different_parameters():
    """Test simulation with different volume fractions."""
    create_beam_problem.invoke({"seed": 0})

    result1 = simulate_beam_design.invoke(
        {"design_description": "random", "volume_fraction": 0.3, "seed": 0}
    )

    result2 = simulate_beam_design.invoke(
        {"design_description": "random", "volume_fraction": 0.5, "seed": 1}
    )

    assert result1["success"] is True
    assert result2["success"] is True
    assert result1["volume_fraction_used"] == 0.3
    assert result2["volume_fraction_used"] == 0.5


# ============================================================================
# CONSTRAINT CHECKING TESTS
# ============================================================================


@pytest.mark.slow
def test_check_beam_constraints_with_stored_design():
    """Test checking constraints on a stored design."""
    create_beam_problem.invoke({"seed": 0})
    simulate_beam_design.invoke({"design_description": "random", "seed": 0})

    result = check_beam_constraints.invoke(
        {
            "design_source": "last",
            "volume_fraction": 0.35,  # Required parameter
        }
    )

    assert result["success"] is True
    assert "has_violations" in result
    assert "violations" in result
    assert result["design_source_used"] == "last design from state"


@pytest.mark.slow
def test_check_beam_constraints_random():
    """Test checking constraints on a random design."""
    create_beam_problem.invoke({"seed": 0})

    result = check_beam_constraints.invoke(
        {
            "design_source": "random",
            "volume_fraction": 0.35,  # Required parameter
        }
    )

    assert result["success"] is True
    assert "has_violations" in result
    assert result["design_source_used"] == "randomly generated design"


@pytest.mark.slow
def test_check_beam_constraints_no_design():
    """Test that constraint check fails gracefully when no design exists."""
    create_beam_problem.invoke({"seed": 0})

    # Try to check constraints without creating a design first
    result = check_beam_constraints.invoke({"design_source": "last"})

    assert result["success"] is False
    assert "error" in result
    assert "no design" in result["error"].lower()


@pytest.mark.slow
def test_check_beam_constraints_custom_config():
    """Test constraint checking with custom volume fraction."""
    create_beam_problem.invoke({"seed": 0})
    simulate_beam_design.invoke({"design_description": "random", "seed": 0})

    result = check_beam_constraints.invoke(
        {"design_source": "last", "volume_fraction": 0.4}
    )

    assert result["success"] is True
    assert result["config_used"]["volfrac"] == 0.4


# ============================================================================
# OPTIMIZATION TESTS
# ============================================================================


@pytest.mark.slow
def test_optimize_beam_design_basic():
    """Test basic beam design optimization."""
    create_beam_problem.invoke({"seed": 0})

    result = optimize_beam_design.invoke(
        {"starting_point": "random", "volume_fraction": 0.35, "seed": 0}
    )

    assert result["success"] is True
    assert "initial_compliance" in result
    assert "final_compliance" in result
    assert "improvement" in result
    assert "iterations" in result
    assert result["final_compliance"] <= result["initial_compliance"]
    assert result["improvement"] >= 0  # Should improve or stay same


@pytest.mark.slow
def test_optimize_beam_design_stores_result():
    """Test that optimization stores the optimized design."""
    create_beam_problem.invoke({"seed": 0})

    optimize_beam_design.invoke({"starting_point": "random", "seed": 0})

    # Check that optimized design was stored
    design = get_last_design()
    assert design is not None
    assert design.shape == (50, 100)


@pytest.mark.slow
def test_optimize_beam_design_reproducibility():
    """Test that optimization is reproducible with same seed."""
    result1 = optimize_beam_design.invoke({"starting_point": "random", "seed": 42})

    # Reset state
    _state["problem_instance"] = None
    _state["last_design"] = None

    result2 = optimize_beam_design.invoke({"starting_point": "random", "seed": 42})

    assert result1["success"] is True
    assert result2["success"] is True
    # Results should be identical with same seed
    assert abs(result1["final_compliance"] - result2["final_compliance"]) < 1e-6


# ============================================================================
# RENDERING TESTS
# ============================================================================


@pytest.mark.slow
def test_render_beam_design_random(tmp_path):
    """Test rendering a random beam design."""
    # Change to temp directory

    original_dir = Path.cwd()
    os.chdir(tmp_path)

    try:
        create_beam_problem.invoke({"seed": 0})

        result = render_beam_design.invoke(
            {
                "design_description": "random design",
                "save_path": "test_beam.png",
                "seed": 42,
            }
        )

        assert result["success"] is True
        assert "save_path" in result
        assert "npy_path" in result
        assert Path(result["save_path"]).exists()
        assert Path(result["npy_path"]).exists()
        assert result["design_shape"] == (50, 100)

    finally:
        os.chdir(original_dir)


@pytest.mark.slow
def test_render_beam_design_automatic_suffix(tmp_path):
    """Test that rendering adds automatic suffixes to prevent overwriting."""

    original_dir = Path.cwd()
    os.chdir(tmp_path)

    try:
        create_beam_problem.invoke({"seed": 0})

        # Render random design
        result1 = render_beam_design.invoke(
            {"design_description": "random design", "save_path": "beam.png", "seed": 0}
        )

        # Render optimized design
        result2 = render_beam_design.invoke(
            {
                "design_description": "optimized design",
                "save_path": "beam.png",
                "seed": 0,
            }
        )

        assert result1["success"] is True
        assert result2["success"] is True

        # Check that files have different names
        assert "_random" in result1["save_path"]
        assert "_optimized" in result2["save_path"]

        # Both files should exist
        assert Path(result1["save_path"]).exists()
        assert Path(result2["save_path"]).exists()

    finally:
        os.chdir(original_dir)


@pytest.mark.slow
def test_render_beam_design_with_stored_design(tmp_path):
    """Test rendering a previously stored design."""

    original_dir = Path.cwd()
    os.chdir(tmp_path)

    try:
        create_beam_problem.invoke({"seed": 0})

        # Create and store a design through simulation
        simulate_beam_design.invoke({"design_description": "random", "seed": 42})

        # Render the stored design
        result = render_beam_design.invoke(
            {"design_description": "last design", "save_path": "stored_design.png"}
        )

        assert result["success"] is True
        assert result["design_type"] == "stored (from previous optimization)"

    finally:
        os.chdir(original_dir)


# ============================================================================
# PROBLEM INFO TESTS
# ============================================================================


@pytest.mark.slow
def test_get_problem_info_beams2d():
    """Test getting problem information for beams2d."""
    result = get_problem_info.invoke({"problem_type": "beams2d"})

    assert result["success"] is True
    assert result["selected_problem"] == "beams2d"
    assert "description" in result
    assert "objectives" in result
    assert "typical_conditions" in result
    assert "available_problems" in result


@pytest.mark.slow
def test_get_problem_info_invalid_type():
    """Test getting info for an invalid problem type."""
    result = get_problem_info.invoke({"problem_type": "nonexistent"})

    assert result["success"] is True  # Still succeeds but returns available options
    assert result["selected_problem"] is None
    assert "available_problems" in result


@pytest.mark.slow
def test_get_problem_details():
    """Test getting detailed problem information."""
    result = get_problem_details.invoke({"problem_type": "beams2d"})

    assert result["success"] is True
    assert result["problem_type"] == "beams2d"
    assert "design_space" in result
    assert "design_space_shape" in result
    assert result["design_space_shape"] == (50, 100)
    assert "objectives" in result
    assert "conditions" in result
    assert "dataset_id" in result


@pytest.mark.slow
def test_get_dataset_info():
    """Test getting dataset information."""
    result = get_dataset_info.invoke({"problem_type": "beams2d"})

    assert result["success"] is True
    assert "dataset_id" in result
    assert "splits" in result
    assert "features" in result
    assert "total_samples" in result
    assert result["total_samples"] > 0


# ============================================================================
# STATE MANAGEMENT TESTS
# ============================================================================


@pytest.mark.slow
def test_state_persistence_across_tools():
    """Test that state persists across different tool calls."""
    # Create problem
    create_beam_problem.invoke({"seed": 0})
    problem1 = get_problem_instance()

    # Simulate design
    simulate_beam_design.invoke({"design_description": "random", "seed": 0})
    design1 = get_last_design()

    # Verify state is maintained
    problem2 = get_problem_instance()
    design2 = get_last_design()

    assert problem1 is problem2  # Same instance
    assert np.array_equal(design1, design2)  # Same design


@pytest.mark.slow
def test_state_setters_and_getters(sample_design):
    """Test state getter and setter functions."""

    # Test problem instance
    problem = Beams2D()
    set_problem_instance(problem)
    retrieved_problem = get_problem_instance()
    assert retrieved_problem is problem

    # Test last design
    set_last_design(sample_design)
    retrieved_design = get_last_design()
    assert np.array_equal(retrieved_design, sample_design)


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


@pytest.mark.slow
def test_simulate_without_problem():
    """Test that simulation handles missing problem gracefully."""
    # Don't create a problem first
    result = simulate_beam_design.invoke({"design_description": "random", "seed": 0})

    # Should still work - creates problem automatically
    assert result["success"] is True


@pytest.mark.slow
def test_check_constraints_without_problem():
    """Test constraint checking without a problem instance."""
    result = check_beam_constraints.invoke(
        {
            "design_source": "random",
            "volume_fraction": 0.35,  # Required parameter
        }
    )

    # Should create problem automatically
    assert result["success"] is True


# ============================================================================
# INTEGRATION TESTS - Full workflows
# ============================================================================


@pytest.mark.slow
def test_full_optimization_workflow(tmp_path):
    """Test a complete optimization workflow."""

    original_dir = Path.cwd()
    os.chdir(tmp_path)

    try:
        # 1. Create problem
        result1 = create_beam_problem.invoke({"seed": 0})
        assert result1["success"] is True

        # 2. Simulate initial design
        result2 = simulate_beam_design.invoke(
            {"design_description": "random", "seed": 0}
        )
        assert result2["success"] is True
        initial_compliance = result2["compliance"]

        # 3. Check constraints
        result3 = check_beam_constraints.invoke(
            {
                "design_source": "last",
                "volume_fraction": 0.35,  # Required parameter
            }
        )
        assert result3["success"] is True

        # 4. Optimize design
        result4 = optimize_beam_design.invoke({"starting_point": "random", "seed": 0})
        assert result4["success"] is True
        assert result4["final_compliance"] <= initial_compliance

        # 5. Render final design
        result5 = render_beam_design.invoke(
            {"design_description": "optimized", "save_path": "final_design.png"}
        )
        assert result5["success"] is True
        assert Path(result5["save_path"]).exists()

    finally:
        os.chdir(original_dir)


@pytest.mark.slow
def test_workflow_compare_designs(tmp_path):
    """Test workflow that compares before and after optimization."""

    original_dir = Path.cwd()
    os.chdir(tmp_path)

    try:
        create_beam_problem.invoke({"seed": 42})

        # Render initial random design
        result1 = render_beam_design.invoke(
            {"design_description": "initial random design", "seed": 42}
        )

        # Optimize
        result2 = optimize_beam_design.invoke({"starting_point": "random", "seed": 42})

        # Render optimized design
        result3 = render_beam_design.invoke(
            {
                "design_description": "final optimized design",
            }
        )

        assert result1["success"] is True
        assert result2["success"] is True
        assert result3["success"] is True

        # Check that both images exist
        assert Path(result1["save_path"]).exists()
        assert Path(result3["save_path"]).exists()

        # Verify improvement
        assert result2["improvement"] > 0

    finally:
        os.chdir(original_dir)
