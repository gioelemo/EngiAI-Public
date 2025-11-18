"""
Tests for unified EngiBench tools.

These tests focus on the unified tool API (create_problem, simulate_design, etc.)
and utility functions that don't depend on legacy tool implementations.

NOTE: Tests marked as @pytest.mark.slow are SKIPPED in CI (runs with -m "not slow").
They only run when you explicitly run pytest locally without the marker filter.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pytest

# Skip these tests if engibench is not available
pytest.importorskip("scipy")
pytest.importorskip("cvxopt")
pytest.importorskip("engibench")

from src.tools.engibench import (
    EXPECTED_ARRAY_DIMENSIONS,
    _problem_states,
    create_problem,
    get_dataset_info,
    get_problem_details,
    get_problem_info,
    get_unified_last_design,
    optimize_design,
    render_design,
    simulate_design,
)

# Supported problems for these unified tests. Add more problem keys here to extend coverage.
PROBLEM_TYPES = ["beams2d", "thermoelastic2d"]


@pytest.mark.unit
@pytest.mark.parametrize("problem", PROBLEM_TYPES)
def test_create_problem_basic(problem):
    """Create a problem instance for each supported problem type."""
    result = create_problem.invoke({"problem_type": problem, "seed": 0})

    assert result["success"] is True
    assert result["problem_type"] == problem
    assert "design_space" in result


@pytest.mark.unit
@pytest.mark.parametrize("problem", PROBLEM_TYPES)
def test_simulate_design_random_and_reuse(problem):
    """Simulate a random design and then reuse the stored design."""
    # Ensure problem instance exists
    create_problem.invoke({"problem_type": problem, "seed": 0})

    # First call: random design (suppress numerical warnings from FEM)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        r1 = simulate_design.invoke(
            {"problem_type": problem, "design_description": "random design", "seed": 1}
        )
    assert r1["success"] is True

    # Last design should be stored
    last = get_unified_last_design(problem)
    assert last is not None

    # Second call: request last design
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        r2 = simulate_design.invoke(
            {"problem_type": problem, "design_description": "last design", "seed": 2}
        )
    assert r2["success"] is True


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

    # Reset unified engibench state
    _problem_states.clear()
    yield
    # Cleanup after test
    _problem_states.clear()


# ============================================================================
# UNIT TESTS - Fast tests without engibench dependency
# ============================================================================


@pytest.mark.unit
def test_state_getters_and_setters_basic():
    """Test basic state management with _problem_states."""
    assert isinstance(_problem_states, dict)
    assert len(_problem_states) == 0  # Should start empty


@pytest.mark.unit
def test_get_problem_info_structure():
    """Test that get_problem_info returns expected structure."""
    result1 = get_problem_info.invoke({"problem_type": "beams2d"})

    assert result1["success"] is True
    assert result1["selected_problem"] == "beams2d"
    assert "description" in result1
    assert "objectives" in result1
    assert "typical_conditions" in result1
    assert "available_problems" in result1

    # Test thermoelastic2d support
    result2 = get_problem_info.invoke({"problem_type": "thermoelastic2d"})

    assert result2["success"] is True
    assert result2["selected_problem"] == "thermoelastic2d"
    assert "description" in result2
    assert "objectives" in result2

    # Test default (no argument)
    result3 = get_problem_info.invoke({})
    assert result3["success"] is True
    assert result3["selected_problem"] == "beams2d"


@pytest.mark.unit
def test_get_problem_info_unknown_type():
    """Test get_problem_info with unknown problem type."""
    result = get_problem_info.invoke({"problem_type": "unknown_type"})

    assert result["success"] is True
    assert result["selected_problem"] is None
    assert "available_problems" in result
    assert len(result["available_problems"]) > 0


@pytest.mark.unit
def test_get_problem_info_case_insensitive():
    """Test that problem type matching is case-insensitive."""
    result = get_problem_info.invoke({"problem_type": "BEAMS2D"})

    assert result["success"] is True
    assert result["selected_problem"] == "beams2d"


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
    """Test getting detailed problem information for supported problem types."""
    for problem in PROBLEM_TYPES:
        result = get_problem_details.invoke({"problem_type": problem})

        assert result["success"] is True
        assert result["problem_type"] == problem
        assert "design_space" in result
        assert "design_space_shape" in result
        # Problem-specific shape expectations
        if problem == "beams2d":
            assert result["design_space_shape"] == (50, 100)
        elif problem == "thermoelastic2d":
            assert result["design_space_shape"] == (64, 64)

        assert "objectives" in result
        assert "conditions" in result
        assert "dataset_id" in result


@pytest.mark.slow
def test_get_dataset_info():
    """Test getting dataset information for supported problems."""
    for problem in PROBLEM_TYPES:
        result = get_dataset_info.invoke({"problem_type": problem})

        assert result["success"] is True
        assert "dataset_id" in result
        assert "splits" in result
        assert "features" in result
        # total_samples may vary; ensure key exists and is non-negative when present
        if "total_samples" in result:
            assert isinstance(result["total_samples"], int)
            assert result["total_samples"] >= 0


@pytest.mark.slow
@pytest.mark.parametrize("problem", PROBLEM_TYPES)
def test_render_design_writes_files(tmp_path, problem):
    """Test that render_design writes PNG and NPY files to the provided outputs dir."""
    # Create outputs path under tmp_path so we don't pollute repo
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    create_problem.invoke({"problem_type": problem, "seed": 0})

    save_path = str(out_dir / "test_design.png")
    res = render_design.invoke(
        {
            "problem_type": problem,
            "design_description": "random design",
            "save_path": save_path,
        }
    )
    assert res["success"] is True

    # Check files exist
    png_path = out_dir / Path(res["save_path"]).name
    npy_path = out_dir / Path(res["npy_path"]).name
    assert png_path.exists()
    assert npy_path.exists()


@pytest.mark.slow
@pytest.mark.parametrize("problem", PROBLEM_TYPES)
def test_optimize_design_smoke(problem):
    """Smoke test for optimize_design (kept slow)."""
    create_problem.invoke({"problem_type": problem, "seed": 0})

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        res = optimize_design.invoke(
            {
                "problem_type": problem,
                "starting_point": "random",
                "config": {},
                "seed": 0,
                "save_result": False,
            }
        )
    assert res["success"] is True
    assert "design_shape" in res
