"""
Tests for unified EngiBench tools.

These tests focus on the unified tool API (create_problem, simulate_design, etc.)
and utility functions that don't depend on legacy tool implementations.

NOTE: Tests marked as @pytest.mark.slow are SKIPPED in CI (runs with -m "not slow").
They only run when you explicitly run pytest locally without the marker filter.
"""

import matplotlib
import pytest

# Skip these tests if engibench is not available
pytest.importorskip("scipy")
pytest.importorskip("cvxopt")
pytest.importorskip("engibench")

from src.tools.engibench import (
    EXPECTED_ARRAY_DIMENSIONS,
    _problem_states,
    get_dataset_info,
    get_problem_details,
    get_problem_info,
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
