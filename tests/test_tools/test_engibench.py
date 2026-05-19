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
    _session_id_var,
    clear_session_state,
    create_problem,
    get_current_session_id,
    get_dataset_info,
    get_final_beta,
    get_initial_design,
    get_problem_class,
    get_problem_details,
    get_problem_state,
    get_unified_last_design,
    optimize_design,
    render_design,
    set_final_beta,
    set_initial_design,
    set_session_id,
    set_unified_last_design,
    simulate_design,
)
from src.tools.problems import SUPPORTED_PROBLEMS


@pytest.mark.unit
@pytest.mark.parametrize("problem", SUPPORTED_PROBLEMS)
def test_create_problem_basic(problem):
    """Create a problem instance for each supported problem type."""
    result = create_problem.invoke({"problem_type": problem, "seed": 0})

    assert result["success"] is True
    assert result["problem_type"] == problem
    assert "design_space" in result


@pytest.mark.unit
@pytest.mark.parametrize("problem", SUPPORTED_PROBLEMS)
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
# STATE MANAGEMENT TESTS
# ============================================================================


@pytest.fixture(autouse=True)
def _clean_engibench_state():
    """Reset problem states and session ID between tests."""
    _problem_states.clear()
    _session_id_var.set(None)
    yield
    _problem_states.clear()
    _session_id_var.set(None)


@pytest.mark.unit
def test_get_problem_class_valid():
    """Known problem type returns a class."""
    cls = get_problem_class("beams2d")
    assert isinstance(cls, type)


@pytest.mark.unit
def test_get_problem_class_case_insensitive():
    """Problem type lookup is case-insensitive."""
    cls = get_problem_class("Beams2D")
    assert isinstance(cls, type)


@pytest.mark.unit
def test_get_problem_class_invalid():
    """Unknown problem type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown problem type"):
        get_problem_class("nope")


@pytest.mark.unit
def test_session_id_set_get():
    """set_session_id / get_current_session_id roundtrip."""
    set_session_id("sess_42")
    assert get_current_session_id() == "sess_42"


@pytest.mark.unit
def test_session_id_default():
    """Unset session ID defaults to 'default'."""
    assert get_current_session_id() == "default"


@pytest.mark.unit
def test_get_problem_state_creates_new():
    """First call creates a fresh state dict with expected keys."""
    set_session_id("s1")
    state = get_problem_state("beams2d")
    assert state["last_design"] is None
    assert state["initial_design"] is None
    assert state["final_beta"] is None


@pytest.mark.unit
def test_get_problem_state_returns_existing():
    """Calling twice returns the same dict object."""
    set_session_id("s1")
    state1 = get_problem_state("beams2d")
    state2 = get_problem_state("beams2d")
    assert state1 is state2


@pytest.mark.unit
def test_unified_last_design_set_get():
    """set/get_unified_last_design roundtrip."""
    set_session_id("s1")
    get_problem_state("beams2d")  # initialize
    arr = np.ones((10, 10))
    set_unified_last_design("beams2d", arr)
    result = get_unified_last_design("beams2d")
    assert result is not None
    np.testing.assert_array_equal(result, arr)


@pytest.mark.unit
def test_initial_design_set_get():
    """set/get_initial_design roundtrip."""
    set_session_id("s1")
    get_problem_state("beams2d")
    arr = np.zeros((5, 5))
    set_initial_design("beams2d", arr)
    result = get_initial_design("beams2d")
    assert result is not None
    np.testing.assert_array_equal(result, arr)


@pytest.mark.unit
def test_final_beta_set_get():
    """set/get_final_beta roundtrip."""
    set_session_id("s1")
    get_problem_state("beams2d")
    set_final_beta("beams2d", 3.14)
    assert get_final_beta("beams2d") == 3.14


@pytest.mark.unit
def test_clear_session_state():
    """clear_session_state removes the session's data."""
    set_session_id("s1")
    get_problem_state("beams2d")
    assert "s1" in _problem_states
    clear_session_state("s1")
    assert "s1" not in _problem_states


@pytest.mark.unit
def test_clear_session_state_current():
    """clear_session_state() with no arg clears current session."""
    set_session_id("s1")
    get_problem_state("beams2d")
    clear_session_state()
    assert "s1" not in _problem_states


@pytest.mark.unit
def test_session_isolation():
    """Two sessions do not share state."""
    set_session_id("s1")
    get_problem_state("beams2d")
    set_unified_last_design("beams2d", np.ones((3, 3)))

    set_session_id("s2")
    assert get_unified_last_design("beams2d") is None


# ============================================================================
# PROBLEM INFO TESTS
# ============================================================================


@pytest.mark.slow
@pytest.mark.parametrize("problem", SUPPORTED_PROBLEMS)
def test_get_problem_details(problem):
    """Test getting detailed problem information for all supported problems."""
    result = get_problem_details.invoke({"problem_type": problem})

    assert result["success"] is True
    assert result["problem_type"] == problem
    assert "design_space" in result
    assert "design_space_shape" in result
    # Verify shape is a tuple with 2 dimensions (for 2D problems)
    assert isinstance(result["design_space_shape"], tuple)
    assert len(result["design_space_shape"]) == 2

    assert "objectives" in result
    assert "conditions" in result
    assert "dataset_id" in result


@pytest.mark.slow
@pytest.mark.parametrize("problem", SUPPORTED_PROBLEMS)
def test_get_dataset_info(problem):
    """Test getting dataset information for all supported problems."""
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
@pytest.mark.parametrize("problem", SUPPORTED_PROBLEMS)
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
@pytest.mark.parametrize("problem", SUPPORTED_PROBLEMS)
def test_optimize_design_smoke(problem):
    """Smoke test for optimize_design (kept slow)."""
    create_problem.invoke({"problem_type": problem, "seed": 0})

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        res = optimize_design.invoke(
            {
                "problem_type": problem,
                "starting_point": "random",
                "constraints": {},
                "seed": 0,
                "save_result": False,
            }
        )
    assert res["success"] is True
    assert "design_shape" in res
