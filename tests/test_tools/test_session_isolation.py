"""
Test session isolation in EngiBench tools.

These tests verify that state management is properly isolated between
different evaluation sessions to prevent data corruption in batch evaluations.
"""

import numpy as np
import pytest

# Skip these tests if engibench is not available
pytest.importorskip("scipy")
pytest.importorskip("cvxopt")
pytest.importorskip("engibench")

from src.tools.engibench import (
    clear_session_state,
    get_current_session_id,
    get_unified_last_design,
    optimize_design,
    set_session_id,
    set_unified_last_design,
)


@pytest.mark.unit
def test_default_session():
    """Test that default session is used when no session_id is set."""
    session_id = get_current_session_id()
    assert session_id == "default"


@pytest.mark.unit
def test_set_session_id():
    """Test that session ID can be set and retrieved."""
    set_session_id("test_session_123")
    assert get_current_session_id() == "test_session_123"

    # Clean up
    set_session_id("default")


@pytest.mark.unit
def test_session_isolation():
    """Test that different sessions have isolated state."""
    rng = np.random.default_rng(seed=42)

    # Create a design in session 1
    set_session_id("session_1")
    design_1 = rng.random((10, 10))
    set_unified_last_design("beams2d", design_1)

    # Create a different design in session 2
    set_session_id("session_2")
    design_2 = rng.random((10, 10))
    set_unified_last_design("beams2d", design_2)

    # Verify session 1 still has design_1
    set_session_id("session_1")
    retrieved_design_1 = get_unified_last_design("beams2d")
    assert retrieved_design_1 is not None
    assert np.array_equal(retrieved_design_1, design_1)

    # Verify session 2 still has design_2
    set_session_id("session_2")
    retrieved_design_2 = get_unified_last_design("beams2d")
    assert retrieved_design_2 is not None
    assert np.array_equal(retrieved_design_2, design_2)

    # Verify designs are different
    assert not np.array_equal(design_1, design_2)

    # Clean up
    clear_session_state("session_1")
    clear_session_state("session_2")


@pytest.mark.unit
def test_clear_session_state():
    """Test that session state can be cleared."""
    rng = np.random.default_rng(seed=42)
    set_session_id("test_session")
    design = rng.random((10, 10))
    set_unified_last_design("beams2d", design)

    # Verify design was stored
    assert get_unified_last_design("beams2d") is not None

    # Clear session state
    clear_session_state("test_session")

    # Verify state was cleared
    assert get_unified_last_design("beams2d") is None


@pytest.mark.slow
@pytest.mark.integration
def test_optimize_design_session_isolation():
    """Test that optimize_design respects session isolation."""
    # Run optimization in session 1
    set_session_id("opt_session_1")
    result_1 = optimize_design.invoke(
        {
            "problem_type": "beams2d",
            "constraints": {"volfrac": 0.3},
            "seed": 42,
            "save_result": False,
        }
    )
    assert result_1["success"] is True
    design_1 = get_unified_last_design("beams2d")

    # Run optimization in session 2 with different config
    set_session_id("opt_session_2")
    result_2 = optimize_design.invoke(
        {
            "problem_type": "beams2d",
            "constraints": {"volfrac": 0.5},
            "seed": 43,
            "save_result": False,
        }
    )
    assert result_2["success"] is True
    design_2 = get_unified_last_design("beams2d")

    # Verify session 1 design is unchanged
    set_session_id("opt_session_1")
    design_1_check = get_unified_last_design("beams2d")
    assert design_1_check is not None
    assert np.array_equal(design_1_check, design_1)

    # Verify designs are different (different configs should produce different results)
    assert not np.array_equal(design_1, design_2)

    # Clean up
    clear_session_state("opt_session_1")
    clear_session_state("opt_session_2")
