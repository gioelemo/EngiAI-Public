"""
Tests for EngiBench tools.

Simple tests for beam creation, optimization, and rendering.
"""

import pytest

from src.tools.engibench import create_beam_problem, optimize_beam_design


@pytest.mark.unit
def test_create_beam_problem():
    """Test creating a beam problem."""
    result = create_beam_problem.invoke({"seed": 42})

    assert result["success"] is True
    assert "problem_id" in result or "message" in result


@pytest.mark.slow
def test_optimize_beam_design():
    """Test beam optimization (marked as slow)."""
    # First create a problem
    create_beam_problem.invoke({"seed": 0})

    # Then optimize
    result = optimize_beam_design.invoke(
        {
            "volume_fraction": 0.35,
            "seed": 0,
        }
    )

    assert result["success"] is True
    assert "final_compliance" in result
    assert "initial_compliance" in result
