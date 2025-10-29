"""
Tests for EngiBench tools.

These tests require engibench and its dependencies to be fully installed.
Mark as slow since they require heavy dependencies and test actual optimization.

NOTE: These tests are SKIPPED in CI (runs with -m "not slow").
They only run when you explicitly run pytest locally without the marker filter.
"""

import pytest

# Skip these tests if engibench is not available
pytest.importorskip("scipy")
pytest.importorskip("cvxopt")
pytest.importorskip("engibench")

from src.tools.engibench import create_beam_problem


@pytest.mark.slow
def test_create_beam_problem():
    """Test creating a beam problem (requires full engibench install)."""
    result = create_beam_problem.invoke({"seed": 42})

    assert result["success"] is True
    assert "problem_id" in result or "message" in result
