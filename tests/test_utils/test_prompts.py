"""
Tests for src/utils/prompts.py utility functions.

Covers _is_eval_mode, strip_suggested_prompts, and _get_problem_examples_text.
All tests are pure-Python with no external dependencies.
"""

from unittest.mock import patch

import pytest

from src.utils.prompts import (
    _get_problem_examples_text,
    _is_eval_mode,
    strip_suggested_prompts,
)

# ============================================================================
# _is_eval_mode
# ============================================================================


@pytest.mark.unit
def test_is_eval_mode_true(monkeypatch):
    """EVAL_MODE=true → True."""
    monkeypatch.setenv("EVAL_MODE", "true")
    assert _is_eval_mode() is True


@pytest.mark.unit
def test_is_eval_mode_case_insensitive(monkeypatch):
    """EVAL_MODE=TRUE (uppercase) → True."""
    monkeypatch.setenv("EVAL_MODE", "TRUE")
    assert _is_eval_mode() is True


@pytest.mark.unit
def test_is_eval_mode_false_when_unset(monkeypatch):
    """Absent env var defaults to False."""
    monkeypatch.delenv("EVAL_MODE", raising=False)
    assert _is_eval_mode() is False


@pytest.mark.unit
def test_is_eval_mode_false_explicit(monkeypatch):
    """EVAL_MODE=false → False."""
    monkeypatch.setenv("EVAL_MODE", "false")
    assert _is_eval_mode() is False


# ============================================================================
# strip_suggested_prompts
# ============================================================================


@pytest.mark.unit
def test_strip_suggested_prompts_removes_section():
    """Section present → removed from the end of the prompt."""
    prompt = "Main content here.\n## Suggested Next Prompts\nSome suggestion\nAnother"
    result = strip_suggested_prompts(prompt)
    assert result == "Main content here."
    assert "## Suggested Next Prompts" not in result


@pytest.mark.unit
def test_strip_suggested_prompts_noop_without_section():
    """No section present → string returned unchanged."""
    prompt = "This is a regular prompt with no suggestions block."
    assert strip_suggested_prompts(prompt) == prompt


@pytest.mark.unit
def test_strip_suggested_prompts_empty_string():
    """Empty input → empty output."""
    assert strip_suggested_prompts("") == ""


@pytest.mark.unit
def test_strip_suggested_prompts_only_section():
    """Prompt consisting only of the section → empty string returned."""
    prompt = "## Suggested Next Prompts\n- Do something\n- Do another thing"
    result = strip_suggested_prompts(prompt)
    assert result == ""


# ============================================================================
# _get_problem_examples_text
# ============================================================================


@pytest.mark.unit
def test_problem_examples_text_two_problems():
    """Two problems → joined with ' or '."""
    with patch("src.utils.prompts.SUPPORTED_PROBLEMS", ["beams2d", "photonics2d"]):
        result = _get_problem_examples_text()
    assert result == "'beams2d' or 'photonics2d'"


@pytest.mark.unit
def test_problem_examples_text_three_or_more():
    """Three or more problems → Oxford comma list with 'or' before the last."""
    with patch(
        "src.utils.prompts.SUPPORTED_PROBLEMS",
        ["beams2d", "photonics2d", "thermoelastic2d"],
    ):
        result = _get_problem_examples_text()
    assert result == "'beams2d', 'photonics2d', or 'thermoelastic2d'"


@pytest.mark.unit
def test_problem_examples_text_one_problem():
    """Single problem → returned as-is (no 'or')."""
    with patch("src.utils.prompts.SUPPORTED_PROBLEMS", ["beams2d"]):
        result = _get_problem_examples_text()
    assert result == "'beams2d'"
