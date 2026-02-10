"""Tests for the ask_human_for_clarification tool."""

import json

import pytest

from src.tools.human_input import ask_human_for_clarification


@pytest.mark.unit
def test_returns_valid_json():
    """Tool output must be parseable JSON."""
    result = ask_human_for_clarification.invoke(
        {"clarification_request": "What volume fraction should I use?"}
    )
    parsed = json.loads(result)
    assert isinstance(parsed, dict)


@pytest.mark.unit
def test_required_keys_present():
    """Result must contain success, question, and message keys."""
    result = ask_human_for_clarification.invoke(
        {"clarification_request": "What filter radius should I use?"}
    )
    parsed = json.loads(result)
    assert "success" in parsed
    assert "question" in parsed
    assert "message" in parsed


@pytest.mark.unit
def test_success_is_true():
    """success field must be True."""
    result = ask_human_for_clarification.invoke(
        {"clarification_request": "Please specify the load direction."}
    )
    parsed = json.loads(result)
    assert parsed["success"] is True


@pytest.mark.unit
def test_question_matches_input():
    """question field must echo back the clarification_request argument."""
    question = "What volume fraction between 0.1 and 0.9 should I use?"
    result = ask_human_for_clarification.invoke({"clarification_request": question})
    parsed = json.loads(result)
    assert parsed["question"] == question


@pytest.mark.unit
def test_handles_special_characters():
    """Output must be valid JSON even when the question contains quotes and apostrophes."""
    question = "It's unclear: use \"full\" or 'half' load? Specify ratio (e.g. 0.5)."
    result = ask_human_for_clarification.invoke({"clarification_request": question})
    parsed = json.loads(result)
    assert parsed["question"] == question
    assert parsed["success"] is True
