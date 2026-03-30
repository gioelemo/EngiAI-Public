"""
Tests for src/ui/chat_management.py — generate_chat_title.

generate_chat_title makes HTTP calls via requests but has no Streamlit dependency.
All tests mock requests.post and config.
"""

from unittest.mock import MagicMock, patch

import pytest

# generate_chat_title is the only function we can unit-test without st.*
from src.ui.chat_management import generate_chat_title


def _openai_response(title: str) -> MagicMock:
    """Build a mock response matching the OpenAI chat completions shape."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": title}}]}
    return resp


def _google_response(title: str) -> MagicMock:
    """Build a mock response matching the Google generateContent shape."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": title}]}}]}
    return resp


# --- OpenAI provider ---


@pytest.mark.unit
def test_generate_title_openai_success():
    """OpenAI provider → title extracted from choices[0].message.content."""
    with (
        patch("src.ui.chat_management.config") as mock_cfg,
        patch(
            "requests.post", return_value=_openai_response("Beam Optimization")
        ) as mock_post,
    ):
        mock_cfg.llm_model = "openai:gpt-4o"
        mock_cfg.openai_api_key = "sk-test"

        result = generate_chat_title("Optimize a cantilever beam with volfrac 0.3")

    assert result == "Beam Optimization"
    mock_post.assert_called_once()


@pytest.mark.unit
def test_generate_title_openai_strips_quotes():
    """Surrounding quotes in the API response are stripped."""
    with (
        patch("src.ui.chat_management.config") as mock_cfg,
        patch("requests.post", return_value=_openai_response('"Beam Design"')),
    ):
        mock_cfg.llm_model = "openai:gpt-4o"
        mock_cfg.openai_api_key = "sk-test"

        result = generate_chat_title("Design a beam")

    assert result == "Beam Design"


@pytest.mark.unit
def test_generate_title_truncates_long_api_title():
    """API returns a title longer than 50 chars → truncated to 47 + '...'."""
    long_title = "A" * 60
    with (
        patch("src.ui.chat_management.config") as mock_cfg,
        patch("requests.post", return_value=_openai_response(long_title)),
    ):
        mock_cfg.llm_model = "openai:gpt-4o"
        mock_cfg.openai_api_key = "sk-test"

        result = generate_chat_title("some request")

    assert len(result) == 50
    assert result.endswith("...")


# --- Google provider ---


@pytest.mark.unit
def test_generate_title_google_success():
    """Google provider → title extracted from candidates[0].content.parts[0].text."""
    with (
        patch("src.ui.chat_management.config") as mock_cfg,
        patch("requests.post", return_value=_google_response("3D Print Setup")),
    ):
        mock_cfg.llm_model = "google:gemini-2.0-flash"
        mock_cfg.google_api_key = "AIza-test"

        result = generate_chat_title("Set up the printer for a new job")

    assert result == "3D Print Setup"


# --- fallback on error ---


@pytest.mark.unit
def test_generate_title_falls_back_on_request_exception():
    """Network error → function falls back to the user message (no raise)."""
    with (
        patch("src.ui.chat_management.config") as mock_cfg,
        patch("requests.post", side_effect=ConnectionError("unreachable")),
    ):
        mock_cfg.llm_model = "openai:gpt-4o"
        mock_cfg.openai_api_key = "sk-test"

        result = generate_chat_title("Short message")

    assert result == "Short message"


@pytest.mark.unit
def test_generate_title_fallback_truncates_long_message():
    """Fallback path: message longer than 50 chars → truncated with '...'."""
    long_msg = "x" * 80
    with (
        patch("src.ui.chat_management.config") as mock_cfg,
        patch("requests.post", side_effect=RuntimeError("boom")),
    ):
        mock_cfg.llm_model = "openai:gpt-4o"
        mock_cfg.openai_api_key = "sk-test"

        result = generate_chat_title(long_msg)

    assert len(result) == 53  # 50 chars + "..."
    assert result.endswith("...")


@pytest.mark.unit
def test_generate_title_fallback_short_message_unchanged():
    """Fallback path: short message → returned as-is, no truncation."""
    with (
        patch("src.ui.chat_management.config") as mock_cfg,
        patch("requests.post", side_effect=RuntimeError("boom")),
    ):
        mock_cfg.llm_model = "openai:gpt-4o"
        mock_cfg.openai_api_key = "sk-test"

        result = generate_chat_title("Help me")

    assert result == "Help me"
