"""
Tests for src/ui/chat_management.py — generate_chat_title.

generate_chat_title uses LangChain's init_chat_model, so all tests mock that
instead of provider-specific HTTP calls.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.ui.chat_management import generate_chat_title


def _mock_llm(content: str | list) -> MagicMock:
    """Build a mock LLM whose .invoke() returns an AIMessage with the given content."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=content)
    return llm


# --- success cases ---


@pytest.mark.unit
def test_generate_title_success():
    """init_chat_model → llm.invoke() → title extracted from response content."""
    mock_llm = _mock_llm("Beam Optimization")
    with (
        patch("src.ui.chat_management.config") as mock_cfg,
        patch(
            "src.ui.chat_management.init_chat_model", return_value=mock_llm
        ) as mock_init,
    ):
        mock_cfg.llm_model = "openai:gpt-4o"

        result = generate_chat_title("Optimize a cantilever beam with volfrac 0.3")

    assert result == "Beam Optimization"
    mock_init.assert_called_once_with("openai:gpt-4o", temperature=0.3)


@pytest.mark.unit
def test_generate_title_google_genai():
    """google_genai provider works the same — no special handling needed."""
    mock_llm = _mock_llm("Topology Analysis")
    with (
        patch("src.ui.chat_management.config") as mock_cfg,
        patch(
            "src.ui.chat_management.init_chat_model", return_value=mock_llm
        ) as mock_init,
    ):
        mock_cfg.llm_model = "google_genai:gemini-3-flash-preview"

        result = generate_chat_title("Analyze topology optimization results")

    assert result == "Topology Analysis"
    mock_init.assert_called_once_with(
        "google_genai:gemini-3-flash-preview", temperature=0.3
    )


@pytest.mark.unit
def test_generate_title_thinking_model_content_blocks():
    """Thinking models return content as list of blocks — text is extracted."""
    content_blocks = [
        {"type": "text", "text": "Print Setup", "extras": {"signature": "abc123"}},
    ]
    mock_llm = _mock_llm(content_blocks)
    with (
        patch("src.ui.chat_management.config") as mock_cfg,
        patch("src.ui.chat_management.init_chat_model", return_value=mock_llm),
    ):
        mock_cfg.llm_model = "google_genai:gemini-3-flash-preview"

        result = generate_chat_title("Set up the 3D printer")

    assert result == "Print Setup"


@pytest.mark.unit
def test_generate_title_strips_quotes():
    """Surrounding quotes in the LLM response are stripped."""
    mock_llm = _mock_llm('"Beam Design"')
    with (
        patch("src.ui.chat_management.config") as mock_cfg,
        patch("src.ui.chat_management.init_chat_model", return_value=mock_llm),
    ):
        mock_cfg.llm_model = "openai:gpt-4o"

        result = generate_chat_title("Design a beam")

    assert result == "Beam Design"


@pytest.mark.unit
def test_generate_title_truncates_long_title():
    """LLM returns a title longer than 50 chars → truncated to 47 + '...'."""
    mock_llm = _mock_llm("A" * 60)
    with (
        patch("src.ui.chat_management.config") as mock_cfg,
        patch("src.ui.chat_management.init_chat_model", return_value=mock_llm),
    ):
        mock_cfg.llm_model = "openai:gpt-4o"

        result = generate_chat_title("some request")

    assert len(result) == 50
    assert result.endswith("...")


# --- fallback on error ---


@pytest.mark.unit
def test_generate_title_falls_back_on_exception():
    """LLM error → function falls back to the user message (no raise)."""
    with (
        patch("src.ui.chat_management.config") as mock_cfg,
        patch(
            "src.ui.chat_management.init_chat_model",
            side_effect=ConnectionError("unreachable"),
        ),
    ):
        mock_cfg.llm_model = "openai:gpt-4o"

        result = generate_chat_title("Short message")

    assert result == "Short message"


@pytest.mark.unit
def test_generate_title_fallback_truncates_long_message():
    """Fallback path: message longer than 50 chars → truncated with '...'."""
    long_msg = "x" * 80
    with (
        patch("src.ui.chat_management.config") as mock_cfg,
        patch(
            "src.ui.chat_management.init_chat_model",
            side_effect=RuntimeError("boom"),
        ),
    ):
        mock_cfg.llm_model = "openai:gpt-4o"

        result = generate_chat_title(long_msg)

    assert len(result) == 50  # 47 chars + "..."
    assert result.endswith("...")


@pytest.mark.unit
def test_generate_title_fallback_short_message_unchanged():
    """Fallback path: short message → returned as-is, no truncation."""
    with (
        patch("src.ui.chat_management.config") as mock_cfg,
        patch(
            "src.ui.chat_management.init_chat_model",
            side_effect=RuntimeError("boom"),
        ),
    ):
        mock_cfg.llm_model = "openai:gpt-4o"

        result = generate_chat_title("Help me")

    assert result == "Help me"
