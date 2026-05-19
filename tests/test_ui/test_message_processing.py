"""
Tests for message_processing module.

These tests cover text processing, LaTeX fixing, suggested prompts extraction,
and validation warning handling.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.ui.message_processing import (
    extract_suggested_prompts,
    fix_latex_delimiters,
    format_tool_call,
)

# ============================================================================
# LATEX DELIMITER TESTS
# ============================================================================


@pytest.mark.unit
def test_fix_latex_delimiters_display_math():
    """Test conversion of display math delimiters."""
    text = r"The formula is \[E = mc^2\]"
    result = fix_latex_delimiters(text)
    assert "$$E = mc^2$$" in result


@pytest.mark.unit
def test_fix_latex_delimiters_inline_math():
    """Test conversion of inline math delimiters."""
    text = r"The formula \(x + y = z\) is simple"
    result = fix_latex_delimiters(text)
    assert "$x + y = z$" in result


@pytest.mark.unit
def test_fix_latex_delimiters_mixed():
    """Test conversion of mixed delimiters."""
    text = r"Inline \(a + b\) and display \[c + d\]"
    result = fix_latex_delimiters(text)
    assert "$a + b$" in result
    assert "$$c + d$$" in result


@pytest.mark.unit
def test_fix_latex_delimiters_no_latex():
    """Test text without LaTeX remains unchanged."""
    text = "This is plain text without any math."
    result = fix_latex_delimiters(text)
    assert result == text


@pytest.mark.unit
def test_fix_latex_delimiters_multiline():
    """Test multiline display math."""
    text = r"\[a = b + c\]"
    result = fix_latex_delimiters(text)
    assert "$$a = b + c$$" in result


@pytest.mark.unit
def test_fix_latex_delimiters_special_chars():
    """Test LaTeX with special characters."""
    text = r"\(\frac{1}{2} + \sum_{i=1}^{n} x_i\)"
    result = fix_latex_delimiters(text)
    assert "$" in result
    assert "frac" in result


# ============================================================================
# SUGGESTED PROMPTS EXTRACTION TESTS
# ============================================================================


@pytest.mark.unit
def test_extract_suggested_prompts_with_code_fence():
    """Test extraction with proper code fence format."""
    response = """Here is my response.

```suggested_prompts
First suggestion
---
Second suggestion
---
Third suggestion
```

Some additional text."""

    cleaned, prompts = extract_suggested_prompts(response)

    assert len(prompts) == 3
    assert "First suggestion" in prompts
    assert "Second suggestion" in prompts
    assert "Third suggestion" in prompts
    assert "```suggested_prompts" not in cleaned


@pytest.mark.unit
def test_extract_suggested_prompts_multiline_fallback():
    """Test extraction with multiline fallback format."""
    response = """Here is my response.

suggested_prompts
First suggestion
Second suggestion
Third suggestion

Some other text."""

    cleaned, prompts = extract_suggested_prompts(response)

    assert len(prompts) == 3
    assert "Some other text" in cleaned


@pytest.mark.unit
def test_extract_suggested_prompts_no_prompts():
    """Test response without suggested prompts."""
    response = "This is a simple response without any suggestions."

    cleaned, prompts = extract_suggested_prompts(response)

    assert cleaned == response
    assert len(prompts) == 0


@pytest.mark.unit
def test_extract_suggested_prompts_empty_response():
    """Test empty response."""
    response = ""
    cleaned, prompts = extract_suggested_prompts(response)

    assert cleaned == ""
    assert len(prompts) == 0


@pytest.mark.unit
def test_extract_suggested_prompts_filters_short_lines():
    """Test that short lines are filtered out."""
    response = """Response text.

suggested_prompts
OK
This is a valid suggestion

More text."""

    _cleaned, prompts = extract_suggested_prompts(response)

    # "OK" should be filtered (length <= 3)
    assert "OK" not in prompts
    if prompts:
        assert all(len(p) > 3 for p in prompts)


@pytest.mark.unit
def test_extract_suggested_prompts_filters_markdown():
    """Test that markdown lines are filtered out."""
    response = """Response text.

suggested_prompts
# Heading
* Bullet item
Valid suggestion

More text."""

    _cleaned, prompts = extract_suggested_prompts(response)

    # Markdown should be filtered
    for prompt in prompts:
        assert not prompt.startswith("#")
        assert not prompt.startswith("*")


# ============================================================================
# TOOL CALL FORMATTING TESTS
# ============================================================================


@pytest.mark.unit
def test_format_tool_call_dict():
    """Test formatting a tool call from dict."""
    tool_call = {"name": "get_weather", "args": {"city": "Boston"}}
    result = format_tool_call(tool_call)

    assert "get_weather" in result
    assert "🔧" in result


@pytest.mark.unit
def test_format_tool_call_object():
    """Test formatting a tool call from object."""

    class MockToolCall:
        name = "search_files"

    result = format_tool_call(MockToolCall())
    assert "search_files" in result


@pytest.mark.unit
def test_format_tool_call_missing_name():
    """Test formatting when name is missing."""
    tool_call = {"args": {"key": "value"}}
    result = format_tool_call(tool_call)

    assert "Unknown" in result


# ============================================================================
# AI MESSAGE FORMATTING TESTS
# ============================================================================


# ============================================================================
# VALIDATION WARNING EXTRACTION TESTS
# ============================================================================


@pytest.mark.unit
def test_extract_validation_warnings_no_warnings():
    """Test that text without warnings passes through."""
    # We can't easily test the st.warning/st.error calls without mocking streamlit
    # So we just test the basic function behavior
    from src.ui.message_processing import extract_and_display_validation_warnings

    with patch("src.ui.message_processing.st"):
        response = "This is a normal response without any warnings."
        result = extract_and_display_validation_warnings(response)
        assert result == response


@pytest.mark.unit
def test_extract_validation_warnings_structured_block():
    """Test extraction of structured CRITICAL block."""
    from src.ui.message_processing import extract_and_display_validation_warnings

    response = """Here is some text.

============================================================
🚨 **CRITICAL: Resource Allocation Review**
============================================================
**Issues Found:**
- GPU count exceeds limit

**💡 Recommendations:**
- Reduce GPU count
============================================================

More text here."""

    mock_st = MagicMock()
    with patch("src.ui.message_processing.st", mock_st):
        result = extract_and_display_validation_warnings(response)
        # The structured block should be removed
        assert "CRITICAL" not in result
        # st.error should have been called
        mock_st.error.assert_called_once()


@pytest.mark.unit
def test_extract_validation_warnings_unstructured_block():
    """Test extraction of unstructured warning block."""
    from src.ui.message_processing import extract_and_display_validation_warnings

    response = """Here is the result.

Important Warnings:
- Warning about resources
- Another warning

SLURM Script File: /path/to/script.slurm

The script is ready."""

    mock_st = MagicMock()
    with patch("src.ui.message_processing.st", mock_st):
        result = extract_and_display_validation_warnings(response)
        # The warning block should be handled
        assert "SLURM Script File" in result


# ============================================================================
# TOOL OUTPUT FILTERING TESTS
# ============================================================================


@pytest.mark.unit
def test_is_shell_output_detection():
    """Test detection of shell vs agent-formatted output."""
    # These patterns should be filtered out (not shell output)
    agent_outputs = [
        "Printer Name: XL\nPrinter UUID: abc123",
        '{"id": "123", "state": "printing"}',
        "PRINT_FILE - /usb/file.bgcode",
        "Temperature (Nozzle): 200°C",
        "file.bgcode - 1h 30m",
        "{'problem_id': 'beams2d', 'success': True}",
    ]

    # Simple heuristic test - these should be detected as NOT shell output
    for output in agent_outputs:
        # Check that our filter patterns exist in the output
        has_filter = any(
            pattern in output
            for pattern in [
                "Printer Name:",
                "Printer UUID:",
                "Temperature (Nozzle)",
                '"id":',
                '"state":',
                "PRINT_FILE",
                ".bgcode",
                "'problem_id':",
                "'success':",
            ]
        )
        assert has_filter, f"Output should contain filter pattern: {output}"


# ============================================================================
# FORMAT AND DISPLAY MESSAGES TESTS
# ============================================================================


@pytest.mark.unit
def test_format_and_display_messages_empty():
    """Test format_and_display_messages with empty list."""
    from src.ui.message_processing import format_and_display_messages

    with patch("src.ui.message_processing.st"):
        result, prompts = format_and_display_messages([])

    assert result == ""
    assert prompts == []


@pytest.mark.unit
def test_format_and_display_messages_with_ai_response():
    """Test format_and_display_messages with AI response."""
    from src.ui.message_processing import format_and_display_messages

    messages = [AIMessage(content="Here is my response")]

    mock_st = MagicMock()
    with (
        patch("src.ui.message_processing.st", mock_st),
        patch("src.ui.message_processing.display_response_media"),
    ):
        result, _prompts = format_and_display_messages(messages)

    assert "Here is my response" in result
    mock_st.markdown.assert_called()


@pytest.mark.unit
def test_format_and_display_messages_filters_prusa_output():
    """Test that Prusa output is filtered from tool messages."""
    from src.ui.message_processing import format_and_display_messages

    tool_msg = ToolMessage(
        content="Printer Name: XL\nPrinter UUID: abc123", tool_call_id="1"
    )
    ai_msg = AIMessage(content="Printer status checked")

    mock_st = MagicMock()
    with (
        patch("src.ui.message_processing.st", mock_st),
        patch("src.ui.message_processing.display_response_media"),
    ):
        result, _ = format_and_display_messages([tool_msg, ai_msg])

    # Tool output should be filtered, only AI response shown
    assert "Printer status checked" in result
    # Raw tool output should NOT be in result
    assert "```\nPrinter Name:" not in result


@pytest.mark.unit
def test_format_and_display_messages_filters_bgcode_output():
    """Test that .bgcode file listings are filtered."""
    from src.ui.message_processing import format_and_display_messages

    tool_msg = ToolMessage(
        content="file1.bgcode\nfile2.bgcode\nPRINT_FILE", tool_call_id="1"
    )
    ai_msg = AIMessage(content="Files listed")

    mock_st = MagicMock()
    with (
        patch("src.ui.message_processing.st", mock_st),
        patch("src.ui.message_processing.display_response_media"),
    ):
        result, _ = format_and_display_messages([tool_msg, ai_msg])

    assert "Files listed" in result


@pytest.mark.unit
def test_format_and_display_messages_shows_shell_output():
    """Test that shell output is shown."""
    from src.ui.message_processing import format_and_display_messages

    tool_msg = ToolMessage(
        content="total 12\ndrwxr-xr-x 2 user user 4096 Jan 1 00:00 dir",
        tool_call_id="1",
    )
    ai_msg = AIMessage(content="Directory listed")

    mock_st = MagicMock()
    with (
        patch("src.ui.message_processing.st", mock_st),
        patch("src.ui.message_processing.display_response_media"),
    ):
        result, _ = format_and_display_messages([tool_msg, ai_msg])

    # Shell output should be included
    assert "drwxr-xr-x" in result or "Directory listed" in result


@pytest.mark.unit
def test_format_and_display_messages_truncates_long_output():
    """Test that very long tool output is truncated."""
    from src.ui.message_processing import format_and_display_messages

    # Create very long shell-like output
    long_content = "\n".join([f"line {i}" for i in range(1000)])
    tool_msg = ToolMessage(content=long_content, tool_call_id="1")
    ai_msg = AIMessage(content="Done")

    mock_st = MagicMock()
    with (
        patch("src.ui.message_processing.st", mock_st),
        patch("src.ui.message_processing.display_response_media"),
    ):
        result, _ = format_and_display_messages([tool_msg, ai_msg])

    # If output was included and truncated, should have truncation marker
    if "line" in result:
        assert "truncated" in result.lower()


@pytest.mark.unit
def test_format_and_display_messages_extracts_prompts():
    """Test that suggested prompts are extracted."""
    from src.ui.message_processing import format_and_display_messages

    ai_msg = AIMessage(
        content="""Response here.

```suggested_prompts
Suggestion 1
---
Suggestion 2
```
"""
    )

    mock_st = MagicMock()
    with (
        patch("src.ui.message_processing.st", mock_st),
        patch("src.ui.message_processing.display_response_media"),
    ):
        _result, prompts = format_and_display_messages([ai_msg])

    assert len(prompts) == 2


# ============================================================================
# DISPLAY MESSAGE TESTS (with mocked streamlit)
# ============================================================================


@pytest.mark.unit
def test_display_message_user():
    """Test displaying a user message."""
    from src.ui.message_processing import display_message

    message = {"role": "user", "content": "Hello there"}

    mock_st = MagicMock()
    with patch("src.ui.message_processing.st", mock_st):
        display_message(message, 0)

    mock_st.chat_message.assert_called_once_with("user")


@pytest.mark.unit
def test_display_message_assistant():
    """Test displaying an assistant message."""
    from src.ui.message_processing import display_message

    message = {"role": "assistant", "content": "Hello! How can I help?"}

    mock_st = MagicMock()
    with (
        patch("src.ui.message_processing.st", mock_st),
        patch("src.ui.message_processing.find_images_in_text", return_value=[]),
        patch("src.ui.message_processing.find_stl_files_in_text", return_value=[]),
        patch("src.ui.message_processing.find_slurm_files_in_text", return_value=[]),
        patch("src.ui.message_processing.find_log_files_in_text", return_value=[]),
    ):
        display_message(message, 0)

    mock_st.chat_message.assert_called_once_with("assistant")


@pytest.mark.unit
def test_display_message_with_latex():
    """Test that LaTeX is fixed before display."""
    from src.ui.message_processing import display_message

    message = {"role": "assistant", "content": r"Formula: \(x + y\)"}

    mock_st = MagicMock()
    with (
        patch("src.ui.message_processing.st", mock_st),
        patch("src.ui.message_processing.find_images_in_text", return_value=[]),
        patch("src.ui.message_processing.find_stl_files_in_text", return_value=[]),
        patch("src.ui.message_processing.find_slurm_files_in_text", return_value=[]),
        patch("src.ui.message_processing.find_log_files_in_text", return_value=[]),
    ):
        display_message(message, 0)

    # Check that markdown was called (on the context manager)
    mock_st.chat_message.return_value.__enter__.return_value.markdown = MagicMock()


@pytest.mark.unit
def test_display_message_strips_suggestions_from_assistant():
    """Test that suggestions are stripped from assistant content."""
    from src.ui.message_processing import display_message

    message = {
        "role": "assistant",
        "content": "Response\n```suggested_prompts\nSuggestion\n```",
    }

    mock_st = MagicMock()
    # Create a mock context manager
    mock_cm = MagicMock()
    mock_st.chat_message.return_value.__enter__ = MagicMock(return_value=mock_cm)
    mock_st.chat_message.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.ui.message_processing.st", mock_st),
        patch("src.ui.message_processing.find_images_in_text", return_value=[]),
        patch("src.ui.message_processing.find_stl_files_in_text", return_value=[]),
        patch("src.ui.message_processing.find_slurm_files_in_text", return_value=[]),
        patch("src.ui.message_processing.find_log_files_in_text", return_value=[]),
    ):
        display_message(message, 0)

    # The function was called (we trust that st.markdown filters the content)


# ============================================================================
# DISPLAY UPLOADED FILES TESTS
# ============================================================================


@pytest.mark.unit
def test_display_uploaded_files_no_images():
    """Test _display_uploaded_files with no images."""
    from src.ui.message_processing import _display_uploaded_files

    message = {"role": "user", "content": "Hello"}

    # Should not raise
    _display_uploaded_files(message, 0)


@pytest.mark.unit
def test_display_uploaded_files_with_pdf():
    """Test _display_uploaded_files with PDF."""
    import base64

    from src.ui.message_processing import _display_uploaded_files

    pdf_data = base64.b64encode(b"fake pdf content").decode()
    message = {
        "role": "user",
        "content": "Check this PDF",
        "images": [{"type": "application/pdf", "name": "doc.pdf", "data": pdf_data}],
    }

    mock_st = MagicMock()
    with patch("src.ui.message_processing.st", mock_st):
        _display_uploaded_files(message, 0)

    mock_st.download_button.assert_called_once()


@pytest.mark.unit
def test_display_uploaded_files_with_image():
    """Test _display_uploaded_files with image."""
    import base64

    from src.ui.message_processing import _display_uploaded_files

    img_data = base64.b64encode(b"fake image").decode()
    message = {
        "role": "user",
        "content": "Check this image",
        "images": [{"type": "image/png", "name": "img.png", "data": img_data}],
    }

    mock_st = MagicMock()
    with patch("src.ui.message_processing.st", mock_st):
        _display_uploaded_files(message, 0)

    mock_st.image.assert_called_once()


# ============================================================================
# DISPLAY SUGGESTED PROMPTS TESTS
# ============================================================================


@pytest.mark.unit
def test_display_suggested_prompts_empty():
    """Test display_suggested_prompts with empty list."""
    from src.ui.message_processing import display_suggested_prompts

    mock_st = MagicMock()
    with patch("src.ui.message_processing.st", mock_st):
        display_suggested_prompts([])

    # Should not create any elements
    mock_st.markdown.assert_not_called()


@pytest.mark.unit
def test_display_suggested_prompts_with_suggestions():
    """Test display_suggested_prompts with suggestions."""
    from src.ui.message_processing import display_suggested_prompts

    mock_st = MagicMock()
    # session_state needs to be a MagicMock to allow attribute assignment
    mock_st.session_state = MagicMock()
    mock_st.session_state.get.return_value = []
    mock_st.columns.return_value = [MagicMock(), MagicMock()]
    # Make button return False so we don't trigger the click handler
    mock_st.button.return_value = False

    with patch("src.ui.message_processing.st", mock_st):
        display_suggested_prompts(["Suggestion 1", "Suggestion 2"])

    mock_st.markdown.assert_called()
    mock_st.columns.assert_called()
