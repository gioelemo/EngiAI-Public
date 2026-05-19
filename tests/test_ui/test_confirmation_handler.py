"""
Tests for src/ui/confirmation_handler.py — format_tool_calls_for_display.

format_tool_calls_for_display is a pure function with no Streamlit dependency.
"""

import pytest

from src.ui.confirmation_handler import format_tool_calls_for_display

# --- empty / all-safe input ---


@pytest.mark.unit
def test_empty_list_returns_empty_string():
    """No tool calls → empty string."""
    assert format_tool_calls_for_display([]) == ""


@pytest.mark.unit
def test_only_safe_tool_skipped():
    """open_gui_application is a safe tool and must not appear in output."""
    tool_calls = [{"name": "open_gui_application", "args": {"app": "terminal"}}]
    result = format_tool_calls_for_display(tool_calls)
    assert result == ""


@pytest.mark.unit
def test_multiple_safe_tools_all_skipped():
    """Multiple safe-tool entries → empty output, not an empty-line artifact."""
    tool_calls = [
        {"name": "open_gui_application", "args": {}},
        {"name": "open_gui_application", "args": {}},
    ]
    assert format_tool_calls_for_display(tool_calls) == ""


# --- execute_cli_command ---


@pytest.mark.unit
def test_execute_cli_command_with_all_fields():
    """execute_cli_command → command block, working dir, and timeout all present."""
    tool_calls = [
        {
            "name": "execute_cli_command",
            "args": {
                "command": "ls -la",
                "working_dir": "/home/user",
                "timeout": 60,
            },
        }
    ]
    result = format_tool_calls_for_display(tool_calls)

    assert "ls -la" in result
    assert "/home/user" in result
    assert "60s" in result
    assert "```bash" in result


@pytest.mark.unit
def test_execute_cli_command_without_working_dir():
    """execute_cli_command without working_dir → no working directory line."""
    tool_calls = [
        {
            "name": "execute_cli_command",
            "args": {"command": "pwd"},
        }
    ]
    result = format_tool_calls_for_display(tool_calls)

    assert "pwd" in result
    assert "Working directory" not in result


@pytest.mark.unit
def test_execute_cli_command_default_timeout():
    """execute_cli_command without explicit timeout → default 300s shown."""
    tool_calls = [
        {
            "name": "execute_cli_command",
            "args": {"command": "echo hi"},
        }
    ]
    result = format_tool_calls_for_display(tool_calls)
    assert "300s" in result


# --- unknown tool ---


@pytest.mark.unit
def test_unknown_tool_shows_name_and_args():
    """Unrecognised tool name → tool name and args both appear in output."""
    tool_calls = [
        {
            "name": "run_simulation",
            "args": {"volfrac": 0.35, "rmin": 3.0},
        }
    ]
    result = format_tool_calls_for_display(tool_calls)

    assert "run_simulation" in result
    assert "volfrac" in result


@pytest.mark.unit
def test_unknown_tool_without_args():
    """Unrecognised tool with empty args dict → name shown, no args line."""
    tool_calls = [{"name": "my_tool", "args": {}}]
    result = format_tool_calls_for_display(tool_calls)

    assert "my_tool" in result
    # Empty args dict is falsy — no args line should be added
    assert "Arguments" not in result


# --- mixed tool calls ---


@pytest.mark.unit
def test_safe_tool_mixed_with_cli_command():
    """Safe tool + execute_cli_command → only the command appears in output."""
    tool_calls = [
        {"name": "open_gui_application", "args": {"app": "finder"}},
        {"name": "execute_cli_command", "args": {"command": "make build"}},
    ]
    result = format_tool_calls_for_display(tool_calls)

    assert "make build" in result
    assert "open_gui_application" not in result
