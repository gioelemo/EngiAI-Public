"""
Tests for pure-logic functions in UI modules that don't require Streamlit runtime.

Covers: streamlit_app.py (extract_job_id, usage status, usage text)
        settings.py (_count_files, _get_file_age_hours, _sanitize_for_json, export data)
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# streamlit_app.py has a circular import chain (chat ↔ streamlit_app).
# Break it by pre-loading lightweight mock modules before the real import.
# ---------------------------------------------------------------------------
for _mod in ("src.ui.chat", "src.ui.home", "src.ui.wandb_report"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# settings.py imports cleanly — no circular issues
from src.ui.settings import (  # noqa: E402
    _count_files_in_directory,
    _get_file_age_hours,
    _get_json_export_data,
    _get_markdown_export_data,
    _sanitize_for_json,
)
from src.ui.streamlit_app import (  # noqa: E402
    _format_usage_text,
    _get_usage_status,
    extract_job_id_from_response,
)
from src.utils.api_usage import UNLIMITED_LIMIT_VALUE  # noqa: E402

# ============================================================================
# streamlit_app.py — extract_job_id_from_response
# ============================================================================


@pytest.mark.unit
def test_extract_job_id_standard():
    assert extract_job_id_from_response("Job ID: 12345678") == "12345678"


@pytest.mark.unit
def test_extract_job_id_lowercase():
    assert extract_job_id_from_response("job id: 99999999") == "99999999"


@pytest.mark.unit
def test_extract_job_id_standalone_number():
    assert extract_job_id_from_response("submitted 12345678 done") == "12345678"


@pytest.mark.unit
def test_extract_job_id_no_match():
    assert extract_job_id_from_response("no id here") is None


@pytest.mark.unit
def test_extract_job_id_short_number_ignored():
    """Numbers shorter than 7 digits should not match."""
    assert extract_job_id_from_response("job 123") is None


# ============================================================================
# streamlit_app.py — _get_usage_status
# ============================================================================


@pytest.mark.unit
def test_get_usage_status_critical():
    color, text = _get_usage_status(96.0)
    assert color == "🔴"
    assert text == "Critical"


@pytest.mark.unit
def test_get_usage_status_warning():
    color, text = _get_usage_status(85.0)
    assert color == "🟡"
    assert text == "High"


@pytest.mark.unit
def test_get_usage_status_good():
    color, text = _get_usage_status(50.0)
    assert color == "🟢"
    assert text == "Good"


# ============================================================================
# streamlit_app.py — _format_usage_text
# ============================================================================


@pytest.mark.unit
def test_format_usage_text_with_limit():
    assert _format_usage_text(100, 1000, 10.0) == "100/1,000 (10%)"


@pytest.mark.unit
def test_format_usage_text_unlimited():
    assert _format_usage_text(100, UNLIMITED_LIMIT_VALUE, 0) == "100 requests"


# ============================================================================
# settings.py — _count_files_in_directory
# ============================================================================


@pytest.mark.unit
def test_count_files_in_directory(tmp_path):
    (tmp_path / "a.txt").touch()
    (tmp_path / "b.txt").touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").touch()
    assert _count_files_in_directory(tmp_path) == 3


@pytest.mark.unit
def test_count_files_empty_dir(tmp_path):
    assert _count_files_in_directory(tmp_path) == 0


@pytest.mark.unit
def test_count_files_nonexistent():
    assert _count_files_in_directory(Path("/nonexistent/dir")) == 0


# ============================================================================
# settings.py — _get_file_age_hours
# ============================================================================


@pytest.mark.unit
def test_get_file_age_hours_exists(tmp_path):
    f = tmp_path / "test.txt"
    f.touch()
    age = _get_file_age_hours(f)
    assert isinstance(age, float)
    assert age >= 0.0


@pytest.mark.unit
def test_get_file_age_hours_missing():
    assert _get_file_age_hours(Path("/nonexistent/file.txt")) == 0.0


# ============================================================================
# settings.py — _sanitize_for_json
# ============================================================================


@pytest.mark.unit
def test_sanitize_for_json_removes_audio_bytes():
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hello"},
                {"audio": b"\x00\x01\x02", "type": "audio"},
            ],
        }
    ]
    result = _sanitize_for_json(msgs)
    assert len(result) == 1
    # Audio item with bytes should be filtered out
    assert all(
        not isinstance(item.get("audio"), bytes) for item in result[0]["content"]
    )


@pytest.mark.unit
def test_sanitize_for_json_keeps_text():
    msgs = [{"role": "user", "content": "hello"}]
    result = _sanitize_for_json(msgs)
    assert result == msgs


# ============================================================================
# settings.py — export data functions
# ============================================================================


@pytest.mark.unit
def test_get_json_export_no_messages():
    assert _get_json_export_data(False) == ""


@pytest.mark.unit
def test_get_json_export_with_messages():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey"},
    ]
    with patch("src.ui.settings.st") as mock_st:
        mock_st.session_state = MagicMock()
        mock_st.session_state.messages = messages
        result = _get_json_export_data(True)
    parsed = json.loads(result)
    assert len(parsed) == 2
    assert parsed[0]["role"] == "user"


@pytest.mark.unit
def test_get_markdown_export_no_messages():
    assert _get_markdown_export_data(False) == ""


@pytest.mark.unit
def test_get_markdown_export_with_messages():
    messages = [{"role": "user", "content": "hello"}]
    with patch("src.ui.settings.st") as mock_st:
        mock_st.session_state = MagicMock()
        mock_st.session_state.messages = messages
        result = _get_markdown_export_data(True)
    assert "# Chat Export" in result
    assert "## User" in result
    assert "hello" in result
