"""
Tests for src/ui/file_processing.py.

process_uploaded_images is a pure function with no Streamlit dependency —
no mocking of st.* required.
"""

import base64
from unittest.mock import Mock

import pytest

from src.ui.file_processing import process_uploaded_images


def _make_file(
    content: bytes, mime_type: str = "image/png", name: str = "img.png"
) -> Mock:
    """Return a minimal file-like mock matching Streamlit's UploadedFile interface."""
    f = Mock()
    f.read.return_value = content
    f.type = mime_type
    f.name = name
    return f


# --- empty input ---


@pytest.mark.unit
def test_empty_list_returns_two_empty_lists():
    """No files → both return lists are empty."""
    display, agent = process_uploaded_images([])
    assert display == []
    assert agent == []


# --- single file ---


@pytest.mark.unit
def test_single_file_base64_encoding():
    """File bytes must be base64-encoded correctly in the display dict."""
    raw = b"\x89PNG\r\n\x1a\n"
    f = _make_file(raw, "image/png", "photo.png")

    display, _agent = process_uploaded_images([f])

    expected_b64 = base64.b64encode(raw).decode("utf-8")
    assert len(display) == 1
    assert display[0]["data"] == expected_b64
    assert display[0]["type"] == "image/png"
    assert display[0]["name"] == "photo.png"
    assert display[0]["bytes"] == raw


@pytest.mark.unit
def test_single_file_agent_data_uri():
    """Agent dict must use the data:<mime>;base64,<data> URI scheme."""
    raw = b"fakeimagedata"
    f = _make_file(raw, "image/jpeg", "shot.jpg")

    _, agent = process_uploaded_images([f])

    b64 = base64.b64encode(raw).decode("utf-8")
    assert agent[0]["type"] == "image_url"
    assert agent[0]["image_url"]["url"] == f"data:image/jpeg;base64,{b64}"


# --- attribute fallbacks ---


@pytest.mark.unit
def test_file_without_type_defaults_to_octet_stream():
    """File lacking .type attribute → 'application/octet-stream'."""
    f = Mock(spec=["read", "name"])  # no .type
    f.read.return_value = b"data"
    f.name = "blob.bin"

    display, _ = process_uploaded_images([f])
    assert display[0]["type"] == "application/octet-stream"


@pytest.mark.unit
def test_file_without_name_defaults_to_unknown():
    """File lacking .name attribute → 'unknown'."""
    f = Mock(spec=["read", "type"])  # no .name
    f.read.return_value = b"data"
    f.type = "image/gif"

    display, _ = process_uploaded_images([f])
    assert display[0]["name"] == "unknown"


# --- multiple files ---


@pytest.mark.unit
def test_multiple_files_correct_count():
    """Three files → three entries in each output list."""
    files = [
        _make_file(b"a", "image/png", "a.png"),
        _make_file(b"b", "image/jpeg", "b.jpg"),
        _make_file(b"c", "application/pdf", "c.pdf"),
    ]
    display, agent = process_uploaded_images(files)
    assert len(display) == 3
    assert len(agent) == 3


@pytest.mark.unit
def test_multiple_files_data_independent():
    """Each file's base64 payload must match its own bytes, not a neighbor's."""
    files = [
        _make_file(b"AAA", "image/png", "first.png"),
        _make_file(b"BBB", "image/png", "second.png"),
    ]
    display, _ = process_uploaded_images(files)

    assert display[0]["data"] == base64.b64encode(b"AAA").decode()
    assert display[1]["data"] == base64.b64encode(b"BBB").decode()
