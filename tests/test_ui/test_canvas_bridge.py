"""Tests for canvas bridge (process_canvas_export)."""

import base64

import pytest

from src.ui.canvas_bridge import (
    InvalidCanvasDataError,
    process_canvas_export,
)

# Minimal valid 1x1 red PNG (67 bytes)
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
_B64_DATA = "data:image/png;base64," + base64.b64encode(_TINY_PNG).decode()


@pytest.mark.unit
def test_process_canvas_export_valid():
    """Valid base64 PNG returns dict with text and files."""
    result = process_canvas_export(_B64_DATA)
    assert "text" in result
    assert "files" in result
    assert len(result["files"]) == 1
    assert result["text"] == "Here's my whiteboard drawing:"


@pytest.mark.unit
def test_process_canvas_export_custom_message():
    """Custom message overrides default text."""
    result = process_canvas_export(_B64_DATA, custom_message="Look at this")
    assert result["text"] == "Look at this"


@pytest.mark.unit
def test_process_canvas_export_mock_file_methods():
    """MockUploadedFile exposes read, seek, getvalue, name, type, size."""
    result = process_canvas_export(_B64_DATA)
    mock_file = result["files"][0]

    assert mock_file.name.endswith(".png")
    assert mock_file.type == "image/png"
    assert mock_file.size == len(_TINY_PNG)

    data = mock_file.read()
    assert data == _TINY_PNG

    mock_file.seek(0)
    assert mock_file.getvalue() == _TINY_PNG


@pytest.mark.unit
def test_process_canvas_export_empty_data():
    with pytest.raises(InvalidCanvasDataError):
        process_canvas_export("")


@pytest.mark.unit
def test_process_canvas_export_no_comma():
    with pytest.raises(InvalidCanvasDataError):
        process_canvas_export("baddata_no_comma")
