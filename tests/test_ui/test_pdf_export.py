"""
Tests for src/ui/pdf_export.py helper functions.

Focuses on pure-Python building blocks that don't require a full
ReportLab document build: _is_bullet_point, _clean_text_for_pdf,
and _create_pdf_image.
"""

import io

import pytest
from PIL import Image as PILImage

from src.ui.pdf_export import (
    MAX_IMAGE_HEIGHT,
    MAX_IMAGE_WIDTH,
    _clean_text_for_pdf,
    _create_pdf_image,
    _is_bullet_point,
)

# ============================================================================
# _is_bullet_point
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "line",
    [
        "- item",
        "* item",
        "• item",
        "  - indented item",
        "  * indented",
    ],
)
def test_is_bullet_point_dash_asterisk_bullet(line):
    """Lines starting with -, *, or • markers → True."""
    assert _is_bullet_point(line) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "line",
    [
        "1. first",
        "2. second",
        "1) item",
        "3) another",
    ],
)
def test_is_bullet_point_numbered(line):
    """Single-digit lines starting with a digit followed by . or ) → True.

    Note: the implementation checks stripped[1:3] so only single-digit
    bullets are detected (e.g. '1.' or '1)'), not '10.'.
    """
    assert _is_bullet_point(line) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "line",
    [
        "Regular text",
        "No bullet here",
        "",
        "  plain indented text",
        "1 not a bullet",  # digit but no dot/paren
    ],
)
def test_is_bullet_point_regular_text(line):
    """Plain text lines → False."""
    assert _is_bullet_point(line) is False


# ============================================================================
# _clean_text_for_pdf
# ============================================================================


@pytest.mark.unit
def test_clean_text_html_escaping():
    """&, <, > must be replaced with XML entities."""
    result = _clean_text_for_pdf("a & b < c > d")
    assert "&amp;" in result
    assert "&lt;" in result
    assert "&gt;" in result
    assert "&" not in result.replace("&amp;", "").replace("&lt;", "").replace(
        "&gt;", ""
    )


@pytest.mark.unit
def test_clean_text_empty_string():
    """Empty input → empty output."""
    assert _clean_text_for_pdf("") == ""


@pytest.mark.unit
def test_clean_text_bullets_get_br_separator():
    """Consecutive bullet lines must be separated by <br/>."""
    text = "- first\n- second\n- third"
    result = _clean_text_for_pdf(text)
    assert "<br/>" in result


@pytest.mark.unit
def test_clean_text_plain_paragraph():
    """Non-bullet text must not gain spurious <br/> tags."""
    text = "This is a sentence.\nThis continues."
    result = _clean_text_for_pdf(text)
    # Consecutive plain lines are joined with a space, not <br/>
    assert "<br/>" not in result


# ============================================================================
# _create_pdf_image
# ============================================================================


def _make_png_bytes(width: int, height: int, mode: str = "RGB") -> bytes:
    """Helper: create minimal in-memory PNG bytes."""
    img = PILImage.new(
        mode,
        (width, height),
        color=(128, 64, 32) if mode == "RGB" else (128, 64, 32, 200),
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.unit
def test_create_pdf_image_rgba_converted_to_rgb():
    """RGBA PNG bytes → RGB conversion succeeds, returns a non-None RLImage."""
    png_bytes = _make_png_bytes(100, 100, mode="RGBA")
    result = _create_pdf_image(png_bytes)
    assert result is not None


@pytest.mark.unit
def test_create_pdf_image_returns_none_on_invalid():
    """Garbage bytes → returns None without raising."""
    result = _create_pdf_image(b"not an image at all \x00\x01\x02")
    assert result is None


@pytest.mark.unit
def test_create_pdf_image_clamps_oversized_width():
    """Image wider than MAX_IMAGE_WIDTH → drawWidth clamped to MAX_IMAGE_WIDTH."""
    # Create a very wide image (5000px wide)
    png_bytes = _make_png_bytes(5000, 100)
    result = _create_pdf_image(png_bytes)
    assert result is not None
    assert result.drawWidth <= MAX_IMAGE_WIDTH + 1  # +1 for float rounding


@pytest.mark.unit
def test_create_pdf_image_clamps_oversized_height():
    """Image taller than MAX_IMAGE_HEIGHT → drawHeight clamped to MAX_IMAGE_HEIGHT."""
    png_bytes = _make_png_bytes(100, 5000)
    result = _create_pdf_image(png_bytes)
    assert result is not None
    assert result.drawHeight <= MAX_IMAGE_HEIGHT + 1


@pytest.mark.unit
def test_create_pdf_image_small_image_not_upscaled():
    """Small image (well within limits) → dimensions preserved as floats, not enlarged."""
    png_bytes = _make_png_bytes(50, 50)
    result = _create_pdf_image(png_bytes)
    assert result is not None
    assert result.drawWidth <= MAX_IMAGE_WIDTH
    assert result.drawHeight <= MAX_IMAGE_HEIGHT
