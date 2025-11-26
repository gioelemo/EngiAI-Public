"""
PDF export functionality for chat conversations.

Exports chat conversations to PDF format with text and embedded images.
"""

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)
from reportlab.platypus import (
    Image as RLImage,
)
from stl import mesh

from src.ui.media_display import find_images_in_text, find_stl_files_in_text

# Page configuration
PAGE_WIDTH = A4[0]
PAGE_HEIGHT = A4[1]
MARGIN = 19 * mm  # approximately 0.75 inches
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN
MAX_IMAGE_WIDTH = CONTENT_WIDTH * 0.8
MAX_IMAGE_HEIGHT = 100 * mm  # approximately 4 inches

# UI color scheme (from .streamlit/config.toml)
UI_COLORS = {
    "primary": colors.HexColor("#cb785c"),
    "background": colors.HexColor("#fdfdf8"),
    "secondary_bg": colors.HexColor("#ecebe3"),
    "text": colors.HexColor("#3d3a2a"),
    "link": colors.HexColor("#3d3a2a"),
    "border": colors.HexColor("#d3d2ca"),
}

# Text processing constants
_MIN_NUMBERED_BULLET_LENGTH = 2


def _register_fonts() -> None:
    """Register custom fonts from the UI."""
    # Check if SpaceGrotesk is already registered to avoid duplicate registration
    if "SpaceGrotesk" in pdfmetrics.getRegisteredFontNames():
        return

    static_dir = Path(__file__).parent / "static"

    try:
        # Register SpaceGrotesk (main font)
        if (static_dir / "SpaceGrotesk-VariableFont_wght.ttf").exists():
            pdfmetrics.registerFont(
                TTFont(
                    "SpaceGrotesk", static_dir / "SpaceGrotesk-VariableFont_wght.ttf"
                )
            )

        # Register SpaceMono (code font)
        if (static_dir / "SpaceMono-Regular.ttf").exists():
            pdfmetrics.registerFont(
                TTFont("SpaceMono", static_dir / "SpaceMono-Regular.ttf")
            )
        if (static_dir / "SpaceMono-Bold.ttf").exists():
            pdfmetrics.registerFont(
                TTFont("SpaceMono-Bold", static_dir / "SpaceMono-Bold.ttf")
            )
        if (static_dir / "SpaceMono-Italic.ttf").exists():
            pdfmetrics.registerFont(
                TTFont("SpaceMono-Italic", static_dir / "SpaceMono-Italic.ttf")
            )
    except Exception:
        # If font registration fails, fall back to default fonts
        pass


def create_pdf_from_conversation(
    conversation_name: str,
    messages: list[dict[str, Any]],
    created_at: datetime | None = None,
) -> bytes:
    """Create a PDF document from a conversation.

    Args:
        conversation_name: Name/title of the conversation
        messages: List of message dictionaries with 'role', 'content', 'images', 'created_at'
        created_at: Conversation creation timestamp

    Returns:
        PDF document as bytes
    """
    # Create PDF buffer
    buffer = io.BytesIO()

    # Create PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=MARGIN,
        leftMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )

    # Container for PDF elements
    story: list[Any] = []

    # Register custom fonts
    _register_fonts()

    # Determine font to use (SpaceGrotesk if available, otherwise default)
    main_font = (
        "SpaceGrotesk"
        if "SpaceGrotesk" in pdfmetrics.getRegisteredFontNames()
        else "Helvetica"
    )

    # Create styles
    pdf_styles = _create_pdf_styles(main_font)

    # Add header section
    _add_pdf_header(story, conversation_name, created_at, pdf_styles)

    # Process each message
    for idx, message in enumerate(messages):
        role = message.get("role", "unknown").capitalize()
        content = message.get("content", "")
        images = message.get("images", [])
        msg_created_at = message.get("created_at")

        # Add role label
        story.append(Paragraph(f"<b>{role}</b>", pdf_styles["role_label"]))

        # Add timestamp if available
        if msg_created_at:
            if isinstance(msg_created_at, str):
                timestamp_str = msg_created_at
            else:
                timestamp_str = msg_created_at.strftime("%Y-%m-%d %H:%M:%S")
            story.append(Paragraph(timestamp_str, pdf_styles["timestamp"]))

        # Add message content
        if content:
            # Clean content for PDF (escape HTML special chars, handle newlines)
            content_cleaned = _clean_text_for_pdf(content)

            # Split by paragraphs and add each
            paragraphs = content_cleaned.split("\n\n")
            style = (
                pdf_styles["user"]
                if role.lower() == "user"
                else pdf_styles["assistant"]
            )

            story.extend(
                Paragraph(para.strip(), style) for para in paragraphs if para.strip()
            )

        # Add images for this message
        _add_message_images(story, images, content, pdf_styles)

        # Add separator between messages
        if idx < len(messages) - 1:
            story.append(Spacer(1, 5 * mm))
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.5,
                    color=UI_COLORS["border"],
                    spaceBefore=0,
                    spaceAfter=0,
                )
            )
            story.append(Spacer(1, 5 * mm))

    # Build PDF
    doc.build(story)

    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


def _create_pdf_styles(main_font: str) -> dict[str, ParagraphStyle]:
    """Create paragraph styles for PDF export.

    Args:
        main_font: Name of the main font to use

    Returns:
        Dictionary of style names to ParagraphStyle objects
    """
    styles = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontName=main_font,
            fontSize=24,
            textColor=UI_COLORS["primary"],
            spaceAfter=12,
            alignment=1,  # Center alignment
        ),
        "metadata": ParagraphStyle(
            "Metadata",
            parent=styles["Normal"],
            fontName=main_font,
            fontSize=10,
            textColor=colors.grey,
            alignment=1,  # Center alignment
            spaceAfter=20,
        ),
        "user": ParagraphStyle(
            "UserMessage",
            parent=styles["Normal"],
            fontName=main_font,
            fontSize=11,
            textColor=UI_COLORS["text"],
            leftIndent=20,
            rightIndent=20,
            spaceAfter=6,
            spaceBefore=6,
        ),
        "assistant": ParagraphStyle(
            "AssistantMessage",
            parent=styles["Normal"],
            fontName=main_font,
            fontSize=11,
            textColor=UI_COLORS["text"],
            leftIndent=20,
            rightIndent=20,
            spaceAfter=6,
            spaceBefore=6,
            backgroundColor=UI_COLORS["secondary_bg"],
        ),
        "role_label": ParagraphStyle(
            "RoleLabel",
            parent=styles["Heading3"],
            fontName=main_font,
            fontSize=12,
            textColor=UI_COLORS["primary"],
            spaceAfter=4,
            spaceBefore=12,
        ),
        "timestamp": ParagraphStyle(
            "Timestamp",
            parent=styles["Normal"],
            fontName=main_font,
            fontSize=9,
            textColor=colors.grey,
            spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "ImageCaption",
            parent=styles["Normal"],
            fontName=main_font,
            fontSize=9,
            textColor=colors.grey,
            alignment=1,  # Center
            spaceAfter=6,
        ),
        "missing": ParagraphStyle(
            "MissingImage",
            parent=styles["Normal"],
            fontName=main_font,
            fontSize=9,
            textColor=colors.HexColor("#e74c3c"),
            leftIndent=20,
            spaceAfter=6,
        ),
        "error": ParagraphStyle(
            "ImageError",
            parent=styles["Normal"],
            fontName=main_font,
            fontSize=9,
            textColor=colors.HexColor("#e67e22"),
            leftIndent=20,
            spaceAfter=6,
        ),
    }


def _add_pdf_header(
    story: list[Any],
    conversation_name: str,
    created_at: datetime | None,
    pdf_styles: dict[str, ParagraphStyle],
) -> None:
    """Add header section to PDF (logo, title, metadata).

    Args:
        story: List to append PDF elements to
        conversation_name: Name of the conversation
        created_at: Conversation creation timestamp
        pdf_styles: Dictionary of paragraph styles
    """
    # Add logo at the top (if available)
    logo_path = Path(__file__).parent.parent.parent / "assets" / "logo_notext.png"
    if logo_path.exists():
        try:
            logo_img = _create_pdf_image_from_path(logo_path)
            if logo_img:
                # Scale logo to reasonable size (13mm height)
                logo_img.drawHeight = 13 * mm
                logo_img.drawWidth = 13 * mm
                story.append(logo_img)
                story.append(Spacer(1, 3 * mm))
        except Exception:
            # If logo fails to load, continue without it
            pass

    # Add title
    story.append(Paragraph(f"Chat Export: {conversation_name}", pdf_styles["title"]))

    # Add metadata (created and exported on same line)
    export_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if created_at:
        created_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
        metadata_text = f"Created: {created_str} | Exported: {export_time}"
    else:
        metadata_text = f"Exported: {export_time}"

    story.append(Paragraph(metadata_text, pdf_styles["metadata"]))
    story.append(Spacer(1, 5 * mm))


def _add_uploaded_images(story: list[Any], images: list[dict[str, Any]]) -> None:
    """Add uploaded images (base64) to PDF story.

    Args:
        story: List to append PDF elements to
        images: List of uploaded images (base64)
    """
    if not images:
        return

    story.append(Spacer(1, 3 * mm))
    for img_data in images:
        img_type = img_data.get("type", "")
        if img_type and img_type.startswith("image/"):
            try:
                img_bytes = base64.b64decode(img_data["data"])
                img_obj = _create_pdf_image(img_bytes)
                if img_obj:
                    story.append(img_obj)
                    story.append(Spacer(1, 3 * mm))
            except Exception:
                pass  # Skip images that can't be decoded


def _add_referenced_images(
    story: list[Any], content: str, pdf_styles: dict[str, ParagraphStyle]
) -> None:
    """Add referenced image files to PDF story.

    Args:
        story: List to append PDF elements to
        content: Message content text
        pdf_styles: Dictionary of paragraph styles
    """
    referenced_images = find_images_in_text(content)
    if not referenced_images:
        return

    story.append(Spacer(1, 3 * mm))
    for img_path in referenced_images:
        try:
            if img_path.exists():
                img_obj = _create_pdf_image_from_path(img_path)
                if img_obj:
                    story.append(img_obj)
                    story.append(Paragraph(img_path.name, pdf_styles["caption"]))
                    story.append(Spacer(1, 3 * mm))
            else:
                story.append(
                    Paragraph(
                        f"⚠️ Image not found: {img_path.name}",
                        pdf_styles["missing"],
                    )
                )
        except Exception as e:
            story.append(
                Paragraph(
                    f"⚠️ Error loading {img_path.name}: {str(e)[:50]}",
                    pdf_styles["error"],
                )
            )


def _add_stl_files(
    story: list[Any], content: str, pdf_styles: dict[str, ParagraphStyle]
) -> None:
    """Add STL files to PDF story as rendered 2D images.

    Args:
        story: List to append PDF elements to
        content: Message content text
        pdf_styles: Dictionary of paragraph styles
    """
    stl_files = find_stl_files_in_text(content)
    print(f"Found {len(stl_files)} STL files in content")

    if not stl_files:
        return

    story.append(Spacer(1, 3 * mm))
    for stl_path in stl_files:
        print(f"Processing STL: {stl_path}")
        try:
            if stl_path.exists():
                print("STL file exists, rendering...")
                stl_img_bytes = _render_stl_to_image(stl_path)
                if stl_img_bytes:
                    print(
                        f"STL rendered successfully, size: {len(stl_img_bytes)} bytes"
                    )
                    img_obj = _create_pdf_image(stl_img_bytes)
                    if img_obj:
                        story.append(img_obj)
                        story.append(
                            Paragraph(
                                f"3D Model: {stl_path.name}",
                                pdf_styles["caption"],
                            )
                        )
                        story.append(Spacer(1, 3 * mm))
                        print("STL added to PDF successfully")
                    else:
                        print("Failed to create PDF image from STL bytes")
                else:
                    print("STL rendering returned None")
            else:
                print(f"STL file does not exist: {stl_path}")
                story.append(
                    Paragraph(
                        f"⚠️ 3D model not found: {stl_path.name}",
                        pdf_styles["missing"],
                    )
                )
        except Exception as e:
            print(f"Exception rendering STL: {e}")
            story.append(
                Paragraph(
                    f"⚠️ Error rendering {stl_path.name}: {str(e)[:50]}",
                    pdf_styles["error"],
                )
            )


def _add_message_images(
    story: list[Any],
    images: list[dict[str, Any]],
    content: str,
    pdf_styles: dict[str, ParagraphStyle],
) -> None:
    """Add images and STL files to PDF story for a message.

    Args:
        story: List to append PDF elements to
        images: List of uploaded images (base64)
        content: Message content text
        pdf_styles: Dictionary of paragraph styles
    """
    _add_uploaded_images(story, images)
    _add_referenced_images(story, content, pdf_styles)
    _add_stl_files(story, content, pdf_styles)


def _is_bullet_point(line: str) -> bool:
    """Check if a line is a bullet point.

    Args:
        line: Line of text to check

    Returns:
        True if line is a bullet point, False otherwise
    """
    stripped = line.strip()
    return stripped.startswith(("- ", "* ", "• ")) or (
        len(stripped) > _MIN_NUMBERED_BULLET_LENGTH
        and stripped[0].isdigit()
        and stripped[1:3] in (". ", ") ")
    )


def _clean_text_for_pdf(text: str) -> str:
    """Clean text for PDF rendering.

    Args:
        text: Raw text content

    Returns:
        Cleaned text suitable for PDF
    """
    # Escape XML/HTML special characters for ReportLab
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    # Split into lines to handle lists properly
    lines = text.split("\n")
    processed_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check if this line is a bullet point
        is_bullet = _is_bullet_point(line)

        # Check if next line is also a bullet (to keep them together)
        next_is_bullet = False
        if i + 1 < len(lines):
            next_is_bullet = _is_bullet_point(lines[i + 1])

        if is_bullet:
            # For bullet points, use <br/> to preserve line breaks
            processed_lines.append(stripped)
            if next_is_bullet or (i + 1 < len(lines) and lines[i + 1].strip()):
                processed_lines.append("<br/>")
        elif stripped == "":
            # Empty line - paragraph break
            if processed_lines and processed_lines[-1] != "\n\n":
                processed_lines.append("\n\n")
        else:
            # Regular text - join with space if previous line wasn't empty
            if processed_lines and not processed_lines[-1].endswith(("\n\n", "<br/>")):
                processed_lines.append(" ")
            processed_lines.append(stripped)

    return "".join(processed_lines)


def _create_pdf_image(img_bytes: bytes) -> RLImage | None:
    """Create a ReportLab Image object from image bytes.

    Args:
        img_bytes: Image data as bytes

    Returns:
        ReportLab Image object or None if failed
    """
    try:
        # Open image with PIL
        pil_img: Image.Image = Image.open(io.BytesIO(img_bytes))

        # Convert to RGB if necessary (removes alpha channel)
        if pil_img.mode in ("RGBA", "LA", "P"):
            background: Image.Image = Image.new("RGB", pil_img.size, (255, 255, 255))
            if pil_img.mode == "P":
                pil_img = pil_img.convert("RGBA")
            background.paste(
                pil_img,
                mask=pil_img.split()[-1] if pil_img.mode in ("RGBA", "LA") else None,
            )
            pil_img = background

        # Save to BytesIO buffer
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        # Calculate dimensions maintaining aspect ratio
        img_width, img_height = pil_img.size
        width = float(img_width)
        height = float(img_height)
        aspect_ratio = width / height

        # Fit within max dimensions
        if width > MAX_IMAGE_WIDTH:
            width = MAX_IMAGE_WIDTH
            height = width / aspect_ratio

        if height > MAX_IMAGE_HEIGHT:
            height = MAX_IMAGE_HEIGHT
            width = height * aspect_ratio

        # Create ReportLab image
        return RLImage(img_buffer, width=width, height=height)

    except Exception:
        return None


def _render_stl_to_image(stl_path: Path, dpi: int = 150) -> bytes | None:
    """Render an STL file to a 2D image.

    Args:
        stl_path: Path to STL file
        dpi: DPI for the rendered image

    Returns:
        Image bytes or None if failed
    """
    try:
        # Load the STL file
        stl_mesh = mesh.Mesh.from_file(str(stl_path))

        # Create a new plot with white background
        fig = plt.figure(figsize=(8, 6), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")

        # Get mesh bounds for proper scaling
        max_range = max(
            stl_mesh.points[:, 0].max() - stl_mesh.points[:, 0].min(),
            stl_mesh.points[:, 1].max() - stl_mesh.points[:, 1].min(),
            stl_mesh.points[:, 2].max() - stl_mesh.points[:, 2].min(),
        )

        mid_x = (stl_mesh.points[:, 0].max() + stl_mesh.points[:, 0].min()) * 0.5
        mid_y = (stl_mesh.points[:, 1].max() + stl_mesh.points[:, 1].min()) * 0.5
        mid_z = (stl_mesh.points[:, 2].max() + stl_mesh.points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range * 0.6, mid_x + max_range * 0.6)
        ax.set_ylim(mid_y - max_range * 0.6, mid_y + max_range * 0.6)
        ax.set_zlim(mid_z - max_range * 0.6, mid_z + max_range * 0.6)

        # Create the mesh collection
        mesh_collection = Poly3DCollection(
            stl_mesh.vectors,
            facecolors="lightblue",
            edgecolors="darkblue",
            linewidths=0.5,
            alpha=0.8,
        )
        ax.add_collection3d(mesh_collection)

        # Set viewing angle
        ax.view_init(elev=30, azim=45)

        # Remove axes for cleaner look
        ax.set_axis_off()

        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)

        return buf.read()

    except Exception as e:
        print(f"Error rendering STL {stl_path}: {e}")  # Debug output
        return None


def _create_pdf_image_from_path(img_path: Path) -> RLImage | None:
    """Create a ReportLab Image object from a file path.

    Args:
        img_path: Path to image file

    Returns:
        ReportLab Image object or None if failed
    """
    try:
        img_bytes = img_path.read_bytes()
        return _create_pdf_image(img_bytes)
    except Exception:
        return None
