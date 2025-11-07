"""
File processing functions for the Streamlit UI.

Handles uploaded files (images and PDFs) processing and conversion.
"""

import base64
import tempfile
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import MathpixPDFLoader


def extract_pdf_text(pdf_files: list[dict[str, str]]) -> str:
    """Extract text from PDF files using MathPixPDFLoader.

    Args:
        pdf_files: List of PDF file data dicts with 'data' (base64) and 'name' keys

    Returns:
        Extracted text from all PDFs
    """
    all_text = []

    for pdf_file in pdf_files:
        # Decode base64 to bytes
        pdf_bytes = base64.b64decode(pdf_file["data"])
        file_name = pdf_file.get("name", "document.pdf")

        # Write to temporary file (MathPixPDFLoader needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name

        try:
            # Use MathPixPDFLoader to extract text
            loader = MathpixPDFLoader(tmp_path)
            docs = loader.load()

            # Combine all pages
            pdf_text = f"\n\n=== {file_name} ===\n\n"
            pdf_text += "\n\n".join(doc.page_content for doc in docs)
            all_text.append(pdf_text)

        finally:
            # Clean up temp file
            tmp_file_path = Path(tmp_path)
            if tmp_file_path.exists():
                tmp_file_path.unlink()

    return "\n\n".join(all_text)


def process_uploaded_images(
    files: list[Any],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Process uploaded image and PDF files to base64 format.

    Args:
        files: List of uploaded file objects from Streamlit

    Returns:
        Tuple of (images_for_display, images_for_agent)
        - images_for_display: List of dicts with 'data' (base64), 'type', and 'name' keys for storage
        - images_for_agent: List of dicts in LangChain vision format
    """
    images_for_display = []
    images_for_agent = []

    for file in files:
        # Read the file bytes
        file_bytes = file.read()

        # Convert to base64
        base64_data = base64.b64encode(file_bytes).decode("utf-8")

        # Determine file type
        file_type = file.type if hasattr(file, "type") else "application/octet-stream"
        file_name = file.name if hasattr(file, "name") else "unknown"

        # For display/storage
        images_for_display.append(
            {
                "data": base64_data,
                "type": file_type,
                "name": file_name,
            }
        )

        # For LangChain agent
        # Note: PDF support varies by model. Claude 3.5 Sonnet and some others support PDFs
        # as if they were images using the same format
        images_for_agent.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{file_type};base64,{base64_data}"},
            }
        )

    return images_for_display, images_for_agent
