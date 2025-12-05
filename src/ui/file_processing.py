"""
File processing functions for the Streamlit UI.

Handles uploaded files (images and PDFs) processing and conversion.
"""

import base64
from typing import Any


def process_uploaded_images(
    files: list[Any],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Process uploaded image and PDF files to base64 format.

    Args:
        files: List of uploaded file objects from Streamlit

    Returns:
        Tuple of (images_for_display, images_for_agent)
        - images_for_display: List of dicts with 'data' (base64), 'type', 'name', and 'bytes' keys for storage
        - images_for_agent: List of dicts in LangChain vision format
    """
    images_for_display = []
    images_for_agent = []

    for file in files:
        # Read the file bytes once
        file_bytes = file.read()

        # Convert to base64 for storage/display
        base64_data = base64.b64encode(file_bytes).decode("utf-8")

        # Determine file type
        file_type = file.type if hasattr(file, "type") else "application/octet-stream"
        file_name = file.name if hasattr(file, "name") else "unknown"

        # For display/storage - include raw bytes to avoid decoding later in the same request
        images_for_display.append(
            {
                "data": base64_data,
                "type": file_type,
                "name": file_name,
                "bytes": file_bytes,  # Keep original bytes for immediate use
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
