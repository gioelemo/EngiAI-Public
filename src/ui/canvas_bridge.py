"""Bridge utilities for Excalidraw canvas export processing."""

import base64
import io
from datetime import datetime

from src.ui.components.excalidraw import excalidraw_whiteboard


class InvalidCanvasDataError(ValueError):
    """Raised when canvas data is invalid or malformed."""


def get_excalidraw_whiteboard(
    height: int = 650, key: str | None = None, trigger_export: bool = False
) -> str | None:
    """
    Render an Excalidraw whiteboard and get export data.

    Parameters
    ----------
    height : int
        Height of the whiteboard in pixels. Default is 650.
    key : str or None
        An optional key that uniquely identifies this component.
    trigger_export : bool
        If True, triggers the export of the canvas. Default is False.

    Returns
    -------
    str or None
        Base64 encoded PNG image data when export is triggered,
        None otherwise.
    """
    return excalidraw_whiteboard(height=height, key=key, trigger_export=trigger_export)


def process_canvas_export(base64_data: str) -> dict:
    """Process the exported canvas data and create a mock file for chat."""

    if not base64_data or "," not in base64_data:
        raise InvalidCanvasDataError

    # Decode the image
    base64_content = base64_data.split(",", 1)[1]
    image_bytes = base64.b64decode(base64_content)

    # Create mock uploaded file
    class MockUploadedFile:
        def __init__(self, data: bytes, name: str, type_: str):
            self._data = io.BytesIO(data)
            self.name = name
            self.type = type_
            self.size = len(data)

        def read(self) -> bytes:
            return self._data.read()

        def seek(self, pos: int) -> int:
            return self._data.seek(pos)

        def getvalue(self) -> bytes:
            return self._data.getvalue()

    mock_file = MockUploadedFile(
        image_bytes,
        f"canvas-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png",
        "image/png",
    )

    return {"text": "Here's my whiteboard drawing:", "files": [mock_file]}
