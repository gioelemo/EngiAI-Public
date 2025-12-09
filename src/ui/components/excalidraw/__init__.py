"""Excalidraw whiteboard component for Streamlit."""

from pathlib import Path

import streamlit.components.v1 as components

# Get the build directory relative to this file
_FRONTEND_BUILD_DIR = Path(__file__).parent / "frontend" / "build"

# Declare the component using v1 API (supports iframe-based React components)
_component = components.declare_component(
    "excalidraw_whiteboard",
    path=str(_FRONTEND_BUILD_DIR),
)


def excalidraw_whiteboard(
    height: int = 650, key: str | None = None, trigger_export: bool = False
) -> str | None:
    """
    Render an Excalidraw whiteboard component.

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
    return _component(
        height=height, key=key, trigger_export=trigger_export, default=None
    )
