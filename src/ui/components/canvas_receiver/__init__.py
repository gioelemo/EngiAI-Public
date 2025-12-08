"""Custom Streamlit component for receiving Excalidraw canvas exports."""

import streamlit as st

# Inline JavaScript for polling localStorage
# Must be an ES module with default export function
_COMPONENT_JS = """
// Track polling intervals per component instance
const instances = new WeakMap();

const CanvasReceiver = (args) => {
    const { parentElement, setStateValue } = args;

    // Check if we already have an interval running for this instance
    if (!instances.has(parentElement)) {
        let lastSent = null;

        // Start polling localStorage for canvas exports
        const intervalId = setInterval(() => {
            try {
                const exportData = localStorage.getItem('excalidraw_pending_export');

                if (exportData && exportData !== lastSent) {
                    // Clear the localStorage item
                    localStorage.removeItem('excalidraw_pending_export');

                    // Send the data back to Streamlit
                    setStateValue("export_data", exportData);
                    lastSent = exportData;
                    console.log('Canvas export sent to Streamlit:', exportData.substring(0, 100) + '...');
                }
            } catch (error) {
                console.error('Error checking for canvas export:', error);
            }
        }, 500); // Check every 500ms

        instances.set(parentElement, { intervalId, lastSent });
    }

    // Cleanup function - called when component is unmounted
    return () => {
        const instanceData = instances.get(parentElement);
        if (instanceData) {
            clearInterval(instanceData.intervalId);
            instances.delete(parentElement);
        }
    };
};

export default CanvasReceiver;
"""

_component = st.components.v2.component(
    "canvas_receiver",
    js=_COMPONENT_JS,
    html='<div style="display: none;"></div>',
)


def _on_export_data_change():
    """Callback function for when export_data changes in the frontend."""


def canvas_receiver(key=None):
    """
    Create a canvas receiver component that polls localStorage for Excalidraw exports.

    Parameters
    ----------
    key: str or None
        An optional key that uniquely identifies this component.

    Returns
    -------
    str or None
        Base64 encoded image data if available, None otherwise.
        (This is the value passed to `setStateValue` on the frontend.)
    """
    component_value = _component(
        key=key,
        default={"export_data": None},
        on_export_data_change=_on_export_data_change,
    )

    # Extract the export_data from the component state
    if component_value and isinstance(component_value, dict):
        return component_value.get("export_data")
    return None
