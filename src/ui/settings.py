"""Settings page for the Engineer Assistant Streamlit app."""

import sys
from pathlib import Path

import streamlit as st

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def render() -> None:
    """Render the settings page."""
    st.markdown("# ⚙️ Settings")
    st.markdown("Configure your preferences and application settings.")

    st.markdown("---")

    # Media save settings
    st.markdown("## 💾 Media Saving")
    st.markdown("Choose where displayed media (images/STL) will be saved on the server")

    # Default save dir inside project outputs
    default_save = str(project_root / "outputs")
    if "media_save_dir" not in st.session_state:
        st.session_state.media_save_dir = default_save

    st.session_state.media_save_dir = st.text_input(
        "Server save directory",
        value=st.session_state.media_save_dir,
        help="Absolute or project-relative path where displayed media will be copied when 'Save' is clicked",
        key="media_save_dir_widget",
    )

    st.session_state.media_auto_save = st.checkbox(
        "Auto-save displayed media",
        value=st.session_state.get("media_auto_save", False),
        help="If enabled, images and STL files shown in the chat will be copied to the server save directory automatically",
        key="media_auto_save_widget",
    )

    st.markdown("---")

    # STL Viewer Settings
    st.markdown("## 🎨 3D Viewer")
    st.info(
        "**Note:** Changing these settings will reload all 3D models in the chat."
    )

    col1, col2 = st.columns(2)

    with col1:
        # Color picker - directly update session state
        st.session_state.stl_color = st.color_picker(
            "Model Color",
            value=st.session_state.get("stl_color", "#0069B4"),
            help="Choose a color for your 3D models",
            key="color_picker_widget",
        )

        # Material selector - directly update session state
        st.session_state.stl_material = st.selectbox(
            "Material",
            options=["material", "flat", "wireframe"],
            index=["material", "flat", "wireframe"].index(
                st.session_state.get("stl_material", "material")
            ),
            help="material: smooth shading, flat: faceted look, wireframe: mesh structure",
            key="material_selector_widget",
        )

        # Opacity slider
        st.session_state.stl_opacity = st.slider(
            "Opacity",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("stl_opacity", 1.0),
            step=0.1,
            help="Adjust the transparency of the model (0 = transparent, 1 = opaque)",
            key="opacity_slider_widget",
        )

    with col2:
        # Height slider - directly update session state
        st.session_state.stl_height = st.slider(
            "Viewer Height (px)",
            min_value=200,
            max_value=800,
            value=st.session_state.get("stl_height", 400),
            step=50,
            help="Adjust the height of the 3D viewer",
            key="height_slider_widget",
        )

        # Shininess slider
        st.session_state.stl_shininess = st.slider(
            "Shininess",
            min_value=0,
            max_value=200,
            value=st.session_state.get("stl_shininess", 100),
            step=10,
            help="Adjust the shininess/glossiness of the surface",
            key="shininess_slider_widget",
        )

        # Auto-rotate toggle
        st.session_state.stl_auto_rotate = st.checkbox(
            "Auto-rotate models",
            value=st.session_state.get("stl_auto_rotate", True),
            help="Automatically rotate 3D models",
            key="auto_rotate_checkbox_widget",
        )

    st.markdown("---")

    # Chat settings section
    st.markdown("## 💬 Chat Settings")

    if st.button("🗑️ Clear Conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.agent_state = {"messages": []}
        st.success("✅ Conversation cleared!")
        st.rerun()

    # Display current conversation stats
    if st.session_state.get("messages"):
        num_messages = len(st.session_state.messages)
        st.info(f"📊 Current conversation has {num_messages} messages")

    st.markdown("---")

    # About section
    st.markdown("## About")
    st.markdown(
        """
        **EngiAI - Engineering Design Assistant**

        Version: 1.0.0

        This application uses:
        - Multi-agent AI system for specialized tasks
        - LangChain for agent orchestration
        - Streamlit for the web interface
        - Custom fonts: [Space Grotesk](https://fonts.google.com/specimen/Space+Grotesk)
          and [Space Mono](https://fonts.google.com/specimen/Space+Mono)
        """
    )


if __name__ == "__main__":
    render()
