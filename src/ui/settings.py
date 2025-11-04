"""Settings page for the Engineer Assistant Streamlit app."""

import sys
from pathlib import Path

import streamlit as st

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ui.database import DatabaseManager  # noqa: E402


def _init_db() -> DatabaseManager:
    """Initialize database manager."""
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    return st.session_state.db_manager


def _load_settings_from_db() -> None:
    """Load all settings from database into session state."""
    db = _init_db()
    default_save = str(project_root / "outputs")

    # Define default settings
    defaults = {
        "media_save_dir": default_save,
        "media_auto_save": False,
        "stl_color": "#0069B4",
        "stl_material": "material",
        "stl_opacity": 1.0,
        "stl_height": 400,
        "stl_shininess": 100,
        "stl_auto_rotate": True,
    }

    # Load settings from database or use defaults
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = db.get_setting(key, default_value)


def _save_setting_to_db(key: str, value: object) -> None:
    """Save a setting to database and update session state.

    Args:
        key: Setting key
        value: Setting value
    """
    db = _init_db()
    db.set_setting(key, value)
    st.session_state[key] = value


def _render_media_settings() -> None:
    """Render media saving settings section."""
    st.markdown("## 💾 Media Saving")
    st.markdown("Choose where displayed media (images/STL) will be saved on the server")

    # Media save directory input
    media_save_dir = st.text_input(
        "Server save directory",
        value=st.session_state.media_save_dir,
        help="Absolute or project-relative path where displayed media will be copied when 'Save' is clicked",
        key="media_save_dir_widget",
    )
    if media_save_dir != st.session_state.media_save_dir:
        _save_setting_to_db("media_save_dir", media_save_dir)

    # Media auto-save checkbox
    media_auto_save = st.checkbox(
        "Auto-save displayed media",
        value=st.session_state.media_auto_save,
        help="If enabled, images and STL files shown in the chat will be copied to the server save directory automatically",
        key="media_auto_save_widget",
    )
    if media_auto_save != st.session_state.media_auto_save:
        _save_setting_to_db("media_auto_save", media_auto_save)


def _render_stl_viewer_settings() -> None:
    """Render 3D viewer settings section."""
    st.markdown("## 🎨 3D Viewer")
    st.info("**Note:** Changing these settings will reload all 3D models in the chat.")

    col1, col2 = st.columns(2)

    with col1:
        # Color picker
        stl_color = st.color_picker(
            "Model Color",
            value=st.session_state.stl_color,
            help="Choose a color for your 3D models",
            key="color_picker_widget",
        )
        if stl_color != st.session_state.stl_color:
            _save_setting_to_db("stl_color", stl_color)

        # Material selector
        material_options = ["material", "flat", "wireframe"]
        material_index = material_options.index(st.session_state.stl_material)
        stl_material = st.selectbox(
            "Material",
            options=material_options,
            index=material_index,
            help="material: smooth shading, flat: faceted look, wireframe: mesh structure",
            key="material_selector_widget",
        )
        if stl_material != st.session_state.stl_material:
            _save_setting_to_db("stl_material", stl_material)

        # Opacity slider
        stl_opacity = st.slider(
            "Opacity",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.stl_opacity,
            step=0.1,
            help="Adjust the transparency of the model (0 = transparent, 1 = opaque)",
            key="opacity_slider_widget",
        )
        if stl_opacity != st.session_state.stl_opacity:
            _save_setting_to_db("stl_opacity", stl_opacity)

    with col2:
        # Height slider
        stl_height = st.slider(
            "Viewer Height (px)",
            min_value=200,
            max_value=800,
            value=st.session_state.stl_height,
            step=50,
            help="Adjust the height of the 3D viewer",
            key="height_slider_widget",
        )
        if stl_height != st.session_state.stl_height:
            _save_setting_to_db("stl_height", stl_height)

        # Shininess slider
        stl_shininess = st.slider(
            "Shininess",
            min_value=0,
            max_value=200,
            value=st.session_state.stl_shininess,
            step=10,
            help="Adjust the shininess/glossiness of the surface",
            key="shininess_slider_widget",
        )
        if stl_shininess != st.session_state.stl_shininess:
            _save_setting_to_db("stl_shininess", stl_shininess)

        # Auto-rotate toggle
        stl_auto_rotate = st.checkbox(
            "Auto-rotate models",
            value=st.session_state.stl_auto_rotate,
            help="Automatically rotate 3D models",
            key="auto_rotate_checkbox_widget",
        )
        if stl_auto_rotate != st.session_state.stl_auto_rotate:
            _save_setting_to_db("stl_auto_rotate", stl_auto_rotate)


def _render_chat_settings() -> None:
    """Render chat settings section."""
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


def _render_about_section() -> None:
    """Render about section."""
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


def render() -> None:
    """Render the settings page."""
    # Load settings from database on first render
    _load_settings_from_db()

    st.markdown("# ⚙️ Settings")
    st.markdown("Configure your preferences and application settings.")

    st.markdown("---")
    _render_media_settings()

    st.markdown("---")
    _render_stl_viewer_settings()

    st.markdown("---")
    _render_chat_settings()

    st.markdown("---")
    _render_about_section()


if __name__ == "__main__":
    render()
