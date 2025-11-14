"""Settings page for the Engineer Assistant Streamlit app."""

import os
import sys
from pathlib import Path

import streamlit as st

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config  # noqa: E402
from scripts.import_local_papers import LocalPaperImporter  # noqa: E402
from src.ui.database import DatabaseManager  # noqa: E402
from src.utils.api_usage import (  # noqa: E402
    USAGE_THRESHOLD_CRITICAL,
    USAGE_THRESHOLD_WARNING,
    get_mathpix_usage,
    get_tavily_usage,
)


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

    if st.button("🗑️ Clear Conversation", width="stretch", type="primary"):
        st.session_state.messages = []
        st.session_state.agent_state = {"messages": []}
        st.success("✅ Conversation cleared!")
        st.rerun()

    # Display current conversation stats
    if st.session_state.get("messages"):
        num_messages = len(st.session_state.messages)
        st.info(f"📊 Current conversation has {num_messages} messages")


def _render_paper_import_settings() -> None:
    """Render paper import settings section."""
    st.markdown("## 📚 Paper Import")
    st.markdown("Bulk import PDF papers into the RAG knowledge base.")

    # Get configuration from environment
    papers_source_dir = os.getenv(
        "PAPERS_SOURCE_DIR", ""
    )  # Container path for operations
    papers_source_dir_host = os.getenv(
        "PAPERS_SOURCE_DIR_HOST", ""
    )  # Host path for display
    papers_state_file = os.getenv("PAPERS_STATE_FILE", "data/local_import_state.json")
    papers_collection = os.getenv("PAPERS_COLLECTION", "engineer_docs")

    # Display configuration (show host path to user)
    display_path = (
        papers_source_dir_host if papers_source_dir_host else papers_source_dir
    )
    st.info(
        f"**Papers directory (host):** `{display_path if display_path else 'Not configured'}`"
    )

    if not papers_source_dir:
        st.warning(
            "⚠️ Papers directory not configured. Set `PAPERS_SOURCE_DIR` in your `.env` file "
            "to enable bulk import."
        )
        return

    # Check if directory exists and is accessible (use container path)
    papers_path = Path(papers_source_dir)
    if not papers_path.exists():
        st.error(
            f"❌ Papers directory does not exist: `{papers_source_dir}`\n\n"
            "Please check your `.env` configuration and ensure the folder is mounted in Docker."
        )
        return

    # Show directory stats
    try:
        pdf_files = list(papers_path.rglob("*.pdf")) + list(papers_path.rglob("*.PDF"))
        pdf_files = [
            f for f in pdf_files if not any(part.startswith(".") for part in f.parts)
        ]
        st.success(f"✅ Found {len(pdf_files)} PDF files in the directory")
    except Exception as e:
        st.error(f"❌ Error scanning directory: {e}")
        return

    # Import button with options
    col1, col2 = st.columns([3, 1])

    with col1:
        dry_run = st.checkbox(
            "Dry run (preview only)",
            value=False,
            help="Preview which files would be imported without actually importing them",
        )

    with col2:
        max_files = st.number_input(
            "Max files",
            min_value=1,
            max_value=1000,
            value=10,
            help="Maximum number of files to import in one batch",
        )

    if st.button("📥 Import Papers", type="primary", use_container_width=True):
        with st.spinner("Importing papers..."):
            try:
                # Create importer
                importer = LocalPaperImporter(
                    source_dir=str(papers_path),
                    state_file=papers_state_file,
                    collection_name=papers_collection,
                )

                # Run import
                stats = importer.run(dry_run=dry_run, max_files=max_files)

                # Display results
                if dry_run:
                    st.info(
                        f"**Dry Run Results:**\n\n"
                        f"- Total files found: {stats['total_files']}\n"
                        f"- New/modified files: {stats['new_files']}\n"
                        f"- Would process: {min(stats['new_files'], max_files)}"
                    )
                elif stats["successful"] > 0:
                    st.success(
                        f"✅ **Import Complete!**\n\n"
                        f"- Total files found: {stats['total_files']}\n"
                        f"- New/modified files: {stats['new_files']}\n"
                        f"- Processed: {stats['processed']}\n"
                        f"- ✓ Successful: {stats['successful']}\n"
                        f"- ✗ Failed: {stats['failed']}\n"
                        f"- Skipped: {stats['skipped']}"
                    )
                elif stats["new_files"] == 0:
                    st.info("All papers are already imported. No new files to process.")
                else:
                    st.warning(
                        f"⚠️ Import completed with errors:\n\n"
                        f"- Failed: {stats['failed']}/{stats['processed']}"
                    )

            except Exception as e:
                st.error(f"❌ Import failed: {e}")


def _render_tavily_usage() -> None:
    """Render Tavily API usage section."""
    st.markdown("### 🔍 Tavily Search API")

    if st.button("🔄 Refresh Usage Stats", key="refresh_tavily_usage"):
        with st.spinner("Fetching Tavily usage..."):
            usage = get_tavily_usage(config.tavily_api_key)

            if usage:
                # Display account-level usage
                st.info(f"**Current Plan:** {usage.current_plan}")

                # Plan usage progress bar
                plan_col1, plan_col2 = st.columns([3, 1])

                with plan_col1:
                    st.progress(
                        usage.plan_percentage / 100,
                        text=f"{usage.plan_usage:,} / {usage.plan_limit:,} requests ({usage.plan_percentage:.1f}%)",
                    )

                with plan_col2:
                    if usage.plan_percentage > USAGE_THRESHOLD_CRITICAL:
                        st.error("🔴 Critical")
                    elif usage.plan_percentage > USAGE_THRESHOLD_WARNING:
                        st.warning("🟡 High")
                    else:
                        st.success("🟢 Good")

                # Pay-as-you-go usage (if applicable)
                if usage.paygo_limit > 0:
                    st.markdown("**Pay-as-you-go Usage:**")
                    paygo_col1, paygo_col2 = st.columns([3, 1])

                    with paygo_col1:
                        st.progress(
                            usage.paygo_percentage / 100,
                            text=f"{usage.paygo_usage:,} / {usage.paygo_limit:,} requests ({usage.paygo_percentage:.1f}%)",
                        )

                    with paygo_col2:
                        if usage.paygo_percentage > USAGE_THRESHOLD_CRITICAL:
                            st.error("🔴 Critical")
                        elif usage.paygo_percentage > USAGE_THRESHOLD_WARNING:
                            st.warning("🟡 High")
                        else:
                            st.success("🟢 Good")

                # Display warnings
                if usage.is_critical:
                    st.error(
                        "⚠️ **Critical Usage Level!** You're approaching your API limits. "
                        "Consider upgrading your plan or reducing usage."
                    )
                elif usage.is_approaching_limit:
                    st.warning(
                        "⚠️ **High Usage Level.** You've used over 80% of your API limits. "
                        "Monitor your usage to avoid hitting the limit."
                    )

            else:
                st.error(
                    "❌ Failed to fetch Tavily usage. Please check your API key configuration."
                )

    st.info(
        "💡 **Tip:** Tavily usage resets on the 1st of each month. "
        "Click 'Refresh Usage Stats' to see your current usage."
    )


def _render_mathpix_usage() -> None:
    """Render Mathpix API usage section."""
    st.markdown("### 📐 Mathpix OCR API")

    if st.button("🔄 Refresh Usage Stats", key="refresh_mathpix_usage"):
        with st.spinner("Fetching Mathpix usage..."):
            usage = get_mathpix_usage(config.mathpix_api_key, config.mathpix_api_id)

            if usage:
                # Display monthly pages usage
                st.info("**Rate Limit:** 50 requests/minute")

                # Pages usage progress bar
                pages_col1, pages_col2 = st.columns([3, 1])

                with pages_col1:
                    st.progress(
                        usage.pages_percentage / 100,
                        text=f"{usage.pages_usage:,} / {usage.pages_limit:,} pages ({usage.pages_percentage:.1f}%)",
                    )

                with pages_col2:
                    if usage.pages_percentage > USAGE_THRESHOLD_CRITICAL:
                        st.error("🔴 Critical")
                    elif usage.pages_percentage > USAGE_THRESHOLD_WARNING:
                        st.warning("🟡 High")
                    else:
                        st.success("🟢 Good")

                # Display warnings
                if usage.is_critical:
                    st.error(
                        "⚠️ **Critical Usage Level!** You're approaching your API limits. "
                        "Consider upgrading your plan or reducing usage."
                    )
                elif usage.is_approaching_limit:
                    st.warning(
                        "⚠️ **High Usage Level.** You've used over 80% of your API limits. "
                        "Monitor your usage to avoid hitting the limit."
                    )

            else:
                st.error(
                    "❌ Failed to fetch Mathpix usage. Please check your API key configuration."
                )

    st.info(
        "💡 **Tip:** Mathpix usage resets on the 5th of each month. "
        "Click 'Refresh Usage Stats' to see your current usage."
    )


def _render_api_usage_section() -> None:
    """Render API usage monitoring section."""
    st.markdown("## 📊 API Usage")
    st.markdown("Monitor your external API usage and limits.")

    # Render Tavily usage
    _render_tavily_usage()

    # Render Mathpix usage
    _render_mathpix_usage()


def _render_about_section() -> None:
    """Render about section."""
    st.markdown("## About")
    st.markdown(
        """
        **EngiAI - Engineering Design Assistant**

        Version: 0.0.1

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
    _render_paper_import_settings()

    st.markdown("---")
    _render_api_usage_section()

    st.markdown("---")
    _render_about_section()


if __name__ == "__main__":
    render()
