"""Settings page for the Engineer Assistant Streamlit app."""

import json
import os
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config  # noqa: E402
from scripts.import_local_papers import LocalPaperImporter  # noqa: E402
from src.tools.connection import HPCConnection  # noqa: E402
from src.tools.hpc import (  # noqa: E402
    CREDENTIAL_EXPIRATION_SECONDS,
    clear_ssh_credentials,
    get_ssh_credentials,
    set_current_session_id,
    set_ssh_credentials,
)
from src.ui.database import DatabaseManager  # noqa: E402
from src.ui.pdf_export import create_pdf_from_conversation  # noqa: E402
from src.utils.api_usage import (  # noqa: E402
    USAGE_THRESHOLD_CRITICAL,
    USAGE_THRESHOLD_WARNING,
    get_tavily_usage,
)

# Constants
SECONDS_PER_HOUR = 3600
SESSION_WARNING_AGE_HOURS = 24


def _inject_custom_css() -> None:
    """Inject custom CSS for card-based layout."""
    st.markdown(
        """
        <style>
            /* Card container styling */
            .settings-card {
                background-color: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                transition: all 0.3s ease;
            }

            .settings-card:hover {
                border-color: rgba(255, 255, 255, 0.2);
                background-color: rgba(255, 255, 255, 0.03);
            }

            /* Card title styling */
            .card-title {
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid rgba(255, 255, 255, 0.1);
            }

            /* Compact spacing for cards */
            .settings-card .stCheckbox {
                margin-top: 0.5rem;
            }

            .settings-card .stButton {
                margin-top: 0.5rem;
            }

            /* Remove excessive padding */
            .settings-card > div > div {
                gap: 0.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
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
        "enable_streaming": True,
        "job_monitor_refresh_interval": 60,  # seconds
        "job_monitor_auto_add": False,  # Ask before monitoring by default
        # SLURM/HPC configuration
        "slurm_venv_path": "~/venvs/engineer_assistant",
        "slurm_project_path": "$HOME/EngiOpt",
        "slurm_email_user": "",
        "slurm_logs_dir": "$SCRATCH/logs",
        "slurm_wandb_entity": "",
        "slurm_wandb_project": "engiopt",
        "hf_home_remote": "$SCRATCH/models",
        "hf_datasets_cache_remote": "$SCRATCH/datasets",
        # Voice interaction settings
        "voice_enabled": False,
        "voice_auto_play": True,
        "voice_selected": "George",
        "voice_input_enabled": True,
        "voice_output_enabled": True,
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


def _count_files_in_directory(directory: Path) -> int:
    """Count files in a directory recursively.

    Args:
        directory: Directory path to count files in

    Returns:
        Number of files in the directory
    """
    if not directory.exists():
        return 0
    try:
        return sum(1 for item in directory.rglob("*") if item.is_file())
    except Exception:
        return 0


def _delete_output_folder_contents(output_dir: Path) -> int:
    """Delete all contents of the output folder.

    Args:
        output_dir: Directory to clear

    Returns:
        Number of files deleted
    """
    deleted_count = 0
    if not output_dir.exists():
        return deleted_count

    # Delete all files and subdirectories
    for item in output_dir.rglob("*"):
        if item.is_file():
            item.unlink()
            deleted_count += 1

    # Remove empty subdirectories
    for item in sorted(output_dir.rglob("*"), reverse=True):
        if item.is_dir() and not any(item.iterdir()):
            item.rmdir()

    return deleted_count


def _get_file_age_hours(file_path: Path) -> float:
    """Get the age of a file in hours.

    Args:
        file_path: Path to the file

    Returns:
        Age of the file in hours, or 0 if file doesn't exist
    """
    if not file_path.exists():
        return 0.0
    try:
        file_mtime = file_path.stat().st_mtime
        current_time = time.time()
        age_seconds = current_time - file_mtime
        return age_seconds / 3600  # Convert to hours
    except Exception:
        return 0.0


def _cleanup_old_connect_state(
    max_age_hours: int = SESSION_WARNING_AGE_HOURS,
) -> tuple[bool, str]:
    """Clean up connect_state.json if it's older than max_age_hours.

    Args:
        max_age_hours: Maximum age in hours before cleanup (default: 24)

    Returns:
        Tuple of (was_deleted, message)
    """
    connect_state_path = project_root / "data" / "connect_state.json"

    if not connect_state_path.exists():
        return False, "File doesn't exist"

    file_age = _get_file_age_hours(connect_state_path)

    if file_age > max_age_hours:
        try:
            connect_state_path.unlink()
        except Exception as e:
            return False, f"Failed to delete: {e}"
        else:
            return True, f"Deleted session file (was {file_age:.1f} hours old)"
    else:
        return (
            False,
            f"File is {file_age:.1f} hours old (keeping until {max_age_hours}h)",
        )


def _render_export_button(
    label: str, data: str | bytes, filename: str, mime: str, has_messages: bool
) -> None:
    """Render an export button with consistent styling.

    Args:
        label: Button label
        data: Data to download
        filename: Download filename
        mime: MIME type
        has_messages: Whether there are messages to export
    """
    if has_messages:
        st.download_button(
            label,
            data,
            filename,
            mime=mime,
            use_container_width=True,
            type="secondary",
        )
    else:
        st.button(
            label,
            use_container_width=True,
            type="secondary",
            disabled=True,
        )


def _get_pdf_export_data() -> tuple[str, datetime | None]:
    """Get conversation data for PDF export.

    Returns:
        Tuple of (conversation_name, created_at)
    """
    conversation_name = "Untitled Conversation"
    created_at = None

    if st.session_state.get("active_chat_id"):
        db = DatabaseManager()
        conv = db.get_conversation(st.session_state["active_chat_id"])
        if conv:
            conversation_name = conv.get("name", "Untitled Conversation")
            created_at = conv.get("created_at")

    return conversation_name, created_at


def _render_chat_quick_settings() -> None:
    """Render compact chat settings for card layout."""
    # Streaming text toggle
    enable_streaming = st.checkbox(
        "Enable streaming text display",
        value=st.session_state.get("enable_streaming", True),
        help="Display AI responses with a typewriter effect. Disable for instant display.",
        key="enable_streaming_card_widget",
    )
    if enable_streaming != st.session_state.get("enable_streaming"):
        _save_setting_to_db("enable_streaming", enable_streaming)

    # Job monitor refresh interval
    refresh_interval = st.number_input(
        "Job Monitor Refresh (seconds)",
        min_value=10,
        max_value=300,
        value=st.session_state.get("job_monitor_refresh_interval", 60),
        step=10,
        help="How often to check SLURM job status (10-300 seconds)",
        key="job_monitor_refresh_card_widget",
    )
    if refresh_interval != st.session_state.get("job_monitor_refresh_interval"):
        _save_setting_to_db("job_monitor_refresh_interval", int(refresh_interval))

    # Auto-add jobs to monitor
    auto_add_jobs = st.checkbox(
        "Automatically monitor submitted jobs",
        value=st.session_state.get("job_monitor_auto_add", False),
        help="If enabled, jobs will be automatically added to the monitor without asking. If disabled, you'll be prompted each time.",
        key="job_monitor_auto_add_card_widget",
    )
    if auto_add_jobs != st.session_state.get("job_monitor_auto_add"):
        _save_setting_to_db("job_monitor_auto_add", auto_add_jobs)

    # Display current conversation stats
    if st.session_state.get("messages"):
        num_messages = len(st.session_state.messages)
        st.caption(f"📊 Current: {num_messages} messages")

    st.markdown("")  # Spacing

    # Conversation management buttons - Row 1: Exports
    col1, col2, col3 = st.columns(3)
    has_messages = bool(st.session_state.get("messages"))

    with col1:
        # JSON export
        json_data = (
            json.dumps(st.session_state.messages, indent=2) if has_messages else ""
        )
        _render_export_button(
            "📥 Export JSON",
            json_data,
            "chat_export.json",
            "application/json",
            has_messages,
        )

    with col2:
        # Markdown export
        md_content = ""
        if has_messages:
            md_content = "# Chat Export\n\n"
            for msg in st.session_state.messages:
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "")
                md_content += f"## {role}\n\n{content}\n\n---\n\n"
        _render_export_button(
            "📄 Export MD", md_content, "chat_export.md", "text/markdown", has_messages
        )

    with col3:
        # PDF export
        pdf_bytes = b""
        if has_messages:
            conversation_name, created_at = _get_pdf_export_data()
            pdf_bytes = create_pdf_from_conversation(
                conversation_name=conversation_name,
                messages=st.session_state.messages,
                created_at=created_at,
            )
        _render_export_button(
            "📑 Export PDF",
            pdf_bytes,
            "chat_export.pdf",
            "application/pdf",
            has_messages,
        )

    # Row 2: Clear button
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.agent_state = {"messages": []}
            st.success("✅ Conversation cleared!")
            st.rerun()


def _render_stl_quick_settings() -> None:
    """Render compact 3D viewer settings for card layout."""
    col1, col2 = st.columns(2)

    with col1:
        # Color picker
        stl_color = st.color_picker(
            "Color",
            value=st.session_state.stl_color,
            help="Choose a color for your 3D models",
            key="color_picker_card_widget",
        )
        if stl_color != st.session_state.stl_color:
            _save_setting_to_db("stl_color", stl_color)

        # Opacity slider
        stl_opacity = st.slider(
            "Opacity",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.stl_opacity,
            step=0.1,
            key="opacity_slider_card_widget",
        )
        if stl_opacity != st.session_state.stl_opacity:
            _save_setting_to_db("stl_opacity", stl_opacity)

    with col2:
        # Material selector
        material_options = ["material", "flat", "wireframe"]
        material_index = material_options.index(st.session_state.stl_material)
        stl_material = st.selectbox(
            "Material",
            options=material_options,
            index=material_index,
            key="material_selector_card_widget",
        )
        if stl_material != st.session_state.stl_material:
            _save_setting_to_db("stl_material", stl_material)

        # Auto-rotate toggle
        stl_auto_rotate = st.checkbox(
            "Auto-rotate",
            value=st.session_state.stl_auto_rotate,
            key="auto_rotate_checkbox_card_widget",
        )
        if stl_auto_rotate != st.session_state.stl_auto_rotate:
            _save_setting_to_db("stl_auto_rotate", stl_auto_rotate)


def _render_voice_settings_card() -> None:
    """Render voice interaction settings for card layout."""
    from src.ui.voices import (  # noqa: PLC0415
        ELEVENLABS_DEFAULT_VOICE,
        ELEVENLABS_VOICE_IDS,
        OPENAI_TTS_VOICE,
        OPENAI_TTS_VOICES,
        VOICE_PROVIDER,
    )

    # Voice enabled toggle
    voice_enabled = st.checkbox(
        "Enable voice features",
        value=st.session_state.get("voice_enabled", False),
        help="Enable voice input (STT) and output (TTS) using ElevenLabs or OpenAI",
        key="voice_enabled_card_widget",
    )
    if voice_enabled != st.session_state.get("voice_enabled"):
        _save_setting_to_db("voice_enabled", voice_enabled)

    if voice_enabled:
        # Voice provider selection (top level)
        st.markdown("**Voice Provider**")
        provider_options = ["ElevenLabs", "OpenAI"]
        current_provider = st.session_state.get("voice_provider", VOICE_PROVIDER)
        # Map internal format to display format
        provider_display = (
            "ElevenLabs" if current_provider.lower() == "elevenlabs" else "OpenAI"
        )

        selected_provider_display = st.radio(
            "Select voice provider",
            options=provider_options,
            index=provider_options.index(provider_display),
            horizontal=True,
            help="Choose between ElevenLabs or OpenAI for voice services",
            key="voice_provider_card_widget",
            label_visibility="collapsed",
        )

        # Convert display format back to internal format
        selected_provider = selected_provider_display.lower()
        if (
            selected_provider
            != st.session_state.get("voice_provider", VOICE_PROVIDER).lower()
        ):
            _save_setting_to_db("voice_provider", selected_provider)
            # Reset voice selection when provider changes
            if selected_provider == "openai":
                _save_setting_to_db("voice_selected", OPENAI_TTS_VOICE)
            else:
                _save_setting_to_db("voice_selected", ELEVENLABS_DEFAULT_VOICE)

        # Voice selection (based on provider)
        if selected_provider == "openai":
            # OpenAI voices
            selected_voice = st.selectbox(
                "Assistant Voice",
                options=OPENAI_TTS_VOICES,
                index=OPENAI_TTS_VOICES.index(
                    st.session_state.get("voice_selected", OPENAI_TTS_VOICE)
                    if st.session_state.get("voice_selected", OPENAI_TTS_VOICE)
                    in OPENAI_TTS_VOICES
                    else OPENAI_TTS_VOICE
                ),
                help="Choose the OpenAI voice for AI responses",
                key="voice_selected_openai_card_widget",
            )
        else:
            # ElevenLabs voices
            selected_voice = st.selectbox(
                "Assistant Voice",
                options=list(ELEVENLABS_VOICE_IDS.keys()),
                index=list(ELEVENLABS_VOICE_IDS.keys()).index(
                    st.session_state.get("voice_selected", ELEVENLABS_DEFAULT_VOICE)
                    if st.session_state.get("voice_selected", ELEVENLABS_DEFAULT_VOICE)
                    in ELEVENLABS_VOICE_IDS
                    else ELEVENLABS_DEFAULT_VOICE
                ),
                help="Choose the ElevenLabs voice for AI responses",
                key="voice_selected_elevenlabs_card_widget",
            )

        if selected_voice != st.session_state.get("voice_selected"):
            _save_setting_to_db("voice_selected", selected_voice)

        st.markdown("**Options**")
        col1, col2 = st.columns(2)

        with col1:
            # Voice input toggle
            voice_input = st.checkbox(
                "Enable microphone input",
                value=st.session_state.get("voice_input_enabled", True),
                help="Allow sending voice messages via microphone (Speech-to-Text)",
                key="voice_input_card_widget",
            )
            if voice_input != st.session_state.get("voice_input_enabled"):
                _save_setting_to_db("voice_input_enabled", voice_input)

            # Voice output toggle
            voice_output = st.checkbox(
                "Enable voice responses",
                value=st.session_state.get("voice_output_enabled", True),
                help="Generate audio for AI responses (Text-to-Speech)",
                key="voice_output_card_widget",
            )
            if voice_output != st.session_state.get("voice_output_enabled"):
                _save_setting_to_db("voice_output_enabled", voice_output)

        with col2:
            # Auto-play toggle
            auto_play = st.checkbox(
                "Auto-play responses",
                value=st.session_state.get("voice_auto_play", True),
                help="Automatically play audio when assistant responds",
                key="voice_auto_play_card_widget",
            )
            if auto_play != st.session_state.get("voice_auto_play"):
                _save_setting_to_db("voice_auto_play", auto_play)
    else:
        st.info("Enable voice features to access voice settings")


def _render_media_quick_settings() -> None:
    """Render compact media settings for card layout."""
    # Media save directory input
    media_save_dir = st.text_input(
        "Save directory",
        value=st.session_state.media_save_dir,
        help="Where displayed media will be saved on the server",
        key="media_save_dir_card_widget",
    )
    if media_save_dir != st.session_state.media_save_dir:
        _save_setting_to_db("media_save_dir", media_save_dir)

    # Media auto-save checkbox
    media_auto_save = st.checkbox(
        "Auto-save displayed media",
        value=st.session_state.media_auto_save,
        help="Automatically save images and STL files to the save directory",
        key="media_auto_save_card_widget",
    )
    if media_auto_save != st.session_state.media_auto_save:
        _save_setting_to_db("media_auto_save", media_auto_save)


def _render_output_folder_clear() -> None:
    """Render output folder clearing UI with confirmation."""
    output_dir = Path(st.session_state.media_save_dir)
    file_count = _count_files_in_directory(output_dir)

    st.caption(f"📁 Output folder: {file_count} file(s)")

    # Initialize confirmation state
    if "confirm_clear_output_card" not in st.session_state:
        st.session_state.confirm_clear_output_card = False

    if not st.session_state.confirm_clear_output_card:
        _render_clear_button(file_count)
    else:
        _render_confirmation_dialog(output_dir, file_count)


def _render_clear_button(file_count: int) -> None:
    """Render the initial clear folder button."""
    if st.button("🗑️ Clear Output Folder", width="stretch", type="secondary"):
        if file_count == 0:
            st.warning("⚠️ Folder is empty!")
        else:
            st.session_state.confirm_clear_output_card = True
            st.rerun()


def _render_confirmation_dialog(output_dir: Path, file_count: int) -> None:
    """Render the confirmation dialog for clearing folder."""
    st.warning(f"Delete {file_count} file(s)?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes", width="stretch", type="primary"):
            _handle_folder_deletion(output_dir)

    with col2:
        if st.button("❌ No", width="stretch", type="secondary"):
            st.session_state.confirm_clear_output_card = False
            st.rerun()


def _handle_folder_deletion(output_dir: Path) -> None:
    """Handle the folder deletion operation."""
    try:
        deleted_count = _delete_output_folder_contents(output_dir)
        st.session_state.confirm_clear_output_card = False
        st.success(f"✅ Deleted {deleted_count} file(s)!")
        st.rerun()
    except Exception as e:
        st.session_state.confirm_clear_output_card = False
        st.error(f"❌ Error: {e}")
        st.rerun()


def _render_prusa_session_compact() -> None:
    """Render compact Prusa Connect session info."""
    connect_state_path = project_root / "data" / "connect_state.json"

    if connect_state_path.exists():
        file_age = _get_file_age_hours(connect_state_path)
        st.caption(f"🔌 Prusa session: {file_age:.1f}h old")
        if st.button(
            "🗑️ Clear Session", width="stretch", type="secondary", key="clear_prusa_card"
        ):
            _handle_prusa_session_clear(connect_state_path)
    else:
        st.caption("🔌 No Prusa session")


def _handle_prusa_session_clear(connect_state_path: Path) -> None:
    """Handle Prusa Connect session clearing."""
    try:
        connect_state_path.unlink()
        st.success("✅ Session cleared!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error: {e}")


def _render_artifact_folder_clear() -> None:
    """Render artifact folder clearing UI with confirmation."""
    artifact_dir = project_root / "artifacts"
    file_count = _count_files_in_directory(artifact_dir)

    st.caption(f"📦 Artifact folder: {file_count} file(s)")

    # Initialize confirmation state
    if "confirm_clear_artifact_card" not in st.session_state:
        st.session_state.confirm_clear_artifact_card = False

    if not st.session_state.confirm_clear_artifact_card:
        if st.button("🗑️ Clear Artifact Folder", width="stretch", type="secondary"):
            if file_count == 0:
                st.warning("⚠️ Folder is empty!")
            else:
                st.session_state.confirm_clear_artifact_card = True
                st.rerun()
    else:
        st.warning(f"Delete {file_count} file(s)?")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Yes", width="stretch", type="primary", key="artifact_yes"):
                try:
                    deleted_count = _delete_output_folder_contents(artifact_dir)
                    st.session_state.confirm_clear_artifact_card = False
                    st.success(f"✅ Deleted {deleted_count} file(s)!")
                    st.rerun()
                except Exception as e:
                    st.session_state.confirm_clear_artifact_card = False
                    st.error(f"❌ Error: {e}")
                    st.rerun()

        with col2:
            if st.button("❌ No", width="stretch", type="secondary", key="artifact_no"):
                st.session_state.confirm_clear_artifact_card = False
                st.rerun()


def _render_file_management_card() -> None:
    """Render compact file management for card layout."""
    # Two columns for output and artifact folders
    col1, col2 = st.columns(2)

    with col1:
        _render_output_folder_clear()

    with col2:
        _render_artifact_folder_clear()

    st.markdown("")  # Spacing
    _render_prusa_session_compact()


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


def _render_ssh_credentials_section() -> None:  # noqa: PLR0912, PLR0915
    """Render SSH/HPC credentials section."""
    st.markdown("## 🔐 HPC Connection")
    st.markdown(
        "Configure SSH credentials for connecting to HPC clusters. "
        "You can use either password authentication or your existing SSH config."
    )

    # Ensure we have a session ID for credential isolation
    if "session_id" not in st.session_state:
        st.session_state.session_id = secrets.token_hex(16)

    # Set the current session ID for the HPC module
    set_current_session_id(st.session_state.session_id)

    # Check current status
    current_creds = get_ssh_credentials(st.session_state.session_id)

    if current_creds:
        expiry_minutes = CREDENTIAL_EXPIRATION_SECONDS // 60
        st.success(
            f"✅ Password authentication configured for **{current_creds['user']}@{current_creds['host']}**\n\n"
            f"Credentials expire in {expiry_minutes} minutes from when they were set."
        )
        if st.button("🗑️ Clear Credentials", key="clear_ssh_creds"):
            clear_ssh_credentials(st.session_state.session_id)
            st.success("Credentials cleared. Will use SSH config for authentication.")
            st.rerun()
    else:
        st.info(
            "Currently using SSH config (`~/.ssh/config`). "
            "Configure password authentication below if SSH config is not available."
        )

    # Auth mode selector
    auth_mode = st.radio(
        "Authentication Mode",
        options=["SSH Config (default)", "Password Authentication"],
        index=0 if not current_creds else 1,
        help="SSH Config uses your ~/.ssh/config file with SSH keys. "
        "Password Authentication uses username/password directly.",
        key="ssh_auth_mode",
    )

    if auth_mode == "Password Authentication":
        st.markdown("### Enter Credentials")

        col1, col2 = st.columns(2)

        with col1:
            ssh_host = st.text_input(
                "Hostname",
                value=current_creds["host"] if current_creds else "euler.ethz.ch",
                placeholder="euler.ethz.ch",
                help="Full hostname of the HPC cluster",
                key="ssh_host_input",
            )

            ssh_user = st.text_input(
                "Username",
                value=current_creds["user"] if current_creds else "",
                placeholder="your_username",
                help="Your HPC cluster username",
                key="ssh_user_input",
            )

        with col2:
            ssh_port = st.number_input(
                "Port",
                min_value=1,
                max_value=65535,
                value=current_creds["port"] if current_creds else 22,
                help="SSH port (usually 22)",
                key="ssh_port_input",
            )

            ssh_password = st.text_input(
                "Password",
                type="password",
                value="",
                placeholder="Enter password",
                help="Your HPC cluster password (stored in memory only)",
                key="ssh_password_input",
            )

        col_save, col_test = st.columns(2)

        with col_save:
            if st.button(
                "💾 Save Credentials", type="primary", use_container_width=True
            ):
                if ssh_host and ssh_user and ssh_password:
                    success, message = set_ssh_credentials(
                        host=ssh_host,
                        user=ssh_user,
                        password=ssh_password,
                        port=int(ssh_port),
                        session_id=st.session_state.session_id,
                    )
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.error("Please fill in all fields (host, username, and password)")

        with col_test:
            if st.button("🔌 Test Connection", use_container_width=True):
                if ssh_host and ssh_user and ssh_password:
                    with st.spinner("Testing connection..."):
                        try:
                            # Validate and set credentials for test
                            success, message = set_ssh_credentials(
                                host=ssh_host,
                                user=ssh_user,
                                password=ssh_password,
                                port=int(ssh_port),
                                session_id=st.session_state.session_id,
                            )

                            if not success:
                                st.error(f"❌ {message}")
                            else:
                                # Try to connect
                                hpc = HPCConnection(
                                    host=ssh_host,
                                    user=ssh_user,
                                    password=ssh_password,
                                    port=int(ssh_port),
                                )
                                result = hpc.run_command("pwd")
                                st.success(
                                    f"✅ Connection successful! Remote directory: `{result}`"
                                )
                        except Exception as e:
                            # Sanitize error message - don't expose full details
                            error_str = str(e)
                            if "Authentication failed" in error_str:
                                st.error(
                                    "❌ Authentication failed. Check username and password."
                                )
                            elif "timed out" in error_str.lower():
                                st.error(
                                    "❌ Connection timed out. Check hostname and network."
                                )
                            elif "refused" in error_str.lower():
                                st.error(
                                    "❌ Connection refused. Check hostname and port."
                                )
                            else:
                                st.error(
                                    "❌ Connection failed. Check your credentials and network."
                                )
                else:
                    st.error("Please fill in all fields before testing")

        # Format expiration time appropriately
        if CREDENTIAL_EXPIRATION_SECONDS >= SECONDS_PER_HOUR:
            expiry_time = f"{CREDENTIAL_EXPIRATION_SECONDS // SECONDS_PER_HOUR} hour(s)"
        else:
            expiry_time = f"{CREDENTIAL_EXPIRATION_SECONDS // 60} minutes"

        st.warning(
            f"⚠️ **Security Note:** Credentials are stored in session memory only and will be "
            f"cleared when you close the browser or after {expiry_time}. They are not saved to disk."
        )

    else:
        # SSH Config mode
        st.markdown("### SSH Config Mode")
        st.markdown(
            f"Using host alias: **{config.hpc_host_alias}** (from `HPC_HOST_ALIAS` in `.env`)"
        )

        if st.button("🔌 Test SSH Config Connection", use_container_width=True):
            with st.spinner("Testing connection..."):
                try:
                    # Clear any password credentials to use SSH config
                    clear_ssh_credentials()

                    hpc = HPCConnection(host_alias=config.hpc_host_alias)
                    result = hpc.run_command("pwd")
                    st.success(
                        f"✅ Connection successful! Remote directory: `{result}`"
                    )
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")
                    st.info(
                        "Make sure:\n"
                        f"1. Host `{config.hpc_host_alias}` is configured in `~/.ssh/config`\n"
                        "2. SSH key is set up correctly\n"
                        "3. SSH agent is running (if using Docker, mount `SSH_AUTH_SOCK`)"
                    )


def _render_slurm_config_section() -> None:
    """Render SLURM/HPC cluster configuration section."""
    st.markdown("## ⚙️ SLURM Cluster Configuration")
    st.markdown(
        "Configure paths and settings for SLURM job submission on HPC clusters. "
        "These settings are used when generating training jobs."
    )

    col1, col2 = st.columns(2)

    with col1:
        # Virtual environment path
        slurm_venv_path = st.text_input(
            "Virtual Environment Path",
            value=st.session_state.get("slurm_venv_path", "~/venvs/engineer_assistant"),
            placeholder="~/venvs/engineer_assistant",
            help="Path to Python virtual environment on the HPC cluster",
            key="slurm_venv_path_widget",
        )
        if slurm_venv_path != st.session_state.get("slurm_venv_path"):
            _save_setting_to_db("slurm_venv_path", slurm_venv_path)

        # EngiOpt path
        slurm_project_path = st.text_input(
            "EngiOpt Path",
            value=st.session_state.get("slurm_project_path", "$HOME/EngiOpt"),
            placeholder="$HOME/EngiOpt",
            help="Path to EngiOpt project directory on the HPC cluster",
            key="slurm_project_path_widget",
        )
        if slurm_project_path != st.session_state.get("slurm_project_path"):
            _save_setting_to_db("slurm_project_path", slurm_project_path)

        # Email user
        slurm_email_user = st.text_input(
            "Email for Job Notifications",
            value=st.session_state.get("slurm_email_user", ""),
            placeholder="username@ethz.ch",
            help="Email address for SLURM job notifications (start, end, fail)",
            key="slurm_email_user_widget",
        )
        if slurm_email_user != st.session_state.get("slurm_email_user"):
            _save_setting_to_db("slurm_email_user", slurm_email_user)

        # WandB Entity
        slurm_wandb_entity = st.text_input(
            "WandB Entity",
            value=st.session_state.get("slurm_wandb_entity", ""),
            placeholder="your-wandb-entity",
            help="WandB entity/username for training job tracking",
            key="slurm_wandb_entity_widget",
        )
        if slurm_wandb_entity != st.session_state.get("slurm_wandb_entity"):
            _save_setting_to_db("slurm_wandb_entity", slurm_wandb_entity)

        # WandB Project
        slurm_wandb_project = st.text_input(
            "WandB Project",
            value=st.session_state.get("slurm_wandb_project", "engiopt"),
            placeholder="engiopt",
            help="WandB project name for training job tracking",
            key="slurm_wandb_project_widget",
        )
        if slurm_wandb_project != st.session_state.get("slurm_wandb_project"):
            _save_setting_to_db("slurm_wandb_project", slurm_wandb_project)

    with col2:
        # Logs directory
        slurm_logs_dir = st.text_input(
            "Logs Directory",
            value=st.session_state.get("slurm_logs_dir", "$SCRATCH/logs"),
            placeholder="$SCRATCH/logs",
            help="Path to directory for SLURM job logs on HPC cluster ($SCRATCH will be expanded)",
            key="slurm_logs_dir_widget",
        )
        if slurm_logs_dir != st.session_state.get("slurm_logs_dir"):
            _save_setting_to_db("slurm_logs_dir", slurm_logs_dir)

        # HuggingFace home remote
        hf_home_remote = st.text_input(
            "HuggingFace Cache (Remote)",
            value=st.session_state.get("hf_home_remote", "$SCRATCH/models"),
            placeholder="$SCRATCH/models",
            help="Path to HuggingFace model cache on HPC cluster ($SCRATCH will be expanded)",
            key="hf_home_remote_widget",
        )
        if hf_home_remote != st.session_state.get("hf_home_remote"):
            _save_setting_to_db("hf_home_remote", hf_home_remote)

        # HuggingFace datasets cache remote
        hf_datasets_cache_remote = st.text_input(
            "HuggingFace Datasets Cache (Remote)",
            value=st.session_state.get("hf_datasets_cache_remote", "$SCRATCH/datasets"),
            placeholder="$SCRATCH/datasets",
            help="Path to HuggingFace datasets cache on HPC cluster ($SCRATCH will be expanded)",
            key="hf_datasets_cache_remote_widget",
        )
        if hf_datasets_cache_remote != st.session_state.get("hf_datasets_cache_remote"):
            _save_setting_to_db("hf_datasets_cache_remote", hf_datasets_cache_remote)

    st.info(
        "💡 **Note:** These paths are used when generating SLURM batch scripts. "
        "Shell variables like `$HOME`, `$SCRATCH` will be expanded on the cluster. "
        "Use `~` for home directory references."
    )


def _render_api_usage_tabs() -> None:
    """Render API usage with tabbed interface."""
    _render_tavily_usage()


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
    """Render the settings page with card-based layout."""
    # Load settings from database on first render
    _load_settings_from_db()

    # Inject custom CSS
    _inject_custom_css()

    # Auto-cleanup old Prusa Connect session on page load (runs once per session)
    if "session_cleanup_done" not in st.session_state:
        was_deleted, message = _cleanup_old_connect_state(
            max_age_hours=SESSION_WARNING_AGE_HOURS
        )
        if was_deleted:
            st.toast(f"🔄 {message}", icon="🔄")
        st.session_state.session_cleanup_done = True

    st.markdown("# ⚙️ Settings")
    st.markdown("Configure your preferences and application settings.")
    st.markdown("---")

    # === QUICK SETTINGS (2-column cards) ===
    st.markdown("### ⚡ Quick Settings")
    col1, col2 = st.columns(2, gap="medium")

    with col1, st.container():
        st.markdown(
            '<div class="card-title">💬 Chat & UI</div>', unsafe_allow_html=True
        )
        _render_chat_quick_settings()

    with col2, st.container():
        st.markdown(
            '<div class="card-title">🎨 3D Viewer</div>', unsafe_allow_html=True
        )
        _render_stl_quick_settings()

    st.markdown("")  # Spacing

    # === VOICE & MEDIA SETTINGS (2-column cards) ===
    col5, col6 = st.columns(2, gap="medium")

    with col5, st.container():
        st.markdown(
            '<div class="card-title">🎤 Voice Interaction</div>', unsafe_allow_html=True
        )
        _render_voice_settings_card()

    with col6, st.container():
        st.markdown(
            '<div class="card-title">💾 Media Saving</div>', unsafe_allow_html=True
        )
        _render_media_quick_settings()

        with st.container():
            st.markdown(
                '<div class="card-title">🗂️ File Management</div>',
                unsafe_allow_html=True,
            )

            _render_file_management_card()

    st.markdown("")  # Spacing

    st.markdown("---")

    # === HPC CONNECTION (full width) ===
    _render_ssh_credentials_section()

    st.markdown("---")

    # === SLURM CLUSTER CONFIGURATION (full width) ===
    _render_slurm_config_section()

    st.markdown("---")

    # === API USAGE (full width, tabbed) ===
    st.markdown("## 📊 API Usage Monitor")
    st.markdown("Track your external API usage and limits.")
    _render_api_usage_tabs()

    st.markdown("---")

    # === ADVANCED SETTINGS (collapsible) ===
    with st.expander("📚 Paper Import (Advanced)", expanded=False):
        _render_paper_import_settings()

    with st.expander("About", expanded=False):
        _render_about_section()


if __name__ == "__main__":
    render()
