"""
Streamlit UI for the Engineer Assistant chatbot.

This provides a web-based chat interface for interacting with the multi-agent system.
"""

import base64
import datetime
import logging
import re
import secrets
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, cast

import streamlit as st
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config  # noqa: E402
from src.tools import MMOREClient  # noqa: E402
from src.tools.hpc import set_current_session_id, set_progress_callback  # noqa: E402
from src.tools.mmore_client import (  # noqa: E402
    set_progress_callback as set_mmore_progress_callback,
)
from src.ui import chat, home, settings, wandb_report  # noqa: E402
from src.ui.chat_management import (  # noqa: E402
    create_new_chat,
    delete_chat,
    get_db,
    initialize_chat_state,
    load_chats_from_database,
    save_active_chat_to_storage,
    switch_to_chat,
    toggle_pin_chat,
)
from src.ui.confirmation_handler import (  # noqa: E402
    check_streamlit_interrupt,
    handle_confirmation_response,
    show_confirmation_prompt,
)
from src.ui.database import DatabaseManager  # noqa: E402
from src.ui.file_processing import (  # noqa: E402
    process_uploaded_images,
)
from src.ui.message_processing import (  # noqa: E402
    format_and_display_messages,
)
from src.utils.api_usage import (  # noqa: E402
    UNLIMITED_LIMIT_VALUE,
    USAGE_THRESHOLD_CRITICAL,
    USAGE_THRESHOLD_WARNING,
    get_tavily_usage,
)

# Suppress Pydantic warnings from LangChain
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic._internal._generate_schema"
)


def _initialize_stl_settings() -> None:
    """Initialize STL viewer settings from database."""
    # Get database manager
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

    db = st.session_state.db_manager

    # Default values
    defaults: dict[str, Any] = {
        "stl_color": "#0069B4",
        "stl_material": "material",
        "stl_height": 400,
        "stl_auto_rotate": True,
        "stl_opacity": 1.0,
        "stl_shininess": 100,
        "media_save_dir": str(Path(__file__).parent.parent.parent / "outputs"),
        "media_auto_save": False,
        "enable_streaming": True,
        "job_monitor_refresh_interval": 60,  # seconds
        "job_monitor_auto_add": False,  # Ask before monitoring by default
    }

    # Load from database or use defaults
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = db.get_setting(key, value)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    # Initialize chat state
    initialize_chat_state()

    # Try to load chats from database on first run
    if "chats_loaded" not in st.session_state:
        load_chats_from_database()
        st.session_state.chats_loaded = True

    # Create first chat if none exists
    if not st.session_state.chats:
        create_new_chat()

    # Page navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"

    # Initialize session ID for SSH credential isolation
    if "session_id" not in st.session_state:
        st.session_state.session_id = secrets.token_hex(16)

    # Set session ID for HPC tools
    set_current_session_id(st.session_state.session_id)

    # Initialize STL viewer settings
    _initialize_stl_settings()


def process_user_input(user_input: str | dict[str, Any] | Any) -> None:  # noqa: PLR0912, PLR0915
    """Process user input and generate response.

    Args:
        user_input: The user's message (string) or dict with 'text' and 'files' keys
    """
    # Parse input - handle both string and dict formats
    # st.chat_input with files returns a ChatInputValue object with 'text' and 'files' attributes
    if isinstance(user_input, str):
        text_content = user_input
        files = []
    elif hasattr(user_input, "text") and hasattr(user_input, "files"):
        # ChatInputValue object from Streamlit
        text_content = str(user_input.text) if user_input.text else ""
        files = list(user_input.files) if user_input.files else []
    elif isinstance(user_input, dict):
        text_content = user_input.get("text", "")
        files = user_input.get("files", [])
    else:
        # Fallback: convert to string
        text_content = str(user_input)
        files = []

    # Process images if any
    images_for_display: list[dict[str, str]] = []
    images_for_agent: list[dict[str, Any]] = []
    if files:
        images_for_display, images_for_agent = process_uploaded_images(files)

    # Check if we're waiting for confirmation from a previous interrupt
    if st.session_state.waiting_for_confirmation:
        handle_confirmation_response(text_content)
        save_active_chat_to_storage()  # Save after confirmation response
        return

    # Check if this is a simple "add to knowledge" request with PDF attachments
    has_pdfs = any(f.get("type") == "application/pdf" for f in images_for_display)
    is_knowledge_upload = has_pdfs and text_content.lower().strip() in [
        "add to my knowledge",
        "add to knowledge",
        "add to my knowledge base",
        "add to knowledge base",
        "upload to knowledge",
        "upload to knowledge base",
    ]

    # Normal flow - add user message to display
    message_dict: dict[str, Any] = {"role": "user", "content": text_content}
    if images_for_display:
        message_dict["images"] = images_for_display
    st.session_state.messages.append(message_dict)

    # Save user message to database (strip 'bytes' field - only store base64)
    if st.session_state.active_chat_id:
        db = get_db()
        # Remove 'bytes' field before saving to database (only keep base64)
        images_for_db = (
            [
                {k: v for k, v in img.items() if k != "bytes"}
                for img in images_for_display
            ]
            if images_for_display
            else None
        )
        db.add_message(
            conversation_id=st.session_state.active_chat_id,
            role="user",
            content=text_content,
            images=images_for_db,
        )

    # Filter out PDFs from images_for_agent (PDFs are handled separately via text extraction)
    non_pdf_images = (
        [
            img
            for img in images_for_agent
            if not any(
                f.get("type") == "application/pdf"
                and f.get("data") in img.get("image_url", {}).get("url", "")
                for f in images_for_display
            )
        ]
        if has_pdfs
        else images_for_agent
    )

    # Create LangChain message with multimodal content if images present (no PDFs)
    if non_pdf_images:
        # Create multimodal content: [text, image1, image2, ...]
        message_content: list[dict[str, Any]] = [
            {"type": "text", "text": text_content},
            *non_pdf_images,
        ]
        human_message = HumanMessage(
            content=cast(str | list[str | dict[Any, Any]], message_content)
        )
    else:
        human_message = HumanMessage(content=text_content)

    # Add user message to agent state (PDFs will be added via text extraction later)
    st.session_state.agent_state["messages"].append(human_message)

    # Display user message
    with st.chat_message("user"):
        st.markdown(text_content)
        # Display uploaded files
        if images_for_display:
            for file_data in images_for_display:
                file_type = file_data.get("type", "")
                file_name = file_data.get("name", "file")

                if file_type == "application/pdf":
                    # Display PDF as info (will be sent to AI)
                    st.info(f"📄 Attached: {file_name}")
                else:
                    # Display as image
                    img_bytes = base64.b64decode(file_data["data"])
                    st.image(img_bytes, width=400)

    # Generate response
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            # Check if we have PDFs - extract and add to RAG system
            pdf_files = [
                f for f in images_for_display if f.get("type") == "application/pdf"
            ]

            # Track messages before invocation
            messages_before = len(st.session_state.agent_state["messages"])

            # Create progress status placeholders FIRST
            hpc_status_placeholder = st.empty()
            hpc_progress_messages = []
            mmore_status_placeholder = st.empty()
            mmore_progress_messages = []

            def hpc_progress_callback(step: str, message: str) -> None:
                """Callback to display HPC operation progress."""
                # Map steps to emoji icons
                step_icons = {
                    "prepare": "📁",
                    "transfer": "📤",
                    "submit": "🚀",
                    "complete": "✅",
                }
                icon = step_icons.get(step, "⚙️")

                # Add to progress messages
                progress_line = f"{icon} {message}"
                hpc_progress_messages.append(progress_line)

                # Display all progress in the placeholder
                with hpc_status_placeholder.container():
                    st.info("\n\n".join(hpc_progress_messages))

                # If job was submitted, handle monitoring based on user preference
                if step == "complete" and "Job ID:" in message:
                    job_id = extract_job_id_from_response(message)
                    if job_id:
                        # Check if auto-add is enabled
                        auto_add = st.session_state.get("job_monitor_auto_add", False)
                        if auto_add:
                            # Automatically add to monitor
                            add_job_to_monitor(job_id, "SUBMITTED")
                            logger.info(f"Job {job_id} automatically added to monitor")
                        else:
                            # Store the job ID temporarily for the prompt
                            if "pending_job_monitor" not in st.session_state:
                                st.session_state.pending_job_monitor = []
                            st.session_state.pending_job_monitor.append(job_id)
                            logger.info(
                                f"Job {job_id} submitted, will prompt user to monitor"
                            )

            def mmore_progress_callback(step: str, message: str) -> None:
                """Callback to display MMORE operation progress."""
                # Map steps to emoji icons
                step_icons = {
                    "prepare": "📄",
                    "upload": "📤",
                    "process": "⚙️",
                    "download": "⬇️",
                    "complete": "✅",
                    "error": "❌",
                }
                icon = step_icons.get(step, "📝")

                # Add to progress messages
                progress_line = f"{icon} {message}"
                mmore_progress_messages.append(progress_line)

                # Display all progress in the placeholder
                with mmore_status_placeholder.container():
                    st.info("\n\n".join(mmore_progress_messages))

            # Set progress callback for HPC operations
            set_progress_callback(hpc_progress_callback)

            # Set progress callback for MMORE operations
            set_mmore_progress_callback(mmore_progress_callback)

            # Upload PDFs to MMORE AFTER setting up progress callbacks
            if pdf_files:
                try:
                    mmore_client = MMOREClient()
                    db = get_db()  # Get database instance

                    # Process and store each PDF in MMORE
                    for pdf_file in pdf_files:
                        # Use original bytes if available (avoids decode), otherwise decode from base64
                        pdf_bytes: bytes = (
                            cast(bytes, pdf_file["bytes"])
                            if "bytes" in pdf_file and pdf_file["bytes"] is not None
                            else base64.b64decode(pdf_file["data"])
                        )
                        file_name = pdf_file.get("name", "document.pdf")

                        # Write to temporary file for MMORE upload
                        with tempfile.NamedTemporaryFile(
                            suffix=".pdf", delete=False
                        ) as tmp_file:
                            tmp_file.write(pdf_bytes)
                            tmp_path = tmp_file.name

                        try:
                            # Upload to MMORE (uses file stem as ID)
                            file_id = Path(file_name).stem
                            mmore_client.upload_file(
                                file_path=tmp_path,
                                file_id=file_id,
                                original_name=file_name,
                            )

                            # Track in database
                            db.add_mmore_document(
                                file_id=file_id, file_name=file_name, file_path=tmp_path
                            )

                            # Report completion through progress callback
                            mmore_progress_callback(
                                "complete",
                                f"✓ Successfully indexed '{file_name}' - ready for queries!",
                            )

                            st.success(
                                f"✓ Added '{file_name}' to MMORE knowledge base (ID: {file_id})"
                            )
                            logger.info(
                                f"Uploaded {file_name} to MMORE with ID {file_id}"
                            )

                        except Exception as e:
                            st.error(f"Error uploading '{file_name}' to MMORE: {e!s}")
                            logger.exception(f"Failed to upload {file_name} to MMORE")
                        finally:
                            # Clean up temp file
                            tmp_file_path = Path(tmp_path)
                            if tmp_file_path.exists():
                                tmp_file_path.unlink()

                except Exception as e:
                    st.error(f"Error processing PDF: {e!s}")
                    # Continue with regular processing

            # If this was just a simple knowledge upload request, provide a direct response
            if is_knowledge_upload:
                # Provide a helpful response without invoking the agent
                uploaded_names = ", ".join(
                    [f"'{f.get('name', 'document.pdf')}'" for f in pdf_files]
                )
                response_text = (
                    f"✅ Successfully added {uploaded_names} to the MMORE knowledge base!\n\n"
                    f"You can now ask questions about {'this document' if len(pdf_files) == 1 else 'these documents'}. "
                    f"For example:\n"
                    f'- "What is this paper about?"\n'
                    f'- "Summarize the key findings"\n'
                    f'- "What methods did they use?"'
                )

                # Display the response
                st.markdown(response_text)

                # Add to display history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                    }
                )

                # Save assistant message to database
                if st.session_state.active_chat_id:
                    db = get_db()
                    db.add_message(
                        conversation_id=st.session_state.active_chat_id,
                        role="assistant",
                        content=response_text,
                    )

                # Save chat state
                save_active_chat_to_storage()
                return

            try:
                # Invoke the agent
                result = st.session_state.agent.invoke(
                    st.session_state.agent_state, st.session_state.config
                )
            finally:
                # Clear progress callbacks after invocation
                set_progress_callback(None)
                set_mmore_progress_callback(None)
                # Clear the status placeholders
                hpc_status_placeholder.empty()
                mmore_status_placeholder.empty()

            # Check if graph was interrupted for confirmation
            is_interrupted, user_request = check_streamlit_interrupt()

            if is_interrupted:
                # Update state
                st.session_state.agent_state = result

                # Check if this is an auto-resume (question, not command)
                if user_request == "__AUTO_RESUME__":
                    # No tool calls detected - auto-resume without confirmation
                    result = st.session_state.agent.invoke(
                        None, st.session_state.config
                    )  # type: ignore[arg-type]
                    st.session_state.agent_state = result
                    # Continue to display messages normally (don't return early)
                else:
                    # Store messages_before for use after confirmation
                    st.session_state.messages_before_confirmation = messages_before
                    # Show confirmation prompt and wait for user response
                    show_confirmation_prompt(user_request)
                    save_active_chat_to_storage()  # Save before waiting
                    return  # Wait for user's confirmation response

            # Update agent state
            st.session_state.agent_state = result

            # Get and display new messages with streaming (based on user preference)
            new_messages = st.session_state.agent_state["messages"][messages_before:]
            use_streaming = st.session_state.get("enable_streaming", True)
            full_response, suggested_prompts = format_and_display_messages(
                new_messages, use_streaming=use_streaming
            )

            if full_response:
                # Save to display history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "suggested_prompts": suggested_prompts,
                    }
                )

                # Save assistant message to database
                if st.session_state.active_chat_id:
                    db = get_db()
                    db.add_message(
                        conversation_id=st.session_state.active_chat_id,
                        role="assistant",
                        content=full_response,
                        suggested_prompts=suggested_prompts,
                    )
            else:
                st.info("Agent is processing... (no response yet)")

            # Save chat state after successful interaction
            save_active_chat_to_storage()

            # Trigger rerun so that chat.py renders all messages with buttons
            st.rerun()

        except Exception as e:
            error_msg = f"❌ **Error:** {e!s}"
            st.error(error_msg)
            # Remove the last user message on error
            if st.session_state.agent_state["messages"]:
                st.session_state.agent_state["messages"].pop()
            # Save even on error
            save_active_chat_to_storage()


def _get_usage_status(max_percentage: float) -> tuple[str, str]:
    """Get status color and text based on usage percentage."""
    if max_percentage > USAGE_THRESHOLD_CRITICAL:
        return "🔴", "Critical"
    if max_percentage > USAGE_THRESHOLD_WARNING:
        return "🟡", "High"
    return "🟢", "Good"


def _format_usage_text(usage: int, limit: int, percentage: float) -> str:
    """Format usage text with or without limit."""
    if limit == UNLIMITED_LIMIT_VALUE:
        return f"{usage:,} requests"
    return f"{usage:,}/{limit:,} ({percentage:.0f}%)"


def render_tavily_usage_widget() -> None:
    """Render a compact Tavily API usage widget in the sidebar."""
    try:
        # Fetch usage data (with caching to avoid too many requests)
        cache_key = "tavily_usage_cache"
        cache_time_key = "tavily_usage_cache_time"
        cache_duration = 300  # 5 minutes

        # Check if we need to fetch fresh data
        current_time = datetime.datetime.now()
        should_fetch = True

        if cache_time_key in st.session_state:
            last_fetch = st.session_state[cache_time_key]
            time_diff = (current_time - last_fetch).total_seconds()
            if time_diff < cache_duration and cache_key in st.session_state:
                should_fetch = False

        if should_fetch:
            usage = get_tavily_usage(config.tavily_api_key)
            st.session_state[cache_key] = usage
            st.session_state[cache_time_key] = current_time
        else:
            usage = st.session_state.get(cache_key)

        if usage:
            # Determine status based on usage percentage
            max_percentage = max(usage.key_percentage, usage.plan_percentage)
            status_color, status_text = _get_usage_status(max_percentage)

            # Display compact usage info
            with st.expander(
                f"{status_color} Tavily API: {status_text}", expanded=False
            ):
                # Key usage
                key_text = _format_usage_text(
                    usage.key_usage, usage.key_limit, usage.key_percentage
                )
                st.caption(f"**Key:** {key_text}")

                # Plan usage
                plan_text = _format_usage_text(
                    usage.plan_usage, usage.plan_limit, usage.plan_percentage
                )
                st.caption(f"**Monthly:** {plan_text}")

                # Link to settings
                st.caption("[View details in Settings →](settings#tavily-search-api)")

    except Exception as e:
        logger.debug(f"Failed to fetch Tavily usage for sidebar: {e}")
        # Silently fail - don't show error in sidebar


def extract_job_id_from_response(response: str) -> str | None:
    """Extract SLURM job ID from agent response text."""
    # Look for patterns like "Job ID: 12345678" or "job_id: 12345678"
    match = re.search(r"[Jj]ob\s*[Ii][Dd][:\s]+(\d+)", response)
    if match:
        return match.group(1)
    # Also look for standalone numbers that might be job IDs (8-10 digits)
    match = re.search(r"\b(\d{7,10})\b", response)
    if match:
        return match.group(1)
    return None


def get_or_create_hpc_connection(host_alias: str | None = None):
    """Get or create a persistent HPC connection for job monitoring.

    This maintains a single SSH connection in session state to avoid
    reconnecting for every status check. Without connection reuse,
    each job status check would create a new SSH connection, leading
    to inefficient reconnections every ~12 seconds.

    Args:
        host_alias: Host alias from ~/.ssh/config. If not provided, uses HPC_HOST_ALIAS from .env

    Returns:
        HPCConnection instance (reused across multiple status checks)
    """
    from config import config  # noqa: PLC0415
    from src.tools.connection import HPCConnection  # noqa: PLC0415

    # Use configured host alias if not provided
    if host_alias is None:
        host_alias = config.hpc_host_alias

    # Initialize connections dict if it doesn't exist
    if "hpc_connections" not in st.session_state:
        st.session_state.hpc_connections = {}

    # Create connection if it doesn't exist for this host
    if host_alias not in st.session_state.hpc_connections:
        logger.info(f"Creating persistent HPC connection for {host_alias}")
        st.session_state.hpc_connections[host_alias] = HPCConnection(
            host_alias=host_alias
        )
    else:
        logger.debug(f"Reusing existing HPC connection for {host_alias}")

    return st.session_state.hpc_connections[host_alias]


def get_job_status_display(
    job_id: str, host_alias: str | None = None
) -> dict[str, Any]:
    """Get formatted job status for display."""
    from config import config  # noqa: PLC0415

    # Use configured host alias if not provided
    if host_alias is None:
        host_alias = config.hpc_host_alias

    try:
        # Use persistent connection instead of creating a new one each time
        hpc = get_or_create_hpc_connection(host_alias)
        status_output = hpc.get_job_status(job_id)

        # Parse status output to extract key info
        is_completed = False
        job_state = "UNKNOWN"

        if "not found" in status_output.lower() or "completed" in status_output.lower():
            is_completed = True
            job_state = "COMPLETED"
        else:
            # Parse squeue output
            lines = [
                line.strip()
                for line in status_output.strip().split("\n")
                if line.strip()
            ]
            if len(lines) <= 1:
                is_completed = True
                job_state = "COMPLETED"
            elif len(lines) > 1:
                # Extract state from output (typically 5th column)
                parts = lines[1].split()
                state_column_index = 4
                if len(parts) > state_column_index:
                    job_state = parts[state_column_index]

    except Exception as e:
        return {
            "job_id": job_id,
            "state": "ERROR",
            "is_completed": False,
            "error": str(e),
            "success": False,
        }
    else:
        return {
            "job_id": job_id,
            "state": job_state,
            "is_completed": is_completed,
            "raw_output": status_output,
            "success": True,
        }


def _render_add_job_form(form_key: str) -> None:
    """Render form to add a new job to monitor."""
    with st.form(form_key):
        st.markdown("**Add Job to Monitor**")
        job_id_input = st.text_input(
            "Job ID", placeholder="Enter SLURM job ID (e.g., 12345678)"
        )
        col_submit, col_cancel = st.columns(2)
        with col_submit:
            submitted = st.form_submit_button(
                "✓ Add", type="primary", use_container_width=True
            )
        with col_cancel:
            cancelled = st.form_submit_button("✕ Cancel", use_container_width=True)

        if submitted and job_id_input:
            add_job_to_monitor(job_id_input.strip())
            st.session_state[f"show_{form_key}"] = False
            st.rerun()
        elif cancelled:
            st.session_state[f"show_{form_key}"] = False
            st.rerun()


def add_job_to_monitor(job_id: str, initial_state: str = "SUBMITTED") -> None:
    """Add a job to the monitoring list."""
    if "monitored_jobs" not in st.session_state:
        st.session_state.monitored_jobs = {}

    st.session_state.monitored_jobs[job_id] = {
        "job_id": job_id,
        "state": initial_state,
        "is_completed": False,
        "last_check": time.time(),
        "added_at": time.time(),
    }


def _display_compact_job_row(job_id: str, status: dict[str, Any]) -> None:
    """Display a single job row in compact format."""
    if not status["success"]:
        col1, col2, col3 = st.columns([2, 3, 1])
        with col1:
            st.code(job_id, language="text")
        with col2:
            st.error(f"❌ Error: {status.get('error', 'Unknown')}")
        with col3:
            if st.button("🗑️", key=f"remove_error_compact_{job_id}", help="Remove"):
                del st.session_state.monitored_jobs[job_id]
                st.rerun()
        return

    state = status["state"]
    is_completed = status["is_completed"]

    # Update job info
    st.session_state.monitored_jobs[job_id]["state"] = state
    st.session_state.monitored_jobs[job_id]["is_completed"] = is_completed
    st.session_state.monitored_jobs[job_id]["last_check"] = time.time()

    # Compact display
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        st.code(job_id, language="text")
    with col2:
        if is_completed:
            st.success(state, icon="✅")
        elif state in {"R", "RUNNING"}:
            st.info(f"{state} (Running)", icon="▶️")
        elif state in {"PD", "PENDING"}:
            st.warning(f"{state} (Pending)", icon="⏳")
        else:
            st.info(state, icon="⚙️")
    with col3:
        if st.button("🗑️", key=f"remove_compact_{job_id}", help="Remove"):
            del st.session_state.monitored_jobs[job_id]
            st.rerun()

    # Show details in sub-expander
    with st.expander(f"Details for {job_id}", expanded=False):
        st.code(status["raw_output"], language="text")


def render_job_monitor_compact() -> None:
    """Render compact job monitoring widget at bottom of chat."""
    # Initialize monitoring state if not exists
    if "monitored_jobs" not in st.session_state:
        st.session_state.monitored_jobs = {}

    if not st.session_state.monitored_jobs:
        return

    # Collapsible section for job monitor
    with st.expander(
        f"🚀 SLURM Job Monitor ({len(st.session_state.monitored_jobs)} jobs)",
        expanded=False,
    ):
        # Buttons row
        col1, col2 = st.columns([5, 1])
        with col1:
            refresh_interval = st.session_state.get("job_monitor_refresh_interval", 60)
            local_time = time.strftime("%H:%M:%S", time.localtime())
            st.caption(
                f"*Auto-refreshing every {refresh_interval}s... (Last update: {local_time})*"
            )
        with col2:
            if st.button("+", key="add_job_btn_compact", help="Add job to monitor"):
                st.session_state.show_add_job_form_compact = True

        # Show add job form if requested
        if st.session_state.get("show_add_job_form_compact", False):
            _render_add_job_form("add_job_form_compact")

        # Display jobs in a table-like format
        for job_id, _job_info in list(st.session_state.monitored_jobs.items()):
            status = get_job_status_display(job_id)
            _display_compact_job_row(job_id, status)

    # Auto-refresh based on configured interval - only if there are active jobs
    if st.session_state.monitored_jobs:
        has_active_jobs = any(
            not job_info.get("is_completed", False)
            for job_info in st.session_state.monitored_jobs.values()
        )

        if has_active_jobs:
            refresh_interval = st.session_state.get("job_monitor_refresh_interval", 60)
            time.sleep(refresh_interval)
            st.rerun()


def render_sidebar() -> None:
    """Render the sidebar with chat management controls."""
    # Constants for chat display
    max_title_length = 35
    truncated_title_length = 32

    # New Chat button at the top
    if st.button(
        "+ New Chat",
        key="new_chat_btn",
        width="stretch",
        type="primary",
        shortcut="Ctrl+K",
    ):
        # Save current chat before creating new one
        save_active_chat_to_storage()
        # Create new chat without auto-generated name (will be generated from first message)
        create_new_chat()
        # Navigate to chat page
        st.session_state._switch_to_chat_page = True
        # The new chat is now active and will show empty message list
        st.rerun()

    # Warning message about API usage and data privacy
    st.warning(
        "Don't share personal data. Limit use - API key usage applies.", icon="⚠️"
    )

    # Tavily API usage widget
    render_tavily_usage_widget()

    st.markdown("---")

    # Filter chats to only show those with at least 1 message OR the active chat
    # (so that newly created empty chats are visible)
    chats_with_messages = {
        chat_id: chat_data
        for chat_id, chat_data in st.session_state.chats.items()
        if len(chat_data.get("messages", [])) >= 1
        or chat_id == st.session_state.active_chat_id
    }

    # Sort chats: pinned first (by created_at desc), then unpinned (by created_at desc)
    sorted_chats = sorted(
        chats_with_messages.items(),
        key=lambda x: (
            not x[1].get(
                "pinned", False
            ),  # Pinned first (False < True, so not reverses it)
            -(
                x[1].get("created_at", datetime.datetime.now()).timestamp()
            ),  # Then by time desc
        ),
    )

    # Display each chat as a clickable item
    for chat_id, chat_data in sorted_chats:
        title = chat_data["title"]
        is_active = chat_id == st.session_state.active_chat_id
        is_pinned = chat_data.get("pinned", False)

        # Truncate title if too long
        display_title = title
        if len(title) > max_title_length:
            display_title = f"{title[:truncated_title_length]}..."

        # Create a container for each chat item with pin, title, and delete buttons
        col1, col2, col3 = st.columns([1, 7, 1])

        with col1:
            # Pin button (star icon when pinned, outline when not)
            pin_icon = "⭐" if is_pinned else "☆"
            if st.button(
                pin_icon,
                key=f"pin_{chat_id}",
                help="Pin/Unpin conversation",
                type="secondary",
            ):
                toggle_pin_chat(chat_id)
                st.rerun()

        with col2:
            # All chats show as buttons for consistent positioning
            button_clicked = st.button(
                display_title,
                key=f"chat_{chat_id}",
                width="stretch",
                type="secondary",
                disabled=False,  # Always clickable to allow navigation from other pages
            )
            # Switch chat if clicked and not active, or navigate if active but not on chat page
            if button_clicked:
                if not is_active:
                    # Switching to a different chat
                    switch_to_chat(chat_id)
                    st.rerun()
                else:
                    # Active chat clicked - just navigate to chat page
                    st.session_state._switch_to_chat_page = True
                    st.rerun()

        with col3:
            # Simple x button for delete
            if st.button(
                "x",
                key=f"delete_{chat_id}",
                help="Delete",
                disabled=is_active and len(chats_with_messages) == 1,
                type="secondary",
            ):
                delete_chat(chat_id)
                st.rerun()

    # Show empty state if no chats
    if not sorted_chats:
        st.info("No conversations yet. Click '+ New Chat' to start!")


def main() -> None:
    """Main Streamlit application."""
    # Use emoji as page icon (more reliable than loading image files)
    # Image icons can cause MediaFileStorageError on reruns
    page_icon = "🤖"

    # Page configuration
    st.set_page_config(
        page_title="EngiAI - Engineering Design Chatbot",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",  # Show sidebar for chat management
    )

    # Custom CSS for sidebar and chat management
    st.markdown(
        """
        <style>
        /* Set sidebar width for chat management - only when expanded */
        [data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 300px;
            max-width: 300px;
        }
        /* Make navigation icons bigger */
        [data-testid="stSidebar"] .stPageLink svg {
            width: 2.5rem !important;
            height: 2.5rem !important;
        }
        [data-testid="stSidebar"] .stPageLink {
            padding: 1rem 0.5rem !important;
        }
        [data-testid="stSidebar"] .stPageLink span {
            font-size: 1.1rem !important;
        }
        /* Chat list styling - Simple and clean */
        [data-testid="stSidebar"] button[kind="secondary"] {
            text-align: left !important;
            border: none !important;
            background: transparent !important;
            padding: 0.25rem 0rem !important;
            font-size: 0.9rem !important;
            font-weight: normal !important;
            box-shadow: none !important;
            justify-content: flex-start !important;
            margin: 0 !important;
        }
        /* Remove internal button padding and force left alignment */
        [data-testid="stSidebar"] button[kind="secondary"] p {
            margin: 0 !important;
            padding: 0 !important;
            text-align: left !important;
            width: 100% !important;
        }
        [data-testid="stSidebar"] button[kind="secondary"] div {
            text-align: left !important;
            justify-content: flex-start !important;
        }
        [data-testid="stSidebar"] button[kind="secondary"]:hover:not(:disabled) {
            background: rgba(128, 128, 128, 0.1) !important;
        }
        /* Disabled buttons (active chat) - same styling as enabled */
        [data-testid="stSidebar"] button[kind="secondary"]:disabled {
            opacity: 1.0 !important;
            color: inherit !important;
            cursor: default !important;
        }
        /* Delete button (x) styling */
        [data-testid="stSidebar"] button[kind="secondary"]:has(p:contains("x")) {
            padding: 0.25rem 0.5rem !important;
            font-size: 1.2rem !important;
            min-width: 2rem !important;
            text-align: center !important;
        }
        /* Remove column padding for chat rows */
        [data-testid="stSidebar"] [data-testid="column"] {
            padding: 0 !important;
        }
        /* Reduce spacing between chat items - very compact */
        [data-testid="stSidebar"] .element-container {
            margin-bottom: 0rem !important;
            padding-bottom: 0rem !important;
        }
        [data-testid="stSidebar"] .row-widget {
            margin-bottom: 0rem !important;
            margin-top: 0rem !important;
            gap: 0 !important;
        }
        /* Target the specific container divs */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
            gap: 0rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    initialize_session_state()

    # Define navigation pages
    home_page = st.Page(
        home.render,
        title="Home",
        icon=":material/home:",
        url_path="home",
        default=True,
    )
    chat_page = st.Page(
        chat.render,
        title="Chat",
        icon=":material/chat:",
        url_path="chat",
    )
    wandb_page = st.Page(
        wandb_report.render,
        title="W&B Report",
        icon=":material/analytics:",
        url_path="wandb-report",
    )
    settings_page = st.Page(
        settings.render,
        title="Settings",
        icon=":material/settings:",
        url_path="settings",
    )

    pages = [home_page, chat_page, wandb_page, settings_page]

    # Check if we need to switch to chat page
    if st.session_state.get("_switch_to_chat_page", False):
        st.session_state._switch_to_chat_page = False
        # Navigate to chat page
        st.switch_page(chat_page)

    # Create navigation
    page = st.navigation(pages)

    # Sidebar
    with st.sidebar:
        render_sidebar()

    # Run the selected page
    page.run()


if __name__ == "__main__":
    main()
