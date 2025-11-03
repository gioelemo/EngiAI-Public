"""
Streamlit UI for the Engineer Assistant chatbot.

This provides a web-based chat interface for interacting with the multi-agent system.
"""

import contextlib
import datetime
import re
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any, cast

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image
from streamlit_stl import stl_from_file  # type: ignore[import-untyped]

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain.chat_models import init_chat_model  # noqa: E402

from config import config  # noqa: E402
from src.agents.supervisor_agent import SupervisorAgent  # noqa: E402
from src.models.state import MessagesState  # noqa: E402
from src.ui import chat, home, settings, wandb  # noqa: E402
from src.ui.database import DatabaseManager  # noqa: E402

# Suppress Pydantic warnings from LangChain
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic._internal._generate_schema"
)


def _generate_chat_title(user_message: str) -> str:
    """Generate a concise title for a chat based on the first user message.

    Args:
        user_message: The first user message in the conversation

    Returns:
        A short, summarized title (max 50 characters)
    """
    max_title_length = 50
    truncate_at = 47

    try:
        # Use a fast model to generate title
        llm = init_chat_model(config.llm_model)

        prompt = f"""Generate a very short title (maximum 4-5 words) that summarizes this question or request:

"{user_message}"

Reply with ONLY the title, nothing else. No quotes, no punctuation at the end."""

        response = llm.invoke(prompt)
        # Ensure content is a string
        content = response.content
        if isinstance(content, list):
            # Handle list content by joining
            title = " ".join(str(item) for item in content).strip()
        else:
            title = str(content).strip()

        # Remove quotes if present
        title = title.strip("\"'")

        # Limit to max length
        if len(title) > max_title_length:
            return title[:truncate_at] + "..."
        else:
            return title
    except Exception:
        # Fallback to truncated message if generation fails
        if len(user_message) > max_title_length:
            return user_message[:max_title_length] + "..."
        else:
            return user_message


def _get_db() -> DatabaseManager:
    """Get or create database manager instance.

    Returns:
        DatabaseManager instance
    """
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    return st.session_state.db_manager


def _create_new_chat(name: str | None = None) -> str:
    """Create a new chat and return its ID.

    Args:
        name: Optional name for the chat. Auto-generated if None

    Returns:
        The ID of the newly created chat (UUID)
    """
    # Save current chat before creating new one
    old_chat_id = st.session_state.active_chat_id
    if old_chat_id:
        _save_active_chat_to_storage()

        # Delete the old chat if it's empty (no messages)
        if old_chat_id in st.session_state.chats:
            old_chat = st.session_state.chats[old_chat_id]
            if len(old_chat.get("messages", [])) == 0:
                # Delete empty chat from memory (not in DB since it was never saved)
                del st.session_state.chats[old_chat_id]

    if name is None:
        st.session_state.chat_counter += 1
        name = f"New Chat {st.session_state.chat_counter}"

    # Generate session ID
    session_id = str(uuid.uuid4())

    # DON'T create in database yet - wait until first message
    # We'll create it when the first message is added

    # Create new chat data in session
    st.session_state.chats[session_id] = {
        "id": session_id,
        "title": name,
        "created_at": datetime.datetime.now(),
        "messages": [],
        "agent_state": {"messages": []},
        "agent": SupervisorAgent(),
        "config": {"configurable": {"thread_id": session_id}},
        "waiting_for_confirmation": False,
        "saved_to_db": False,  # Track if this chat has been saved to DB yet
    }

    # Set as active chat and sync (this will clear the display)
    st.session_state.active_chat_id = session_id
    _sync_active_chat_to_session()

    return session_id


def _sync_active_chat_to_session() -> None:
    """Sync the active chat data to session state for backward compatibility."""
    if (
        st.session_state.active_chat_id
        and st.session_state.active_chat_id in st.session_state.chats
    ):
        active_chat = st.session_state.chats[st.session_state.active_chat_id]
        st.session_state.messages = active_chat["messages"]
        st.session_state.agent = active_chat["agent"]
        st.session_state.agent_state = active_chat["agent_state"]
        st.session_state.config = active_chat["config"]
        st.session_state.waiting_for_confirmation = active_chat[
            "waiting_for_confirmation"
        ]


def _save_active_chat_to_storage() -> None:
    """Save the current session state back to the active chat storage and database."""
    if (
        st.session_state.active_chat_id
        and st.session_state.active_chat_id in st.session_state.chats
    ):
        active_chat = st.session_state.chats[st.session_state.active_chat_id]
        active_chat["messages"] = st.session_state.messages
        active_chat["agent"] = st.session_state.agent
        active_chat["agent_state"] = st.session_state.agent_state
        active_chat["config"] = st.session_state.config
        active_chat["waiting_for_confirmation"] = (
            st.session_state.waiting_for_confirmation
        )

        # Only save to database if chat has at least 1 message
        if len(active_chat["messages"]) < 1:
            return

        # Auto-generate title from first message
        new_title = None
        if active_chat["title"].startswith("New Chat") or active_chat[
            "title"
        ].startswith("Chat "):
            first_user_msg = next(
                (
                    msg["content"]
                    for msg in active_chat["messages"]
                    if msg["role"] == "user"
                ),
                None,
            )
            if first_user_msg:
                # Generate a concise title using AI
                new_title = _generate_chat_title(first_user_msg)
                active_chat["title"] = new_title

        # Save to database
        db = _get_db()

        # Create in database if not already saved
        if not active_chat.get("saved_to_db", False):
            db.create_conversation(
                name=active_chat["title"], session_id=st.session_state.active_chat_id
            )
            active_chat["saved_to_db"] = True

        # Update title if changed
        if new_title:
            db.update_conversation_name(st.session_state.active_chat_id, new_title)

        # Save agent state
        db.save_conversation_state(
            conversation_id=st.session_state.active_chat_id,
            agent_state=st.session_state.agent_state,
            config=st.session_state.config,
            waiting_for_confirmation=st.session_state.waiting_for_confirmation,
        )


def _switch_to_chat(chat_id: str) -> None:
    """Switch to a different chat and navigate to chat page.

    Args:
        chat_id: The ID of the chat to switch to
    """
    if chat_id in st.session_state.chats:
        # Save current chat first
        _save_active_chat_to_storage()

        # Switch to new chat
        st.session_state.active_chat_id = chat_id
        _sync_active_chat_to_session()

        # Store that we want to navigate to chat page
        st.session_state._switch_to_chat_page = True


def _delete_chat(chat_id: str) -> None:
    """Delete a chat from session and database.

    Args:
        chat_id: The ID of the chat to delete
    """
    if chat_id in st.session_state.chats:
        # Delete from session
        del st.session_state.chats[chat_id]

        # Delete from database
        db = _get_db()
        db.delete_conversation(chat_id)

        # If deleted chat was active, switch to another or create new
        if st.session_state.active_chat_id == chat_id:
            if st.session_state.chats:
                # Switch to first available chat
                first_chat_id = next(iter(st.session_state.chats.keys()))
                _switch_to_chat(first_chat_id)
            else:
                # Create new chat if no chats remain
                _create_new_chat()


def _load_chats_from_database() -> None:
    """Load all conversations from database into session state."""
    try:
        db = _get_db()
        conversations = db.get_all_conversations()

        for conv in conversations:
            chat_id = conv["id"]

            # Load messages
            messages = db.get_messages(chat_id)
            display_messages = [
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ]

            # Load state
            state_data = db.get_conversation_state(chat_id)
            if state_data:
                agent_state = state_data["agent_state"]
                config = state_data["config"]
                waiting_for_confirmation = state_data["waiting_for_confirmation"]
            else:
                agent_state = {"messages": []}
                config = {"configurable": {"thread_id": chat_id}}
                waiting_for_confirmation = False

            # Create chat data in session
            st.session_state.chats[chat_id] = {
                "id": chat_id,
                "title": conv["name"],
                "created_at": conv["created_at"],
                "messages": display_messages,
                "agent_state": agent_state,
                "agent": SupervisorAgent(),
                "config": config,
                "waiting_for_confirmation": waiting_for_confirmation,
                "saved_to_db": True,  # Already in database
            }

        # Set the most recently updated as active
        if conversations and not st.session_state.active_chat_id:
            st.session_state.active_chat_id = conversations[0]["id"]
            _sync_active_chat_to_session()

    except Exception:
        # Silently fail - if loading fails, start fresh
        pass


def _initialize_chat_state() -> None:
    """Initialize chat-related session state."""
    defaults: dict[str, Any] = {
        "chats": {},
        "active_chat_id": None,
        "chat_counter": 0,
        "messages": [],
        "agent": SupervisorAgent(),
        "agent_state": {"messages": []},
        "config": {"configurable": {"thread_id": "streamlit-session"}},
        "waiting_for_confirmation": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _initialize_stl_settings() -> None:
    """Initialize STL viewer settings."""
    defaults: dict[str, Any] = {
        "stl_color": "#0069B4",
        "stl_material": "material",
        "stl_height": 400,
        "stl_auto_rotate": True,
        "stl_opacity": 1.0,
        "stl_shininess": 100,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    # Initialize chat state
    _initialize_chat_state()

    # Try to load chats from database on first run
    if "chats_loaded" not in st.session_state:
        _load_chats_from_database()
        st.session_state.chats_loaded = True

    # Create first chat if none exists
    if not st.session_state.chats:
        _create_new_chat()

    # Page navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"

    # Initialize STL viewer settings
    _initialize_stl_settings()


def find_images_in_text(text: str) -> list[Path]:
    """Find image file paths mentioned in text.

    Args:
        text: Text that may contain file paths

    Returns:
        List of valid image file paths
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg"}
    image_paths: list[Path] = []

    # Regex patterns to catch:
    # - explicit outputs/ or assets/ or artifacts/ paths
    # - bare filenames like image.png
    # - markdown image syntax: ![alt](path.png)
    # - HTML <img src="path.png">
    patterns = [
        r"outputs/[\w\-_.]+\.(?:png|jpg|jpeg|gif|bmp|svg)",
        r"assets/[\w\-_.]+\.(?:png|jpg|jpeg|gif|bmp|svg)",
        r"artifacts/[\w\-_.]+\.(?:png|jpg|jpeg|gif|bmp|svg)",
        r"/[^\s)\"]+\.(?:png|jpg|jpeg|gif|bmp|svg)",
        r"[\w\-_.]+\.(?:png|jpg|jpeg|gif|bmp|svg)",
        r"!\[[^\]]*\]\(([^)]+\.(?:png|jpg|jpeg|gif|bmp|svg))\)",
        r"<img[^>]+src=[\"']([^\"']+\.(?:png|jpg|jpeg|gif|bmp|svg))[\"']",
    ]

    seen: set[str] = set()

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            # re groups may return tuples for some patterns, normalize
            candidate_str = m[0] if isinstance(m, tuple) else m

            path = Path(candidate_str)
            # Expand user home if present
            with contextlib.suppress(Exception):
                path = Path(str(path).replace("~", str(Path.home())))

            # If not absolute, try sensible locations: project root, outputs, assets, artifacts
            candidates = [path]
            if not path.is_absolute():
                candidates.extend(
                    [
                        project_root / candidate_str,
                        project_root / "outputs" / candidate_str,
                        project_root / "assets" / candidate_str,
                        project_root / "artifacts" / candidate_str,
                    ]
                )

            for candidate in candidates:
                try:
                    if (
                        candidate.exists()
                        and candidate.suffix.lower() in image_extensions
                    ):
                        key = str(candidate.resolve())
                        if key not in seen:
                            image_paths.append(candidate)
                            seen.add(key)
                        break
                except Exception:
                    # ignore resolution errors and continue
                    continue

    return image_paths


def find_stl_files_in_text(text: str) -> list[Path]:
    """Find STL file paths mentioned in text.

    Args:
        text: Text that may contain file paths

    Returns:
        List of valid STL file paths (only files that currently exist)
    """
    stl_paths = []

    # Look for STL file patterns
    patterns = [
        r"outputs/[\w\-_.]+\.stl",
        r"[\w\-_.]+\.stl",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Try as absolute path first
            path = Path(match)
            if not path.is_absolute():
                # Try relative to project root
                path = project_root / match

            # Only add if file exists and is valid STL
            if path.exists() and path.suffix.lower() == ".stl":
                stl_paths.append(path)

    return list(set(stl_paths))  # Remove duplicates


def find_log_files_in_text(text: str) -> list[Path]:
    """Find .err and .out log files mentioned in text or associated with downloaded jobs.

    Args:
        text: Text that may contain file paths or job IDs

    Returns:
        List of valid log file paths (.err and .out files)
    """
    log_paths = []
    seen: set[str] = set()

    # Look for explicit .err and .out file patterns
    patterns = [
        r"outputs/[\w\-_.]+\.(?:err|out)",
        r"[\w\-_.]+\.(?:err|out)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            path = Path(match)
            if not path.is_absolute():
                path = project_root / match

            if path.exists() and path.suffix.lower() in [".err", ".out"]:
                key = str(path.resolve())
                if key not in seen:
                    log_paths.append(path)
                    seen.add(key)

    # Also look for job IDs and find their corresponding log files
    job_id_pattern = (
        r"job[_ ](?:id|ID)[:\s]*(\d+)|Job ID: (\d+)|job_id[\"']?\s*:\s*[\"']?(\d+)"
    )
    job_matches = re.findall(job_id_pattern, text)

    for match_groups in job_matches:
        # Extract the actual job ID from the groups
        job_id = next((m for m in match_groups if m), None)
        if job_id:
            # Look for log files matching this job ID in outputs directory
            outputs_dir = project_root / "outputs"
            if outputs_dir.exists():
                err_files = list(outputs_dir.glob(f"*{job_id}.err"))
                out_files = list(outputs_dir.glob(f"*{job_id}.out"))
                for log_file in err_files + out_files:
                    key = str(log_file.resolve())
                    if key not in seen:
                        log_paths.append(log_file)
                        seen.add(key)

    return log_paths


def _save_file_to_server(file_path: Path, save_dir_path: Path) -> bool:
    """Save a file to the server directory.

    Args:
        file_path: Source file path
        save_dir_path: Destination directory path

    Returns:
        True if successful, False otherwise
    """
    try:
        save_dir_path.mkdir(parents=True, exist_ok=True)
        dest = save_dir_path / file_path.name
        dest.write_bytes(file_path.read_bytes())
    except Exception:
        return False
    else:
        return True


def _auto_save_if_enabled(file_path: Path) -> None:
    """Automatically save file if auto-save is enabled in session state.

    Args:
        file_path: File to save
    """
    if st.session_state.get("media_auto_save", False):
        save_dir = Path(st.session_state.media_save_dir)
        _save_file_to_server(file_path, save_dir)


def _render_download_save_controls(
    file_path: Path, file_type: str, button_key_prefix: str
) -> None:
    """Render download and save buttons for a media file.

    Args:
        file_path: Path to the file
        file_type: Type of file ('image' or 'stl')
        button_key_prefix: Unique prefix for button keys
    """
    cols = st.columns([1, 1, 2])

    # Create unique key based on file path hash and prefix
    unique_key_base = f"{button_key_prefix}_{abs(hash(str(file_path)))}"

    # Download button
    with cols[0]:
        mime = "image/png" if file_type == "image" else "application/sla"
        label = f"Download {file_type}"
        st.download_button(
            label=label,
            data=file_path.read_bytes(),
            file_name=file_path.name,
            mime=mime,
            key=f"{unique_key_base}_download",
        )

    # Save to server button
    with cols[1]:
        if st.button(
            f"Save to server: {file_path.name}",
            key=f"{unique_key_base}_save",
        ):
            save_dir = Path(st.session_state.media_save_dir)
            if _save_file_to_server(file_path, save_dir):
                st.success(f"Saved {file_path.name} to {save_dir}")
            else:
                st.warning(f"Could not save {file_type}")

    # Auto-save
    _auto_save_if_enabled(file_path)


def _display_image(img_path: Path, button_key_prefix: str) -> None:
    """Display an image with download/save controls.

    Args:
        img_path: Path to image file
        button_key_prefix: Unique prefix for button keys
    """
    # Skip if file doesn't exist (e.g., from previous sessions)
    if not img_path.exists():
        return

    try:
        image = Image.open(img_path)
        st.image(image, caption=img_path.name, width=500)
        _render_download_save_controls(img_path, "image", button_key_prefix)
    except FileNotFoundError:
        # Silently skip - file was deleted
        pass
    except Exception as e:
        # Only show warning for unexpected errors
        if "MediaFileStorageError" not in str(type(e).__name__):
            st.warning(f"Could not display image {img_path.name}: {e}")


def _display_stl(stl_path: Path, idx: int, button_key_prefix: str) -> None:
    """Display an STL file with viewer and download/save controls.

    Args:
        stl_path: Path to STL file
        idx: Index for unique key generation
        button_key_prefix: Unique prefix for button keys
    """
    if not stl_path.exists():
        # Silently skip missing STL files (e.g., from previous sessions)
        return

    try:
        st.markdown(f"**3D Model: {stl_path.name}**")
        # Include button_key_prefix to make key unique across messages
        stable_key = f"{button_key_prefix}_stl_{abs(hash(str(stl_path)))}_{idx}"
        stl_from_file(
            file_path=str(stl_path),
            color=st.session_state.stl_color,
            material=st.session_state.stl_material,
            auto_rotate=st.session_state.stl_auto_rotate,
            height=st.session_state.stl_height,
            opacity=st.session_state.stl_opacity,
            shininess=st.session_state.stl_shininess,
            key=stable_key,
        )
        _render_download_save_controls(stl_path, "stl", button_key_prefix)
    except FileNotFoundError:
        # Silently skip - file was deleted
        pass
    except Exception as e:
        st.warning(f"Could not display 3D model {stl_path.name}: {e}")


def _display_log_file(log_path: Path, button_key_prefix: str) -> None:
    """Display a log file (.err or .out) with expandable content and download controls.

    Args:
        log_path: Path to log file
        button_key_prefix: Unique prefix for button keys
    """
    if not log_path.exists():
        st.info(f"Log file not found: {log_path.name}")
        return

    try:
        # Read log file content
        content = log_path.read_text(encoding="utf-8", errors="replace")

        # Determine file type for styling
        file_type = log_path.suffix.lower()
        icon = "❌" if file_type == ".err" else "📄"
        label = "Error Log" if file_type == ".err" else "Output Log"

        # Display log file in an expander
        with st.expander(f"{icon} **{label}: {log_path.name}**", expanded=False):
            if content.strip():
                # Show first 100 lines by default, full content in code block
                lines = content.split("\n")
                preview_lines = 100

                if len(lines) > preview_lines:
                    st.caption(
                        f"Showing first {preview_lines} lines of {len(lines)} total lines"
                    )
                    st.code("\n".join(lines[:preview_lines]), language="text")

                    # Option to show full content
                    if st.button(
                        "Show full content",
                        key=f"{button_key_prefix}_{abs(hash(str(log_path)))}_full",
                    ):
                        st.code(content, language="text")
                else:
                    st.code(content, language="text")
            else:
                st.info("Log file is empty")

            # Download button for log file
            cols = st.columns([1, 3])
            with cols[0]:
                st.download_button(
                    label=f"Download {file_type} file",
                    data=content,
                    file_name=log_path.name,
                    mime="text/plain",
                    key=f"{button_key_prefix}_{abs(hash(str(log_path)))}_download",
                )

        # Auto-save if enabled
        _auto_save_if_enabled(log_path)

    except Exception as e:
        st.warning(f"Could not display log file {log_path.name}: {e}")


def _fix_latex_delimiters(text: str) -> str:
    """Fix LaTeX delimiters to be Streamlit-compatible.

    Converts \\( \\) to $ $ and \\[ \\] to $$ $$ for proper rendering.

    Args:
        text: Text that may contain LaTeX with incorrect delimiters

    Returns:
        Text with fixed LaTeX delimiters
    """
    # Replace display math: \[ ... \] with $$ ... $$
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)

    # Replace inline math: \( ... \) with $ ... $
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)

    # Also handle single bracket/paren versions that might appear
    text = re.sub(
        r"\[\s*([^]]*?)\s*\](?=\s|$|[.,;!?])",
        lambda m: (
            f"${m.group(1)}$"
            if any(
                c in m.group(1)
                for c in ["\\frac", "\\sum", "\\int", "=", "+", "-", "*", "/", "^", "_"]
            )
            else m.group(0)
        ),
        text,
    )

    return text


def display_message(message: dict, message_idx: int = 0) -> None:
    """Display a single message in the chat interface.

    Args:
        message: Dictionary with 'role' and 'content' keys
        message_idx: Index of the message in the chat history for unique keys
    """
    with st.chat_message(message["role"]):
        # Fix LaTeX delimiters before displaying
        content = _fix_latex_delimiters(message["content"])

        # Display text content
        st.markdown(content)

        # Display images
        images = find_images_in_text(message["content"])
        for img_path in images:
            _display_image(img_path, button_key_prefix=f"msg_{message_idx}_img")

        # Display STL files
        stl_files = find_stl_files_in_text(message["content"])
        for idx, stl_path in enumerate(stl_files):
            _display_stl(stl_path, idx, button_key_prefix=f"msg_{message_idx}_stl")

        # Display log files (.err and .out)
        log_files = find_log_files_in_text(message["content"])
        for log_path in log_files:
            _display_log_file(log_path, button_key_prefix=f"msg_{message_idx}_log")


def format_tool_call(tool_call: Any) -> str:
    """Format a tool call for display.

    Args:
        tool_call: Tool call object (dict-like)

    Returns:
        Formatted string describing the tool call
    """
    # Tool calls can be dict-like objects
    if isinstance(tool_call, dict):
        tool_name = tool_call.get("name", "Unknown")
    else:
        tool_name = getattr(tool_call, "name", "Unknown")
    return f"🔧 **Using tool:** `{tool_name}`"


def format_ai_message(message: AIMessage | ToolMessage) -> str:
    """Format an AI or tool message for display.

    Args:
        message: The message to format

    Returns:
        Formatted message content
    """
    max_content_length = 1000

    if isinstance(message, ToolMessage):
        # Tool results
        content = str(message.content)
        if len(content) > max_content_length:
            content = content[:max_content_length] + "\n\n... (truncated)"
        return f"```\n{content}\n```"

    # AI message
    content = str(message.content) if message.content else ""

    # Add tool calls if present
    if hasattr(message, "tool_calls") and message.tool_calls:
        tool_info = "\n\n".join(
            format_tool_call(tc)
            for tc in message.tool_calls  # type: ignore[arg-type]
        )
        if content:
            return f"{tool_info}\n\n{content}"
        return tool_info

    return content


def _extract_and_display_validation_warnings(response_text: str) -> str:
    """Extract validation warnings from response and display them as Streamlit warnings.

    Args:
        response_text: The response text that may contain validation warnings

    Returns:
        Response text with validation warnings removed (they'll be shown separately)
    """

    # --- Pattern 1: Check for the highly structured "CRITICAL" block ---
    pattern_structured = (
        r"={60,}\n🚨 \*\*CRITICAL: Resource Allocation Review\*\*\n={60,}.*?={60,}"
    )
    match_structured = re.search(pattern_structured, response_text, re.DOTALL)

    if match_structured:
        validation_block = match_structured.group(0)

        # Extract the issues
        issues_match = re.search(
            r"\*\*Issues Found:\*\*\n(.*?)(?=\n\*\*💡|={60})",
            validation_block,
            re.DOTALL,
        )
        recommendations_match = re.search(
            r"\*\*💡 Recommendations:\*\*\n(.*?)(?=\n={60})",
            validation_block,
            re.DOTALL,
        )

        # Display as Streamlit error
        error_message = "### 🚨 Resource Allocation Warning\n\n"

        if issues_match:
            issues_text = issues_match.group(1).strip()
            error_message += "**Issues Found:**\n" + issues_text + "\n\n"

        if recommendations_match:
            rec_text = recommendations_match.group(1).strip()
            error_message += "**💡 Recommendations:**\n" + rec_text + "\n\n"

        error_message += (
            "⚠️ **The script was generated despite exceeding recommended limits.**\n"
        )
        error_message += "⚠️ **Please review and adjust resources before submitting.**"

        st.error(error_message)

        # Remove the validation block from response text
        response_text = response_text.replace(validation_block, "").strip()
        return re.sub(r"\n{3,}", "\n\n", response_text)  # Return cleaned text

    # --- UPDATED: Pattern 2: Check for various unstructured warning blocks ---

    # Define start markers (case-insensitive)
    start_markers = [
        r"Important Validation Notes:",  # <-- ADDED THIS
        r"Important Warnings:",
        r"However, there are important notes and validation warnings:",
        r"Resource Validation/Warnings:",
        r"Resource Allocation Warning",
    ]

    # Define end markers (lookahead, case-insensitive)
    end_markers = [
        r"Generated SLURM Script \(shortened for clarity",  # <-- ADDED THIS
        r"File Location:",  # <-- ADDED THIS
        r"SLURM Script File:",
        r"SLURM Script \(saved to",
        r"The script below is valid",
        r"Example SLURM Script \(view below\):",
        r"##",  # Next markdown heading
        r"\n\nLet me know if",
        r"\n\nIf you want to correct",
        r"\n\nPlease reduce the GPU count",  # Add another common follow-up
    ]

    pattern_unstructured = (
        r"((?:{}).*?)"  # Start: Match any of the start markers
        r"(?={})"  # End: Lookahead for any of the end markers
    ).format("|".join(start_markers), "|".join(end_markers))

    match_unstructured = re.search(
        pattern_unstructured, response_text, re.DOTALL | re.IGNORECASE
    )

    if match_unstructured:
        validation_block = match_unstructured.group(1).strip()

        # Clean up the extracted block for display
        # Remove the introductory line itself to avoid redundancy
        warning_content = re.sub(
            "|".join(start_markers),
            "",
            validation_block,
            flags=re.IGNORECASE,
        ).strip()

        # Format it nicely for the warning box
        warning_msg = f"⚠️ **Resource Allocation Warning**\n\n{warning_content}"

        # Display it as a Streamlit warning
        st.warning(warning_msg)

        # Remove the validation block from the original response text
        response_text = response_text.replace(validation_block, "").strip()

        # Clean up potential double newlines
        response_text = re.sub(r"\n{3,}", "\n\n", response_text)

        return response_text

    # If no patterns matched, return the original text
    return response_text


def display_response_media(response_text: str) -> None:
    """Display images, STL files, and log files found in response text.

    Args:
        response_text: The response text to scan for media files
    """
    # Display images
    images = find_images_in_text(response_text)
    for img_path in images:
        _display_image(img_path, button_key_prefix="resp_img")

    # Display STL files
    stl_files = find_stl_files_in_text(response_text)
    for idx, stl_path in enumerate(stl_files):
        _display_stl(stl_path, idx, button_key_prefix="resp_stl")

    # Display log files (.err and .out)
    log_files = find_log_files_in_text(response_text)
    for log_path in log_files:
        _display_log_file(log_path, button_key_prefix="resp_log")


def _check_streamlit_interrupt() -> tuple[bool, str]:
    """Check if graph execution was interrupted for confirmation.

    Returns:
        Tuple of (is_interrupted, user_request)
    """
    try:
        snapshot = st.session_state.agent.graph.get_state(st.session_state.config)  # type: ignore[attr-defined]
        if hasattr(snapshot, "next") and snapshot.next:
            next_nodes = str(snapshot.next)

            if "cli_agent" in next_nodes:
                # Before showing confirmation, check if CLI agent will actually call tools
                # If it's just an informational question, auto-resume without confirmation
                command_info = _extract_command_info()
                if not command_info:
                    # No tool calls detected - this is just a question, not a command
                    # Auto-resume execution without confirmation
                    # The special marker "AUTO_RESUME" signals the caller to continue
                    return True, "__AUTO_RESUME__"

                # Extract user's original request for context
                user_request = ""
                if st.session_state.agent_state.get("messages"):
                    for msg in reversed(st.session_state.agent_state["messages"]):
                        if hasattr(msg, "type") and msg.type == "human":
                            if isinstance(msg.content, str):
                                user_request = msg.content
                            break
                return True, user_request
    except Exception as e:
        st.error(f"[DEBUG] Exception: {e}")

    return False, ""


def _extract_command_info() -> str:
    """Extract command information from pending CLI tool calls.

    Returns:
        Formatted string with command details, or empty string if none found
    """
    try:
        # Get the snapshot to access state
        snapshot = st.session_state.agent.graph.get_state(st.session_state.config)  # type: ignore[attr-defined]

        # Check if CLI agent is about to run
        if (
            hasattr(snapshot, "next")
            and snapshot.next
            and "cli_agent" in str(snapshot.next)
            and hasattr(snapshot, "values")
            and "messages" in snapshot.values
        ):
            messages = snapshot.values["messages"]

            # Temporarily invoke the CLI agent to get its plan
            # Use a separate thread ID so we don't affect the main conversation
            try:
                # Check if agent has cli_agent attribute (SupervisorAgent)
                if hasattr(st.session_state.agent, "cli_agent"):
                    cli_agent = st.session_state.agent.cli_agent  # type: ignore[attr-defined]
                    # Invoke CLI agent to get the plan (will be interrupted at tool_node)
                    cli_agent_state = cast(MessagesState, {"messages": messages})
                    cli_config = {
                        "configurable": {"thread_id": "cli_preview_streamlit"}
                    }

                    # Invoke once - it will be interrupted before tool execution
                    _ = cli_agent.invoke(cli_agent_state, cli_config)

                    # Check the CLI agent's state for tool calls
                    cli_snapshot = cli_agent.agent.get_state(cli_config)  # type: ignore[union-attr]

                    if (
                        hasattr(cli_snapshot, "values")
                        and "messages" in cli_snapshot.values
                    ):
                        cli_messages = cli_snapshot.values["messages"]

                        # Look for tool calls in the most recent AI message
                        for msg in reversed(cli_messages):
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                return _format_tool_calls_for_display(msg.tool_calls)

            except Exception:
                # Silently fail and fall back to checking main state
                pass

        # Fallback: Look for tool calls in main state messages
        for msg in reversed(st.session_state.agent_state.get("messages", [])):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return _format_tool_calls_for_display(msg.tool_calls)

    except Exception:
        # Silently fail - just return empty string
        pass

    return ""


def _format_tool_calls_for_display(tool_calls: list) -> str:
    """Format tool calls into a readable string for display.

    Args:
        tool_calls: List of tool call dictionaries

    Returns:
        Formatted string with command details
    """
    info_lines = []
    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "Unknown")
        args = tool_call.get("args", {})

        if tool_name == "execute_cli_command":
            command = args.get("command", "N/A")
            working_dir = args.get("working_dir")
            timeout = args.get("timeout", 300)

            info_lines.append("**Command to execute:**")
            info_lines.append(f"```bash\n$ {command}\n```")
            if working_dir:
                info_lines.append(f"**Working directory:** `{working_dir}`")
            info_lines.append(f"**Timeout:** {timeout}s")

        elif tool_name == "check_cli_tool_available":
            tool = args.get("tool_name", "N/A")
            info_lines.append(f"**Checking availability of tool:** `{tool}`")

        elif tool_name == "list_directory_contents":
            directory = args.get("directory_path", "N/A")
            pattern = args.get("pattern")
            info_lines.append(f"**Listing directory:** `{directory}`")
            if pattern:
                info_lines.append(f"**Pattern:** `{pattern}`")

        else:
            info_lines.append(f"**Tool:** `{tool_name}`")
            if args:
                info_lines.append(f"**Arguments:** `{args}`")

    return "\n".join(info_lines)


def _show_confirmation_prompt(user_request: str) -> None:
    """Show confirmation prompt in Streamlit UI.

    Args:
        user_request: The user's original request
    """
    st.session_state.waiting_for_confirmation = True

    # Build confirmation message
    msg_parts = ["⚠️ **CLI Command Execution Pending**\n"]

    if user_request:
        msg_parts.append(f"Based on your request:\n> *{user_request}*\n")

    # Extract and display the command information
    command_info = _extract_command_info()
    if command_info:
        msg_parts.append(f"\n{command_info}\n")
    else:
        msg_parts.append("\nThe agent will execute a command-line tool.\n")

    msg_parts.append("\nType **'yes'** to proceed or **'no'** to cancel.")

    confirm_msg = "\n".join(msg_parts)
    st.warning(confirm_msg)
    st.session_state.messages.append({"role": "assistant", "content": confirm_msg})


def _format_and_display_messages(new_messages: list) -> str:
    """Format and display new messages from the agent.

    Args:
        new_messages: List of new messages to display

    Returns:
        Formatted response string
    """
    response_parts = []
    for message in new_messages:
        if isinstance(message, HumanMessage):
            continue
        formatted = format_ai_message(message)
        if formatted:
            response_parts.append(formatted)

    full_response = "\n\n".join(response_parts)
    if full_response:
        # Extract and display validation warnings as separate Streamlit components
        cleaned_response = _extract_and_display_validation_warnings(full_response)
        # Display the cleaned response (without validation block)
        st.markdown(cleaned_response)
        # Display media files
        display_response_media(cleaned_response)
    return full_response


def _handle_confirmation_response(user_input: str) -> None:
    """Handle user's confirmation response (yes/no).

    Args:
        user_input: User's confirmation response
    """
    user_response = user_input.lower().strip()

    # Display the confirmation response
    with st.chat_message("user"):
        st.markdown(user_input)

    if user_response in ["yes", "y", "si", "sì", "ok", "proceed", "confermo", "certo"]:
        # User confirmed - resume execution
        with st.chat_message("assistant"), st.spinner("Executing command..."):
            try:
                result = st.session_state.agent.invoke(None, st.session_state.config)  # type: ignore[arg-type]
                st.session_state.waiting_for_confirmation = False
                st.session_state.agent_state = result

                # Use the stored messages_before count to display only new messages
                messages_before = st.session_state.get(
                    "messages_before_confirmation", 0
                )
                new_messages = st.session_state.agent_state["messages"][
                    messages_before:
                ]

                full_response = _format_and_display_messages(new_messages)
                if full_response:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
            except Exception as e:
                st.error(f"Error executing command: {e}")
                st.session_state.waiting_for_confirmation = False
    else:
        # User cancelled
        st.session_state.waiting_for_confirmation = False
        with st.chat_message("assistant"):
            cancel_msg = "❌ Command execution cancelled."
            st.markdown(cancel_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": cancel_msg}
            )


def process_user_input(user_input: str) -> None:
    """Process user input and generate response.

    Args:
        user_input: The user's message
    """
    # Check if we're waiting for confirmation from a previous interrupt
    if st.session_state.waiting_for_confirmation:
        _handle_confirmation_response(user_input)
        _save_active_chat_to_storage()  # Save after confirmation response
        return

    # Normal flow - add user message to display
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Save user message to database
    if st.session_state.active_chat_id:
        db = _get_db()
        db.add_message(
            conversation_id=st.session_state.active_chat_id,
            role="user",
            content=user_input,
        )

    # Add user message to agent state
    st.session_state.agent_state["messages"].append(HumanMessage(content=user_input))

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            # Track messages before invocation
            messages_before = len(st.session_state.agent_state["messages"])

            # Invoke the agent
            result = st.session_state.agent.invoke(
                st.session_state.agent_state, st.session_state.config
            )

            # Check if graph was interrupted for confirmation
            is_interrupted, user_request = _check_streamlit_interrupt()

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
                    _show_confirmation_prompt(user_request)
                    _save_active_chat_to_storage()  # Save before waiting
                    return  # Wait for user's confirmation response

            # Update agent state
            st.session_state.agent_state = result

            # Get and display new messages
            new_messages = st.session_state.agent_state["messages"][messages_before:]
            full_response = _format_and_display_messages(new_messages)

            if full_response:
                # Save to display history
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

                # Save assistant message to database
                if st.session_state.active_chat_id:
                    db = _get_db()
                    db.add_message(
                        conversation_id=st.session_state.active_chat_id,
                        role="assistant",
                        content=full_response,
                    )
            else:
                st.info("Agent is processing... (no response yet)")

            # Save chat state after successful interaction
            _save_active_chat_to_storage()

        except Exception as e:
            error_msg = f"❌ **Error:** {e!s}"
            st.error(error_msg)
            # Remove the last user message on error
            if st.session_state.agent_state["messages"]:
                st.session_state.agent_state["messages"].pop()
            # Save even on error
            _save_active_chat_to_storage()


def render_sidebar() -> None:
    """Render the sidebar with chat management controls."""
    # Constants for chat display
    max_title_length = 35
    truncated_title_length = 32

    # New Chat button at the top
    if st.button(
        "+ New Chat", key="new_chat_btn", use_container_width=True, type="primary"
    ):
        # Save current chat before creating new one
        _save_active_chat_to_storage()
        # Create new chat without auto-generated name (will be generated from first message)
        _create_new_chat()
        # Navigate to chat page
        st.session_state._switch_to_chat_page = True
        # The new chat is now active and will show empty message list
        st.rerun()

    st.markdown("---")

    # Filter chats to only show those with at least 1 message OR the active chat
    # (so that newly created empty chats are visible)
    chats_with_messages = {
        chat_id: chat_data
        for chat_id, chat_data in st.session_state.chats.items()
        if len(chat_data.get("messages", [])) >= 1
        or chat_id == st.session_state.active_chat_id
    }

    # Sort chats by created time (newest first)
    sorted_chats = sorted(
        chats_with_messages.items(),
        key=lambda x: x[1].get("created_at", datetime.datetime.now()),
        reverse=True,
    )

    # Display each chat as a clickable item
    for chat_id, chat_data in sorted_chats:
        title = chat_data["title"]
        is_active = chat_id == st.session_state.active_chat_id

        # Truncate title if too long
        display_title = title
        if len(title) > max_title_length:
            display_title = f"{title[:truncated_title_length]}..."

        # Create a container for each chat item
        col1, col2 = st.columns([9, 1])

        with col1:
            # All chats show as buttons for consistent positioning
            button_clicked = st.button(
                display_title,
                key=f"chat_{chat_id}",
                use_container_width=True,
                type="secondary",
                disabled=False,  # Always clickable to allow navigation from other pages
            )
            # Switch chat if clicked and not active, or navigate if active but not on chat page
            if button_clicked:
                if not is_active:
                    # Switching to a different chat
                    _switch_to_chat(chat_id)
                    st.rerun()
                else:
                    # Active chat clicked - just navigate to chat page
                    st.session_state._switch_to_chat_page = True
                    st.rerun()

        with col2:
            # Simple x button for delete
            if st.button(
                "x",
                key=f"delete_{chat_id}",
                help="Delete",
                disabled=is_active and len(chats_with_messages) == 1,
                type="secondary",
            ):
                _delete_chat(chat_id)
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
        /* Set sidebar width for chat management */
        [data-testid="stSidebar"] {
            min-width: 300px;
            max-width: 300px;
        }
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
        wandb.render,
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
