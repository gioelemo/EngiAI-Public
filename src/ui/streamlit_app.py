"""
Streamlit UI for the Engineer Assistant chatbot.

This provides a web-based chat interface for interacting with the multi-agent system.
"""

import contextlib
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image
from streamlit_stl import stl_from_file  # type: ignore[import-untyped]

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.supervisor_agent import SupervisorAgent  # noqa: E402

# Suppress Pydantic warnings from LangChain
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic._internal._generate_schema"
)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent" not in st.session_state:
        st.session_state.agent = SupervisorAgent()

    if "agent_state" not in st.session_state:
        st.session_state.agent_state = {"messages": []}

    if "config" not in st.session_state:
        st.session_state.config = {"configurable": {"thread_id": "streamlit-session"}}

    # STL viewer settings
    if "stl_color" not in st.session_state:
        st.session_state.stl_color = "#0069B4"

    if "stl_material" not in st.session_state:
        st.session_state.stl_material = "material"

    if "stl_height" not in st.session_state:
        st.session_state.stl_height = 400

    if "stl_auto_rotate" not in st.session_state:
        st.session_state.stl_auto_rotate = True

    if "stl_opacity" not in st.session_state:
        st.session_state.stl_opacity = 1.0

    if "stl_shininess" not in st.session_state:
        st.session_state.stl_shininess = 100

        st.session_state.stl_auto_rotate = True


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
        List of valid STL file paths
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

            if path.exists() and path.suffix.lower() == ".stl":
                stl_paths.append(path)

    return list(set(stl_paths))  # Remove duplicates


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
    try:
        image = Image.open(img_path)
        st.image(image, caption=img_path.name, width=500)
        _render_download_save_controls(img_path, "image", button_key_prefix)
    except Exception as e:
        st.warning(f"Could not display image {img_path.name}: {e}")


def _display_stl(stl_path: Path, idx: int, button_key_prefix: str) -> None:
    """Display an STL file with viewer and download/save controls.

    Args:
        stl_path: Path to STL file
        idx: Index for unique key generation
        button_key_prefix: Unique prefix for button keys
    """
    if not stl_path.exists():
        st.info(f"3D model file not found: {stl_path.name}")
        return

    try:
        st.markdown(f"**3D Model: {stl_path.name}**")
        stable_key = f"stl_{abs(hash(str(stl_path)))}_{idx}"
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
        st.warning(f"3D model file not found: {stl_path.name}")
    except Exception as e:
        st.warning(f"Could not display 3D model {stl_path.name}: {e}")


def display_message(message: dict) -> None:
    """Display a single message in the chat interface.

    Args:
        message: Dictionary with 'role' and 'content' keys
    """
    with st.chat_message(message["role"]):
        # Display text content
        st.markdown(message["content"])

        # Display images
        images = find_images_in_text(message["content"])
        for img_path in images:
            _display_image(img_path, button_key_prefix="msg_img")

        # Display STL files
        stl_files = find_stl_files_in_text(message["content"])
        for idx, stl_path in enumerate(stl_files):
            _display_stl(stl_path, idx, button_key_prefix="msg_stl")


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


def display_response_media(response_text: str) -> None:
    """Display images and STL files found in response text.

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


def process_user_input(user_input: str) -> None:
    """Process user input and generate response.

    Args:
        user_input: The user's message
    """
    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": user_input})

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

            # Update agent state
            st.session_state.agent_state = result

            # Get new messages
            new_messages = st.session_state.agent_state["messages"][messages_before:]

            # Format and display response
            response_parts = []
            for message in new_messages:
                # Skip re-displaying user messages
                if isinstance(message, HumanMessage):
                    continue

                formatted = format_ai_message(message)
                if formatted:
                    response_parts.append(formatted)

            # Combine all response parts
            full_response = "\n\n".join(response_parts)

            if full_response:
                st.markdown(full_response)
                display_response_media(full_response)

                # Save to display history
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                st.info("Agent is processing... (no response yet)")

        except Exception as e:
            error_msg = f"❌ **Error:** {e!s}"
            st.error(error_msg)
            # Remove the last user message on error
            if st.session_state.agent_state["messages"]:
                st.session_state.agent_state["messages"].pop()


def render_sidebar() -> None:
    """Render the sidebar with controls and galleries."""
    # Display logo at the top
    logo_path = project_root / "assets" / "logo.png"
    if logo_path.exists():
        try:
            logo = Image.open(logo_path)
            st.image(logo, width="stretch")
        except Exception:
            # Fallback to text title if logo fails to load
            st.title("🤖 EngiAI")
    else:
        st.title("🤖 EngiAI")

    st.markdown("**Engineering Design Chatbot**")
    st.markdown("---")

    # Media save settings
    with st.expander("\U0001f4be Media saving", expanded=False):
        st.markdown(
            "Choose where displayed media (images/STL) will be saved on the server"
        )
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

    st.markdown(
        """
    ### Multi-Agent System

    This assistant coordinates specialized agents:

    - 🔧 **Engineering Agent**: Optimization & design
    - 🏗️ **CAD Agent**: STL conversion & 3D printing
    - 🔍 **Search Agent**: Research & information
    """
    )

    st.markdown("---")

    # STL Viewer Settings
    with st.expander("🎨 3D Viewer Settings", expanded=True):
        st.markdown("**Customize 3D Model Display**")
        st.caption("⚠️ Note: Changing settings will reload all 3D models")

        # Color picker - directly update session state
        st.session_state.stl_color = st.color_picker(
            "Model Color",
            value=st.session_state.stl_color,
            help="Choose a color for your 3D models",
            key="color_picker_widget",
        )

        # Material selector - directly update session state
        st.session_state.stl_material = st.selectbox(
            "Material",
            options=["material", "flat", "wireframe"],
            index=["material", "flat", "wireframe"].index(
                st.session_state.stl_material
            ),
            help="material: smooth shading, flat: faceted look, wireframe: mesh structure",
            key="material_selector_widget",
        )

        # Height slider - directly update session state
        st.session_state.stl_height = st.slider(
            "Viewer Height (px)",
            min_value=200,
            max_value=800,
            value=st.session_state.stl_height,
            step=50,
            help="Adjust the height of the 3D viewer",
            key="height_slider_widget",
        )

        # Opacity slider
        st.session_state.stl_opacity = st.slider(
            "Opacity",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.stl_opacity,
            step=0.1,
            help="Adjust the transparency of the model (0 = transparent, 1 = opaque)",
            key="opacity_slider_widget",
        )

        # Shininess slider
        st.session_state.stl_shininess = st.slider(
            "Shininess",
            min_value=0,
            max_value=200,
            value=st.session_state.stl_shininess,
            step=10,
            help="Adjust the shininess/glossiness of the surface",
            key="shininess_slider_widget",
        )

        # Auto-rotate toggle
        if "stl_auto_rotate" not in st.session_state:
            st.session_state.stl_auto_rotate = True

        st.session_state.stl_auto_rotate = st.checkbox(
            "Auto-rotate models",
            value=st.session_state.stl_auto_rotate,
            help="Automatically rotate 3D models",
            key="auto_rotate_checkbox_widget",
        )

    st.markdown("---")

    # Clear conversation button
    if st.button("🗑️ Clear Conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.agent_state = {"messages": []}
        st.rerun()


def main() -> None:
    """Main Streamlit application."""
    # Try to use logo as page icon
    logo_path = project_root / "assets" / "engiai_logo.jpg"
    page_icon: str | Image.Image = "🤖"
    if logo_path.exists():
        try:
            page_icon = Image.open(logo_path)
        except Exception:
            page_icon = "🤖"

    # Page configuration
    st.set_page_config(
        page_title="EngiAI - Engineering Design Chatbot",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Initialize session state
    initialize_session_state()

    # Sidebar (collapsed by default)
    with st.sidebar:
        render_sidebar()

    # Main centered chat interface
    if not st.session_state.messages:
        # Welcome screen with logo - centered vertically
        st.markdown('<div class="welcome-content">', unsafe_allow_html=True)
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        if logo_path.exists():
            try:
                logo = Image.open(logo_path)
                # Center the logo using columns
                _, logo_col, _ = st.columns([1, 1, 1])
                with logo_col:
                    st.image(logo, width=140)
            except Exception:
                st.markdown(
                    '<h1 class="welcome-title">💬 EngiAI</h1>', unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<h1 class="welcome-title">💬 EngiAI</h1>', unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<p class="welcome-subtitle">Your AI-powered engineering design assistant</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Small logo at top when chat is active
        if logo_path.exists():
            try:
                logo = Image.open(logo_path)
                _, logo_col, _ = st.columns([2, 1, 2])
                with logo_col:
                    st.image(logo, width=60)
            except Exception:
                pass

        # Display chat history
        for message in st.session_state.messages:
            display_message(message)

    # Chat input (centered on welcome screen, bottom-fixed during chat)
    if prompt := st.chat_input("Ask me anything about engineering design..."):
        process_user_input(prompt)


if __name__ == "__main__":
    main()
