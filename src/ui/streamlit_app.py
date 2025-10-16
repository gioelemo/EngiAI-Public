"""
Streamlit UI for the Engineer Assistant chatbot.

This provides a web-based chat interface for interacting with the multi-agent system.
"""

import re
import sys
import warnings
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image

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


def find_images_in_text(text: str) -> list[Path]:
    """Find image file paths mentioned in text.

    Args:
        text: Text that may contain file paths

    Returns:
        List of valid image file paths
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg"}
    image_paths = []

    # Look for common path patterns
    patterns = [
        r"outputs/[\w\-_.]+\.(?:png|jpg|jpeg|gif|bmp|svg)",
        r"[\w\-_.]+\.(?:png|jpg|jpeg|gif|bmp|svg)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Try as absolute path first
            path = Path(match)
            if not path.is_absolute():
                # Try relative to project root
                path = project_root / match

            if path.exists() and path.suffix.lower() in image_extensions:
                image_paths.append(path)

    return list(set(image_paths))  # Remove duplicates


def display_message(message: dict) -> None:
    """Display a single message in the chat interface.

    Args:
        message: Dictionary with 'role' and 'content' keys
    """
    with st.chat_message(message["role"]):
        # Display text content
        st.markdown(message["content"])

        # Check for and display any images mentioned in the message
        images = find_images_in_text(message["content"])
        if images:
            for img_path in images:
                try:
                    image = Image.open(img_path)
                    st.image(
                        image,
                        caption=img_path.name,
                        width="stretch",
                    )
                except Exception as e:
                    st.warning(f"Could not display image {img_path.name}: {e}")


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

                # Check for and display any images mentioned in the response
                images = find_images_in_text(full_response)
                if images:
                    for img_path in images:
                        try:
                            image = Image.open(img_path)
                            st.image(
                                image,
                                caption=img_path.name,
                                width="stretch",
                            )
                        except Exception as e:
                            st.warning(f"Could not display image {img_path.name}: {e}")

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
    st.title("🤖 Engineer Assistant")
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

    # Clear conversation button
    if st.button("🗑️ Clear Conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.agent_state = {"messages": []}
        st.rerun()

    st.markdown("---")

    # Image gallery section
    render_image_gallery()

    # Info section
    render_info_section()

    # Download section for outputs
    render_downloads_section()


def render_image_gallery() -> None:
    """Render the image gallery in the sidebar."""
    output_dir = Path("outputs")
    if not output_dir.exists():
        return

    image_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
    if not image_files:
        return

    with st.expander("🖼️ Image Gallery", expanded=False):
        st.markdown("**Generated Designs & Results**")
        for img_path in sorted(image_files):
            try:
                image = Image.open(img_path)
                st.image(
                    image,
                    caption=img_path.name,
                    width="stretch",
                )
                # Add download button for each image
                with img_path.open("rb") as f:
                    st.download_button(
                        label=f"⬇️ {img_path.name}",
                        data=f,
                        file_name=img_path.name,
                        mime="image/png",
                        key=f"img_{img_path.name}",
                        width="stretch",
                    )
                st.markdown("---")
            except Exception as e:
                st.warning(f"Could not load {img_path.name}: {e}")


def render_info_section() -> None:
    """Render the info/about section."""
    with st.expander("ℹ About"):  # noqa: RUF001
        st.markdown(
            """
        This is a Jarvis-style multimodal AI assistant for
        mechanical engineering design and manufacturing.

        **Capabilities:**
        - Structural optimization
        - Design generation
        - 3D model export (STL)
        - Engineering research
        - Code execution
        """
        )


def render_downloads_section() -> None:
    """Render the downloads section for .npy files."""
    output_dir = Path("outputs")
    if not output_dir.exists():
        return

    with st.expander("📥 Download Outputs"):
        for file_path in output_dir.glob("*.npy"):
            with file_path.open("rb") as f:
                st.download_button(
                    label=f"⬇️ {file_path.name}",
                    data=f,
                    file_name=file_path.name,
                    mime="application/octet-stream",
                    width="stretch",
                )


def render_quick_start_examples() -> None:
    """Render quick start example buttons."""
    st.markdown("### 🎯 Quick Start Examples")
    st.markdown("Click any example below to get started:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "🔧 Optimize a 2D beam structure",
            width="stretch",
        ):
            process_user_input("Optimize a 2D beam structure for minimum compliance")

        if st.button(
            "🔍 Search topology optimization",
            width="stretch",
        ):
            process_user_input("What are the latest advances in topology optimization?")

    with col2:
        if st.button(
            "🏗️ Convert design to STL",
            width="stretch",
        ):
            process_user_input("Convert the optimized beam design to STL format")

        if st.button(
            "📊 Multi-step workflow",
            width="stretch",
        ):
            process_user_input("Optimize a beam structure and then convert it to STL")

    st.markdown("---")


def main() -> None:
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Engineer Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        render_sidebar()

    # Main chat interface
    st.title("💬 Chat with Engineer Assistant")

    # Show example prompts if no messages yet
    if not st.session_state.messages:
        render_quick_start_examples()

    # Display chat history
    for message in st.session_state.messages:
        display_message(message)

    # Chat input
    if prompt := st.chat_input("Ask me anything about engineering design..."):
        process_user_input(prompt)


if __name__ == "__main__":
    main()
