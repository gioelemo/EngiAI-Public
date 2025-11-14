"""
Streamlit UI for the Engineer Assistant chatbot.

This provides a web-based chat interface for interacting with the multi-agent system.
"""

import base64
import datetime
import logging
import sys
import tempfile
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
from src.tools import EngineerRAGStore, MultimodalDocumentProcessor  # noqa: E402
from src.ui import chat, home, settings, wandb  # noqa: E402
from src.ui.chat_management import (  # noqa: E402
    create_new_chat,
    delete_chat,
    get_db,
    initialize_chat_state,
    load_chats_from_database,
    save_active_chat_to_storage,
    switch_to_chat,
)
from src.ui.confirmation_handler import (  # noqa: E402
    check_streamlit_interrupt,
    handle_confirmation_response,
    show_confirmation_prompt,
)
from src.ui.database import DatabaseManager  # noqa: E402
from src.ui.file_processing import (  # noqa: E402
    extract_pdf_text,
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

    # Normal flow - add user message to display
    message_dict: dict[str, Any] = {"role": "user", "content": text_content}
    if images_for_display:
        message_dict["images"] = images_for_display
    st.session_state.messages.append(message_dict)

    # Save user message to database
    if st.session_state.active_chat_id:
        db = get_db()
        db.add_message(
            conversation_id=st.session_state.active_chat_id,
            role="user",
            content=text_content,
            images=images_for_display if images_for_display else None,
        )

    # Check if we have PDFs - they will be extracted and added as text
    has_pdfs = any(f.get("type") == "application/pdf" for f in images_for_display)

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

            if pdf_files:
                # Extract PDF text using MathPixPDFLoader AND add to RAG vector store
                try:
                    pdf_text = extract_pdf_text(pdf_files)

                    # ALSO add PDFs to RAG system for persistent storage
                    processor = MultimodalDocumentProcessor()
                    vector_store = EngineerRAGStore(collection_name="engineer_docs")

                    # Process and store each PDF in the RAG system
                    for pdf_file in pdf_files:
                        pdf_bytes = base64.b64decode(pdf_file["data"])
                        file_name = pdf_file.get("name", "document.pdf")

                        # Write to temporary file for processing
                        with tempfile.NamedTemporaryFile(
                            suffix=".pdf", delete=False
                        ) as tmp_file:
                            tmp_file.write(pdf_bytes)
                            tmp_path = tmp_file.name

                        try:
                            # Process PDF and add to vector store
                            docs = processor.process_file(tmp_path)

                            # Update source metadata to use original filename instead of temp path
                            for doc in docs:
                                doc.metadata["source"] = file_name
                                doc.metadata["original_name"] = file_name

                            # Add to vector store
                            vector_store.add_documents(docs)

                            # Count non-empty documents
                            non_empty = sum(
                                1 for doc in docs if doc.page_content.strip()
                            )
                            st.success(
                                f"✓ Added '{file_name}' to knowledge base ({non_empty} text chunks)"
                            )

                        except Exception as e:
                            st.error(f"Error processing '{file_name}': {e!s}")
                            logger.exception(f"Failed to process {file_name}")
                        finally:
                            # Clean up temp file
                            tmp_file_path = Path(tmp_path)
                            if tmp_file_path.exists():
                                tmp_file_path.unlink()

                    # Update the user message in agent state to include PDF content
                    # Remove the last message (text-only) and replace with PDF-enhanced version
                    st.session_state.agent_state["messages"].pop()

                    # Create enhanced message with PDF text
                    enhanced_content = (
                        f"PDF Content:\n\n{pdf_text}\n\nUser Question: {text_content}"
                    )
                    st.session_state.agent_state["messages"].append(
                        HumanMessage(content=enhanced_content)
                    )

                except Exception as e:
                    st.error(f"Error processing PDF: {e!s}")
                    # Continue with regular processing

            # Track messages before invocation
            messages_before = len(st.session_state.agent_state["messages"])

            # Invoke the agent
            result = st.session_state.agent.invoke(
                st.session_state.agent_state, st.session_state.config
            )

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

            # Get and display new messages
            new_messages = st.session_state.agent_state["messages"][messages_before:]
            full_response, suggested_prompts = format_and_display_messages(new_messages)

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


def render_sidebar() -> None:
    """Render the sidebar with chat management controls."""
    # Constants for chat display
    max_title_length = 35
    truncated_title_length = 32

    # New Chat button at the top
    if st.button("+ New Chat", key="new_chat_btn", width="stretch", type="primary"):
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

        with col2:
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
