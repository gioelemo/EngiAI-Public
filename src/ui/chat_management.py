"""
Chat management functions for the Streamlit UI.

Handles chat creation, switching, deletion, and state management.
"""

import datetime
import logging
import uuid
from typing import Any

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from config import config
from src.agents.supervisor_agent import SupervisorAgent
from src.ui.database import DatabaseManager

logger = logging.getLogger(__name__)


def create_supervisor_for_chat() -> SupervisorAgent:
    """Create a new SupervisorAgent instance for a chat.

    Creates a separate agent instance for each chat to maintain conversation
    isolation. Each chat gets its own SupervisorAgent instance with independent
    conversation state and sub-agents (RAG, ArXiv, etc.).

    Knowledge persistence is now handled by MMORE which provides a shared
    knowledge base across all chats via the database.

    Returns:
        New SupervisorAgent instance
    """
    logger.info("Creating new SupervisorAgent for chat")

    # Create new agent instance
    # Each sub-agent (RAG, ArXiv) now uses MMORE which provides shared knowledge
    return SupervisorAgent()


_supervisor_state: dict[str, str | None] = {"init_error": None}


def get_supervisor_init_error() -> str | None:
    """Return the last SupervisorAgent initialization error message, or None."""
    return _supervisor_state["init_error"]


def _try_create_supervisor() -> "SupervisorAgent | None":
    """Create a SupervisorAgent, returning None if initialization fails (e.g. missing API key).

    Short-circuits after the first failure to avoid repeated attempts and log spam
    when loading many conversations from the database.
    """
    if _supervisor_state["init_error"] is not None:
        return None
    try:
        return create_supervisor_for_chat()
    except Exception as e:
        logger.exception("Failed to initialize SupervisorAgent")
        _supervisor_state["init_error"] = str(e)
        return None


def generate_chat_title(user_message: str) -> str:
    """Generate a concise title for a chat based on the first user message.

    Uses LangChain's init_chat_model to handle all providers (OpenAI, Google,
    Ollama, etc.) uniformly, consistent with how agents create LLM instances.

    Args:
        user_message: The first user message in the conversation

    Returns:
        A short, summarized title (max 50 characters)
    """
    max_title_length = 50
    truncate_at = 47

    try:
        llm = init_chat_model(config.llm_model, temperature=0.3)

        prompt = f"""Generate a very short title (maximum 4-5 words) that summarizes this question or request:

"{user_message}"

Reply with ONLY the title, nothing else. No quotes, no punctuation at the end."""

        response = llm.invoke([HumanMessage(content=prompt)])

        # Thinking models (e.g. Gemini 3 Flash) return content as a list of
        # blocks like [{"type": "text", "text": "..."}] instead of a string.
        raw_content = response.content
        if isinstance(raw_content, list):
            content = ""
            for block in raw_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    content = str(block["text"])
                    break
        else:
            content = str(raw_content)
        title = content.strip().strip("\"'")

        if len(title) > max_title_length:
            return title[:truncate_at] + "..."
        else:
            return title
    except Exception:
        logger.exception("Failed to generate chat title")
        if len(user_message) > max_title_length:
            return user_message[:truncate_at] + "..."
        return user_message


def get_db() -> DatabaseManager:
    """Get or create database manager instance.

    Returns:
        DatabaseManager instance
    """
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    return st.session_state.db_manager


def create_new_chat(name: str | None = None) -> str:
    """Create a new chat and return its ID.

    Args:
        name: Optional name for the chat. Auto-generated if None

    Returns:
        The ID of the newly created chat (UUID)
    """
    # Save current chat before creating new one
    old_chat_id = st.session_state.active_chat_id
    if old_chat_id:
        save_active_chat_to_storage()

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

    # Get current voice selection from session state
    current_voice = st.session_state.get("voice_selected", None)

    # Create new chat data in session
    st.session_state.chats[session_id] = {
        "id": session_id,
        "title": name,
        "created_at": datetime.datetime.now(),
        "messages": [],
        "agent_state": {"messages": []},
        "agent": _try_create_supervisor(),  # Each chat gets its own agent instance
        "config": {"configurable": {"thread_id": session_id}},
        "waiting_for_confirmation": False,
        "saved_to_db": False,  # Track if this chat has been saved to DB yet
        "pinned": False,
        "voice_id": current_voice,  # Store voice selection
    }

    # Set as active chat and sync (this will clear the display)
    st.session_state.active_chat_id = session_id
    sync_active_chat_to_session()

    return session_id


def sync_active_chat_to_session() -> None:
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
        # Restore voice selection for this conversation
        if "voice_id" in active_chat and active_chat["voice_id"] is not None:
            st.session_state.voice_selected = active_chat["voice_id"]
        if (
            "voice_provider" in active_chat
            and active_chat["voice_provider"] is not None
        ):
            st.session_state.voice_provider = active_chat["voice_provider"]
        # Set flag to prevent auto-playing audio when switching chats
        st.session_state._switching_chat = True


def save_active_chat_to_storage() -> None:
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
        # Save current voice selection and provider
        active_chat["voice_id"] = st.session_state.get("voice_selected", None)
        active_chat["voice_provider"] = st.session_state.get("voice_provider", None)
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
                new_title = generate_chat_title(first_user_msg)
                active_chat["title"] = new_title

        # Save to database
        db = get_db()

        # Create in database if not already saved
        if not active_chat.get("saved_to_db", False):
            db.create_conversation(
                name=active_chat["title"],
                session_id=st.session_state.active_chat_id,
                voice_id=active_chat.get("voice_id"),
                voice_provider=active_chat.get("voice_provider"),
            )
            active_chat["saved_to_db"] = True
        else:
            # Update voice if it changed
            db.update_conversation_voice(
                st.session_state.active_chat_id,
                active_chat.get("voice_id"),
                active_chat.get("voice_provider"),
            )

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


def switch_to_chat(chat_id: str) -> None:
    """Switch to a different chat and navigate to chat page.

    Args:
        chat_id: The ID of the chat to switch to
    """
    if chat_id in st.session_state.chats:
        # Save current chat first
        save_active_chat_to_storage()

        # Switch to new chat
        st.session_state.active_chat_id = chat_id
        sync_active_chat_to_session()

        # Store that we want to navigate to chat page
        st.session_state._switch_to_chat_page = True


def delete_chat(chat_id: str) -> None:
    """Delete a chat from session and database.

    Args:
        chat_id: The ID of the chat to delete
    """
    if chat_id in st.session_state.chats:
        # Delete from session
        del st.session_state.chats[chat_id]

        # Delete from database
        db = get_db()
        db.delete_conversation(chat_id)

        # If deleted chat was active, switch to another or create new
        if st.session_state.active_chat_id == chat_id:
            if st.session_state.chats:
                # Switch to first available chat
                first_chat_id = next(iter(st.session_state.chats.keys()))
                switch_to_chat(first_chat_id)
            else:
                # Create new chat if no chats remain
                create_new_chat()


def load_chats_from_database() -> None:
    """Load all conversations from database into session state."""
    try:
        db = get_db()
        conversations = db.get_all_conversations()

        for conv in conversations:
            chat_id = conv["id"]

            # Load messages
            messages = db.get_messages(chat_id)
            display_messages = []
            for msg in messages:
                display_msg = {"role": msg["role"], "content": msg["content"]}
                if msg.get("images"):
                    display_msg["images"] = msg["images"]
                if msg.get("suggested_prompts"):
                    display_msg["suggested_prompts"] = msg["suggested_prompts"]
                if msg.get("audio"):
                    display_msg["audio"] = msg["audio"]
                display_messages.append(display_msg)

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
                "agent": _try_create_supervisor(),  # Each chat gets its own agent instance
                "config": config,
                "waiting_for_confirmation": waiting_for_confirmation,
                "saved_to_db": True,  # Already in database
                "pinned": conv.get("pinned", False),
                "voice_id": conv.get("voice_id"),  # Restore voice selection
                "voice_provider": conv.get("voice_provider"),  # Restore voice provider
            }

        # Set the most recently updated as active
        if conversations and not st.session_state.active_chat_id:
            st.session_state.active_chat_id = conversations[0]["id"]
            sync_active_chat_to_session()

    except Exception:
        # Silently fail - if loading fails, start fresh
        pass


def toggle_pin_chat(chat_id: str) -> None:
    """Toggle the pinned status of a chat.

    Args:
        chat_id: The ID of the chat to pin/unpin
    """
    if chat_id in st.session_state.chats:
        # Toggle pinned status in database
        db = get_db()
        new_status = db.toggle_pin_conversation(chat_id)

        # Update session state
        st.session_state.chats[chat_id]["pinned"] = new_status

        # Reload chats to update order
        load_chats_from_database()
        sync_active_chat_to_session()


def initialize_chat_state() -> None:
    """Initialize chat-related session state."""
    # Initialize basic state first
    defaults: dict[str, Any] = {
        "chats": {},
        "active_chat_id": None,
        "chat_counter": 0,
        "messages": [],
        "agent_state": {"messages": []},
        "config": {"configurable": {"thread_id": "streamlit-session"}},
        "waiting_for_confirmation": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Initialize agent separately - each chat will get its own instance
    if "agent" not in st.session_state:
        st.session_state.agent = _try_create_supervisor()
