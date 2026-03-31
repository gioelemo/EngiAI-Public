"""
Tier-2 UI tests: functions that read/write st.session_state.

Strategy: patch `src.ui.chat_management.st` with a lightweight fake that
supports attribute access, item access, and `.get()` — mirroring
Streamlit's SessionState API without starting a Streamlit runtime.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.ui.chat_management import (
    save_active_chat_to_storage,
    sync_active_chat_to_session,
)
from tests.test_ui.conftest import make_st

# ============================================================================
# Helpers
# ============================================================================


def _sample_chat(
    messages=None,
    title="Test Chat",
    saved_to_db=True,
    voice_id=None,
    voice_provider=None,
):
    chat_id = "chat-abc-123"
    return chat_id, {
        "id": chat_id,
        "title": title,
        "messages": messages if messages is not None else [],
        "agent": MagicMock(),
        "agent_state": {"messages": []},
        "config": {"configurable": {"thread_id": chat_id}},
        "waiting_for_confirmation": False,
        "saved_to_db": saved_to_db,
        "voice_id": voice_id,
        "voice_provider": voice_provider,
    }


# ============================================================================
# sync_active_chat_to_session
# ============================================================================


@pytest.mark.unit
def test_sync_noop_when_active_chat_id_is_none():
    """active_chat_id is None → guard fails, session state unchanged."""
    st = make_st(active_chat_id=None, chats={})
    with patch("src.ui.chat_management.st", st):
        sync_active_chat_to_session()

    # Nothing should have been set (no 'messages' key in store)
    assert "messages" not in st.session_state


@pytest.mark.unit
def test_sync_noop_when_active_chat_not_in_chats():
    """active_chat_id not present in chats → guard fails, session state unchanged."""
    st = make_st(active_chat_id="missing-id", chats={})
    with patch("src.ui.chat_management.st", st):
        sync_active_chat_to_session()

    assert "messages" not in st.session_state


@pytest.mark.unit
def test_sync_copies_core_fields():
    """Valid active chat → messages, agent, agent_state, config, waiting all synced."""
    chat_id, chat = _sample_chat(messages=[{"role": "user", "content": "hi"}])
    st = make_st(active_chat_id=chat_id, chats={chat_id: chat})

    with patch("src.ui.chat_management.st", st):
        sync_active_chat_to_session()

    assert st.session_state.messages == chat["messages"]
    assert st.session_state.agent is chat["agent"]
    assert st.session_state.agent_state == chat["agent_state"]
    assert st.session_state.config == chat["config"]
    assert st.session_state.waiting_for_confirmation is False


@pytest.mark.unit
def test_sync_sets_switching_chat_flag():
    """After sync, _switching_chat must be True to suppress audio auto-play."""
    chat_id, chat = _sample_chat()
    st = make_st(active_chat_id=chat_id, chats={chat_id: chat})

    with patch("src.ui.chat_management.st", st):
        sync_active_chat_to_session()

    assert st.session_state._switching_chat is True


@pytest.mark.unit
def test_sync_restores_voice_id_when_present():
    """voice_id present in chat → voice_selected updated in session state."""
    chat_id, chat = _sample_chat(voice_id="George", voice_provider="elevenlabs")
    st = make_st(active_chat_id=chat_id, chats={chat_id: chat})

    with patch("src.ui.chat_management.st", st):
        sync_active_chat_to_session()

    assert st.session_state.voice_selected == "George"
    assert st.session_state.voice_provider == "elevenlabs"


@pytest.mark.unit
def test_sync_does_not_overwrite_voice_when_none():
    """voice_id is None in chat → voice_selected must NOT be touched."""
    chat_id, chat = _sample_chat(voice_id=None)
    st = make_st(active_chat_id=chat_id, chats={chat_id: chat}, voice_selected="alloy")

    with patch("src.ui.chat_management.st", st):
        sync_active_chat_to_session()

    # Original voice_selected untouched
    assert st.session_state.voice_selected == "alloy"


# ============================================================================
# save_active_chat_to_storage — early-return paths (no DB needed)
# ============================================================================


@pytest.mark.unit
def test_save_noop_when_active_chat_id_is_none():
    """active_chat_id is None → function exits without touching anything."""
    st = make_st(active_chat_id=None, chats={})
    mock_db = MagicMock()

    with (
        patch("src.ui.chat_management.st", st),
        patch("src.ui.chat_management.get_db", return_value=mock_db),
    ):
        save_active_chat_to_storage()

    mock_db.save_conversation_state.assert_not_called()


@pytest.mark.unit
def test_save_early_return_when_chat_has_no_messages():
    """Chat with 0 messages → early return, no DB write."""
    chat_id, chat = _sample_chat(messages=[])
    st = make_st(
        active_chat_id=chat_id,
        chats={chat_id: chat},
        messages=[],
        agent=MagicMock(),
        agent_state={"messages": []},
        config={},
        waiting_for_confirmation=False,
    )
    mock_db = MagicMock()

    with (
        patch("src.ui.chat_management.st", st),
        patch("src.ui.chat_management.get_db", return_value=mock_db),
    ):
        save_active_chat_to_storage()

    mock_db.save_conversation_state.assert_not_called()
    mock_db.create_conversation.assert_not_called()


# ============================================================================
# save_active_chat_to_storage — DB-write paths
# ============================================================================


@pytest.mark.unit
def test_save_calls_create_conversation_when_not_yet_in_db():
    """Chat not yet persisted → db.create_conversation called and saved_to_db set True."""
    chat_id, chat = _sample_chat(
        messages=[{"role": "user", "content": "hello"}],
        title="My Chat",
        saved_to_db=False,
    )
    st = make_st(
        active_chat_id=chat_id,
        chats={chat_id: chat},
        messages=chat["messages"],
        agent=MagicMock(),
        agent_state={"messages": []},
        config={"configurable": {"thread_id": chat_id}},
        waiting_for_confirmation=False,
    )
    mock_db = MagicMock()

    with (
        patch("src.ui.chat_management.st", st),
        patch("src.ui.chat_management.get_db", return_value=mock_db),
        patch("src.ui.chat_management.generate_chat_title", return_value="My Chat"),
    ):
        save_active_chat_to_storage()

    mock_db.create_conversation.assert_called_once()
    assert chat["saved_to_db"] is True


@pytest.mark.unit
def test_save_calls_update_voice_when_already_in_db():
    """Chat already persisted → db.update_conversation_voice called, not create_conversation."""
    chat_id, chat = _sample_chat(
        messages=[{"role": "user", "content": "hello"}],
        title="My Chat",
        saved_to_db=True,
    )
    st = make_st(
        active_chat_id=chat_id,
        chats={chat_id: chat},
        messages=chat["messages"],
        agent=MagicMock(),
        agent_state={"messages": []},
        config={"configurable": {"thread_id": chat_id}},
        waiting_for_confirmation=False,
    )
    mock_db = MagicMock()

    with (
        patch("src.ui.chat_management.st", st),
        patch("src.ui.chat_management.get_db", return_value=mock_db),
        patch("src.ui.chat_management.generate_chat_title", return_value="My Chat"),
    ):
        save_active_chat_to_storage()

    mock_db.create_conversation.assert_not_called()
    mock_db.update_conversation_voice.assert_called_once()


@pytest.mark.unit
def test_save_generates_title_for_new_chat_prefix():
    """Chat title starting with 'New Chat' → generate_chat_title called with first user message."""
    chat_id, chat = _sample_chat(
        messages=[{"role": "user", "content": "optimize a beam"}],
        title="New Chat 1",
        saved_to_db=True,
    )
    st = make_st(
        active_chat_id=chat_id,
        chats={chat_id: chat},
        messages=chat["messages"],
        agent=MagicMock(),
        agent_state={"messages": []},
        config={"configurable": {"thread_id": chat_id}},
        waiting_for_confirmation=False,
    )
    mock_db = MagicMock()

    with (
        patch("src.ui.chat_management.st", st),
        patch("src.ui.chat_management.get_db", return_value=mock_db),
        patch(
            "src.ui.chat_management.generate_chat_title",
            return_value="Beam Optimization",
        ) as mock_title,
    ):
        save_active_chat_to_storage()

    mock_title.assert_called_once_with("optimize a beam")
    assert chat["title"] == "Beam Optimization"


@pytest.mark.unit
def test_save_skips_title_generation_for_custom_title():
    """Chat with a custom title → generate_chat_title NOT called."""
    chat_id, chat = _sample_chat(
        messages=[{"role": "user", "content": "hello"}],
        title="My Custom Title",
        saved_to_db=True,
    )
    st = make_st(
        active_chat_id=chat_id,
        chats={chat_id: chat},
        messages=chat["messages"],
        agent=MagicMock(),
        agent_state={"messages": []},
        config={"configurable": {"thread_id": chat_id}},
        waiting_for_confirmation=False,
    )
    mock_db = MagicMock()

    with (
        patch("src.ui.chat_management.st", st),
        patch("src.ui.chat_management.get_db", return_value=mock_db),
        patch("src.ui.chat_management.generate_chat_title") as mock_title,
    ):
        save_active_chat_to_storage()

    mock_title.assert_not_called()
    assert chat["title"] == "My Custom Title"


@pytest.mark.unit
def test_save_always_calls_save_conversation_state():
    """For any chat with messages, db.save_conversation_state is always called."""
    chat_id, chat = _sample_chat(
        messages=[{"role": "user", "content": "hi"}],
        title="My Chat",
        saved_to_db=True,
    )
    agent_state = {"messages": []}
    config = {"configurable": {"thread_id": chat_id}}
    st = make_st(
        active_chat_id=chat_id,
        chats={chat_id: chat},
        messages=chat["messages"],
        agent=MagicMock(),
        agent_state=agent_state,
        config=config,
        waiting_for_confirmation=False,
    )
    mock_db = MagicMock()

    with (
        patch("src.ui.chat_management.st", st),
        patch("src.ui.chat_management.get_db", return_value=mock_db),
        patch("src.ui.chat_management.generate_chat_title", return_value="My Chat"),
    ):
        save_active_chat_to_storage()

    mock_db.save_conversation_state.assert_called_once_with(
        conversation_id=chat_id,
        agent_state=agent_state,
        config=config,
        waiting_for_confirmation=False,
    )
