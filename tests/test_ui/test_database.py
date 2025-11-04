"""
Tests for database module (conversations, messages, states, and settings).

These tests use SQLite in-memory database for fast, isolated testing.
"""

import uuid
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.ui.database import DatabaseManager

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def db_manager():
    """Create an in-memory SQLite database for testing."""
    # Use in-memory SQLite database
    return DatabaseManager(database_url="sqlite:///:memory:")


@pytest.fixture
def sample_conversation_id(db_manager):
    """Create a sample conversation and return its ID."""
    return db_manager.create_conversation("Test Conversation")


# ============================================================================
# CONVERSATION TESTS
# ============================================================================


@pytest.mark.unit
def test_create_conversation(db_manager):
    """Test creating a new conversation."""
    conv_id = db_manager.create_conversation("My Test Chat")

    assert conv_id is not None
    assert isinstance(conv_id, str)

    # Verify it was created
    conv = db_manager.get_conversation(conv_id)
    assert conv is not None
    assert conv["name"] == "My Test Chat"
    assert conv["message_count"] == 0


@pytest.mark.unit
def test_create_conversation_with_custom_id(db_manager):
    """Test creating conversation with custom session ID."""
    custom_id = str(uuid.uuid4())
    conv_id = db_manager.create_conversation("Custom ID Chat", session_id=custom_id)

    assert conv_id == custom_id
    conv = db_manager.get_conversation(conv_id)
    assert conv["name"] == "Custom ID Chat"


@pytest.mark.unit
def test_get_all_conversations(db_manager):
    """Test retrieving all conversations."""
    # Create multiple conversations
    db_manager.create_conversation("Chat 1")
    db_manager.create_conversation("Chat 2")
    id3 = db_manager.create_conversation("Chat 3")

    conversations = db_manager.get_all_conversations()

    assert len(conversations) == 3
    assert all("id" in conv for conv in conversations)
    assert all("name" in conv for conv in conversations)
    # Should be ordered by updated_at desc (newest first)
    assert conversations[0]["id"] == id3


@pytest.mark.unit
def test_get_nonexistent_conversation(db_manager):
    """Test getting a conversation that doesn't exist."""
    result = db_manager.get_conversation("nonexistent-id")
    assert result is None


@pytest.mark.unit
def test_update_conversation_name(db_manager, sample_conversation_id):
    """Test updating a conversation's name."""
    db_manager.update_conversation_name(sample_conversation_id, "New Name")

    conv = db_manager.get_conversation(sample_conversation_id)
    assert conv["name"] == "New Name"


@pytest.mark.unit
def test_delete_conversation(db_manager, sample_conversation_id):
    """Test deleting a conversation."""
    # Add some messages first
    db_manager.add_message(sample_conversation_id, "user", "Hello")
    db_manager.add_message(sample_conversation_id, "assistant", "Hi there")

    # Delete conversation
    db_manager.delete_conversation(sample_conversation_id)

    # Verify it's gone
    conv = db_manager.get_conversation(sample_conversation_id)
    assert conv is None

    # Verify messages are also gone
    messages = db_manager.get_messages(sample_conversation_id)
    assert len(messages) == 0


# ============================================================================
# MESSAGE TESTS
# ============================================================================


@pytest.mark.unit
def test_add_message(db_manager, sample_conversation_id):
    """Test adding a message to a conversation."""
    db_manager.add_message(sample_conversation_id, "user", "Hello, world!")

    messages = db_manager.get_messages(sample_conversation_id)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello, world!"
    assert messages[0]["suggested_prompts"] is None


@pytest.mark.unit
def test_add_message_with_suggested_prompts(db_manager, sample_conversation_id):
    """Test adding a message with suggested prompts."""
    prompts = ["Tell me more", "Show example", "Explain further"]
    db_manager.add_message(
        sample_conversation_id,
        "assistant",
        "Here's the answer",
        suggested_prompts=prompts,
    )

    messages = db_manager.get_messages(sample_conversation_id)
    assert len(messages) == 1
    assert messages[0]["suggested_prompts"] == prompts


@pytest.mark.unit
def test_multiple_messages(db_manager, sample_conversation_id):
    """Test adding multiple messages in order."""
    db_manager.add_message(sample_conversation_id, "user", "First message")
    db_manager.add_message(sample_conversation_id, "assistant", "First response")
    db_manager.add_message(sample_conversation_id, "user", "Second message")
    db_manager.add_message(sample_conversation_id, "assistant", "Second response")

    messages = db_manager.get_messages(sample_conversation_id)
    assert len(messages) == 4
    assert messages[0]["content"] == "First message"
    assert messages[1]["content"] == "First response"
    assert messages[2]["content"] == "Second message"
    assert messages[3]["content"] == "Second response"


@pytest.mark.unit
def test_message_count_updates(db_manager, sample_conversation_id):
    """Test that message count is updated correctly."""
    db_manager.add_message(sample_conversation_id, "user", "Message 1")
    db_manager.add_message(sample_conversation_id, "assistant", "Message 2")

    conv = db_manager.get_conversation(sample_conversation_id)
    assert conv["message_count"] == 2


# ============================================================================
# CONVERSATION STATE TESTS
# ============================================================================


@pytest.mark.unit
def test_save_and_get_conversation_state(db_manager, sample_conversation_id):
    """Test saving and retrieving conversation state."""
    agent_state = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
    }
    config = {"configurable": {"thread_id": sample_conversation_id}}

    db_manager.save_conversation_state(
        sample_conversation_id, agent_state, config, waiting_for_confirmation=False
    )

    retrieved = db_manager.get_conversation_state(sample_conversation_id)
    assert retrieved is not None
    assert len(retrieved["agent_state"]["messages"]) == 2
    assert retrieved["config"] == config
    assert retrieved["waiting_for_confirmation"] is False


@pytest.mark.unit
def test_conversation_state_message_serialization(db_manager, sample_conversation_id):
    """Test that LangChain messages are properly serialized and deserialized."""
    agent_state = {
        "messages": [
            HumanMessage(content="User question", id="msg1"),
            AIMessage(content="AI response", id="msg2"),
        ]
    }
    config = {"configurable": {"thread_id": sample_conversation_id}}

    db_manager.save_conversation_state(sample_conversation_id, agent_state, config)

    retrieved = db_manager.get_conversation_state(sample_conversation_id)
    messages = retrieved["agent_state"]["messages"]

    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "User question"
    assert messages[1].content == "AI response"


@pytest.mark.unit
def test_update_conversation_state(db_manager, sample_conversation_id):
    """Test updating existing conversation state."""
    # Initial state
    initial_state = {"messages": [HumanMessage(content="First")]}
    config = {"configurable": {"thread_id": sample_conversation_id}}
    db_manager.save_conversation_state(sample_conversation_id, initial_state, config)

    # Update state
    updated_state = {
        "messages": [
            HumanMessage(content="First"),
            AIMessage(content="Response"),
        ]
    }
    db_manager.save_conversation_state(sample_conversation_id, updated_state, config)

    # Verify updated
    retrieved = db_manager.get_conversation_state(sample_conversation_id)
    assert len(retrieved["agent_state"]["messages"]) == 2


@pytest.mark.unit
def test_conversation_state_waiting_for_confirmation(
    db_manager, sample_conversation_id
):
    """Test waiting_for_confirmation flag."""
    agent_state = {"messages": [HumanMessage(content="Test")]}
    config = {"configurable": {"thread_id": sample_conversation_id}}

    db_manager.save_conversation_state(
        sample_conversation_id, agent_state, config, waiting_for_confirmation=True
    )

    retrieved = db_manager.get_conversation_state(sample_conversation_id)
    assert retrieved["waiting_for_confirmation"] is True


@pytest.mark.unit
def test_get_nonexistent_state(db_manager):
    """Test getting state for nonexistent conversation."""
    result = db_manager.get_conversation_state("nonexistent-id")
    assert result is None


# ============================================================================
# SETTINGS TESTS
# ============================================================================


@pytest.mark.unit
def test_set_and_get_setting(db_manager):
    """Test setting and getting a simple setting."""
    db_manager.set_setting("theme", "dark")

    value = db_manager.get_setting("theme")
    assert value == "dark"


@pytest.mark.unit
def test_get_setting_with_default(db_manager):
    """Test getting nonexistent setting returns default."""
    value = db_manager.get_setting("nonexistent_key", "default_value")
    assert value == "default_value"


@pytest.mark.unit
def test_update_existing_setting(db_manager):
    """Test updating an existing setting."""
    db_manager.set_setting("volume", 50)
    db_manager.set_setting("volume", 75)

    value = db_manager.get_setting("volume")
    assert value == 75


@pytest.mark.unit
def test_setting_different_types(db_manager):
    """Test settings with different data types."""
    # String
    db_manager.set_setting("name", "Test User")
    assert db_manager.get_setting("name") == "Test User"

    # Integer
    db_manager.set_setting("count", 42)
    assert db_manager.get_setting("count") == 42

    # Float
    db_manager.set_setting("opacity", 0.75)
    assert db_manager.get_setting("opacity") == 0.75

    # Boolean
    db_manager.set_setting("enabled", True)
    assert db_manager.get_setting("enabled") is True

    # Dictionary
    db_manager.set_setting("config", {"key": "value", "nested": {"a": 1}})
    assert db_manager.get_setting("config") == {"key": "value", "nested": {"a": 1}}

    # List
    db_manager.set_setting("items", [1, 2, 3, "four"])
    assert db_manager.get_setting("items") == [1, 2, 3, "four"]


@pytest.mark.unit
def test_get_all_settings(db_manager):
    """Test getting all settings at once."""
    db_manager.set_setting("setting1", "value1")
    db_manager.set_setting("setting2", 123)
    db_manager.set_setting("setting3", True)

    all_settings = db_manager.get_all_settings()

    assert len(all_settings) == 3
    assert all_settings["setting1"] == "value1"
    assert all_settings["setting2"] == 123
    assert all_settings["setting3"] is True


@pytest.mark.unit
def test_delete_setting(db_manager):
    """Test deleting a setting."""
    db_manager.set_setting("temp_setting", "temporary")
    assert db_manager.get_setting("temp_setting") == "temporary"

    db_manager.delete_setting("temp_setting")
    assert db_manager.get_setting("temp_setting") is None


@pytest.mark.unit
def test_delete_nonexistent_setting(db_manager):
    """Test deleting a setting that doesn't exist (should not raise error)."""
    db_manager.delete_setting("nonexistent_setting")  # Should not raise


# ============================================================================
# DATABASE FALLBACK TESTS
# ============================================================================


@pytest.mark.unit
def test_database_fallback_to_sqlite():
    """Test that database falls back to SQLite on connection failure."""
    with patch("src.ui.database.create_engine") as mock_create_engine:
        # First call fails (PostgreSQL), second succeeds (SQLite)
        mock_create_engine.side_effect = [
            Exception("Connection refused"),
            mock_create_engine.return_value,
        ]

        # Should fall back to SQLite without raising
        db = DatabaseManager(database_url="postgresql://bad_url")
        assert db is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.unit
def test_full_conversation_workflow(db_manager):
    """Test a complete conversation workflow."""
    # Create conversation
    conv_id = db_manager.create_conversation("Complete Workflow Test")

    # Add messages
    db_manager.add_message(conv_id, "user", "Hello!")
    db_manager.add_message(conv_id, "assistant", "Hi! How can I help?")
    db_manager.add_message(conv_id, "user", "I need help with optimization")

    # Save agent state
    agent_state = {
        "messages": [
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi! How can I help?"),
            HumanMessage(content="I need help with optimization"),
        ]
    }
    config = {"configurable": {"thread_id": conv_id}}
    db_manager.save_conversation_state(conv_id, agent_state, config)

    # Verify everything
    conv = db_manager.get_conversation(conv_id)
    assert conv["message_count"] == 3

    messages = db_manager.get_messages(conv_id)
    assert len(messages) == 3

    state = db_manager.get_conversation_state(conv_id)
    assert len(state["agent_state"]["messages"]) == 3

    # Update name
    db_manager.update_conversation_name(conv_id, "Updated Name")
    conv = db_manager.get_conversation(conv_id)
    assert conv["name"] == "Updated Name"

    # Delete
    db_manager.delete_conversation(conv_id)
    assert db_manager.get_conversation(conv_id) is None


@pytest.mark.unit
def test_multiple_conversations_isolation(db_manager):
    """Test that multiple conversations remain isolated."""
    conv1_id = db_manager.create_conversation("Conversation 1")
    conv2_id = db_manager.create_conversation("Conversation 2")

    # Add messages to each
    db_manager.add_message(conv1_id, "user", "Message in conv 1")
    db_manager.add_message(conv2_id, "user", "Message in conv 2")

    # Verify isolation
    conv1_messages = db_manager.get_messages(conv1_id)
    conv2_messages = db_manager.get_messages(conv2_id)

    assert len(conv1_messages) == 1
    assert len(conv2_messages) == 1
    assert conv1_messages[0]["content"] == "Message in conv 1"
    assert conv2_messages[0]["content"] == "Message in conv 2"
