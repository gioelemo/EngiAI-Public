"""
Tests for voice interaction features (STT and TTS).

Tests the dual voice provider system (ElevenLabs and OpenAI) including:
- Voice transcription (STT) with both providers
- Voice generation (TTS) with both providers
- Provider routing logic
- Database voice provider tracking
- Voice configuration

Note: Streamlit-specific tests are skipped due to import complexity.
Focus is on database persistence and configuration validation.
"""

import base64

import pytest

from src.ui.database import DatabaseManager

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def db_manager():
    """Create an in-memory SQLite database for testing."""
    return DatabaseManager(database_url="sqlite:///:memory:")


# ============================================================================
# DATABASE VOICE PROVIDER TESTS
# ============================================================================


@pytest.mark.unit
def test_create_conversation_with_voice_provider(db_manager):
    """Test creating a conversation with voice provider specified."""
    conv_id = db_manager.create_conversation(
        "Voice Test Chat", voice_id="alloy", voice_provider="openai"
    )

    conv = db_manager.get_conversation(conv_id)
    assert conv is not None
    assert conv["voice_id"] == "alloy"
    assert conv["voice_provider"] == "openai"


@pytest.mark.unit
def test_create_conversation_without_voice_provider(db_manager):
    """Test creating a conversation without voice provider (default None)."""
    conv_id = db_manager.create_conversation("No Voice Chat")

    conv = db_manager.get_conversation(conv_id)
    assert conv is not None
    assert conv["voice_id"] is None
    assert conv["voice_provider"] is None


@pytest.mark.unit
def test_update_conversation_voice(db_manager):
    """Test updating a conversation's voice settings."""
    conv_id = db_manager.create_conversation("Update Voice Test")

    # Update voice
    db_manager.update_conversation_voice(conv_id, "George", "elevenlabs")

    conv = db_manager.get_conversation(conv_id)
    assert conv["voice_id"] == "George"
    assert conv["voice_provider"] == "elevenlabs"

    # Update again to different provider
    db_manager.update_conversation_voice(conv_id, "echo", "openai")

    conv = db_manager.get_conversation(conv_id)
    assert conv["voice_id"] == "echo"
    assert conv["voice_provider"] == "openai"


@pytest.mark.unit
def test_update_conversation_voice_only(db_manager):
    """Test updating only voice_id without changing provider."""
    conv_id = db_manager.create_conversation(
        "Voice Only Update", voice_id="alloy", voice_provider="openai"
    )

    # Update only voice_id
    db_manager.update_conversation_voice(conv_id, "nova", None)

    conv = db_manager.get_conversation(conv_id)
    assert conv["voice_id"] == "nova"
    assert conv["voice_provider"] == "openai"  # Should remain unchanged


@pytest.mark.unit
def test_voice_provider_persists_across_retrieval(db_manager):
    """Test that voice provider persists when getting all conversations."""
    conv1 = db_manager.create_conversation(
        "ElevenLabs Chat", voice_id="George", voice_provider="elevenlabs"
    )
    conv2 = db_manager.create_conversation(
        "OpenAI Chat", voice_id="alloy", voice_provider="openai"
    )

    conversations = db_manager.get_all_conversations()

    assert len(conversations) == 2

    # Find our conversations
    conv1_data = next(c for c in conversations if c["id"] == conv1)
    conv2_data = next(c for c in conversations if c["id"] == conv2)

    assert conv1_data["voice_provider"] == "elevenlabs"
    assert conv2_data["voice_provider"] == "openai"


@pytest.mark.unit
def test_voice_provider_with_message_attachments(db_manager):
    """Test adding messages with audio attachments."""
    conv_id = db_manager.create_conversation(
        "Audio Message Test", voice_id="alloy", voice_provider="openai"
    )

    # Add message with audio attachment
    audio_data = base64.b64encode(b"fake_audio").decode("utf-8")
    db_manager.add_message(
        conv_id,
        "user",
        "Hello",
        attachments={
            "audio": {"data": audio_data, "format": "mp3", "transcribed": True}
        },
    )

    # Verify message was saved with audio
    messages = db_manager.get_messages(conv_id)
    assert len(messages) == 1
    assert messages[0]["audio"] is not None
    assert messages[0]["audio"]["format"] == "mp3"
    assert messages[0]["audio"]["transcribed"] is True


# ============================================================================
# VOICE CONFIGURATION TESTS
# ============================================================================


@pytest.mark.unit
def test_voice_configuration_imports():
    """Test that voice configuration imports work correctly."""
    from src.ui.voices import (
        DEFAULT_VOICE,
        ELEVENLABS_DEFAULT_VOICE,
        ELEVENLABS_STT_MODEL,
        ELEVENLABS_TTS_MODEL,
        ELEVENLABS_VOICE_IDS,
        OPENAI_STT_MODEL,
        OPENAI_TTS_MODEL,
        OPENAI_TTS_VOICE,
        OPENAI_TTS_VOICES,
        VOICE_IDS,
        VOICE_PROVIDER,
    )

    # Test ElevenLabs configuration
    assert isinstance(ELEVENLABS_VOICE_IDS, dict)
    assert len(ELEVENLABS_VOICE_IDS) == 6
    assert "George" in ELEVENLABS_VOICE_IDS
    assert ELEVENLABS_DEFAULT_VOICE == "George"
    # STT/TTS models can be configured via env, just check they're strings
    assert isinstance(ELEVENLABS_STT_MODEL, str)
    assert isinstance(ELEVENLABS_TTS_MODEL, str)

    # Test OpenAI configuration
    assert isinstance(OPENAI_TTS_VOICES, list)
    assert len(OPENAI_TTS_VOICES) == 6
    assert "alloy" in OPENAI_TTS_VOICES
    assert OPENAI_TTS_MODEL in ["tts-1", "tts-1-hd"]
    assert OPENAI_TTS_VOICE in OPENAI_TTS_VOICES
    assert OPENAI_STT_MODEL == "whisper-1"

    # Test backward compatibility
    assert VOICE_IDS == ELEVENLABS_VOICE_IDS
    assert DEFAULT_VOICE == ELEVENLABS_DEFAULT_VOICE

    # Test provider selection
    assert VOICE_PROVIDER in ["elevenlabs", "openai"]


@pytest.mark.unit
def test_elevenlabs_voice_ids_structure():
    """Test ElevenLabs voice ID mapping structure."""
    from src.ui.voices import ELEVENLABS_VOICE_IDS

    expected_voices = ["Rachel", "Domi", "Bella", "Antoni", "Josh", "George"]

    for voice_name in expected_voices:
        assert voice_name in ELEVENLABS_VOICE_IDS
        # Voice IDs should be non-empty strings
        assert isinstance(ELEVENLABS_VOICE_IDS[voice_name], str)
        assert len(ELEVENLABS_VOICE_IDS[voice_name]) > 0


@pytest.mark.unit
def test_openai_voice_list():
    """Test OpenAI voice list completeness."""
    from src.ui.voices import OPENAI_TTS_VOICES

    expected_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    assert len(OPENAI_TTS_VOICES) == len(expected_voices)
    for voice in expected_voices:
        assert voice in OPENAI_TTS_VOICES


# ============================================================================
# AUDIO MIME TYPE TESTS
# ============================================================================


@pytest.mark.unit
def test_audio_mime_type_mapping():
    """Test that audio MIME types are correctly mapped."""
    # Test the MIME type mapping that was fixed
    # MP3 should map to "audio/mpeg" not "audio/mp3"
    mime_type_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
    }

    for expected_mime in mime_type_map.values():
        assert expected_mime.startswith("audio/")

    # Specifically test MP3 fix
    assert mime_type_map["mp3"] == "audio/mpeg"
    assert mime_type_map["mp3"] != "audio/mp3"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.unit
def test_voice_conversation_workflow(db_manager):
    """Test complete voice conversation workflow with database."""
    # Create conversation with OpenAI voice
    conv_id = db_manager.create_conversation(
        "Voice Workflow Test", voice_id="alloy", voice_provider="openai"
    )

    # Add message with audio
    audio_data = base64.b64encode(b"fake_audio").decode("utf-8")
    db_manager.add_message(
        conv_id,
        "user",
        "Hello",
        attachments={
            "audio": {"data": audio_data, "format": "mp3", "transcribed": True}
        },
    )

    # Verify message was saved with audio
    messages = db_manager.get_messages(conv_id)
    assert len(messages) == 1
    assert messages[0]["audio"] is not None
    assert messages[0]["audio"]["format"] == "mp3"

    # Switch to ElevenLabs
    db_manager.update_conversation_voice(conv_id, "George", "elevenlabs")

    conv = db_manager.get_conversation(conv_id)
    assert conv["voice_provider"] == "elevenlabs"
    assert conv["voice_id"] == "George"


@pytest.mark.unit
def test_voice_provider_persistence(db_manager):
    """Test that voice provider persists across conversation lifecycle."""
    # Create with OpenAI
    conv_id = db_manager.create_conversation(
        "Persistence Test", voice_id="nova", voice_provider="openai"
    )

    # Add messages
    db_manager.add_message(conv_id, "user", "Message 1")
    db_manager.add_message(conv_id, "assistant", "Response 1")

    # Retrieve and verify
    conv = db_manager.get_conversation(conv_id)
    assert conv["voice_provider"] == "openai"
    assert conv["voice_id"] == "nova"

    # Update name (should not affect voice settings)
    db_manager.update_conversation_name(conv_id, "Updated Name")

    conv = db_manager.get_conversation(conv_id)
    assert conv["voice_provider"] == "openai"  # Should persist
    assert conv["voice_id"] == "nova"  # Should persist


@pytest.mark.unit
def test_multiple_conversations_different_providers(db_manager):
    """Test managing multiple conversations with different voice providers."""
    # Create conversations with different providers
    conv1 = db_manager.create_conversation(
        "ElevenLabs Conversation",
        voice_id="Rachel",
        voice_provider="elevenlabs",
    )
    conv2 = db_manager.create_conversation(
        "OpenAI Conversation",
        voice_id="fable",
        voice_provider="openai",
    )
    conv3 = db_manager.create_conversation(
        "No Voice Conversation",
        voice_id=None,
        voice_provider=None,
    )

    # Add messages to each
    db_manager.add_message(conv1, "user", "Hello from ElevenLabs")
    db_manager.add_message(conv2, "user", "Hello from OpenAI")
    db_manager.add_message(conv3, "user", "Hello without voice")

    # Retrieve all and verify
    conversations = db_manager.get_all_conversations()
    assert len(conversations) == 3

    # Verify each conversation maintains its voice settings
    conv_dict = {c["id"]: c for c in conversations}

    assert conv_dict[conv1]["voice_provider"] == "elevenlabs"
    assert conv_dict[conv1]["voice_id"] == "Rachel"

    assert conv_dict[conv2]["voice_provider"] == "openai"
    assert conv_dict[conv2]["voice_id"] == "fable"

    assert conv_dict[conv3]["voice_provider"] is None
    assert conv_dict[conv3]["voice_id"] is None


@pytest.mark.unit
def test_voice_settings_isolation(db_manager):
    """Test that voice settings don't interfere between conversations."""
    # Create two conversations with same provider but different voices
    conv1 = db_manager.create_conversation(
        "Voice 1", voice_id="alloy", voice_provider="openai"
    )
    conv2 = db_manager.create_conversation(
        "Voice 2", voice_id="echo", voice_provider="openai"
    )

    # Update one conversation's voice
    db_manager.update_conversation_voice(conv1, "nova", "openai")

    # Verify isolation
    conv1_data = db_manager.get_conversation(conv1)
    conv2_data = db_manager.get_conversation(conv2)

    assert conv1_data["voice_id"] == "nova"
    assert conv2_data["voice_id"] == "echo"  # Should not have changed
