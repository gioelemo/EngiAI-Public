"""Voice configuration for text-to-speech and speech-to-text.

This module defines the available voices and their IDs for both ElevenLabs and OpenAI
voice interaction features. Voice configuration can be customized via environment variables.
"""

import os

# Voice provider selection
VOICE_PROVIDER = os.getenv("VOICE_PROVIDER", "elevenlabs").lower()

# ===== ElevenLabs Configuration =====

# Default voice mappings (fallback if not configured in .env)
_DEFAULT_ELEVENLABS_VOICE_IDS = {
    "Rachel": "21m00Tcm4TlvDq8ikWAM",
    "Domi": "AZnzlk1XvdvUeBnXmlld",
    "Bella": "EXAVITQu4vr4xnSDxMaL",
    "Antoni": "ErXwobaYiN019PkySvjV",
    "Josh": "TxGEqnHWrfWFTfGW9XjX",
    "George": "JBFqnCBsd6RMkjVDRZzb",
}


def _parse_voices_from_env() -> dict[str, str]:
    """Parse voice mappings from ELEVENLABS_VOICES environment variable.

    Format: VoiceName:voice_id,AnotherVoice:voice_id
    Returns: Dictionary mapping voice names to IDs
    """
    voices_str = os.getenv("ELEVENLABS_VOICES", "")
    if not voices_str:
        return _DEFAULT_ELEVENLABS_VOICE_IDS.copy()

    voices = {}
    for pair_str in voices_str.split(","):
        cleaned = pair_str.strip()
        if ":" in cleaned:
            name, voice_id = cleaned.split(":", 1)
            voices[name.strip()] = voice_id.strip()

    return voices or _DEFAULT_ELEVENLABS_VOICE_IDS.copy()


# ElevenLabs voice ID mappings (loaded from env or defaults)
ELEVENLABS_VOICE_IDS = _parse_voices_from_env()

# ElevenLabs default voice (from env or fallback)
ELEVENLABS_DEFAULT_VOICE = os.getenv("ELEVENLABS_DEFAULT_VOICE", "George")

# ElevenLabs model configurations
ELEVENLABS_STT_MODEL = os.getenv("ELEVENLABS_STT_MODEL", "eleven_multilingual_v2")
ELEVENLABS_TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")

# ===== OpenAI Configuration =====

# OpenAI TTS voices (these are the available voices from OpenAI)
OPENAI_TTS_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# OpenAI TTS models (Text-to-Speech)
OPENAI_TTS_MODELS = ["tts-1", "tts-1-hd"]

# OpenAI STT models (Speech-to-Text via Whisper)
OPENAI_STT_MODELS = ["whisper-1"]

# OpenAI default configurations from env
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")

# ===== Backward Compatibility =====
# These maintain backward compatibility with existing code that uses the old names
VOICE_IDS = ELEVENLABS_VOICE_IDS  # For backward compatibility
DEFAULT_VOICE = ELEVENLABS_DEFAULT_VOICE  # For backward compatibility
STT_MODEL = ELEVENLABS_STT_MODEL  # For backward compatibility
TTS_MODEL = ELEVENLABS_TTS_MODEL  # For backward compatibility
