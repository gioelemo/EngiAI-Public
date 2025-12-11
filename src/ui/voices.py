"""ElevenLabs voice configuration for text-to-speech.

This module defines the available voices and their IDs for the voice interaction feature.
Voice configuration can be customized via environment variables.
"""

import os

# Default voice mappings (fallback if not configured in .env)
_DEFAULT_VOICE_IDS = {
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
        return _DEFAULT_VOICE_IDS.copy()

    voices = {}
    for pair_str in voices_str.split(","):
        cleaned = pair_str.strip()
        if ":" in cleaned:
            name, voice_id = cleaned.split(":", 1)
            voices[name.strip()] = voice_id.strip()

    return voices if voices else _DEFAULT_VOICE_IDS.copy()


# ElevenLabs voice ID mappings (loaded from env or defaults)
VOICE_IDS = _parse_voices_from_env()

# Default voice (from env or fallback)
DEFAULT_VOICE = os.getenv("ELEVENLABS_DEFAULT_VOICE", "George")

# Model configurations
STT_MODEL = os.getenv("ELEVENLABS_STT_MODEL", "eleven_multilingual_v2")
TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
