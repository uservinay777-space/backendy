"""Text-to-Speech package for PhantomFlow.

Provides:
- TTSManager: Orchestrator with ElevenLabs primary and Sarvam Bulbul backup (via FallbackAdapter).
- ElevenLabsTTSClient & create_elevenlabs_tts: Primary ElevenLabs (Flash v2.5).
- SarvamTTSClient & create_sarvam_tts: Backup Sarvam Bulbul (v3).
- generate_speech: High-level streaming speech synthesis function.
- LocalAudioPlayer: Speaker output.
"""

from speech.audio.playback import LiveKitAudioOutputTrack, LocalAudioPlayer
from speech.tts.elevenlabs_tts import ElevenLabsTTSClient, create_elevenlabs_tts
from speech.tts.sarvam_tts import SarvamTTSClient, create_sarvam_tts
from speech.tts.tts_manager import (
    TTSManager,
    TTSMetrics,
    detect_target_language,
    generate_speech,
)

__all__ = [
    "TTSManager",
    "ElevenLabsTTSClient",
    "create_elevenlabs_tts",
    "SarvamTTSClient",
    "create_sarvam_tts",
    "TTSMetrics",
    "detect_target_language",
    "generate_speech",
    "LocalAudioPlayer",
    "LiveKitAudioOutputTrack",
]
