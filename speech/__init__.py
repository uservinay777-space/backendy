"""PhantomFlow Speech & Audio Engineering Module (Role 02).

Provides the real-time speech pipeline:
Microphone → LiveKit WebRTC → Sarvam Saaras STT → ElevenLabs Primary / Sarvam Backup TTS → Live Playback.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from speech.audio.microphone import (
        LiveKitMicrophonePublisher,
        MicrophoneCapture,
        generate_synthetic_pcm,
    )
    from speech.audio.playback import (
        LiveKitAudioOutputTrack,
        LocalAudioPlayer,
    )
    from speech.config.settings import SpeechSettings, settings
    from speech.pipeline import SpeechInputPipeline
    from speech.stt.saaras import SarvamSaarasSTT, normalize_to_language_code
    from speech.tts.elevenlabs_tts import ElevenLabsTTSClient, create_elevenlabs_tts
    from speech.tts.sarvam_tts import SarvamTTSClient, create_sarvam_tts
    from speech.tts.tts_manager import (
        TTSManager,
        TTSMetrics,
        detect_target_language,
        generate_speech,
    )

__all__ = [
    "SpeechInputPipeline",
    "SarvamSaarasSTT",
    "TTSManager",
    "ElevenLabsTTSClient",
    "create_elevenlabs_tts",
    "SarvamTTSClient",
    "create_sarvam_tts",
    "TTSMetrics",
    "MicrophoneCapture",
    "LiveKitMicrophonePublisher",
    "LocalAudioPlayer",
    "LiveKitAudioOutputTrack",
    "SpeechSettings",
    "settings",
    "normalize_to_language_code",
    "generate_synthetic_pcm",
    "detect_target_language",
    "generate_speech",
]


def __getattr__(name: str):
    if name in ("SpeechInputPipeline", "TranscriptDisplay"):
        from speech.pipeline import SpeechInputPipeline
        return SpeechInputPipeline
    elif name in ("SarvamSaarasSTT", "normalize_to_language_code"):
        from speech.stt.saaras import SarvamSaarasSTT, normalize_to_language_code
        return SarvamSaarasSTT if name == "SarvamSaarasSTT" else normalize_to_language_code
    elif name in (
        "TTSManager",
        "TTSMetrics",
        "detect_target_language",
        "generate_speech",
        "ElevenLabsTTSClient",
        "create_elevenlabs_tts",
        "SarvamTTSClient",
        "create_sarvam_tts",
    ):
        import speech.tts as tts_mod
        return getattr(tts_mod, name)
    elif name in ("LiveKitMicrophonePublisher", "MicrophoneCapture", "generate_synthetic_pcm"):
        import speech.audio.microphone as mic_mod
        return getattr(mic_mod, name)
    elif name in ("LocalAudioPlayer", "LiveKitAudioOutputTrack"):
        import speech.audio.playback as play_mod
        return getattr(play_mod, name)
    elif name in ("SpeechSettings", "settings"):
        import speech.config.settings as cfg_mod
        return getattr(cfg_mod, name)
    raise AttributeError(f"module 'speech' has no attribute '{name}'")
