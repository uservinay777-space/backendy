"""Configuration and environment settings for PhantomFlow Speech & Audio module.

Handles LiveKit credentials, Sarvam AI API keys, and audio capture parameters.
Loads values from environment variables or a local .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Automatically load environment variables from .env in project root if present
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()


class SpeechSettings:
    """Settings for the Speech and Audio input pipeline."""

    def __init__(self) -> None:
        # --- Sarvam AI Settings ---
        self.sarvam_api_key: Optional[str] = os.getenv("SARVAM_API_KEY")
        self.sarvam_base_url: str = os.getenv(
            "SARVAM_BASE_URL",
            "wss://api.sarvam.ai/speech-to-text-realtime/ws",
        )
        # Speech-to-Text configuration
        # auto = automatic language identification (Hindi, English, Indian languages)
        self.stt_language: str = os.getenv("SARVAM_STT_LANGUAGE", "auto")
        # fast = ultra-low latency streaming partials; balanced = accuracy balance
        self.stt_stream_type: str = os.getenv("SARVAM_STT_STREAM_TYPE", "fast")
        # codemix = optimized for Hinglish and mixed language conversations
        self.stt_mode: str = os.getenv("SARVAM_STT_MODE", "codemix")
        self.stt_sample_rate: int = int(os.getenv("SARVAM_STT_SAMPLE_RATE", "16000"))
        # vad = server-side VAD endpointing; manual = external endpointing
        self.stt_endpointing: str = os.getenv("SARVAM_STT_ENDPOINTING", "vad")

        # Text-to-Speech: Primary Provider (ElevenLabs)
        # Read from ELEVEN_API_KEY (preferred per spec) or ELEVENLABS_API_KEY
        self.eleven_api_key: Optional[str] = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_voice_id: Optional[str] = os.getenv("ELEVENLABS_VOICE_ID")
        self.elevenlabs_model: str = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
        self.elevenlabs_auto_mode: bool = os.getenv("ELEVENLABS_AUTO_MODE", "true").lower() in ("true", "1", "yes")
        self.elevenlabs_streaming_latency: int = int(os.getenv("ELEVENLABS_STREAMING_LATENCY", "3"))

        # Text-to-Speech: Backup Provider (Sarvam Bulbul)
        self.tts_model: str = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
        raw_speaker = os.getenv("SARVAM_TTS_SPEAKER", os.getenv("SARVAM_TTS_VOICE_ID", os.getenv("SARVAM_SPEAKER", "shubh_hi_customer")))
        self.tts_speaker: str = "shubh_hi_customer" if (raw_speaker == "shubh" and "v4" in self.tts_model) else raw_speaker
        self.tts_language: str = os.getenv("SARVAM_TTS_LANGUAGE", "hi-IN")
        self.tts_sample_rate: int = int(os.getenv("SARVAM_TTS_SAMPLE_RATE", "22050"))
        self.tts_pace: float = float(os.getenv("SARVAM_TTS_PACE", "1.0"))
        self.tts_enable_local_playback: bool = os.getenv("AUDIO_LOCAL_PLAYBACK", "true").lower() in ("true", "1", "yes")
        self.tts_fallback_enabled: bool = os.getenv("TTS_FALLBACK_ENABLED", "true").lower() in ("true", "1", "yes")

        # --- LiveKit WebRTC Settings ---
        self.livekit_url: Optional[str] = os.getenv("LIVEKIT_URL")
        self.livekit_api_key: Optional[str] = os.getenv("LIVEKIT_API_KEY")
        self.livekit_api_secret: Optional[str] = os.getenv("LIVEKIT_API_SECRET")
        self.livekit_room_name: str = os.getenv("LIVEKIT_ROOM_NAME", "phantomflow-room")

        # --- Hugging Face Token (Lahaja Evaluation) ---
        self.hf_token: Optional[str] = os.getenv("HF_TOKEN")

        # --- Audio Capture Hardware Settings ---
        self.audio_sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
        self.audio_channels: int = int(os.getenv("AUDIO_CHANNELS", "1"))
        self.audio_chunk_duration_ms: int = int(os.getenv("AUDIO_CHUNK_DURATION_MS", "20"))
        
        device_idx_raw = os.getenv("AUDIO_INPUT_DEVICE_INDEX")
        self.audio_device_index: Optional[int] = (
            int(device_idx_raw) if device_idx_raw is not None and device_idx_raw.strip().isdigit() else None
        )

    @property
    def is_elevenlabs_configured(self) -> bool:
        """Check if ElevenLabs API key is configured."""
        return bool(self.eleven_api_key and self.eleven_api_key.strip() and not self.eleven_api_key.startswith("your_"))

    @property
    def is_hf_configured(self) -> bool:
        """Check if Hugging Face token is provided."""
        return bool(self.hf_token and self.hf_token.strip() and not self.hf_token.startswith("your_"))

    @property
    def is_sarvam_configured(self) -> bool:
        """Check if Sarvam API key is available."""
        return bool(self.sarvam_api_key and self.sarvam_api_key.strip() and not self.sarvam_api_key.startswith("your_"))

    @property
    def is_livekit_configured(self) -> bool:
        """Check if LiveKit connection credentials are fully provided."""
        return bool(
            self.livekit_url
            and not self.livekit_url.startswith("wss://your-")
            and self.livekit_api_key
            and not self.livekit_api_key.startswith("your_")
            and self.livekit_api_secret
            and not self.livekit_api_secret.startswith("your_")
        )

    def generate_livekit_token(
        self,
        participant_identity: str,
        participant_name: Optional[str] = None,
        room_name: Optional[str] = None,
        can_publish: bool = True,
        can_subscribe: bool = True,
    ) -> str:
        """Generate a LiveKit JWT access token for a room participant.
        
        Requires LIVEKIT_API_KEY and LIVEKIT_API_SECRET to be configured.
        """
        if not self.livekit_api_key or not self.livekit_api_secret:
            raise ValueError(
                "LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be configured to generate an access token."
            )
        try:
            from livekit import api
        except ImportError as exc:
            raise RuntimeError("livekit-api package is required for token generation") from exc

        target_room = room_name or self.livekit_room_name
        token = (
            api.AccessToken(self.livekit_api_key, self.livekit_api_secret)
            .with_identity(participant_identity)
            .with_name(participant_name or participant_identity)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=target_room,
                    can_publish=can_publish,
                    can_subscribe=can_subscribe,
                )
            )
        )
        return token.to_jwt()

    def print_diagnostics(self) -> None:
        """Print diagnostics of active configuration (masking secrets)."""
        sarvam_status = "CONFIGURED" if self.is_sarvam_configured else "NOT CONFIGURED / PLACEHOLDER"
        eleven_status = "CONFIGURED" if self.is_elevenlabs_configured else "NOT CONFIGURED / PLACEHOLDER"
        livekit_status = "CONFIGURED" if self.is_livekit_configured else "NOT CONFIGURED (Will use Direct Mic mode)"

        masked_sarvam = f"{self.sarvam_api_key[:4]}...{self.sarvam_api_key[-4:]}" if self.is_sarvam_configured and len(self.sarvam_api_key or "") > 8 else "N/A"
        masked_eleven = f"{self.eleven_api_key[:4]}...{self.eleven_api_key[-4:]}" if self.is_elevenlabs_configured and len(self.eleven_api_key or "") > 8 else "N/A"
        
        print("\n=== PhantomFlow Speech & Audio Configuration ===")
        print(f"  Primary TTS:        ElevenLabs ({self.elevenlabs_model}) [{eleven_status}] ({masked_eleven})")
        print(f"    Voice ID:         {self.elevenlabs_voice_id or 'default'}")
        print(f"  Backup TTS:         Sarvam Bulbul ({self.tts_model}) [{sarvam_status}] ({masked_sarvam})")
        print(f"    Speaker / Lang:   {self.tts_speaker} / {self.tts_language} ({self.tts_sample_rate} Hz)")
        print(f"  Fallback Enabled:   {self.tts_fallback_enabled}")
        print(f"  STT Provider:       Sarvam Saaras Realtime [{sarvam_status}]")
        print(f"    STT Config:       lang={self.stt_language}, stream_type={self.stt_stream_type}, mode={self.stt_mode}")
        print(f"  LiveKit WebRTC:     {self.livekit_url or 'None'} [{livekit_status}] (room: {self.livekit_room_name})")
        print(f"  Audio Format:       {self.audio_sample_rate} Hz, {self.audio_channels} ch, {self.audio_chunk_duration_ms} ms chunk")
        print("================================================\n")


# Global singleton settings instance
settings = SpeechSettings()

