"""Unit tests for speech module configuration and settings."""

import os
import pytest
from speech.config.settings import SpeechSettings


def test_speech_settings_defaults(monkeypatch):
    """Test default values when no environment variables are set."""
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

    s = SpeechSettings()
    assert s.stt_language == "auto"
    assert s.stt_stream_type == "fast"
    assert s.stt_mode == "codemix"
    assert s.stt_endpointing == "vad"
    assert s.stt_sample_rate == 16000
    assert s.audio_sample_rate == 16000
    assert s.audio_channels == 1
    assert s.audio_chunk_duration_ms == 20
    assert s.is_sarvam_configured is False
    assert s.is_livekit_configured is False


def test_speech_settings_with_env(monkeypatch):
    """Test that environment variables properly configure settings."""
    monkeypatch.setenv("SARVAM_API_KEY", "real_test_key_12345")
    monkeypatch.setenv("SARVAM_STT_LANGUAGE", "hi-IN")
    monkeypatch.setenv("SARVAM_STT_STREAM_TYPE", "balanced")
    monkeypatch.setenv("SARVAM_STT_MODE", "transcribe")
    monkeypatch.setenv("LIVEKIT_URL", "wss://livekit.example.com")
    monkeypatch.setenv("LIVEKIT_API_KEY", "api_key_abc")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret_key_xyz_012345678901234567890123456789")

    s = SpeechSettings()
    assert s.is_sarvam_configured is True
    assert s.is_livekit_configured is True
    assert s.stt_language == "hi-IN"
    assert s.stt_stream_type == "balanced"
    assert s.stt_mode == "transcribe"

    # Test LiveKit token generation
    token = s.generate_livekit_token(participant_identity="test-agent")
    assert isinstance(token, str)
    assert len(token) > 20

