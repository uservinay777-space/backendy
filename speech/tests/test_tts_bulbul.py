"""Unit tests for Sarvam Bulbul TTS wrapper and metrics."""

import asyncio
import pytest
from speech.config.settings import SpeechSettings
from speech.tts.bulbul import (
    LocalAudioPlayer,
    SarvamBulbulTTS,
    TTSMetrics,
    detect_target_language,
)


def test_detect_target_language_scenarios():
    """Verify automatic language detection for Hindi, English, and Hinglish."""
    # Devanagari Hindi
    assert detect_target_language("नमस्ते, क्या हाल है?") == "hi-IN"
    # English
    assert detect_target_language("Hello! How can I help you today?") == "en-IN"
    # Hinglish with common markers
    assert detect_target_language("Mujhe train ticket chahiye") == "hi-IN"
    assert detect_target_language("Sure, main aapki train booking mein help kar sakta hoon.") == "hi-IN"
    # Fallback when empty
    assert detect_target_language("", fallback_language="hi-IN") == "hi-IN"


def test_tts_missing_key_raises_error(monkeypatch):
    """Verify that missing SARVAM_API_KEY raises ValueError on synthesis."""
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    s = SpeechSettings()
    tts = SarvamBulbulTTS(config=s, api_key=None)

    with pytest.raises(ValueError, match="SARVAM_API_KEY is not set"):
        tts._get_or_create_tts("hi-IN", "shubh")


def test_tts_metrics_calculations():
    """Verify that TTFB, generation time, and audio duration are computed accurately."""
    m = TTSMetrics(
        text="Sample text",
        target_language="hi-IN",
        speaker="shubh",
        sample_rate=22050,
        request_started_at=100.0,
    )
    m.first_chunk_received_at = 100.180  # 180ms TTFB
    m.first_chunk_played_at = 100.185    # 185ms first play
    m.generation_completed_at = 100.350  # 350ms total generation
    # 22050 samples = 44100 bytes = 1.0 second (1000ms)
    m.total_bytes = 44100
    m.total_chunks = 4

    assert pytest.approx(m.time_to_first_audio_ms, 0.1) == 180.0
    assert pytest.approx(m.time_to_first_playback_ms, 0.1) == 185.0
    assert pytest.approx(m.total_generation_time_ms, 0.1) == 350.0
    assert pytest.approx(m.total_audio_duration_ms, 0.1) == 1000.0

    summary = m.format_summary()
    assert "180.0ms" in summary
    assert "350.0ms" in summary
    assert "1000ms" in summary


def test_tts_initialization_defaults():
    """Verify that TTS is initialized with expected model and speaker defaults."""
    tts = SarvamBulbulTTS(
        api_key="dummy_key",
        model="bulbul:v3",
        speaker="shubh",
        sample_rate=22050,
        pace=1.0,
    )
    assert tts.model == "bulbul:v3"
    assert tts.speaker == "shubh"
    assert tts.sample_rate == 22050
    assert tts.pace == 1.0


@pytest.mark.asyncio
async def test_stop_playback_interrupts():
    """Verify stop_playback cancels turn synthesis."""
    tts = SarvamBulbulTTS(api_key="dummy_key")
    turn_id = "turn-test-123"

    await tts.stop_playback(turn_id)
    assert turn_id in tts._stopped_turns

