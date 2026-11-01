"""Unit tests for microphone capture and audio frames."""

import pytest
from speech.audio.microphone import MicrophoneCapture, generate_synthetic_pcm
from phantomflow.contracts.audio_contracts import AudioInputChunk


def test_microphone_chunk_calculations():
    """Verify that sample and byte size calculations match 16kHz 16-bit mono PCM."""
    mic = MicrophoneCapture(
        sample_rate=16000,
        channels=1,
        chunk_duration_ms=20,
    )
    # 16000 samples/sec * 0.02s = 320 samples
    assert mic.samples_per_chunk == 320
    # 320 samples * 1 channel * 2 bytes = 640 bytes
    assert mic.bytes_per_chunk == 640


def test_generate_synthetic_pcm():
    """Verify synthetic PCM generator produces valid 16-bit signed PCM audio bytes."""
    pcm = generate_synthetic_pcm(duration_s=0.1, sample_rate=16000)
    # 0.1s at 16000Hz = 1600 samples = 3200 bytes
    assert len(pcm) == 3200
    assert isinstance(pcm, bytes)


def test_list_input_devices():
    """Verify device listing returns a list structure."""
    devices = MicrophoneCapture.list_input_devices()
    assert isinstance(devices, list)

