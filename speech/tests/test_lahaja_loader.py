"""Unit tests for the AI4Bharat Lahaja dataset loader."""

import pytest
from pathlib import Path
from speech.evaluation.lahaja_loader import LahajaDatasetLoader


def test_lahaja_loader_manifest_and_metadata():
    """Verify Lahaja loader parses manifest.json and preserves regional/speaker metadata."""
    loader = LahajaDatasetLoader()
    samples = loader.load_samples()

    assert len(samples) >= 35
    assert "AI4Bharat Lahaja" in loader.dataset_name
    assert "CC BY-NC" in loader.license

    first = samples[0]
    meta = first.metadata
    assert meta["native_state"] == "Uttar Pradesh"
    assert meta["native_district"] == "Lucknow"
    assert meta["scenario"] in ("read", "extempore")
    assert meta["sp_id"].startswith("spk_")
    assert len(first.reference_transcript) > 0


def test_lahaja_loader_filtering():
    """Verify filtering by scenario, state, and native language."""
    loader = LahajaDatasetLoader()

    # Filter read speech
    read_samples = loader.filter_samples(scenario="read")
    assert len(read_samples) > 0
    assert all(s.metadata["scenario"] == "read" for s in read_samples)

    # Filter extempore speech
    extempore_samples = loader.filter_samples(scenario="extempore")
    assert len(extempore_samples) > 0
    assert all(s.metadata["scenario"] == "extempore" for s in extempore_samples)

    # Filter by state
    up_samples = loader.filter_samples(state="Uttar Pradesh")
    assert len(up_samples) >= 7
    assert all(s.metadata["native_state"] == "Uttar Pradesh" for s in up_samples)

    # Filter by native language
    bhojpuri_samples = loader.filter_samples(native_language="Bhojpuri")
    assert len(bhojpuri_samples) >= 2


def test_lahaja_audio_pcm_reading():
    """Verify audio files can be read as 16kHz PCM mono."""
    loader = LahajaDatasetLoader()
    samples = loader.load_samples()
    first = samples[0]

    assert first.file_exists
    pcm_bytes, sr = loader.read_pcm16_mono(first.audio_path, target_sample_rate=16000)
    assert sr == 16000
    assert len(pcm_bytes) > 0
    assert len(pcm_bytes) % 2 == 0  # 16-bit
