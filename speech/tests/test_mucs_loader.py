"""Unit tests for the MUCS 2021 dataset loader."""

import pytest
from pathlib import Path
from speech.evaluation.mucs_loader import MUCSDatasetLoader


def test_mucs_loader_manifest_and_samples():
    """Verify loader reads manifest.json and loads AudioSample objects."""
    loader = MUCSDatasetLoader()
    samples = loader.load_samples()

    assert len(samples) > 0
    assert loader.dataset_name == "MUCS 2021 (OpenSLR 104)"
    assert loader.subset_name == "test-1hr-reproducible"
    assert loader.license == "CC BY-SA 4.0"

    first = samples[0]
    assert first.sample_id == "mucs_test_hien_0001"
    assert "Saturday" in first.reference_transcript
    assert first.domain == "railway_inquiry"


def test_mucs_loader_validation():
    """Verify dataset validation reports valid and missing files correctly."""
    loader = MUCSDatasetLoader()
    report = loader.validate_dataset()

    assert report["total_samples"] > 0
    assert report["valid_samples"] > 0
    assert report["empty_transcript_count"] == 0
    assert "01:00:" in report["total_duration_formatted"]


def test_pcm16_mono_reading():
    """Verify reading and format validation of WAV audio."""
    loader = MUCSDatasetLoader()
    sample = loader.get_sample_by_id("mucs_test_hien_0001")
    assert sample is not None

    if sample.file_exists:
        pcm_bytes, sr = loader.read_pcm16_mono(sample.audio_path, target_sample_rate=16000)
        assert sr == 16000
        assert len(pcm_bytes) > 0
        assert len(pcm_bytes) % 2 == 0  # 16-bit (2 bytes per sample)


def test_audio_chunking():
    """Verify chunking audio bytes into 20ms frames."""
    loader = MUCSDatasetLoader()
    # 1 second of 16kHz 16-bit mono = 32000 bytes
    dummy_bytes = b"\x00" * 32000
    chunks = list(loader.chunk_audio_pcm(dummy_bytes, sample_rate=16000, chunk_duration_ms=20))
    # 20ms chunk = 16000 * 0.02 * 2 = 640 bytes
    # 1000ms / 20ms = 50 chunks
    assert len(chunks) == 50
    assert len(chunks[0]) == 640

