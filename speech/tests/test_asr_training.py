"""Unit tests for local ASR model, feature extraction, and training loop (Milestone 6)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from speech.asr_experiment.preprocessing.audio_preprocessing import (
    AudioPreprocessor,
    create_mel_filterbank,
    read_audio_file,
    save_wav_file,
)
from speech.asr_experiment.preprocessing.load_dataset import ASRSample
from speech.asr_experiment.preprocessing.transcript_cleaning import clean_asr_transcript
from speech.asr_experiment.training.dataset_split import split_dataset_speaker_aware
from speech.asr_experiment.training.train import (
    ASRExperimentTrainer,
    LocalASRModel,
)


def test_audio_preprocessor_log_mel():
    """Verify 80-channel Log-Mel spectrogram computation and normalization."""
    preprocessor = AudioPreprocessor(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80)
    fb = create_mel_filterbank(sample_rate=16000, n_fft=512, n_mels=80)
    assert fb.shape == (80, 257)

    # 1 second of 440Hz sine wave
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    log_mel = preprocessor.compute_log_mel_spectrogram(audio)
    assert log_mel.shape[0] == 80
    assert log_mel.shape[1] > 0
    assert np.all(log_mel >= -1.0) and np.all(log_mel <= 1.0)


def test_wav_save_and_read(tmp_path: Path):
    """Verify saving and reading 16kHz 16-bit PCM WAV audio."""
    wav_path = tmp_path / "test_audio.wav"
    t = np.linspace(0, 0.5, 8000, endpoint=False, dtype=np.float32)
    original_audio = 0.6 * np.sin(2 * np.pi * 300.0 * t)

    saved_path = save_wav_file(wav_path, original_audio, sample_rate=16000)
    assert saved_path.is_file()

    read_audio, sr = read_audio_file(saved_path)
    assert sr == 16000
    assert len(read_audio) == 8000
    np.testing.assert_allclose(read_audio, original_audio, atol=1e-3)


def test_transcript_cleaning():
    """Verify Unicode NFC and punctuation normalization on Hindi and English text."""
    raw_hi = "कृपया   कल सुबह 9:00 बजे का अलार्म लगा दीजिए!  "
    cleaned_hi = clean_asr_transcript(raw_hi)
    assert "!" not in cleaned_hi
    assert "  " not in cleaned_hi

    raw_en = "Please check the PNR STATUS for 2839182390???"
    cleaned_en = clean_asr_transcript(raw_en)
    assert "???" not in cleaned_en
    assert "status" in cleaned_en


def test_speaker_aware_splitting():
    """Verify train, val, and test partitions have zero speaker overlap."""
    samples = []
    for i in range(20):
        spk = f"spk_{i % 5}"
        s = ASRSample(
            sample_id=f"sample_{i}",
            source_dataset="indicvoices_hindi",
            audio_path="",
            duration_sec=3.0,
            raw_transcript="परीक्षण वाक्य",
            cleaned_transcript="परीक्षण वाक्य",
            language="hi",
            speaker_id=spk,
            is_train_eligible=True,
        )
        samples.append(s)

    train_s, val_s, test_s = split_dataset_speaker_aware(
        samples,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=42,
    )

    train_speakers = {s.speaker_id for s in train_s}
    val_speakers = {s.speaker_id for s in val_s}
    test_speakers = {s.speaker_id for s in test_s}

    assert train_speakers.isdisjoint(val_speakers)
    assert train_speakers.isdisjoint(test_speakers)
    assert val_speakers.isdisjoint(test_speakers)


def test_local_asr_model_checkpoint(tmp_path: Path):
    """Verify model initialization, forward pass, LoRA parameter update, and checkpoint save/load."""
    model = LocalASRModel(architecture="whisper-small-lora", hidden_dim=64)

    assert model.total_params > 0
    assert model.trainable_params > 0
    assert model.trainable_params < model.total_params

    # Dummy log-mel: (80, 50)
    dummy_mel = np.random.uniform(-0.5, 0.5, (80, 50)).astype(np.float32)
    hidden = model.forward_features(dummy_mel)
    assert hidden.shape == (50, 64)

    # Train step: LoRA B matrix and prediction head weights update upon gradient step
    initial_lora_B = model.lora_B.copy()
    initial_head = model.head_weights.copy()
    loss = model.train_step(dummy_mel, "कल सुबह नौ बजे")
    assert loss > 0.0
    assert not np.array_equal(model.lora_B, initial_lora_B)
    assert not np.array_equal(model.head_weights, initial_head)

    # Save and reload checkpoint
    ckpt_path = tmp_path / "test_checkpoint.json"
    model.save_checkpoint(ckpt_path, metadata={"epoch": 1, "test_loss": loss})
    assert ckpt_path.is_file()

    model_loaded = LocalASRModel(architecture="whisper-small-lora", hidden_dim=64)
    meta = model_loaded.load_checkpoint(ckpt_path)
    assert meta["epoch"] == 1
    np.testing.assert_allclose(model_loaded.lora_A, model.lora_A)
