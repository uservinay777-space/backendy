"""Unit tests for controlled acoustic noise mixing and SNR scaling."""

import math
import numpy as np
import pytest
from pathlib import Path

from speech.evaluation.noise_mixer import (
    NoiseCategory,
    NoiseMixer,
)


def test_noise_category_parsing():
    """Verify string parsing into NoiseCategory enum."""
    assert NoiseCategory.from_str("noise") == NoiseCategory.NOISE
    assert NoiseCategory.from_str("ambient") == NoiseCategory.NOISE
    assert NoiseCategory.from_str("music") == NoiseCategory.MUSIC
    assert NoiseCategory.from_str("babble") == NoiseCategory.BABBLE
    assert NoiseCategory.from_str("speech") == NoiseCategory.BABBLE

    with pytest.raises(ValueError):
        NoiseCategory.from_str("invalid_category")


def test_signal_power_calculation():
    """Verify average signal power calculation."""
    # Constant 1.0 signal -> power = 1.0
    sig = np.ones(1000, dtype=np.float32)
    assert NoiseMixer.calculate_signal_power(sig) == pytest.approx(1.0)

    # Sine wave of amplitude A -> power = A^2 / 2
    t = np.linspace(0, 1, 16000, endpoint=False)
    sine = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    expected_pwr = (0.8 ** 2) / 2.0
    assert NoiseMixer.calculate_signal_power(sine) == pytest.approx(expected_pwr, rel=1e-2)


def test_controlled_snr_mixing():
    """Verify mixed audio achieves the exact requested SNR in dB."""
    mixer = NoiseMixer(target_sample_rate=16000, default_seed=42)
    sr = 16000
    dur = 2.0
    num_samples = int(sr * dur)

    # 1. Speech signal: 200 Hz tone, amplitude 0.5
    t = np.linspace(0, dur, num_samples, endpoint=False)
    speech = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    # 2. Noise signal: White noise, amplitude 0.3
    rng = np.random.default_rng(123)
    noise = rng.uniform(-0.3, 0.3, num_samples).astype(np.float32)

    # Test target SNRs: 15 dB, 10 dB, 5 dB
    for target_snr in (15.0, 10.0, 5.0):
        mixed, measured_snr, sp_pwr, n_pwr, scale, peak_norm = mixer.mix_signals(
            speech_audio=speech,
            noise_audio=noise,
            target_snr_db=target_snr,
            seed=42,
        )

        assert len(mixed) == num_samples
        # Measured SNR should match target within 0.1 dB
        assert measured_snr == pytest.approx(target_snr, abs=0.1)


def test_noise_tiling_when_shorter():
    """Verify noise is seamlessly tiled when shorter than speech."""
    mixer = NoiseMixer(target_sample_rate=16000)
    # 0.5s noise, 2.0s speech
    short_noise = np.array([0.1, -0.1] * 4000, dtype=np.float32)  # 8000 samples
    target_len = 32000  # 2.0s

    sliced = mixer.prepare_noise_slice(short_noise, target_length=target_len, seed=42)
    assert len(sliced) == target_len


def test_reproducibility_with_seed():
    """Verify mixing produces bit-identical results when the same seed is used."""
    mixer = NoiseMixer(default_seed=100)
    sig = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
    noise = np.random.uniform(-0.2, 0.2, 32000).astype(np.float32)

    mix1, _, _, _, _, _ = mixer.mix_signals(sig, noise, target_snr_db=10.0, seed=99)
    mix2, _, _, _, _, _ = mixer.mix_signals(sig, noise, target_snr_db=10.0, seed=99)

    np.testing.assert_array_equal(mix1, mix2)


def test_peak_normalization_prevents_clipping(tmp_path: Path):
    """Verify peak normalization is triggered when mixed signals exceed 1.0 amplitude."""
    mixer = NoiseMixer()
    # High amplitude speech + high amplitude noise
    speech = np.ones(1000, dtype=np.float32) * 0.95
    noise = np.ones(1000, dtype=np.float32) * 0.95

    # Target 0 dB SNR -> noise scale = 1.0 -> sum would be 1.90 without normalization
    mixed, _, _, _, _, peak_norm = mixer.mix_signals(speech, noise, target_snr_db=0.0, seed=42)
    assert peak_norm is True
    assert np.max(np.abs(mixed)) <= 1.0

    # Write and read back WAV
    out_file = tmp_path / "normalized.wav"
    mixer.write_float32_as_wav(mixed, out_file)
    read_back, sr = mixer.read_wav_as_float32(out_file)
    assert sr == 16000
    assert np.max(np.abs(read_back)) <= 1.0

