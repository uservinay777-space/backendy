"""PhantomFlow Controlled Noise Mixing Module (Role 02 — Speech & Audio Engineer).

Implements realistic acoustic noise mixing for ASR robustness benchmarking using MUSAN:
- Exact signal power and Signal-to-Noise Ratio (SNR) in decibels (dB)
- Configurable target SNRs (e.g. 15 dB, 10 dB, 5 dB)
- Three distinct acoustic categories:
  * 'noise': ambient, environmental, traffic, background machinery
  * 'music': instrumental, melodic background
  * 'babble': overlapping multi-speaker cocktail party chatter
- Reproducible noise segment selection via fixed pseudo-random seeds
- Dynamic duration matching (tiling if noise is shorter, seeded slicing if longer)
- Peak normalization to prevent audio clipping and integer overflow
- Format validation: 16 kHz 16-bit signed PCM mono
"""

from __future__ import annotations

import logging
import math
import random
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("phantomflow.evaluation.noise_mixer")


class NoiseCategory(str, Enum):
    """Acoustic noise category per OpenSLR 17 (MUSAN)."""

    NOISE = "noise"
    MUSIC = "music"
    BABBLE = "babble"

    @classmethod
    def from_str(cls, val: str) -> NoiseCategory:
        v = val.lower().strip()
        if v in ("noise", "ambient", "technical"):
            return cls.NOISE
        elif v in ("music", "instrumental"):
            return cls.MUSIC
        elif v in ("babble", "speech", "chatter"):
            return cls.BABBLE
        raise ValueError(f"Unknown noise category '{val}'. Expected 'noise', 'music', or 'babble'.")


@dataclass
class MixedAudioResult:
    """Telemetry and file metadata for a generated noisy audio recording."""

    clean_audio_path: Path
    noise_audio_path: Path
    output_audio_path: Path
    category: NoiseCategory
    target_snr_db: float
    measured_snr_db: float
    speech_duration_sec: float
    noise_duration_sec: float
    sample_rate: int
    speech_power: float
    noise_power: float
    noise_scale_factor: float
    peak_normalized: bool
    seed: int


class NoiseMixer:
    """Manages audio loading, sample-rate alignment, and controlled SNR mixing."""

    def __init__(
        self,
        target_sample_rate: int = 16000,
        default_seed: int = 42,
    ) -> None:
        self.target_sample_rate = target_sample_rate
        self.default_seed = default_seed

    def read_wav_as_float32(
        self,
        audio_path: Path | str,
    ) -> Tuple[np.ndarray, int]:
        """Read a WAV audio file and convert to mono float32 array in [-1.0, 1.0]."""
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            num_frames = wf.getnframes()
            frames = wf.readframes(num_frames)

        # Convert to numpy array
        if sample_width == 2:
            raw_data = np.frombuffer(frames, dtype=np.int16)
            float_data = raw_data.astype(np.float32) / 32768.0
        elif sample_width == 1:
            raw_data = np.frombuffer(frames, dtype=np.uint8)
            float_data = (raw_data.astype(np.float32) - 128.0) / 128.0
        elif sample_width == 4:
            raw_data = np.frombuffer(frames, dtype=np.int32)
            float_data = raw_data.astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width} bytes in {path}")

        # Convert multi-channel to mono
        if channels == 2:
            float_data = float_data.reshape(-1, 2).mean(axis=1)
        elif channels > 2:
            float_data = float_data.reshape(-1, channels).mean(axis=1)

        # Resample to target_sample_rate if needed
        if framerate != self.target_sample_rate and len(float_data) > 0:
            num_target_samples = int(len(float_data) * self.target_sample_rate / framerate)
            indices = np.linspace(0, len(float_data) - 1, num_target_samples)
            float_data = np.interp(indices, np.arange(len(float_data)), float_data).astype(np.float32)
            framerate = self.target_sample_rate

        return float_data, framerate

    def write_float32_as_wav(
        self,
        audio_data: np.ndarray,
        output_path: Path | str,
        sample_rate: int = 16000,
    ) -> None:
        """Save a float32 audio array in [-1.0, 1.0] as 16-bit PCM mono WAV."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Clip safety
        clipped = np.clip(audio_data, -1.0, 1.0)
        int16_data = (clipped * 32767.0).astype(np.int16)

        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_data.tobytes())

    @staticmethod
    def calculate_signal_power(signal: np.ndarray) -> float:
        """Compute average signal power (mean squared amplitude)."""
        if len(signal) == 0:
            return 0.0
        return float(np.mean(signal ** 2))

    def prepare_noise_slice(
        self,
        noise_audio: np.ndarray,
        target_length: int,
        seed: int,
    ) -> np.ndarray:
        """Extract or tile a noise slice of exact target_length reproducibly."""
        if target_length <= 0:
            return np.zeros(0, dtype=np.float32)

        noise_len = len(noise_audio)
        if noise_len == 0:
            return np.zeros(target_length, dtype=np.float32)

        # If noise is shorter than speech, repeat it until sufficiently long
        if noise_len < target_length:
            repeats = int(math.ceil(target_length / noise_len))
            noise_audio = np.tile(noise_audio, repeats)
            noise_len = len(noise_audio)

        # Slice noise segment reproducibly using seed
        rng = random.Random(seed)
        max_start = noise_len - target_length
        start_idx = rng.randint(0, max_start) if max_start > 0 else 0
        return noise_audio[start_idx : start_idx + target_length].copy()

    def mix_signals(
        self,
        speech_audio: np.ndarray,
        noise_audio: np.ndarray,
        target_snr_db: float,
        seed: int,
    ) -> Tuple[np.ndarray, float, float, float, float, bool]:
        """Mix speech and noise arrays at the exact target SNR in decibels.

        Returns:
            Tuple of:
            (mixed_audio, measured_snr_db, speech_power, noise_power, scale_factor, peak_normalized)
        """
        num_samples = len(speech_audio)
        if num_samples == 0:
            return np.zeros(0, dtype=np.float32), target_snr_db, 0.0, 0.0, 0.0, False

        noise_slice = self.prepare_noise_slice(noise_audio, num_samples, seed=seed)

        speech_pwr = self.calculate_signal_power(speech_audio)
        noise_pwr = self.calculate_signal_power(noise_slice)

        # Avoid division by zero on silent audio
        if speech_pwr < 1e-12:
            logger.warning("Speech power is near zero (silent audio). Returning clean audio.")
            return speech_audio.copy(), target_snr_db, speech_pwr, noise_pwr, 0.0, False

        if noise_pwr < 1e-12:
            logger.warning("Noise power is near zero (silent noise). Returning clean audio.")
            return speech_audio.copy(), target_snr_db, speech_pwr, noise_pwr, 0.0, False

        # Scale factor for desired SNR:
        # SNR_linear = 10^(SNR_db / 10) = P_speech / (scale^2 * P_noise)
        # scale = sqrt(P_speech / (P_noise * 10^(SNR_db / 10)))
        snr_linear = 10.0 ** (target_snr_db / 10.0)
        scale = math.sqrt(speech_pwr / (noise_pwr * snr_linear))

        scaled_noise = noise_slice * scale
        mixed = speech_audio + scaled_noise

        # Calculate actual measured SNR
        scaled_noise_pwr = self.calculate_signal_power(scaled_noise)
        measured_snr = (
            10.0 * math.log10(speech_pwr / scaled_noise_pwr)
            if scaled_noise_pwr > 1e-12
            else target_snr_db
        )

        # Check for peak clipping and apply soft peak normalization if exceeded
        peak = float(np.max(np.abs(mixed)))
        peak_normalized = False
        if peak > 0.99:
            # Scale down safely by peak to preserve waveform dynamics without clipping
            mixed = mixed * (0.95 / peak)
            peak_normalized = True

        return mixed, measured_snr, speech_pwr, noise_pwr, scale, peak_normalized

    def mix_and_save(
        self,
        clean_audio_path: Path | str,
        noise_audio_path: Path | str,
        output_audio_path: Path | str,
        category: Union[NoiseCategory, str],
        target_snr_db: float,
        seed: Optional[int] = None,
    ) -> MixedAudioResult:
        """Mix a clean audio file with a noise file at target SNR and save to disk."""
        effective_seed = seed if seed is not None else self.default_seed
        cat = category if isinstance(category, NoiseCategory) else NoiseCategory.from_str(category)

        speech, sr_speech = self.read_wav_as_float32(clean_audio_path)
        noise, sr_noise = self.read_wav_as_float32(noise_audio_path)

        mixed, measured_snr, sp_pwr, n_pwr, scale, peak_norm = self.mix_signals(
            speech_audio=speech,
            noise_audio=noise,
            target_snr_db=target_snr_db,
            seed=effective_seed,
        )

        self.write_float32_as_wav(mixed, output_audio_path, sample_rate=self.target_sample_rate)

        return MixedAudioResult(
            clean_audio_path=Path(clean_audio_path),
            noise_audio_path=Path(noise_audio_path),
            output_audio_path=Path(output_audio_path),
            category=cat,
            target_snr_db=target_snr_db,
            measured_snr_db=round(measured_snr, 2),
            speech_duration_sec=round(len(speech) / self.target_sample_rate, 2),
            noise_duration_sec=round(len(noise) / self.target_sample_rate, 2),
            sample_rate=self.target_sample_rate,
            speech_power=float(sp_pwr),
            noise_power=float(n_pwr),
            noise_scale_factor=float(scale),
            peak_normalized=peak_norm,
            seed=effective_seed,
        )

