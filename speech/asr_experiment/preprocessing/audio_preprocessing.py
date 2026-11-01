"""Audio feature extraction and Mel-spectrogram processing for local ASR fine-tuning.

Extracts standard 80-channel Log-Mel Spectrogram features matching OpenAI Whisper /
modern multilingual ASR architectures:
- Sample rate: 16,000 Hz
- FFT size: 512, Window length: 400 (25 ms), Hop length: 160 (10 ms)
- Filterbanks: 80 triangular Mel bins spanning 0 Hz to 8000 Hz
- Log compression: log10(max(mel, 1e-5))
"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def hz_to_mel(hz: float) -> float:
    """Convert Frequency in Hertz to Mel scale."""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """Convert Mel scale to Frequency in Hertz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sample_rate: int = 16000,
    n_fft: int = 512,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """Construct an 80-channel triangular Mel filterbank matrix of shape (n_mels, 1 + n_fft // 2)."""
    n_freqs = 1 + n_fft // 2
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    # Linearly spaced points in Mel scale
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]

        if f_center > f_left:
            for k in range(f_left, f_center):
                if k < n_freqs:
                    filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        if f_right > f_center:
            for k in range(f_center, f_right):
                if k < n_freqs:
                    filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    return filterbank


class AudioPreprocessor:
    """Preprocesses 16 kHz raw audio waveforms into normalized Log-Mel Spectrograms."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,  # 25ms
        hop_length: int = 160,  # 10ms
        n_mels: int = 80,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.window = np.hanning(self.win_length).astype(np.float32)
        self.filterbank = create_mel_filterbank(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
        )

    def compute_log_mel_spectrogram(
        self,
        audio_array: np.ndarray,
        target_frames: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the 80-channel Log-Mel Spectrogram for a 1D float32 audio waveform.

        Args:
            audio_array: 1D numpy array of audio samples in [-1.0, 1.0].
            target_frames: Optional number of frames to pad or truncate to (e.g., 3000 for 30s).

        Returns:
            2D numpy array of shape (n_mels, n_frames).
        """
        if len(audio_array) == 0:
            return np.zeros((self.n_mels, target_frames or 1), dtype=np.float32)

        # 1. Framing & STFT
        num_samples = len(audio_array)
        n_frames = max(1, 1 + (num_samples - self.win_length) // self.hop_length)

        # Pad end if needed
        pad_needed = (n_frames - 1) * self.hop_length + self.win_length - num_samples
        if pad_needed > 0:
            padded_audio = np.pad(audio_array, (0, pad_needed), mode="constant")
        else:
            padded_audio = audio_array

        stft_matrix = np.zeros((1 + self.n_fft // 2, n_frames), dtype=np.complex64)
        for t in range(n_frames):
            start = t * self.hop_length
            end = start + self.win_length
            segment = padded_audio[start:end] * self.window
            # Zero-pad to n_fft
            fft_res = np.fft.rfft(segment, n=self.n_fft)
            stft_matrix[:, t] = fft_res

        # 2. Power Spectrogram
        power_spec = (np.abs(stft_matrix) ** 2).astype(np.float32)

        # 3. Apply Mel Filterbank: shape (n_mels, n_frames)
        mel_spec = np.dot(self.filterbank, power_spec)

        # 4. Log Compression
        log_mel = np.log10(np.maximum(mel_spec, 1e-5))

        # 5. Dynamic Normalization (zero mean, unit variance approximation)
        max_val = np.max(log_mel)
        log_mel = (log_mel - max_val + 4.0) / 4.0
        log_mel = np.clip(log_mel, -1.0, 1.0)

        # 6. Optional Target Frames Padding / Truncation
        if target_frames is not None:
            cur_frames = log_mel.shape[1]
            if cur_frames < target_frames:
                log_mel = np.pad(log_mel, ((0, 0), (0, target_frames - cur_frames)), mode="constant")
            elif cur_frames > target_frames:
                log_mel = log_mel[:, :target_frames]

        return log_mel.astype(np.float32)


def read_audio_file(audio_path: Path | str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Read a WAV audio file into a 1D float32 numpy array [-1.0, 1.0] and sample rate."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 2:
        audio_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio_data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        audio_data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)

    if framerate != target_sr and len(audio_data) > 0:
        target_len = int(len(audio_data) * target_sr / framerate)
        indices = np.linspace(0, len(audio_data) - 1, target_len)
        audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data).astype(np.float32)
        framerate = target_sr

    return audio_data, framerate


def save_wav_file(audio_path: Path | str, audio_array: np.ndarray, sample_rate: int = 16000) -> Path:
    """Save a 1D float32 numpy array as a standard 16-bit mono PCM WAV file."""
    path = Path(audio_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio_array, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())
    return path

