"""Abstract Base Class and data models for ASR Evaluation Dataset Loaders.

Enables seamless extension to future evaluation datasets such as:
- Lahaja (regional Hindi accents)
- MUSAN (additive acoustic noise and babble)
- Custom Railway and Doctor appointment call recordings (30-50 samples)
"""

from __future__ import annotations

import io
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple
import numpy as np


@dataclass
class AudioSample:
    """Represents a single audio recording and its reference transcript in an ASR evaluation set."""

    sample_id: str
    audio_path: Path
    reference_transcript: str
    duration_sec: Optional[float] = None
    language_hint: str = "hi-en"
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def file_exists(self) -> bool:
        return self.audio_path.is_file()

    def get_actual_duration_sec(self) -> float:
        """Inspect the audio file header to get precise duration in seconds."""
        if not self.file_exists:
            return self.duration_sec or 0.0

        try:
            with wave.open(str(self.audio_path), "rb") as wf:
                framerate = wf.getframerate()
                frames = wf.getnframes()
                return frames / float(framerate) if framerate > 0 else 0.0
        except Exception:
            return self.duration_sec or 0.0


class BaseASRDatasetLoader(ABC):
    """Abstract interface for loading and preparing speech datasets for ASR evaluation."""

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = Path(root_dir)

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Human-readable dataset name."""
        ...

    @abstractmethod
    def load_samples(self) -> List[AudioSample]:
        """Load and return the list of AudioSample items in this dataset."""
        ...

    def validate_dataset(self) -> Dict[str, Any]:
        """Verify presence of audio files and reference transcripts.

        Returns:
            Dictionary with counts of total, valid, missing audio, and empty transcript samples.
        """
        samples = self.load_samples()
        total = len(samples)
        valid = 0
        missing_audio = []
        empty_transcripts = []
        total_duration = 0.0

        for s in samples:
            duration = s.get_actual_duration_sec()
            total_duration += duration

            if not s.file_exists:
                missing_audio.append(s.sample_id)
            elif not s.reference_transcript.strip():
                empty_transcripts.append(s.sample_id)
            else:
                valid += 1

        return {
            "dataset_name": self.dataset_name,
            "total_samples": total,
            "valid_samples": valid,
            "missing_audio_count": len(missing_audio),
            "missing_audio_samples": missing_audio[:10],
            "empty_transcript_count": len(empty_transcripts),
            "total_duration_sec": total_duration,
            "total_duration_formatted": f"{int(total_duration // 3600):02d}:{int((total_duration % 3600) // 60):02d}:{int(total_duration % 60):02d}",
        }

    def read_pcm16_mono(
        self,
        audio_path: Path | str,
        target_sample_rate: int = 16000,
    ) -> Tuple[bytes, int]:
        """Read an audio file, validate or convert to 16-bit signed PCM mono, and return bytes.

        Args:
            audio_path: Path to the WAV audio file.
            target_sample_rate: Required sample rate (default 16000 Hz for Sarvam Saaras).

        Returns:
            Tuple of (raw_pcm16_bytes, actual_sample_rate)
        """
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        # Validate format
        if sample_width != 2:
            # Convert to 16-bit PCM if needed
            audio_array = np.frombuffer(frames, dtype=np.int32 if sample_width == 4 else np.int16)
            audio_array = (audio_array / (2 ** (sample_width * 8 - 1)) * 32767).astype(np.int16)
            frames = audio_array.tobytes()

        # Convert stereo to mono if needed
        if channels == 2:
            audio_array = np.frombuffer(frames, dtype=np.int16)
            mono_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
            frames = mono_array.tobytes()
        elif channels > 2:
            raise ValueError(f"Unsupported channel count {channels} in audio file {path}")

        # Resample if sample rate doesn't match
        if framerate != target_sample_rate:
            audio_array = np.frombuffer(frames, dtype=np.int16)
            num_target_samples = int(len(audio_array) * target_sample_rate / framerate)
            indices = np.linspace(0, len(audio_array) - 1, num_target_samples)
            resampled = np.interp(indices, np.arange(len(audio_array)), audio_array).astype(np.int16)
            frames = resampled.tobytes()
            framerate = target_sample_rate

        return frames, framerate

    def chunk_audio_pcm(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 20,
    ) -> Iterator[bytes]:
        """Split raw PCM bytes into fixed-duration frames (e.g. 20ms = 640 bytes)."""
        bytes_per_sample = 2  # 16-bit mono
        chunk_size = int(sample_rate * (chunk_duration_ms / 1000.0) * bytes_per_sample)

        for offset in range(0, len(pcm_bytes), chunk_size):
            yield pcm_bytes[offset : offset + chunk_size]

