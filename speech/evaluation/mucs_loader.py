"""MUCS 2021 (OpenSLR 104) Dataset Loader.

Loads the official 1-hour reproducible Hindi-English code-switched test subset.
Validates audio files, converts/resamples to 16kHz 16-bit mono PCM, and prepares
utterance batches for the Sarvam Saaras ASR benchmark runner.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from speech.evaluation.base_loader import AudioSample, BaseASRDatasetLoader
except ImportError:
    from evaluation.base_loader import AudioSample, BaseASRDatasetLoader

logger = logging.getLogger("phantomflow.evaluation.mucs_loader")

DEFAULT_MUCS_DIR = Path(__file__).resolve().parent.parent / "datasets" / "mucs"


class MUCSDatasetLoader(BaseASRDatasetLoader):
    """Loader for the MUCS 2021 (OpenSLR 104) Hindi-English test evaluation set."""

    def __init__(
        self,
        root_dir: Optional[Path | str] = None,
        manifest_path: Optional[Path | str] = None,
        audio_dir: Optional[Path | str] = None,
    ) -> None:
        resolved_root = Path(root_dir) if root_dir else DEFAULT_MUCS_DIR
        super().__init__(resolved_root)

        self.manifest_path = Path(manifest_path) if manifest_path else self.root_dir / "manifest.json"
        self.audio_dir = Path(audio_dir) if audio_dir else self.root_dir / "audio"
        self._cached_samples: Optional[List[AudioSample]] = None
        self._manifest_metadata: Dict[str, Any] = {}

    @property
    def dataset_name(self) -> str:
        return self._manifest_metadata.get("dataset_name", "MUCS 2021 (OpenSLR 104)")

    @property
    def subset_name(self) -> str:
        return self._manifest_metadata.get("subset", "test-1hr-reproducible")

    @property
    def license(self) -> str:
        return self._manifest_metadata.get("license", "CC BY-SA 4.0")

    def load_manifest(self) -> Dict[str, Any]:
        """Read and parse the raw manifest.json file."""
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"MUCS manifest file not found at: {self.manifest_path}\n"
                f"Please ensure manifest.json exists or pass a custom manifest_path."
            )

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._manifest_metadata = {
            "dataset_name": data.get("dataset_name", "MUCS 2021 (OpenSLR 104)"),
            "subset": data.get("subset", "test-1hr-reproducible"),
            "language_pair": data.get("language_pair", "hi-en"),
            "license": data.get("license", "CC BY-SA 4.0"),
            "total_samples": data.get("total_samples", 0),
            "total_duration_sec": data.get("total_duration_sec", 0.0),
            "attribution": data.get("attribution", ""),
        }
        return data

    def load_samples(self) -> List[AudioSample]:
        """Load, validate, and return the list of AudioSample objects."""
        if self._cached_samples is not None:
            return self._cached_samples

        manifest_data = self.load_manifest()
        raw_samples = manifest_data.get("samples", [])
        samples: List[AudioSample] = []

        for item in raw_samples:
            sample_id = item.get("id") or item.get("audio_file", "").split(".")[0]
            audio_filename = item.get("audio_file") or f"{sample_id}.wav"
            audio_path = self.audio_dir / audio_filename

            # If audio doesn't exist in audio_dir, check if absolute or relative to root
            if not audio_path.is_file():
                alt_path = self.root_dir / audio_filename
                if alt_path.is_file():
                    audio_path = alt_path

            reference_raw = item.get("reference_raw", "")
            duration_sec = item.get("duration_sec")
            domain = item.get("domain", "general")
            language = item.get("language", "hi-en")

            sample = AudioSample(
                sample_id=sample_id,
                audio_path=audio_path,
                reference_transcript=reference_raw,
                duration_sec=duration_sec,
                language_hint=language,
                domain=domain,
                metadata={
                    "original_audio_file": audio_filename,
                    "dataset": self.dataset_name,
                    "subset": self.subset_name,
                },
            )
            samples.append(sample)

        self._cached_samples = samples
        return samples

    def get_sample_by_id(self, sample_id: str) -> Optional[AudioSample]:
        """Lookup an audio sample by its unique ID."""
        for s in self.load_samples():
            if s.sample_id == sample_id:
                return s
        return None

