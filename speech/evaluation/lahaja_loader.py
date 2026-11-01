"""AI4Bharat Lahaja Dataset Loader.

Loads the official Lahaja Hindi accent evaluation recordings and metadata:
- Supports manifest.json and meta-data-lahaja.csv formats
- Preserves verbatim, normalized (primary reference), and text fields
- Preserves speaker and regional metadata: native_state, native_district, native_language, scenario, sp_id
- Converts/validates audio into 16kHz 16-bit mono PCM
- Provides filtering by state, district, native language, scenario, and speaker
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from speech.evaluation.base_loader import AudioSample, BaseASRDatasetLoader
except ImportError:
    from evaluation.base_loader import AudioSample, BaseASRDatasetLoader

logger = logging.getLogger("phantomflow.evaluation.lahaja_loader")

DEFAULT_LAHAJA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "lahaja"


class LahajaDatasetLoader(BaseASRDatasetLoader):
    """Dataset loader for AI4Bharat Lahaja Hindi accent and regional dialect evaluation."""

    def __init__(
        self,
        root_dir: Optional[Path | str] = None,
        manifest_path: Optional[Path | str] = None,
        csv_path: Optional[Path | str] = None,
        audio_dir: Optional[Path | str] = None,
        reference_field: str = "normalized",
    ) -> None:
        resolved_root = Path(root_dir) if root_dir else DEFAULT_LAHAJA_DIR
        super().__init__(resolved_root)

        self.manifest_path = Path(manifest_path) if manifest_path else self.root_dir / "manifest.json"
        self.csv_path = Path(csv_path) if csv_path else self.root_dir / "meta-data-lahaja.csv"
        self.audio_dir = Path(audio_dir) if audio_dir else self.root_dir / "audio"
        self.reference_field = reference_field  # Default is 'normalized' per specification

        self._cached_samples: Optional[List[AudioSample]] = None
        self._manifest_metadata: Dict[str, Any] = {}

    @property
    def dataset_name(self) -> str:
        return self._manifest_metadata.get("dataset_name", "AI4Bharat Lahaja")

    @property
    def license(self) -> str:
        return self._manifest_metadata.get("license", "AI4Bharat Research License / CC BY-NC-SA 4.0")

    def _load_from_manifest(self) -> List[AudioSample]:
        """Load audio samples from manifest.json."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._manifest_metadata = {
            "dataset_name": data.get("dataset_name", "AI4Bharat Lahaja"),
            "license": data.get("license", "CC BY-NC-SA 4.0"),
            "repository": data.get("repository", "https://github.com/AI4Bharat/Lahaja"),
            "huggingface": data.get("huggingface", "https://huggingface.co/datasets/ai4bharat/Lahaja"),
            "total_samples": data.get("total_samples", 0),
        }

        raw_samples = data.get("samples", [])
        samples: List[AudioSample] = []

        for item in raw_samples:
            fname = item.get("fname") or item.get("audio_file") or f"{item.get('id', 'sample')}.wav"
            audio_path = self.audio_dir / fname
            if not audio_path.is_file():
                alt = self.root_dir / fname
                if alt.is_file():
                    audio_path = alt

            # Upstream normalized transcript is primary reference
            primary_ref = (
                item.get(self.reference_field)
                or item.get("normalized")
                or item.get("text")
                or item.get("verbatim")
                or ""
            )

            sample_id = Path(fname).stem
            sample = AudioSample(
                sample_id=sample_id,
                audio_path=audio_path,
                reference_transcript=primary_ref,
                duration_sec=item.get("duration"),
                language_hint="hi",
                domain=f"lahaja_{item.get('scenario', 'general')}",
                metadata={
                    "fname": fname,
                    "sp_id": item.get("sp_id", "unknown_spk"),
                    "native_state": item.get("native_state", "Unknown"),
                    "native_district": item.get("native_district", "Unknown"),
                    "native_language": item.get("native_language", "Hindi"),
                    "scenario": item.get("scenario", "read"),
                    "verbatim": item.get("verbatim", ""),
                    "normalized": item.get("normalized", ""),
                    "text": item.get("text", ""),
                },
            )
            samples.append(sample)

        return samples

    def _load_from_csv(self) -> List[AudioSample]:
        """Load audio samples from official meta-data-lahaja.csv."""
        samples: List[AudioSample] = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("fname") or row.get("audio_file") or ""
                audio_path = self.audio_dir / fname
                primary_ref = (
                    row.get(self.reference_field)
                    or row.get("normalized")
                    or row.get("text")
                    or row.get("verbatim")
                    or ""
                )

                sample_id = Path(fname).stem
                sample = AudioSample(
                    sample_id=sample_id,
                    audio_path=audio_path,
                    reference_transcript=primary_ref,
                    duration_sec=float(row.get("duration", 0.0)) if row.get("duration") else None,
                    language_hint="hi",
                    domain=f"lahaja_{row.get('scenario', 'general')}",
                    metadata={
                        "fname": fname,
                        "sp_id": row.get("sp_id", "unknown_spk"),
                        "native_state": row.get("native_state", "Unknown"),
                        "native_district": row.get("native_district", "Unknown"),
                        "native_language": row.get("native_language", "Hindi"),
                        "scenario": row.get("scenario", "read"),
                        "verbatim": row.get("verbatim", ""),
                        "normalized": row.get("normalized", ""),
                        "text": row.get("text", ""),
                    },
                )
                samples.append(sample)
        return samples

    def load_samples(self) -> List[AudioSample]:
        """Load audio samples from manifest.json (preferred) or meta-data-lahaja.csv."""
        if self._cached_samples is not None:
            return self._cached_samples

        if self.manifest_path.is_file():
            self._cached_samples = self._load_from_manifest()
        elif self.csv_path.is_file():
            self._cached_samples = self._load_from_csv()
        else:
            raise FileNotFoundError(
                f"Neither manifest ({self.manifest_path}) nor CSV metadata ({self.csv_path}) was found."
            )

        return self._cached_samples

    def filter_samples(
        self,
        scenario: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        native_language: Optional[str] = None,
        speaker: Optional[str] = None,
    ) -> List[AudioSample]:
        """Filter loaded samples by specific demographic or scenario metadata."""
        all_samples = self.load_samples()
        filtered = []

        for s in all_samples:
            meta = s.metadata
            if scenario and meta.get("scenario", "").lower() != scenario.lower():
                continue
            if state and meta.get("native_state", "").lower() != state.lower():
                continue
            if district and meta.get("native_district", "").lower() != district.lower():
                continue
            if native_language and meta.get("native_language", "").lower() != native_language.lower():
                continue
            if speaker and meta.get("sp_id", "").lower() != speaker.lower():
                continue
            filtered.append(s)

        return filtered

    def get_sample_by_id(self, sample_id: str) -> Optional[AudioSample]:
        """Lookup sample by filename stem / ID."""
        for s in self.load_samples():
            if s.sample_id == sample_id or s.metadata.get("fname") == sample_id:
                return s
        return None

