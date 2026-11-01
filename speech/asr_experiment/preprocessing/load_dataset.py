"""ASR Dataset Loader with strict Benchmark Leakage Prevention.

Enforces critical data hygiene:
- Drops all Lahaja samples (active evaluation benchmark)
- Drops all Svarah samples (public evaluation benchmark)
- Drops all official MUCS 2021 test audio
- Normalizes transcripts and extracts 16 kHz audio paths
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Set, Tuple

try:
    from speech.asr_experiment.preprocessing.transcript_cleaning import clean_asr_transcript
except ImportError:
    from preprocessing.transcript_cleaning import clean_asr_transcript

logger = logging.getLogger("phantomflow.asr_experiment.loader")

# Excluded benchmark dataset identifiers to prevent test-set leakage
EXCLUDED_BENCHMARK_SOURCES: Set[str] = {
    "lahaja",
    "lahaja_hindi",
    "svarah",
    "svarah_english",
    "mucs",
    "mucs_2021",
}


@dataclass
class ASRSample:
    """Represents a clean, leakage-filtered speech recognition training sample."""

    sample_id: str
    source_dataset: str
    audio_path: str
    duration_sec: float
    raw_transcript: str
    cleaned_transcript: str
    language: str  # "hi", "en", or "hi-en"
    speaker_id: str = "unknown"
    is_train_eligible: bool = True
    exclusion_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetLeakageFilter:
    """Inspects and filters samples to guarantee zero contamination from evaluation benchmarks."""

    def __init__(self, excluded_sources: Optional[Set[str]] = None) -> None:
        self.excluded_sources = excluded_sources or EXCLUDED_BENCHMARK_SOURCES
        self.total_scanned = 0
        self.total_accepted = 0
        self.total_excluded_lahaja = 0
        self.total_excluded_svarah = 0
        self.total_excluded_mucs = 0
        self.total_excluded_invalid = 0

    def evaluate_sample(
        self,
        sample_id: str,
        source: str,
        raw_transcript: str,
        duration_sec: float,
    ) -> Tuple[bool, Optional[str]]:
        """Determine if a sample is safe for training or must be quarantined."""
        self.total_scanned += 1
        s_lower = source.lower().strip()
        id_lower = sample_id.lower().strip()

        # 1. Check Lahaja contamination
        if "lahaja" in s_lower or id_lower.startswith("lhj_") or "lahaja" in id_lower:
            self.total_excluded_lahaja += 1
            return False, "EXCLUDED: Lahaja test set leakage prevention"

        # 2. Check Svarah contamination
        if "svarah" in s_lower or id_lower.startswith("svarah_") or "svarah" in id_lower:
            self.total_excluded_svarah += 1
            return False, "EXCLUDED: Svarah public benchmark leakage prevention"

        # 3. Check MUCS test contamination
        if "mucs" in s_lower or id_lower.startswith("mucs_test_"):
            self.total_excluded_mucs += 1
            return False, "EXCLUDED: MUCS official test set leakage prevention"

        # 4. Check for empty transcript or invalid duration (< 0.5s or > 35s)
        if not raw_transcript or not raw_transcript.strip():
            self.total_excluded_invalid += 1
            return False, "EXCLUDED: Empty transcript"

        if duration_sec <= 0.4 or duration_sec > 35.0:
            self.total_excluded_invalid += 1
            return False, f"EXCLUDED: Invalid duration ({duration_sec:.1f}s)"

        self.total_accepted += 1
        return True, None

    def get_summary(self) -> Dict[str, int]:
        return {
            "total_scanned": self.total_scanned,
            "total_accepted": self.total_accepted,
            "total_excluded_lahaja": self.total_excluded_lahaja,
            "total_excluded_svarah": self.total_excluded_svarah,
            "total_excluded_mucs": self.total_excluded_mucs,
            "total_excluded_invalid": self.total_excluded_invalid,
        }


class ASRExperimentDatasetLoader:
    """Loads, cleans, and partitions ASR-EN-HI-400H datasets."""

    def __init__(
        self,
        manifest_path: Path | str,
        leakage_filter: Optional[DatasetLeakageFilter] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.leakage_filter = leakage_filter or DatasetLeakageFilter()
        self._cached_samples: Optional[List[ASRSample]] = None

    def load_manifest(self, reload: bool = False) -> List[ASRSample]:
        """Read and validate samples from manifest.json."""
        if self._cached_samples is not None and not reload:
            return self._cached_samples

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"ASR experiment manifest not found at: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_list = data.get("samples", [])
        samples: List[ASRSample] = []

        for item in raw_list:
            sample_id = item.get("id") or item.get("sample_id") or "sample"
            source = item.get("source") or item.get("source_dataset", "unknown")
            raw_text = item.get("raw_transcript") or item.get("transcript") or item.get("text", "")
            duration = float(item.get("duration_sec") or item.get("duration", 0.0))

            is_safe, reason = self.leakage_filter.evaluate_sample(
                sample_id=sample_id,
                source=source,
                raw_transcript=raw_text,
                duration_sec=duration,
            )

            cleaned_text = clean_asr_transcript(raw_text) if is_safe else ""

            sample = ASRSample(
                sample_id=sample_id,
                source_dataset=source,
                audio_path=item.get("audio_path") or item.get("audio_file", ""),
                duration_sec=duration,
                raw_transcript=raw_text,
                cleaned_transcript=cleaned_text,
                language=item.get("language", "hi-en"),
                speaker_id=item.get("speaker_id", "unknown"),
                is_train_eligible=is_safe,
                exclusion_reason=reason,
            )
            samples.append(sample)

        self._cached_samples = samples
        return samples

    def get_train_eligible_samples(self) -> List[ASRSample]:
        """Return only samples that passed all benchmark leakage checks."""
        all_s = self.load_manifest()
        return [s for s in all_s if s.is_train_eligible]
