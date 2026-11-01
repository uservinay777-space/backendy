"""Accent and Regional Robustness Analysis for AI4Bharat Lahaja.

Aggregates and computes comparative ASR performance metrics across:
- Overall dataset performance
- Speaking scenarios (Read Speech vs. Extempore Speech)
- Native states (regional Hindi accents)
- Native language backgrounds (Awadhi, Bhojpuri, Maithili, Magahi, Braj, Marwari, etc.)
- Speaker-level aggregated performance (using pseudonymous speaker IDs)

Key Methodology:
- Uses exact Levenshtein Dynamic Programming edit distance counts (S + D + I) / N_ref
- Computes aggregated micro Corpus WER/CER for each group
- Flags groups with insufficient sample sizes (< min_samples threshold) to avoid biased generalizations
- Preserves Devanagari and Latin characters without script transliteration
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from speech.evaluation.metrics import CorpusMetrics, SampleMetrics, aggregate_corpus_metrics
except ImportError:
    from evaluation.metrics import CorpusMetrics, SampleMetrics, aggregate_corpus_metrics


@dataclass
class GroupAnalysisMetric:
    """Aggregated evaluation metrics for a specific regional, demographic, or scenario group."""

    group_type: str  # "overall", "scenario", "state", "native_language", "speaker"
    group_name: str  # e.g., "Uttar Pradesh", "read", "Bhojpuri", "spk_up_001"
    sample_count: int
    speaker_count: int
    total_duration_sec: float
    total_ref_words: int
    total_hyp_words: int
    total_errors: int
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    corpus_wer: float
    corpus_cer: float
    macro_wer: float
    macro_cer: float
    avg_latency_ms: float
    success_rate_pct: float
    insufficient_data: bool = False
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AccentRobustnessAnalyzer:
    """Analyzes ASR predictions across regional accents and speech scenarios."""

    def __init__(self, min_sample_threshold: int = 3) -> None:
        self.min_sample_threshold = min_sample_threshold

    def _aggregate_records(
        self,
        group_type: str,
        group_name: str,
        records: Sequence[Any],
    ) -> GroupAnalysisMetric:
        """Aggregate a slice of evaluation records into a GroupAnalysisMetric."""
        sample_count = len(records)
        if sample_count == 0:
            return GroupAnalysisMetric(
                group_type=group_type,
                group_name=group_name,
                sample_count=0,
                speaker_count=0,
                total_duration_sec=0.0,
                total_ref_words=0,
                total_hyp_words=0,
                total_errors=0,
                substitutions=0,
                deletions=0,
                insertions=0,
                hits=0,
                corpus_wer=0.0,
                corpus_cer=0.0,
                macro_wer=0.0,
                macro_cer=0.0,
                avg_latency_ms=0.0,
                success_rate_pct=0.0,
                insufficient_data=True,
                warning="No samples available",
            )

        unique_speakers = len(set(getattr(r, "sp_id", "spk") for r in records))
        total_dur = sum(getattr(r, "duration_sec", 0.0) for r in records)
        latencies = [getattr(r, "latency_ms", 0.0) for r in records if getattr(r, "latency_ms", 0.0) > 0]
        avg_lat = float(sum(latencies) / len(latencies)) if latencies else 0.0

        valid_records = [r for r in records if getattr(r, "status", "") in ("SUCCESS", "DRY_RUN")]
        success_rate = (len(valid_records) / sample_count) * 100.0

        sample_metrics = [
            SampleMetrics(
                wer=r.wer,
                substitutions=r.substitutions,
                deletions=r.deletions,
                insertions=r.insertions,
                hits=r.hits,
                ref_word_count=r.word_count,
                hyp_word_count=r.hits + r.substitutions + r.insertions,
                cer=r.cer,
                char_substitutions=r.char_substitutions,
                char_deletions=r.char_deletions,
                char_insertions=r.char_insertions,
                char_hits=r.char_hits,
                ref_char_count=r.char_count,
                hyp_char_count=r.char_hits + r.char_substitutions + r.char_insertions,
            )
            for r in valid_records
        ]

        corpus = aggregate_corpus_metrics(sample_metrics)
        insufficient = sample_count < self.min_sample_threshold
        warning_msg = (
            f"Low sample size (< {self.min_sample_threshold} samples); interpret with caution."
            if insufficient
            else None
        )

        return GroupAnalysisMetric(
            group_type=group_type,
            group_name=group_name,
            sample_count=sample_count,
            speaker_count=unique_speakers,
            total_duration_sec=round(total_dur, 2),
            total_ref_words=corpus.total_ref_words,
            total_hyp_words=corpus.total_hyp_words,
            total_errors=corpus.total_substitutions + corpus.total_deletions + corpus.total_insertions,
            substitutions=corpus.total_substitutions,
            deletions=corpus.total_deletions,
            insertions=corpus.total_insertions,
            hits=corpus.total_hits,
            corpus_wer=corpus.corpus_wer,
            corpus_cer=corpus.corpus_cer,
            macro_wer=corpus.macro_wer,
            macro_cer=corpus.macro_cer,
            avg_latency_ms=round(avg_lat, 2),
            success_rate_pct=round(success_rate, 1),
            insufficient_data=insufficient,
            warning=warning_msg,
        )

    def analyze_dataset(self, records: Sequence[Any]) -> Dict[str, Any]:
        """Perform comprehensive multi-dimensional accent and scenario analysis."""
        # 1. Overall
        overall_metric = self._aggregate_records("overall", "Overall Dataset", records)

        # 2. Scenario Analysis: Read vs Extempore
        by_scenario: Dict[str, List[Any]] = {}
        for r in records:
            scen = getattr(r, "scenario", "unknown")
            by_scenario.setdefault(scen, []).append(r)

        scenario_metrics = [
            self._aggregate_records("scenario", scen, recs)
            for scen, recs in sorted(by_scenario.items())
        ]

        # 3. State-wise Analysis
        by_state: Dict[str, List[Any]] = {}
        for r in records:
            st = getattr(r, "native_state", "Unknown")
            by_state.setdefault(st, []).append(r)

        state_metrics = [
            self._aggregate_records("state", st, recs)
            for st, recs in sorted(by_state.items(), key=lambda x: len(x[1]), reverse=True)
        ]

        # 4. Native Language Background Analysis
        by_lang: Dict[str, List[Any]] = {}
        for r in records:
            nl = getattr(r, "native_language", "Unknown")
            by_lang.setdefault(nl, []).append(r)

        language_metrics = [
            self._aggregate_records("native_language", nl, recs)
            for nl, recs in sorted(by_lang.items(), key=lambda x: len(x[1]), reverse=True)
        ]

        # 5. Speaker-level Analysis (Pseudonymous)
        by_speaker: Dict[str, List[Any]] = {}
        for r in records:
            spk = getattr(r, "sp_id", "unknown_spk")
            by_speaker.setdefault(spk, []).append(r)

        speaker_metrics = [
            self._aggregate_records("speaker", spk, recs)
            for spk, recs in sorted(by_speaker.items(), key=lambda x: x[0])
        ]

        return {
            "overall": overall_metric,
            "scenarios": scenario_metrics,
            "states": state_metrics,
            "native_languages": language_metrics,
            "speakers": speaker_metrics,
        }

    def export_regional_csv(
        self,
        analysis_data: Dict[str, Any],
        output_path: Path | str,
    ) -> Path:
        """Export state and native language group metrics to regional_analysis.csv."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "group_type",
            "group_name",
            "sample_count",
            "speaker_count",
            "total_duration_sec",
            "total_ref_words",
            "corpus_wer",
            "corpus_cer",
            "macro_wer",
            "avg_latency_ms",
            "success_rate_pct",
            "insufficient_data",
            "warning",
        ]

        all_groups = []
        all_groups.append(analysis_data["overall"])
        all_groups.extend(analysis_data["states"])
        all_groups.extend(analysis_data["native_languages"])

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for g in all_groups:
                d = g.to_dict()
                row = {k: d.get(k) for k in fieldnames}
                writer.writerow(row)

        return out

    def export_scenario_csv(
        self,
        analysis_data: Dict[str, Any],
        output_path: Path | str,
    ) -> Path:
        """Export Read vs Extempore comparison to scenario_comparison.csv."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "scenario",
            "sample_count",
            "speaker_count",
            "total_ref_words",
            "corpus_wer",
            "corpus_cer",
            "macro_wer",
            "avg_latency_ms",
            "success_rate_pct",
        ]

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in analysis_data["scenarios"]:
                d = s.to_dict()
                row = {
                    "scenario": d["group_name"],
                    "sample_count": d["sample_count"],
                    "speaker_count": d["speaker_count"],
                    "total_ref_words": d["total_ref_words"],
                    "corpus_wer": d["corpus_wer"],
                    "corpus_cer": d["corpus_cer"],
                    "macro_wer": d["macro_wer"],
                    "avg_latency_ms": d["avg_latency_ms"],
                    "success_rate_pct": d["success_rate_pct"],
                }
                writer.writerow(row)

        return out

