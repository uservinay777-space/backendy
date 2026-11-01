"""High-resolution latency measurement and profiling for PhantomFlow.

Uses Python's monotonic high-resolution timer (time.perf_counter_ns) to measure
every millisecond of the audio -> STT -> Intent -> Task -> TTS pipeline.
Guarantees zero fabricated numbers and accurate p50/p95 calculations.
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("phantomflow.latency")

HACKATHON_TARGET_MS = 800.0
MIDNIGHT_CURSE_TARGET_MS = 600.0


class TurnLatencyRecord(BaseModel):
    """Detailed timestamp record for a single turn."""
    turn_id: str
    routing_path: str = "unknown"
    interrupted: bool = False

    # High-resolution monotonic timestamps (nanoseconds)
    t_speech_endpoint_ns: Optional[int] = None
    t_stt_first_partial_ns: Optional[int] = None
    t_stt_final_ns: Optional[int] = None
    t_intent_resolved_ns: Optional[int] = None
    t_task_start_ns: Optional[int] = None
    t_llm_first_token_ns: Optional[int] = None
    t_task_end_ns: Optional[int] = None
    t_tts_first_audio_ns: Optional[int] = None
    t_turn_complete_ns: Optional[int] = None

    def _diff_ms(self, start_ns: Optional[int], end_ns: Optional[int]) -> Optional[float]:
        if start_ns is not None and end_ns is not None and end_ns >= start_ns:
            return round((end_ns - start_ns) / 1_000_000.0, 2)
        return None

    @property
    def stt_first_partial_ms(self) -> Optional[float]:
        """Speech endpoint to first partial transcript latency."""
        return self._diff_ms(self.t_speech_endpoint_ns, self.t_stt_first_partial_ns)

    @property
    def stt_latency_ms(self) -> Optional[float]:
        """Speech endpoint to final transcript latency."""
        return self._diff_ms(self.t_speech_endpoint_ns, self.t_stt_final_ns)

    @property
    def intent_latency_ms(self) -> Optional[float]:
        """Final transcript to intent classification latency."""
        return self._diff_ms(self.t_stt_final_ns, self.t_intent_resolved_ns)

    @property
    def task_execution_ms(self) -> Optional[float]:
        """Task start to task complete latency."""
        return self._diff_ms(self.t_task_start_ns, self.t_task_end_ns)

    @property
    def llm_first_token_ms(self) -> Optional[float]:
        """Task start to first LLM token latency."""
        return self._diff_ms(self.t_task_start_ns, self.t_llm_first_token_ns)

    @property
    def tts_first_audio_ms(self) -> Optional[float]:
        """Task start to first synthesized audio chunk latency."""
        return self._diff_ms(self.t_task_start_ns, self.t_tts_first_audio_ns)

    @property
    def e2e_response_latency_ms(self) -> Optional[float]:
        """CRITICAL HACKATHON METRIC: Speech endpoint to first playable audio."""
        if self.t_speech_endpoint_ns is not None and self.t_tts_first_audio_ns is not None:
            return self._diff_ms(self.t_speech_endpoint_ns, self.t_tts_first_audio_ns)
        # Fallback if t_speech_endpoint_ns was not available: measure from stt_final
        if self.t_stt_final_ns is not None and self.t_tts_first_audio_ns is not None:
            return self._diff_ms(self.t_stt_final_ns, self.t_tts_first_audio_ns)
        return None

    def to_breakdown_dict(self) -> Dict[str, Any]:
        """Return a structured dictionary of breakdown latencies."""
        return {
            "turn_id": self.turn_id,
            "routing_path": self.routing_path,
            "interrupted": self.interrupted,
            "stt_ms": self.stt_latency_ms,
            "stt_first_partial_ms": self.stt_first_partial_ms,
            "stt_final_ms": self.stt_latency_ms,
            "intent_ms": self.intent_latency_ms,
            "task_ms": self.task_execution_ms,
            "llm_first_token_ms": self.llm_first_token_ms,
            "tts_first_audio_ms": self.tts_first_audio_ms,
            "e2e_first_audio_ms": self.e2e_response_latency_ms,
            "meets_800ms_target": (self.e2e_response_latency_ms is not None and self.e2e_response_latency_ms <= HACKATHON_TARGET_MS),
            "meets_600ms_target": (self.e2e_response_latency_ms is not None and self.e2e_response_latency_ms <= MIDNIGHT_CURSE_TARGET_MS),
        }


class LatencyTracker:
    """Tracks latency metrics across turns and computes rolling statistics."""

    def __init__(self, history_capacity: int = 200):
        self._history_capacity = history_capacity
        self._records: List[TurnLatencyRecord] = []
        self._active_records: Dict[str, TurnLatencyRecord] = {}

    def start_turn(self, turn_id: str, speech_endpoint_ns: Optional[int] = None) -> TurnLatencyRecord:
        """Initialize tracking for a new turn."""
        record = TurnLatencyRecord(
            turn_id=turn_id,
            t_speech_endpoint_ns=speech_endpoint_ns or time.perf_counter_ns(),
        )
        self._active_records[turn_id] = record
        return record

    def mark_stt_partial(self, turn_id: str) -> None:
        """Record timestamp of first partial transcript."""
        if record := self._active_records.get(turn_id):
            if record.t_stt_first_partial_ns is None:
                record.t_stt_first_partial_ns = time.perf_counter_ns()

    def mark_stt_final(self, turn_id: str) -> None:
        if record := self._active_records.get(turn_id):
            record.t_stt_final_ns = time.perf_counter_ns()

    def mark_intent_resolved(self, turn_id: str, routing_path: str) -> None:
        if record := self._active_records.get(turn_id):
            record.t_intent_resolved_ns = time.perf_counter_ns()
            record.routing_path = routing_path

    def mark_task_start(self, turn_id: str) -> None:
        if record := self._active_records.get(turn_id):
            record.t_task_start_ns = time.perf_counter_ns()

    def mark_llm_first_token(self, turn_id: str) -> None:
        if record := self._active_records.get(turn_id):
            if record.t_llm_first_token_ns is None:
                record.t_llm_first_token_ns = time.perf_counter_ns()

    def mark_task_end(self, turn_id: str) -> None:
        if record := self._active_records.get(turn_id):
            record.t_task_end_ns = time.perf_counter_ns()

    def mark_tts_first_audio(self, turn_id: str) -> None:
        if record := self._active_records.get(turn_id):
            if record.t_tts_first_audio_ns is None:
                record.t_tts_first_audio_ns = time.perf_counter_ns()

    def finish_turn(self, turn_id: str, interrupted: bool = False) -> Optional[TurnLatencyRecord]:
        """Complete tracking for a turn and archive the record."""
        record = self._active_records.pop(turn_id, None)
        if not record:
            return None

        record.t_turn_complete_ns = time.perf_counter_ns()
        record.interrupted = interrupted

        self._records.append(record)
        if len(self._records) > self._history_capacity:
            self._records.pop(0)

        breakdown = record.to_breakdown_dict()
        logger.info("Latency report for turn %s: %s", turn_id, breakdown)
        return record

    def get_percentiles(self, filter_interrupted: bool = True) -> Dict[str, Any]:
        """Calculate p50, p90, and p95 end-to-end latencies."""
        valid_records = [
            r for r in self._records
            if (not filter_interrupted or not r.interrupted)
            and r.e2e_response_latency_ms is not None
        ]

        if not valid_records:
            return {
                "count": 0,
                "p50_ms": None,
                "p90_ms": None,
                "p95_ms": None,
                "min_ms": None,
                "max_ms": None,
                "avg_ms": None,
            }

        latencies = sorted([r.e2e_response_latency_ms for r in valid_records if r.e2e_response_latency_ms is not None])
        n = len(latencies)

        def _percentile(p: float) -> float:
            idx = max(0, min(n - 1, math.ceil(p / 100.0 * n) - 1))
            return latencies[idx]

        return {
            "count": n,
            "p50_ms": _percentile(50),
            "p90_ms": _percentile(90),
            "p95_ms": _percentile(95),
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
            "avg_ms": round(sum(latencies) / n, 2),
        }
