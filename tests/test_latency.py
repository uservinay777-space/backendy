"""Tests for high-resolution latency tracking and statistical percentiles."""

import time
import pytest
from phantomflow.core.latency import LatencyTracker, TurnLatencyRecord


def test_turn_latency_breakdown():
    tracker = LatencyTracker()
    turn_id = "turn_lat_1"

    t0 = time.perf_counter_ns()
    tracker.start_turn(turn_id, speech_endpoint_ns=t0)

    # Simulate 3ms STT partial
    time.sleep(0.003)
    tracker.mark_stt_partial(turn_id)

    # Simulate 5ms STT final
    time.sleep(0.005)
    tracker.mark_stt_final(turn_id)

    # Simulate 2ms intent resolution
    time.sleep(0.002)
    tracker.mark_intent_resolved(turn_id, routing_path="fast")

    # Simulate 3ms task execution
    time.sleep(0.003)
    tracker.mark_task_start(turn_id)
    tracker.mark_task_end(turn_id)

    # Simulate 10ms TTS first audio
    time.sleep(0.010)
    tracker.mark_tts_first_audio(turn_id)

    record = tracker.finish_turn(turn_id, interrupted=False)
    assert record is not None
    breakdown = record.to_breakdown_dict()

    assert breakdown["turn_id"] == turn_id
    assert breakdown["routing_path"] == "fast"
    assert breakdown["interrupted"] is False
    assert breakdown["stt_first_partial_ms"] is not None and breakdown["stt_first_partial_ms"] > 0
    assert breakdown["stt_ms"] is not None and breakdown["stt_ms"] > 0
    assert breakdown["intent_ms"] is not None and breakdown["intent_ms"] > 0
    assert breakdown["tts_first_audio_ms"] is not None and breakdown["tts_first_audio_ms"] > 0
    assert breakdown["e2e_first_audio_ms"] is not None and breakdown["e2e_first_audio_ms"] > 0
    assert breakdown["meets_800ms_target"] is True


def test_percentile_calculations():
    tracker = LatencyTracker()

    # Seed 10 synthetic turns with varying latencies
    for i in range(1, 11):
        turn_id = f"t_{i}"
        t_start = 1_000_000_000 * i
        # Simulate latencies: 100ms, 200ms, ... 1000ms
        t_audio = t_start + (i * 100 * 1_000_000)

        record = tracker.start_turn(turn_id, speech_endpoint_ns=t_start)
        record.t_tts_first_audio_ns = t_audio
        tracker.finish_turn(turn_id, interrupted=False)

    stats = tracker.get_percentiles(filter_interrupted=True)
    assert stats["count"] == 10
    assert stats["min_ms"] == 100.0
    assert stats["max_ms"] == 1000.0
    # p50 of 10 items (100 to 1000) should be around 500
    assert stats["p50_ms"] == 500.0
    # p95 should be close to 1000
    assert stats["p95_ms"] == 1000.0
