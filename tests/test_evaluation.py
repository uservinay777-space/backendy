"""Unit tests for Evaluation Framework, WER, TCR, and CSV/JSON export."""

import pytest
from phantomflow.contracts.common import Domain
from phantomflow.evaluation.metrics import (
    calculate_slot_f1,
    calculate_wer,
    calculate_word_errors,
    normalize_text,
    summarize_run,
)
from phantomflow.evaluation.runner import EvaluationRunner
from phantomflow.evaluation.schemas import CallEvaluationRecord


def test_normalize_text():
    raw = "  Mujhe, Saturday ke liye... Delhi se Mumbai train chahiye?!  "
    norm = normalize_text(raw)
    assert norm == "mujhe saturday ke liye delhi se mumbai train chahiye"


def test_wer_calculation_identical():
    ref = "Mujhe Saturday ke liye Delhi se Mumbai train chahiye"
    hyp = "mujhe saturday ke liye delhi se mumbai train chahiye"
    assert calculate_wer(ref, hyp) == 0.0


def test_wer_calculation_with_errors():
    ref = "delhi se mumbai train chahiye"  # 5 words
    hyp = "delhi se bombay train"           # 4 words: 1 sub (bombay), 1 del (chahiye) -> 2 errors
    wer = calculate_wer(ref, hyp)
    assert wer == 0.4  # 2 / 5 = 0.4


def test_wer_calculation_empty_cases():
    assert calculate_wer("", "") == 0.0
    assert calculate_wer("test", "") == 1.0
    assert calculate_wer("", "test") == 1.0


def test_aggregate_wer_calculation_weighted():
    """Verify aggregate WER computes sum(edits) / sum(ref_words), not average of percentages."""
    # Sentence 1: 2 words, 1 error -> 50%
    # Sentence 2: 10 words, 1 error -> 10%
    # Blind avg: (50% + 10%) / 2 = 30%
    # Weighted corpus WER: (1 + 1) / (2 + 10) = 2 / 12 = 16.67%
    records = [
        CallEvaluationRecord(
            call_id="c1",
            dataset_name="Test",
            domain=Domain.RAILWAY,
            audio_condition="clean",
            reference_transcript="go now",
            hypothesis_transcript="go later",  # 1 sub out of 2 words
            task_completed=True,
            e2e_latency_ms=100.0,
        ),
        CallEvaluationRecord(
            call_id="c2",
            dataset_name="Test",
            domain=Domain.RAILWAY,
            audio_condition="clean",
            reference_transcript="one two three four five six seven eight nine ten",
            hypothesis_transcript="one two three four five six seven eight nine zero",  # 1 sub out of 10 words
            task_completed=True,
            e2e_latency_ms=100.0,
        ),
    ]
    run = summarize_run(records, run_id="agg_wer_test", run_name="Aggregate WER")
    assert run.avg_wer == 16.67


def test_calculate_slot_f1():
    expected = {"origin": "Delhi", "destination": "Mumbai", "date": "Saturday"}
    # Exact match
    assert calculate_slot_f1(expected, {"origin": "delhi", "destination": "mumbai", "date": "saturday"}) == 1.0

    # Partial match
    partial = {"origin": "Delhi", "destination": "Pune", "date": "Saturday"}
    f1 = calculate_slot_f1(expected, partial)
    assert 0.6 < f1 < 0.7  # 2 matches out of 3


def test_calculate_slot_f1_edge_cases():
    # Both empty -> 1.0
    assert calculate_slot_f1({}, {}) == 1.0
    # Expected empty, actual has slots -> 0.0
    assert calculate_slot_f1({}, {"origin": "Delhi"}) == 0.0
    # Expected has slots, actual empty -> 0.0
    assert calculate_slot_f1({"origin": "Delhi"}, {}) == 0.0
    # Disjoint keys
    assert calculate_slot_f1({"origin": "Delhi"}, {"destination": "Delhi"}) == 0.0


def test_summarize_run_and_percentiles():
    records = [
        CallEvaluationRecord(
            call_id="call_001",
            dataset_name="Test Set",
            domain=Domain.RAILWAY,
            audio_condition="clean",
            task_completed=True,
            e2e_latency_ms=210.0,
            meets_800ms=True,
            meets_600ms=True,
            wer=0.05,
        ),
        CallEvaluationRecord(
            call_id="call_002",
            dataset_name="Test Set",
            domain=Domain.HEALTHCARE,
            audio_condition="noisy_musan",
            task_completed=True,
            e2e_latency_ms=450.0,
            meets_800ms=True,
            meets_600ms=True,
            wer=0.10,
        ),
        CallEvaluationRecord(
            call_id="call_003",
            dataset_name="Test Set",
            domain=Domain.RAILWAY,
            audio_condition="netem_3g",
            task_completed=False,
            e2e_latency_ms=900.0,
            meets_800ms=False,
            meets_600ms=False,
            wer=0.20,
        ),
    ]

    run = summarize_run(records, run_id="run_test", run_name="Unit Test Run")
    assert run.total_calls == 3
    assert run.overall_tcr == 66.7  # 2 / 3
    assert run.railway_tcr == 50.0  # 1 / 2
    assert run.appointment_tcr == 100.0  # 1 / 1
    assert run.clean_tcr == 100.0
    assert run.noisy_tcr == 50.0
    assert run.p50_e2e_ms == 450.0
    assert run.pct_under_800ms == 66.7


def test_json_and_csv_export_import():
    runner = EvaluationRunner()
    records = [
        CallEvaluationRecord(
            call_id="call_export_1",
            dataset_name="Volunteer Calls",
            domain=Domain.RAILWAY,
            scenario="Check PNR",
            audio_condition="clean",
            ground_truth_transcript="Check PNR 1234567890",
            actual_transcript="Check PNR 1234567890",
            wer=0.0,
            expected_intent="check_pnr",
            actual_intent="check_pnr",
            intent_correct=True,
            task_completed=True,
            e2e_latency_ms=180.5,
            meets_800ms=True,
            meets_600ms=True,
        )
    ]
    run = summarize_run(records, run_id="run_export", run_name="Export Test")

    # JSON export & import
    json_out = runner.export_to_json(run)
    imported_run = runner.import_from_json(json_out)
    assert imported_run.run_id == "run_export"
    assert len(imported_run.records) == 1
    assert imported_run.records[0].call_id == "call_export_1"

    # CSV export & import
    csv_out = runner.export_to_csv(run)
    imported_csv_run = runner.import_from_csv(csv_out, run_id="run_csv_test")
    assert len(imported_csv_run.records) == 1
    assert imported_csv_run.records[0].e2e_latency_ms == 180.5
    assert imported_csv_run.records[0].task_completed is True


@pytest.mark.asyncio
async def test_synthetic_benchmark_runner():
    runner = EvaluationRunner()
    test_cases = [
        {
            "call_id": "test_call_001",
            "utterance": "Delhi to Mumbai train",
            "domain": Domain.RAILWAY,
            "expected_intent": "search_trains",
            "expected_slots": {"origin": "Delhi", "destination": "Mumbai"},
            "actual_intent": "search_trains",
            "actual_slots": {"origin": "Delhi", "destination": "Mumbai"},
            "e2e_latency_ms": 150.0,
        }
    ]
    run = await runner.run_synthetic_benchmark(test_cases, run_name="Test Bench")
    assert run.total_calls == 1
    assert run.overall_tcr == 100.0
    assert run.records[0].task_completed is True
