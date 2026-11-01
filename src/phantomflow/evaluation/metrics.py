"""Evaluation metrics calculation module for PhantomFlow.

Provides exact implementations for:
- Word Error Rate (WER) using Levenshtein distance on normalized word tokens.
- Task Completion Rate (TCR) across overall, railway, healthcare domains.
- Robustness comparison (clean vs noisy conditions).
- Latency percentiles (p50, p90, p95) and target percentages (<800ms, <600ms).
- Slot F1 extraction accuracy.
"""

import math
import re
from typing import Any, Dict, List, Optional

from phantomflow.contracts.common import Domain
from phantomflow.evaluation.schemas import CallEvaluationRecord, EvaluationRun


def normalize_text(text: str) -> str:
    """Normalize text for fair WER calculation across English, Hindi, and Hinglish.

    - Lowercases text.
    - Strips punctuation symbols: .,!?;:'"-()[]{}/*
    - Normalizes multi-spaces to a single space.
    - Trims leading/trailing whitespace.
    """
    if not text:
        return ""
    lowered = text.lower()
    # Strip punctuation while preserving Devanagari / Latin unicode word characters
    cleaned = re.sub(r"[.,!?;:'\"\-()[\]{}/*\\<>|`~@#$%^&_+=]+", " ", lowered)
    return " ".join(cleaned.split())


def calculate_word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Compute (edit_distance, ref_word_count) between reference and hypothesis strings.

    Returns tuple of (substitutions + deletions + insertions, reference_token_count).
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    r = len(ref_words)
    h = len(hyp_words)

    if r == 0:
        return (h, 0)

    # DP matrix of dimensions (r + 1) x (h + 1)
    dp = [[0] * (h + 1) for _ in range(r + 1)]

    for i in range(r + 1):
        dp[i][0] = i
    for j in range(h + 1):
        dp[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)

    return (dp[r][h], r)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) between reference and hypothesis strings.

    WER = (Substitutions + Deletions + Insertions) / Total Reference Words
    Uses dynamic programming Levenshtein distance on normalized word tokens.
    """
    edits, r = calculate_word_errors(reference, hypothesis)
    if r == 0:
        return 0.0 if edits == 0 else 1.0
    wer = edits / float(r)
    return round(wer, 4)


def calculate_slot_f1(expected: Dict[str, Any], actual: Dict[str, Any]) -> float:
    """Calculate F1 score of extracted slot key-value pairs."""
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0

    # Normalize values to string comparisons
    norm_expected = {str(k).lower(): str(v).lower() for k, v in expected.items()}
    norm_actual = {str(k).lower(): str(v).lower() for k, v in actual.items()}

    tp = sum(1 for k, v in norm_actual.items() if k in norm_expected and norm_expected[k] == v)
    fp = len(norm_actual) - tp
    fn = sum(1 for k in norm_expected if k not in norm_actual or norm_actual[k] != norm_expected[k])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if (precision + recall) == 0.0:
        return 0.0
    return round(2 * (precision * recall) / (precision + recall), 4)


def summarize_run(
    records: List[CallEvaluationRecord],
    run_id: str,
    run_name: str,
    is_mock: bool = False,
    notes: str = "",
) -> EvaluationRun:
    """Aggregate a list of call records into a complete EvaluationRun."""
    total = len(records)
    if total == 0:
        return EvaluationRun(
            run_id=run_id,
            run_name=run_name,
            is_mock=is_mock,
            notes=notes,
            total_calls=0,
            overall_tcr=0.0,
            railway_tcr=0.0,
            appointment_tcr=0.0,
        )

    # Task completion
    completed = sum(1 for r in records if r.task_completed)
    overall_tcr = round((completed / total) * 100.0, 1)

    railway_records = [r for r in records if r.domain == Domain.RAILWAY]
    railway_tcr = (
        round((sum(1 for r in railway_records if r.task_completed) / len(railway_records)) * 100.0, 1)
        if railway_records else 0.0
    )

    appointment_records = [r for r in records if r.domain == Domain.HEALTHCARE]
    appointment_tcr = (
        round((sum(1 for r in appointment_records if r.task_completed) / len(appointment_records)) * 100.0, 1)
        if appointment_records else 0.0
    )

    # Clean vs Noisy TCR
    clean_records = [r for r in records if r.audio_condition == "clean"]
    noisy_records = [r for r in records if r.audio_condition in ("noisy_musan", "netem_3g")]

    clean_tcr = (
        round((sum(1 for r in clean_records if r.task_completed) / len(clean_records)) * 100.0, 1)
        if clean_records else 0.0
    )
    noisy_tcr = (
        round((sum(1 for r in noisy_records if r.task_completed) / len(noisy_records)) * 100.0, 1)
        if noisy_records else 0.0
    )

    # Interruption success rate
    interruption_records = [r for r in records if r.interruption_tested]
    interruption_success_rate = (
        round((sum(1 for r in interruption_records if r.interruption_success) / len(interruption_records)) * 100.0, 1)
        if interruption_records else None
    )

    # Aggregate WER: total edit distance over total reference words
    total_edits = 0
    total_ref_words = 0
    for r in records:
        if r.reference_transcript and r.hypothesis_transcript:
            edits, ref_count = calculate_word_errors(r.reference_transcript, r.hypothesis_transcript)
            total_edits += edits
            total_ref_words += ref_count

    if total_ref_words > 0:
        avg_wer = round((total_edits / total_ref_words) * 100.0, 2)
    else:
        wer_records = [r for r in records if r.wer is not None]
        avg_wer = (
            round((sum(r.wer for r in wer_records) / len(wer_records)) * 100.0, 2)
            if wer_records else None
        )

    # Latency percentiles
    valid_latencies = sorted([r.e2e_latency_ms for r in records if r.e2e_latency_ms is not None])
    n_lat = len(valid_latencies)

    def _percentile(p: float) -> Optional[float]:
        if n_lat == 0:
            return None
        idx = max(0, min(n_lat - 1, math.ceil(p / 100.0 * n_lat) - 1))
        return valid_latencies[idx]

    p50_e2e_ms = _percentile(50)
    p90_e2e_ms = _percentile(90)
    p95_e2e_ms = _percentile(95)

    pct_under_800ms = (
        round((sum(1 for r in records if r.meets_800ms) / total) * 100.0, 1)
    )
    pct_under_600ms = (
        round((sum(1 for r in records if r.meets_600ms) / total) * 100.0, 1)
    )

    return EvaluationRun(
        run_id=run_id,
        run_name=run_name,
        is_mock=is_mock,
        notes=notes,
        total_calls=total,
        overall_tcr=overall_tcr,
        railway_tcr=railway_tcr,
        appointment_tcr=appointment_tcr,
        p50_e2e_ms=p50_e2e_ms,
        p90_e2e_ms=p90_e2e_ms,
        p95_e2e_ms=p95_e2e_ms,
        pct_under_800ms=pct_under_800ms,
        pct_under_600ms=pct_under_600ms,
        clean_tcr=clean_tcr,
        noisy_tcr=noisy_tcr,
        interruption_success_rate=interruption_success_rate,
        avg_wer=avg_wer,
        records=records,
    )
