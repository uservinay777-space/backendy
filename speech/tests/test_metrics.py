"""Unit tests for Word Error Rate (WER) and Character Error Rate (CER) calculation."""

import pytest
from speech.evaluation.metrics import (
    SampleMetrics,
    aggregate_corpus_metrics,
    calculate_cer,
    calculate_wer,
    compute_edit_distance,
    evaluate_sample,
)


def test_wer_identical_strings():
    """Verify WER is 0.0 for identical sentences."""
    ref = "Saturday ke liye Delhi se Mumbai train chahiye"
    hyp = "Saturday ke liye Delhi se Mumbai train chahiye"
    res = calculate_wer(ref, hyp)
    assert res.error_rate == 0.0
    assert res.distance == 0
    assert res.substitutions == 0
    assert res.deletions == 0
    assert res.insertions == 0
    assert res.hits == 8


def test_wer_single_substitution():
    """Verify single substitution increases distance by 1."""
    ref = "Saturday ke liye train chahiye"
    hyp = "Sunday ke liye train chahiye"
    res = calculate_wer(ref, hyp)
    assert res.substitutions == 1
    assert res.deletions == 0
    assert res.insertions == 0
    assert res.hits == 4
    assert res.ref_length == 5
    assert res.error_rate == pytest.approx(1 / 5)


def test_wer_insertion_and_deletion():
    """Verify insertions and deletions are properly categorized."""
    ref = "Delhi se Mumbai train"
    hyp = "Delhi Mumbai fast train"
    # 'se' was deleted, 'fast' was inserted
    res = calculate_wer(ref, hyp)
    assert res.deletions == 1
    assert res.insertions == 1
    assert res.hits == 3
    assert res.ref_length == 4
    assert res.error_rate == pytest.approx(2 / 4)


def test_cer_calculation():
    """Verify character-level error rate."""
    ref = "train"
    hyp = "brain"  # 1 substitution ('t' -> 'b') out of 5 chars
    res = calculate_cer(ref, hyp)
    assert res.substitutions == 1
    assert res.ref_length == 5
    assert res.error_rate == pytest.approx(0.20)


def test_empty_reference_handling():
    """Verify edge case with empty reference string."""
    res_both_empty = calculate_wer("", "")
    assert res_both_empty.error_rate == 0.0

    res_hyp_only = calculate_wer("", "extra words here")
    assert res_hyp_only.error_rate == 1.0
    assert res_hyp_only.insertions == 3


def test_evaluate_sample_structure():
    """Verify evaluate_sample returns both WER and CER in a SampleMetrics object."""
    ref = "Doctor appointment book karni hai"
    hyp = "Doctor appointment book karni hai urgently"
    s = evaluate_sample(ref, hyp)
    assert isinstance(s, SampleMetrics)
    assert s.ref_word_count == 5
    assert s.insertions == 1
    assert s.wer == pytest.approx(1 / 5)


def test_corpus_aggregation_micro_vs_macro():
    """Verify corpus WER (micro) is computed from aggregated counts, not mean of rates."""
    # Utterance 1: 1 word, 1 error -> WER = 1.0 (100%)
    s1 = SampleMetrics(
        wer=1.0,
        substitutions=1,
        deletions=0,
        insertions=0,
        hits=0,
        ref_word_count=1,
        hyp_word_count=1,
        cer=1.0,
        char_substitutions=4,
        char_deletions=0,
        char_insertions=0,
        char_hits=0,
        ref_char_count=4,
        hyp_char_count=4,
    )

    # Utterance 2: 9 words, 1 error -> WER = 1/9 (11.11%)
    s2 = SampleMetrics(
        wer=1 / 9,
        substitutions=1,
        deletions=0,
        insertions=0,
        hits=8,
        ref_word_count=9,
        hyp_word_count=9,
        cer=0.1,
        char_substitutions=4,
        char_deletions=0,
        char_insertions=0,
        char_hits=36,
        ref_char_count=40,
        hyp_char_count=40,
    )

    corpus = aggregate_corpus_metrics([s1, s2])
    # Micro WER: (1 + 1) / (1 + 9) = 2 / 10 = 0.20 (20%)
    # Macro WER: (1.0 + 1/9) / 2 = 0.5555... (55.56%)
    assert corpus.corpus_wer == pytest.approx(0.20)
    assert corpus.macro_wer == pytest.approx((1.0 + 1 / 9) / 2, rel=1e-3)
    assert corpus.corpus_wer != corpus.macro_wer

