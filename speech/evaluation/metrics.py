"""ASR evaluation metrics: Word Error Rate (WER) and Character Error Rate (CER).

Computes exact Dynamic Programming Levenshtein distances with full breakdown
(Substitutions, Deletions, Insertions, Hits/Matches), token alignments, and
proper aggregated corpus-level (micro) and average (macro) statistics.

Definitions:
- Word Error Rate (WER) = (S + D + I) / N_ref
- Character Error Rate (CER) = (S_char + D_char + I_char) / N_ref_char
- Corpus WER = sum(S + D + I) / sum(N_ref)
- Corpus CER = sum(S_char + D_char + I_char) / sum(N_ref_char)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from speech.evaluation.text_normalization import (
        normalize_transcript,
        tokenize_characters,
        tokenize_words,
    )
except ImportError:
    from evaluation.text_normalization import (
        normalize_transcript,
        tokenize_characters,
        tokenize_words,
    )


@dataclass
class EditDistanceResult:
    """Detailed breakdown of Levenshtein edit distance between two token sequences."""

    distance: int
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_length: int
    hyp_length: int
    error_rate: float
    alignment: List[Tuple[str, Optional[str], Optional[str]]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SampleMetrics:
    """Complete evaluation metrics for a single audio recording turn."""

    # Word-level (WER)
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_word_count: int
    hyp_word_count: int

    # Character-level (CER)
    cer: float
    char_substitutions: int
    char_deletions: int
    char_insertions: int
    char_hits: int
    ref_char_count: int
    hyp_char_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CorpusMetrics:
    """Aggregated evaluation metrics across an entire evaluation corpus / dataset."""

    total_samples: int
    successful_samples: int
    failed_samples: int

    # Overall Corpus Scores (Micro / Aggregated Edit Distance)
    corpus_wer: float
    corpus_cer: float

    # Average Scores (Macro / Simple Mean of Sample Rates)
    macro_wer: float
    macro_cer: float

    # Aggregated Word Counts
    total_substitutions: int
    total_deletions: int
    total_insertions: int
    total_hits: int
    total_ref_words: int
    total_hyp_words: int

    # Aggregated Character Counts
    total_char_substitutions: int
    total_char_deletions: int
    total_char_insertions: int
    total_char_hits: int
    total_ref_chars: int
    total_hyp_chars: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_edit_distance(
    reference: Sequence[str],
    hypothesis: Sequence[str],
    compute_alignment: bool = False,
) -> EditDistanceResult:
    """Compute dynamic programming Levenshtein distance with operation breakdown.

    Operations:
    - Match / Hit: cost 0
    - Substitution: cost 1
    - Deletion (present in ref, absent in hyp): cost 1
    - Insertion (absent in ref, present in hyp): cost 1

    Args:
        reference: Sequence of reference tokens (words or characters).
        hypothesis: Sequence of hypothesis tokens (words or characters).
        compute_alignment: If True, reconstructs the alignment path.

    Returns:
        EditDistanceResult with distances, operation counts, and error rate.
    """
    n = len(reference)
    m = len(hypothesis)

    if n == 0:
        if m == 0:
            return EditDistanceResult(
                distance=0,
                substitutions=0,
                deletions=0,
                insertions=0,
                hits=0,
                ref_length=0,
                hyp_length=0,
                error_rate=0.0,
            )
        else:
            return EditDistanceResult(
                distance=m,
                substitutions=0,
                deletions=0,
                insertions=m,
                hits=0,
                ref_length=0,
                hyp_length=m,
                error_rate=1.0,
            )

    # Standard NIST sclite / ASR Alignment Cost Weights:
    # Match: 0, Substitution: 4, Deletion: 3, Insertion: 3
    # Property:
    # - 1 Substitution (cost 4) < 1 Del + 1 Ins (cost 6) -> favors substitution for single token change
    # - 1 Match + 1 Del + 1 Ins (cost 0 + 6 = 6) < 2 Substitutions (cost 8) -> favors finding real matches
    COST_MATCH = 0
    COST_SUB = 4
    COST_DEL = 3
    COST_INS = 3

    # DP Matrix: dp[i][j] = min alignment cost
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i * COST_DEL
    for j in range(m + 1):
        dp[0][j] = j * COST_INS

    for i in range(1, n + 1):
        ref_tok = reference[i - 1]
        for j in range(1, m + 1):
            hyp_tok = hypothesis[j - 1]
            if ref_tok == hyp_tok:
                dp[i][j] = dp[i - 1][j - 1] + COST_MATCH
            else:
                cost_sub = dp[i - 1][j - 1] + COST_SUB
                cost_del = dp[i - 1][j] + COST_DEL
                cost_ins = dp[i][j - 1] + COST_INS
                dp[i][j] = min(cost_sub, cost_del, cost_ins)

    # Backtrack to count operations
    i, j = n, m
    substitutions = 0
    deletions = 0
    insertions = 0
    hits = 0
    alignment: List[Tuple[str, Optional[str], Optional[str]]] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] and dp[i][j] == dp[i - 1][j - 1] + COST_MATCH:
            hits += 1
            if compute_alignment:
                alignment.append(("match", reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + COST_SUB:
            substitutions += 1
            if compute_alignment:
                alignment.append(("sub", reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + COST_DEL:
            deletions += 1
            if compute_alignment:
                alignment.append(("del", reference[i - 1], None))
            i -= 1
        else:
            insertions += 1
            if compute_alignment:
                alignment.append(("ins", None, hypothesis[j - 1]))
            j -= 1

    alignment.reverse()
    distance = substitutions + deletions + insertions
    error_rate = distance / n if n > 0 else (1.0 if m > 0 else 0.0)

    return EditDistanceResult(
        distance=distance,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        hits=hits,
        ref_length=n,
        hyp_length=m,
        error_rate=error_rate,
        alignment=alignment,
    )


def calculate_wer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
    compute_alignment: bool = False,
) -> EditDistanceResult:
    """Calculate Word Error Rate (WER) between reference and hypothesis transcripts.

    WER = (Substitutions + Deletions + Insertions) / Total Reference Words
    """
    ref_norm = normalize_transcript(reference) if normalize else reference
    hyp_norm = normalize_transcript(hypothesis) if normalize else hypothesis

    ref_words = tokenize_words(ref_norm)
    hyp_words = tokenize_words(hyp_norm)

    return compute_edit_distance(ref_words, hyp_words, compute_alignment=compute_alignment)


def calculate_cer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
    ignore_whitespace: bool = True,
    compute_alignment: bool = False,
) -> EditDistanceResult:
    """Calculate Character Error Rate (CER) between reference and hypothesis transcripts.

    CER = (Substitutions + Deletions + Insertions) / Total Reference Characters
    """
    ref_norm = normalize_transcript(reference) if normalize else reference
    hyp_norm = normalize_transcript(hypothesis) if normalize else hypothesis

    ref_chars = tokenize_characters(ref_norm, ignore_whitespace=ignore_whitespace)
    hyp_chars = tokenize_characters(hyp_norm, ignore_whitespace=ignore_whitespace)

    return compute_edit_distance(ref_chars, hyp_chars, compute_alignment=compute_alignment)


def evaluate_sample(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
) -> SampleMetrics:
    """Compute complete WER and CER metrics for an individual transcript comparison."""
    wer_res = calculate_wer(reference, hypothesis, normalize=normalize)
    cer_res = calculate_cer(reference, hypothesis, normalize=normalize)

    return SampleMetrics(
        wer=wer_res.error_rate,
        substitutions=wer_res.substitutions,
        deletions=wer_res.deletions,
        insertions=wer_res.insertions,
        hits=wer_res.hits,
        ref_word_count=wer_res.ref_length,
        hyp_word_count=wer_res.hyp_length,
        cer=cer_res.error_rate,
        char_substitutions=cer_res.substitutions,
        char_deletions=cer_res.deletions,
        char_insertions=cer_res.insertions,
        char_hits=cer_res.hits,
        ref_char_count=cer_res.ref_length,
        hyp_char_count=cer_res.hyp_length,
    )


def aggregate_corpus_metrics(samples: Sequence[SampleMetrics]) -> CorpusMetrics:
    """Calculate aggregated corpus-level metrics from a collection of sample results.

    Crucial: Computes micro Corpus WER and CER from aggregated operation counts,
    not merely by averaging per-sample rates.
    """
    total_samples = len(samples)
    if total_samples == 0:
        return CorpusMetrics(
            total_samples=0,
            successful_samples=0,
            failed_samples=0,
            corpus_wer=0.0,
            corpus_cer=0.0,
            macro_wer=0.0,
            macro_cer=0.0,
            total_substitutions=0,
            total_deletions=0,
            total_insertions=0,
            total_hits=0,
            total_ref_words=0,
            total_hyp_words=0,
            total_char_substitutions=0,
            total_char_deletions=0,
            total_char_insertions=0,
            total_char_hits=0,
            total_ref_chars=0,
            total_hyp_chars=0,
        )

    # Word aggregations
    tot_sub = sum(s.substitutions for s in samples)
    tot_del = sum(s.deletions for s in samples)
    tot_ins = sum(s.insertions for s in samples)
    tot_hits = sum(s.hits for s in samples)
    tot_ref_words = sum(s.ref_word_count for s in samples)
    tot_hyp_words = sum(s.hyp_word_count for s in samples)

    # Character aggregations
    tot_c_sub = sum(s.char_substitutions for s in samples)
    tot_c_del = sum(s.char_deletions for s in samples)
    tot_c_ins = sum(s.char_insertions for s in samples)
    tot_c_hits = sum(s.char_hits for s in samples)
    tot_ref_chars = sum(s.ref_char_count for s in samples)
    tot_hyp_chars = sum(s.hyp_char_count for s in samples)

    # Micro / Corpus error rates
    corpus_wer = (
        (tot_sub + tot_del + tot_ins) / tot_ref_words if tot_ref_words > 0 else 0.0
    )
    corpus_cer = (
        (tot_c_sub + tot_c_del + tot_c_ins) / tot_ref_chars if tot_ref_chars > 0 else 0.0
    )

    # Macro / Simple average error rates
    macro_wer = sum(s.wer for s in samples) / total_samples
    macro_cer = sum(s.cer for s in samples) / total_samples

    return CorpusMetrics(
        total_samples=total_samples,
        successful_samples=total_samples,
        failed_samples=0,
        corpus_wer=round(corpus_wer, 6),
        corpus_cer=round(corpus_cer, 6),
        macro_wer=round(macro_wer, 6),
        macro_cer=round(macro_cer, 6),
        total_substitutions=tot_sub,
        total_deletions=tot_del,
        total_insertions=tot_ins,
        total_hits=tot_hits,
        total_ref_words=tot_ref_words,
        total_hyp_words=tot_hyp_words,
        total_char_substitutions=tot_c_sub,
        total_char_deletions=tot_c_del,
        total_char_insertions=tot_c_ins,
        total_char_hits=tot_c_hits,
        total_ref_chars=tot_ref_chars,
        total_hyp_chars=tot_hyp_chars,
    )
