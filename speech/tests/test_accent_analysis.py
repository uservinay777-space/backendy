"""Unit tests for accent robustness analysis across regional states, dialects, and scenarios."""

import pytest
from pathlib import Path
from speech.evaluation.accent_analysis import AccentRobustnessAnalyzer


class DummyRecord:
    def __init__(self, sp_id, native_state, native_language, scenario, wer, cer, word_count=10, latency_ms=10.0, status="SUCCESS"):
        self.sp_id = sp_id
        self.native_state = native_state
        self.native_language = native_language
        self.scenario = scenario
        self.wer = wer
        self.cer = cer
        self.word_count = word_count
        self.substitutions = int(wer * word_count)
        self.deletions = 0
        self.insertions = 0
        self.hits = word_count - self.substitutions
        self.char_count = word_count * 5
        self.char_substitutions = int(cer * self.char_count)
        self.char_deletions = 0
        self.char_insertions = 0
        self.char_hits = self.char_count - self.char_substitutions
        self.latency_ms = latency_ms
        self.duration_sec = 6.0
        self.status = status


def test_accent_analyzer_groupings():
    """Verify analyzer correctly segments records across states and scenarios."""
    analyzer = AccentRobustnessAnalyzer(min_sample_threshold=3)

    records = [
        # UP records (4 samples, 2 speakers)
        DummyRecord("spk_01", "Uttar Pradesh", "Awadhi", "read", 0.0, 0.0),
        DummyRecord("spk_01", "Uttar Pradesh", "Awadhi", "extempore", 0.1, 0.02),
        DummyRecord("spk_02", "Uttar Pradesh", "Bhojpuri", "read", 0.0, 0.0),
        DummyRecord("spk_02", "Uttar Pradesh", "Bhojpuri", "extempore", 0.2, 0.04),
        # Bihar records (2 samples, 1 speaker) -> under threshold!
        DummyRecord("spk_03", "Bihar", "Maithili", "read", 0.0, 0.0),
        DummyRecord("spk_03", "Bihar", "Maithili", "extempore", 0.1, 0.02),
    ]

    analysis = analyzer.analyze_dataset(records)

    # 1. Overall
    overall = analysis["overall"]
    assert overall.sample_count == 6
    assert overall.speaker_count == 3
    assert overall.insufficient_data is False

    # 2. Scenarios
    scenarios = {s.group_name: s for s in analysis["scenarios"]}
    assert "read" in scenarios
    assert "extempore" in scenarios
    assert scenarios["read"].sample_count == 3
    assert scenarios["extempore"].sample_count == 3
    assert scenarios["read"].corpus_wer < scenarios["extempore"].corpus_wer

    # 3. State-wise & Threshold Flagging
    states = {st.group_name: st for st in analysis["states"]}
    assert "Uttar Pradesh" in states
    assert "Bihar" in states
    # UP has 4 samples >= 3 -> not insufficient
    assert states["Uttar Pradesh"].insufficient_data is False
    # Bihar has 2 samples < 3 -> insufficient flag must be True
    assert states["Bihar"].insufficient_data is True
    assert "Low sample size" in states["Bihar"].warning


def test_export_regional_and_scenario_csvs(tmp_path: Path):
    """Verify CSV export of regional analysis and scenario comparison."""
    analyzer = AccentRobustnessAnalyzer(min_sample_threshold=2)
    records = [
        DummyRecord("spk_01", "Uttar Pradesh", "Hindi", "read", 0.0, 0.0),
        DummyRecord("spk_01", "Uttar Pradesh", "Hindi", "extempore", 0.1, 0.02),
    ]
    analysis = analyzer.analyze_dataset(records)

    reg_path = tmp_path / "regional_analysis.csv"
    scen_path = tmp_path / "scenario_comparison.csv"

    analyzer.export_regional_csv(analysis, reg_path)
    analyzer.export_scenario_csv(analysis, scen_path)

    assert reg_path.is_file()
    assert scen_path.is_file()

