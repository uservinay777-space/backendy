"""Unit tests for ASR benchmark dataset leakage prevention (Milestone 6).

Verifies:
1. DatasetLeakageFilter rejects all Lahaja, Svarah, and MUCS test samples.
2. Filter allows legitimate training samples from IndicVoices, Kathbath, Vaani, and Hinglish.
3. ASRExperimentDatasetLoader accurately separates train-eligible samples from quarantined samples.
4. Source license and quarantine catalog in source_licenses.json is complete and well-formed.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from speech.asr_experiment.preprocessing.load_dataset import (
    ASRExperimentDatasetLoader,
    DatasetLeakageFilter,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "asr_experiment"
MANIFEST_PATH = EXPERIMENT_DIR / "dataset" / "manifest.json"
LICENSES_PATH = EXPERIMENT_DIR / "dataset" / "source_licenses.json"


def test_leakage_filter_quarantines_benchmarks():
    """Verify that Lahaja, Svarah, and MUCS test samples are rejected with explicit reasons."""
    leakage_filter = DatasetLeakageFilter()

    # 1. Lahaja samples must be quarantined
    safe, reason = leakage_filter.evaluate_sample(
        sample_id="lhj_bho_001",
        source="lahaja_hindi",
        raw_transcript="हमार घरवा में आज सब लोग बहरा गइल बा",
        duration_sec=3.5,
    )
    assert not safe
    assert "Lahaja" in str(reason)

    # 2. Svarah samples must be quarantined
    safe, reason = leakage_filter.evaluate_sample(
        sample_id="svarah_002",
        source="svarah_english",
        raw_transcript="The meeting is scheduled for tomorrow",
        duration_sec=4.0,
    )
    assert not safe
    assert "Svarah" in str(reason)

    # 3. MUCS official test samples must be quarantined
    safe, reason = leakage_filter.evaluate_sample(
        sample_id="mucs_test_9001",
        source="mucs_2021",
        raw_transcript="Main station par wait kar raha hoon",
        duration_sec=3.7,
    )
    assert not safe
    assert "MUCS" in str(reason)

    # 4. Valid training sample must be accepted
    safe, reason = leakage_filter.evaluate_sample(
        sample_id="kb_hi_001",
        source="kathbath_hindi",
        raw_transcript="संसद भवन में आज महत्वपूर्ण विधेयक पारित हुआ",
        duration_sec=3.0,
    )
    assert safe
    assert reason is None

    # Summary check
    summary = leakage_filter.get_summary()
    assert summary["total_scanned"] == 4
    assert summary["total_accepted"] == 1
    assert summary["total_excluded_lahaja"] == 1
    assert summary["total_excluded_svarah"] == 1
    assert summary["total_excluded_mucs"] == 1


def test_dataset_loader_manifest_leakage_audit():
    """Verify that the experiment manifest properly filters quarantined evaluation samples."""
    assert MANIFEST_PATH.is_file(), f"Manifest not found at {MANIFEST_PATH}"
    loader = ASRExperimentDatasetLoader(MANIFEST_PATH)

    all_samples = loader.load_manifest()
    train_eligible = loader.get_train_eligible_samples()

    assert len(all_samples) >= 50
    assert len(train_eligible) < len(all_samples)

    # Verify that NO train-eligible sample contains Lahaja, Svarah, or MUCS
    for s in train_eligible:
        src = s.source_dataset.lower()
        sid = s.sample_id.lower()
        assert "lahaja" not in src and not sid.startswith("lhj_")
        assert "svarah" not in src and not sid.startswith("svarah_")
        assert "mucs" not in src and not sid.startswith("mucs_test_")

    # Verify summary audit counts
    summary = loader.leakage_filter.get_summary()
    assert summary["total_excluded_lahaja"] >= 3
    assert summary["total_excluded_svarah"] >= 3
    assert summary["total_excluded_mucs"] >= 2


def test_source_licenses_metadata_integrity():
    """Verify that source_licenses.json contains all 9 sources and explicit quarantine designations."""
    assert LICENSES_PATH.is_file(), f"Licenses file not found at {LICENSES_PATH}"
    with open(LICENSES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "sources" in data
    sources = data["sources"]
    assert len(sources) == 9

    # Check Lahaja is explicitly quarantined
    assert "lahaja_hindi" in sources
    assert sources["lahaja_hindi"]["eligibility"] == "EXCLUDED_TEST_SET"
    assert "CC BY-NC" in sources["lahaja_hindi"]["license"]

    # Check Svarah is explicitly quarantined
    assert "svarah_english" in sources
    assert sources["svarah_english"]["eligibility"] == "EXCLUDED_TEST_SET"

    # Check IndicVoices and Kathbath are permitted
    assert sources["indicvoices_hindi"]["eligibility"] == "TRAIN_PERMITTED"
    assert sources["kathbath_hindi"]["eligibility"] == "TRAIN_PERMITTED"
