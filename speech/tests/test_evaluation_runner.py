"""Unit tests for the MUCS 2021 evaluation runner, checkpointing, and reporting."""

import json
import pytest
from pathlib import Path
from speech.evaluation.evaluate_mucs import MUCSEvaluator
from speech.evaluation.mucs_loader import MUCSDatasetLoader


@pytest.mark.asyncio
async def test_mucs_evaluator_dry_run(tmp_path: Path):
    """Verify evaluation runner works end-to-end in dry-run mode and creates reports."""
    loader = MUCSDatasetLoader()
    evaluator = MUCSEvaluator(
        loader=loader,
        output_dir=tmp_path,
        concurrency=1,
        dry_run=True,
    )

    records, corpus_metrics = await evaluator.run_evaluation(limit=3)
    assert len(records) == 3
    assert corpus_metrics.total_samples == 3
    assert corpus_metrics.successful_samples == 3

    reports = evaluator.export_reports(records, corpus_metrics)
    assert reports["csv"].is_file()
    assert reports["json"].is_file()
    assert reports["summary_md"].is_file()

    # Read exported JSON
    with open(reports["json"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["metadata"]["dataset"] == "MUCS 2021 (OpenSLR 104)"
        assert len(data["records"]) == 3


@pytest.mark.asyncio
async def test_mucs_evaluator_resumption(tmp_path: Path):
    """Verify evaluator loads existing checkpoints and does not re-process completed samples."""
    loader = MUCSDatasetLoader()
    evaluator_1 = MUCSEvaluator(
        loader=loader,
        output_dir=tmp_path,
        concurrency=1,
        resume=True,
        dry_run=True,
    )

    # Process 2 samples
    records_1, _ = await evaluator_1.run_evaluation(limit=2)
    assert len(records_1) == 2

    # Verify checkpoint file exists
    checkpoint_file = tmp_path / "checkpoint_results.json"
    assert checkpoint_file.is_file()

    # Create new evaluator pointing to same directory and resume
    evaluator_2 = MUCSEvaluator(
        loader=loader,
        output_dir=tmp_path,
        concurrency=1,
        resume=True,
        dry_run=True,
    )
    loaded_count = evaluator_2.load_checkpoint()
    assert loaded_count == 2
    assert "mucs_test_hien_0001" in evaluator_2.records

