"""Unit tests for the Lahaja evaluation runner, pilot mode, and report generation."""

import json
import pytest
from pathlib import Path
from speech.evaluation.evaluate_lahaja import LahajaEvaluator
from speech.evaluation.lahaja_loader import LahajaDatasetLoader


@pytest.mark.asyncio
async def test_lahaja_evaluator_dry_run(tmp_path: Path):
    """Verify LahajaEvaluator runs in dry-run mode and exports all required report files."""
    loader = LahajaDatasetLoader()
    evaluator = LahajaEvaluator(
        loader=loader,
        output_dir=tmp_path,
        min_sample_threshold=2,
        concurrency=1,
        resume=False,
        dry_run=True,
    )

    records, analysis = await evaluator.run_evaluation(limit=5, pilot=True)
    assert len(records) == 5
    assert analysis["overall"].sample_count == 5

    reports = evaluator.export_reports(records, analysis)
    assert reports["predictions_csv"].is_file()
    assert reports["metrics_json"].is_file()
    assert reports["regional_csv"].is_file()
    assert reports["scenario_csv"].is_file()
    assert reports["summary_md"].is_file()
    assert reports["summary_txt"].is_file()

    # Verify JSON content
    with open(reports["metrics_json"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["metadata"]["dataset"] == "AI4Bharat Lahaja"
        assert "overall" in data
        assert "scenarios" in data
        assert "states" in data


@pytest.mark.asyncio
async def test_lahaja_evaluator_checkpoint_resumption(tmp_path: Path):
    """Verify evaluator resumes from existing checkpoint without re-evaluating."""
    loader = LahajaDatasetLoader()
    evaluator_1 = LahajaEvaluator(
        loader=loader,
        output_dir=tmp_path,
        concurrency=1,
        resume=True,
        dry_run=True,
    )

    records_1, _ = await evaluator_1.run_evaluation(limit=3)
    assert len(records_1) == 3

    # Checkpoint exists
    checkpoint = tmp_path / "checkpoint_lahaja.json"
    assert checkpoint.is_file()

    evaluator_2 = LahajaEvaluator(
        loader=loader,
        output_dir=tmp_path,
        concurrency=1,
        resume=True,
        dry_run=True,
    )
    loaded_count = evaluator_2.load_checkpoint()
    assert loaded_count == 3

