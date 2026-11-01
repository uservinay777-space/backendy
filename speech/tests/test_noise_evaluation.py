"""Unit tests for the MUSAN noise robustness evaluation runner and reporting."""

import json
import pytest
from pathlib import Path

from speech.evaluation.evaluate_noise import NoiseRobustnessEvaluator
from speech.evaluation.mucs_loader import MUCSDatasetLoader


@pytest.mark.asyncio
async def test_noise_evaluator_dry_run(tmp_path: Path):
    """Verify NoiseRobustnessEvaluator runs cleanly in dry-run mode and exports reports."""
    clean_loader = MUCSDatasetLoader()
    musan_raw_dir = Path("speech/datasets/musan/raw")
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "reports"

    evaluator = NoiseRobustnessEvaluator(
        clean_loader=clean_loader,
        musan_raw_dir=musan_raw_dir,
        processed_audio_dir=processed_dir,
        output_dir=output_dir,
        snr_levels=[15.0, 10.0, 5.0],
        categories=["noise", "music", "babble"],
        concurrency=1,
        random_seed=42,
        resume=False,
        dry_run=True,
    )

    payload = await evaluator.run_evaluation(limit=1)
    records = payload["records"]
    summaries = payload["condition_summaries"]

    # 1 clean + (3 categories * 3 SNRs = 9 noisy) = 10 records
    assert len(records) == 10
    assert "clean" in summaries
    assert "noise_15dB" in summaries
    assert "music_10dB" in summaries
    assert "babble_5dB" in summaries

    # Export reports
    paths = evaluator.export_reports(payload)
    assert paths["clean_csv"].is_file()
    assert paths["noisy_csv"].is_file()
    assert paths["comparison_json"].is_file()
    assert paths["summary_md"].is_file()
    assert paths["summary_txt"].is_file()

    # Verify JSON content
    with open(paths["comparison_json"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["metadata"]["total_clean_samples"] == 1
        assert data["metadata"]["total_noisy_samples"] == 9
        assert "clean" in data["condition_summaries"]


@pytest.mark.asyncio
async def test_noise_evaluator_checkpoint_resumption(tmp_path: Path):
    """Verify evaluator loads existing noise checkpoints and skips completed turns."""
    clean_loader = MUCSDatasetLoader()
    musan_raw_dir = Path("speech/datasets/musan/raw")
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "reports"

    evaluator_1 = NoiseRobustnessEvaluator(
        clean_loader=clean_loader,
        musan_raw_dir=musan_raw_dir,
        processed_audio_dir=processed_dir,
        output_dir=output_dir,
        snr_levels=[15.0],
        categories=["noise"],
        concurrency=1,
        resume=True,
        dry_run=True,
    )

    payload_1 = await evaluator_1.run_evaluation(limit=1)
    # 1 clean + 1 noise_15dB = 2 records
    assert len(payload_1["records"]) == 2

    evaluator_2 = NoiseRobustnessEvaluator(
        clean_loader=clean_loader,
        musan_raw_dir=musan_raw_dir,
        processed_audio_dir=processed_dir,
        output_dir=output_dir,
        snr_levels=[15.0],
        categories=["noise"],
        concurrency=1,
        resume=True,
        dry_run=True,
    )
    loaded_count = evaluator_2.load_checkpoint()
    assert loaded_count == 2
    assert "mucs_test_hien_0001::clean" in evaluator_2.records
    assert "mucs_test_hien_0001::noise_15dB" in evaluator_2.records

