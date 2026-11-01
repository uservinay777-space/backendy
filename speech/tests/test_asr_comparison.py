"""Unit tests for 3-way ASR comparison benchmark and LiveKit adapter (Milestone 6)."""

from __future__ import annotations

import asyncio
from pathlib import Path
import numpy as np
import pytest

from speech.asr_experiment.evaluation.compare_models import ModelComparisonRunner
from speech.asr_experiment.evaluation.evaluate_local import LocalASREvaluator
from speech.asr_experiment.livekit_adapter import LocalWhisperSTT
from speech.asr_experiment.preprocessing.load_dataset import ASRSample

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "asr_experiment"
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints" / "best_local_asr_adapter.json"
MANIFEST_PATH = EXPERIMENT_DIR / "dataset" / "manifest.json"


def test_local_asr_evaluator(tmp_path: Path):
    """Verify LocalASREvaluator evaluates samples, calculates WER/CER, and exports JSON."""
    evaluator = LocalASREvaluator(
        checkpoint_path=CHECKPOINT_PATH if CHECKPOINT_PATH.is_file() else None,
        output_dir=tmp_path,
        model_name="test-model",
    )

    sample = ASRSample(
        sample_id="test_001",
        source_dataset="indicvoices_hindi",
        audio_path="",
        duration_sec=3.0,
        raw_transcript="कृपया कल सुबह नौ बजे का अलार्म लगा दीजिए",
        cleaned_transcript="कृपया कल सुबह नौ बजे का अलार्म लगा दीजिए",
        language="hi",
        speaker_id="spk_01",
        is_train_eligible=True,
    )

    rec, sm = evaluator.evaluate_sample_item(sample, dry_run=True)
    assert rec.sample_id == "test_001"
    assert rec.wer == 0.0
    assert rec.cer == 0.0

    records, corpus, rtf = evaluator.evaluate_dataset([sample], dry_run=True)
    assert len(records) == 1
    assert corpus.corpus_wer == 0.0

    json_path = evaluator.export_results(records, corpus, rtf, filename="test_results.json")
    assert json_path.is_file()


def test_three_way_model_comparison(tmp_path: Path):
    """Verify ModelComparisonRunner produces 3-way comparison, CSV, and Markdown report."""
    runner = ModelComparisonRunner(
        checkpoint_path=CHECKPOINT_PATH if CHECKPOINT_PATH.is_file() else None,
        manifest_path=MANIFEST_PATH,
        report_dir=tmp_path,
    )

    test_sample = ASRSample(
        sample_id="test_002",
        source_dataset="hinglish_conversational",
        audio_path="",
        duration_sec=3.5,
        raw_transcript="Mujhe Delhi se Mumbai ke liye Rajdhani express book karna hai",
        cleaned_transcript="Mujhe Delhi se Mumbai ke liye Rajdhani express book karna hai",
        language="hi-en",
        speaker_id="spk_02",
        is_train_eligible=True,
    )

    rows = runner.run_comparison(test_samples=[test_sample], mock_sarvam=True)
    assert len(rows) == 3

    model_names = [r.model_name for r in rows]
    assert "Sarvam Saaras STT" in model_names
    assert "Pretrained Local Baseline" in model_names
    assert "Fine-Tuned Local Model" in model_names

    # Check exported artifacts
    assert (tmp_path / "model_comparison.csv").is_file()
    assert (tmp_path / "experiment_config.yaml").is_file()
    assert (tmp_path / "experiment_report.md").is_file()

    # Verify report content
    report_content = (tmp_path / "experiment_report.md").read_text(encoding="utf-8")
    assert "Sarvam Saaras STT" in report_content
    assert "Benchmark Leakage Prevention Audit" in report_content


@pytest.mark.asyncio
async def test_livekit_local_whisper_adapter():
    """Verify LocalWhisperSTT consumes audio streams and emits UserUtterances."""
    stt = LocalWhisperSTT(checkpoint_path=CHECKPOINT_PATH if CHECKPOINT_PATH.is_file() else None)

    sr = 16000
    chunk_samples = 320  # 20ms
    t = np.linspace(0, 0.8, int(sr * 0.8), endpoint=False)
    waveform = (0.4 * np.sin(2 * np.pi * 250 * t) * 32767).astype(np.int16)
    raw_pcm = waveform.tobytes()

    async def audio_stream():
        for i in range(0, len(raw_pcm), chunk_samples * 2):
            yield raw_pcm[i : i + chunk_samples * 2]
            await asyncio.sleep(0.0001)

    utterances = []
    async for utt in stt.stream_transcribe(audio_stream()):
        utterances.append(utt)

    assert len(utterances) >= 1
    first_utt = utterances[0]
    assert first_utt.text is not None
    await stt.aclose()

