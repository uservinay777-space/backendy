"""PhantomFlow ASR Evaluation Module.

Provides:
- MUCSEvaluator & evaluate_mucs: Evaluator for MUCS 2021 (OpenSLR 104) benchmark.
- NoiseRobustnessEvaluator & evaluate_noise: Evaluator for MUSAN (OpenSLR 17) noise robustness benchmark.
- LahajaEvaluator & evaluate_lahaja: Evaluator for AI4Bharat Lahaja Hindi accent robustness benchmark.
- AccentRobustnessAnalyzer, GroupAnalysisMetric: Regional and scenario accent analysis.
- NoiseMixer, NoiseCategory: SNR-controlled acoustic noise mixing.
- MUCSDatasetLoader: Loader for Hindi-English code-switched dataset.
- LahajaDatasetLoader: Loader for multi-dialectal Hindi dataset.
- BaseASRDatasetLoader: Extensible base interface for evaluation datasets.
- calculate_wer, calculate_cer, evaluate_sample, aggregate_corpus_metrics: Evaluation metrics.
- normalize_transcript: Text normalization preserving Hindi and English scripts.
"""

try:
    from speech.evaluation.accent_analysis import (
        AccentRobustnessAnalyzer,
        GroupAnalysisMetric,
    )
    from speech.evaluation.base_loader import AudioSample, BaseASRDatasetLoader
    from speech.evaluation.lahaja_loader import LahajaDatasetLoader
    from speech.evaluation.metrics import (
        CorpusMetrics,
        EditDistanceResult,
        SampleMetrics,
        aggregate_corpus_metrics,
        calculate_cer,
        calculate_wer,
        evaluate_sample,
    )
    from speech.evaluation.mucs_loader import MUCSDatasetLoader
    from speech.evaluation.noise_mixer import (
        MixedAudioResult,
        NoiseCategory,
        NoiseMixer,
    )
    from speech.evaluation.text_normalization import (
        normalize_transcript,
        tokenize_characters,
        tokenize_words,
    )
except ImportError:
    from .accent_analysis import (
        AccentRobustnessAnalyzer,
        GroupAnalysisMetric,
    )
    from .base_loader import AudioSample, BaseASRDatasetLoader
    from .lahaja_loader import LahajaDatasetLoader
    from .metrics import (
        CorpusMetrics,
        EditDistanceResult,
        SampleMetrics,
        aggregate_corpus_metrics,
        calculate_cer,
        calculate_wer,
        evaluate_sample,
    )
    from .mucs_loader import MUCSDatasetLoader
    from .noise_mixer import (
        MixedAudioResult,
        NoiseCategory,
        NoiseMixer,
    )
    from .text_normalization import (
        normalize_transcript,
        tokenize_characters,
        tokenize_words,
    )


def get_mucs_evaluator_class():
    try:
        from speech.evaluation.evaluate_mucs import MUCSEvaluator
    except ImportError:
        from .evaluate_mucs import MUCSEvaluator
    return MUCSEvaluator


def get_noise_evaluator_class():
    try:
        from speech.evaluation.evaluate_noise import NoiseRobustnessEvaluator
    except ImportError:
        from .evaluate_noise import NoiseRobustnessEvaluator
    return NoiseRobustnessEvaluator


def get_lahaja_evaluator_class():
    try:
        from speech.evaluation.evaluate_lahaja import LahajaEvaluator
    except ImportError:
        from .evaluate_lahaja import LahajaEvaluator
    return LahajaEvaluator


__all__ = [
    "get_mucs_evaluator_class",
    "get_noise_evaluator_class",
    "get_lahaja_evaluator_class",
    "MUCSDatasetLoader",
    "LahajaDatasetLoader",
    "BaseASRDatasetLoader",
    "AudioSample",
    "AccentRobustnessAnalyzer",
    "GroupAnalysisMetric",
    "NoiseMixer",
    "NoiseCategory",
    "MixedAudioResult",
    "SampleMetrics",
    "CorpusMetrics",
    "EditDistanceResult",
    "calculate_wer",
    "calculate_cer",
    "evaluate_sample",
    "aggregate_corpus_metrics",
    "normalize_transcript",
    "tokenize_words",
    "tokenize_characters",
]
