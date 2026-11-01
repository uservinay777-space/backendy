"""Unit tests for Sarvam Saaras STT wrapper."""

import pytest
from phantomflow.contracts.common import LanguageCode
from speech.config.settings import SpeechSettings
from speech.stt.saaras import SarvamSaarasSTT, normalize_to_language_code


def test_language_normalization():
    """Verify that Indic, English, and Hinglish language tags map accurately."""
    assert normalize_to_language_code("hi") == LanguageCode.HI
    assert normalize_to_language_code("hi-IN") == LanguageCode.HI
    assert normalize_to_language_code("en") == LanguageCode.EN
    assert normalize_to_language_code("en-IN") == LanguageCode.EN
    assert normalize_to_language_code("en-US") == LanguageCode.EN
    assert normalize_to_language_code("hi-en") == LanguageCode.HINGLISH
    assert normalize_to_language_code("codemix") == LanguageCode.HINGLISH
    assert normalize_to_language_code("auto") == LanguageCode.AUTO
    assert normalize_to_language_code(None) == LanguageCode.HINGLISH


def test_stt_missing_api_key_raises_error(monkeypatch):
    """Verify that missing SARVAM_API_KEY raises a descriptive error."""
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    s = SpeechSettings()
    stt = SarvamSaarasSTT(config=s, api_key=None)

    with pytest.raises(ValueError, match="SARVAM_API_KEY is not set"):
        stt._ensure_stt_instance()


def test_stt_initialization_with_key():
    """Verify STT parameters are correctly configured on the wrapper."""
    stt = SarvamSaarasSTT(
        api_key="test_sarvam_key_abc",
        language="auto",
        stream_type="fast",
        mode="codemix",
        endpointing="vad",
        sample_rate=16000,
    )
    assert stt.language == "auto"
    assert stt.stream_type == "fast"
    assert stt.mode == "codemix"
    assert stt.endpointing == "vad"
    assert stt.sample_rate == 16000

