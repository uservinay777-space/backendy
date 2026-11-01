"""Unit tests for text normalization in ASR evaluation."""

import pytest
from speech.evaluation.text_normalization import (
    normalize_transcript,
    tokenize_characters,
    tokenize_words,
)


def test_unicode_nfc_normalization():
    """Verify combining characters and nuktas normalize consistently to NFC."""
    # Devanagari 'ka' + nukta vs precomposed 'qa' (\u0958)
    decomposed = "\u0915\u093c"
    precomposed = "\u0958"
    assert normalize_transcript(decomposed) == normalize_transcript(precomposed)


def test_zero_width_character_removal():
    """Verify zero-width space and joiner characters are cleanly stripped."""
    text_with_zw = "Delhi\u200b \u200cse \u200dMumbai"
    assert normalize_transcript(text_with_zw) == "delhi se mumbai"


def test_punctuation_removal():
    """Verify ASCII and Indic punctuation marks are stripped without harming words."""
    text = "Saturday ke liye train chahiye? Haan, confirm kijiye! नमस्ते।"
    expected = "saturday ke liye train chahiye haan confirm kijiye नमस्ते"
    assert normalize_transcript(text) == expected


def test_hyphen_and_slash_separation():
    """Verify hyphens and slashes are replaced with spaces to avoid token gluing."""
    text = "Delhi-Mumbai / Bengaluru-Chennai"
    assert normalize_transcript(text) == "delhi mumbai bengaluru chennai"


def test_script_preservation():
    """Verify Hindi Devanagari and English Latin words are both preserved without transliteration."""
    text = "PNR status check karna hai confirm seat या waiting list"
    normalized = normalize_transcript(text)
    assert "pnr" in normalized
    assert "seat" in normalized
    assert "या" in normalized


def test_whitespace_collapse():
    """Verify multi-spaces, newlines, and tabs collapse to a single space."""
    text = "   Bengaluru \t\n  se   Chennai   "
    assert normalize_transcript(text) == "bengaluru se chennai"


def test_tokenization():
    """Verify word and character tokenization."""
    norm = "delhi se mumbai train"
    words = tokenize_words(norm)
    assert words == ["delhi", "se", "mumbai", "train"]

    chars = tokenize_characters(norm, ignore_whitespace=True)
    assert "".join(chars) == "delhisemumbaitrain"

