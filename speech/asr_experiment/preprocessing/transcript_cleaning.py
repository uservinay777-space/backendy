"""Bilingual transcript cleaning and normalization for Hindi-English ASR fine-tuning.

Reuses the production text normalization from speech.evaluation.text_normalization:
- Preserves Hindi (Devanagari \u0900-\u097F) and English (Latin a-zA-Z) scripts intact
- Unicode NFC normalization (standardizing nuktas and matras)
- Strips punctuation and extraneous symbols
- Normalizes numbers and collapses whitespace
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

try:
    from speech.evaluation.text_normalization import normalize_transcript
except ImportError:
    try:
        from evaluation.text_normalization import normalize_transcript
    except ImportError:
        # Self-contained fallback
        def normalize_transcript(text: str) -> str:
            if not text:
                return ""
            norm = unicodedata.normalize("NFC", text)
            norm = re.sub(r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”‘’«»—–…\u0964\u0965]""", " ", norm)
            return re.sub(r"\s+", " ", norm).strip().lower()


def clean_asr_transcript(
    text: str,
    lowercase_latin: bool = True,
    strip_punctuation: bool = True,
) -> str:
    """Clean and standardize reference and hypothesis transcripts for fine-tuning.

    Args:
        text: Raw transcript string in Hindi, English, or code-mixed Hinglish.
        lowercase_latin: Whether to lowercase Latin characters (Devanagari is unaffected).
        strip_punctuation: Whether to remove punctuation marks.

    Returns:
        Clean normalized transcript string.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Base normalization
    cleaned = normalize_transcript(
        text,
        remove_punctuation=strip_punctuation,
        lowercase=lowercase_latin,
        unicode_form="NFC",
    )

    # 2. Strip bracketed annotations like [applause], <laughter>, (inaudible)
    cleaned = re.sub(r"\[.*?\]|\<.*?\>|\(.*?\)", " ", cleaned)

    # 3. Final whitespace collapse
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

