"""Text normalization for multilingual and code-switched (Hindi-English / Hinglish) ASR evaluation.

Design Principles:
1. Script Preservation: Preserves Hindi (Devanagari \u0900-\u097F) and English (Latin a-zA-Z) scripts.
   Does not transliterate Hindi into English or English into Hindi.
2. Unicode Normalization: Uses standard Unicode NFC to ensure consistent glyph representation
   (e.g., combining characters, matras, and nuktas).
3. Punctuation Handling: Strips punctuation marks (English and Indic purna viram । / deergh viram ॥)
   to ensure fair acoustic model evaluation without penalizing punctuation differences.
4. Whitespace Normalization: Collapses multi-spaces, tabs, and newlines into single spaces.
5. Case Folding: Lowercases Latin characters; Devanagari remains unaffected.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Set

# Punctuation to remove: ASCII punctuation + Unicode quotes, dashes, and Indic virams
# Hindi Purna Viram (\u0964) and Deergh Viram (\u0965)
INDIC_PUNCTUATION = "\u0964\u0965"
COMMON_PUNCTUATION = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”‘’«»—–…"""
ALL_PUNCTUATION = COMMON_PUNCTUATION + INDIC_PUNCTUATION

# Regex to strip zero-width and invisible control characters
ZERO_WIDTH_CHARS_RE = re.compile(r"[\u200B-\u200D\uFEFF\u200E\u200F]")

# Regex to replace hyphens, slashes, and underscores with space to avoid token gluing
WORD_SEPARATOR_RE = re.compile(r"[-_/]")

# Regex to match punctuation characters
PUNCTUATION_RE = re.compile(f"[{re.escape(ALL_PUNCTUATION)}]")

# Regex to collapse multiple whitespace
WHITESPACE_RE = re.compile(r"\s+")


def normalize_transcript(
    text: str,
    remove_punctuation: bool = True,
    lowercase: bool = True,
    remove_zero_width: bool = True,
    unicode_form: str = "NFC",
) -> str:
    """Normalize a reference or predicted transcript for ASR evaluation.

    Args:
        text: Raw transcript string (Hindi, English, or Hinglish).
        remove_punctuation: Whether to remove punctuation marks.
        lowercase: Whether to lowercase Latin characters.
        remove_zero_width: Whether to strip zero-width and format control characters.
        unicode_form: Unicode normalization form ('NFC', 'NFKC', 'NFD', 'NFKD'). Default is 'NFC'.

    Returns:
        Cleaned, normalized string ready for tokenization and WER/CER calculation.
    """
    if not text:
        return ""

    # 1. Unicode normalization (NFC ensures consistent Devanagari matras and nuktas)
    normalized = unicodedata.normalize(unicode_form, text)

    # 2. Strip zero-width and directional marks
    if remove_zero_width:
        normalized = ZERO_WIDTH_CHARS_RE.sub("", normalized)

    # 3. Replace word-separating hyphens and slashes with space (e.g. "Delhi-Mumbai" -> "Delhi Mumbai")
    if remove_punctuation:
        normalized = WORD_SEPARATOR_RE.sub(" ", normalized)
        # Remove remaining punctuation
        normalized = PUNCTUATION_RE.sub("", normalized)

    # 4. Lowercase Latin characters
    if lowercase:
        normalized = normalized.lower()

    # 5. Collapse and strip whitespace
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()

    return normalized


def tokenize_words(text: str) -> list[str]:
    """Tokenize normalized text into a list of words.

    Splits on whitespace. Preserves both Latin and Devanagari words intact.
    """
    if not text:
        return []
    return [token for token in text.split() if token]


def tokenize_characters(text: str, ignore_whitespace: bool = True) -> list[str]:
    """Tokenize normalized text into characters for Character Error Rate (CER).

    In Indic scripts, combining characters (matras, virama, nukta) are handled as
    individual Unicode characters in NFC form, or user can evaluate over code points.

    Args:
        text: Normalized transcript string.
        ignore_whitespace: If True, whitespace is excluded from character evaluation.

    Returns:
        List of character tokens.
    """
    if not text:
        return []

    if ignore_whitespace:
        return [char for char in text if not char.isspace()]
    return list(text)

