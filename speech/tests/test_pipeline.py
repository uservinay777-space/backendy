"""Unit tests for the speech input pipeline and callbacks."""

import pytest
from phantomflow.contracts.common import LanguageCode, UserUtterance
from speech.config.settings import SpeechSettings
from speech.pipeline import SpeechInputPipeline, TranscriptDisplay


def test_transcript_display():
    """Verify TranscriptDisplay renders partial and final without errors."""
    display = TranscriptDisplay()

    partial = UserUtterance(
        text="Mujhe train chahiye",
        is_final=False,
        language=LanguageCode.HINGLISH,
    )
    display.show_partial(partial)
    assert display.last_was_partial is True

    final = UserUtterance(
        text="Mujhe Saturday ke liye train chahiye",
        is_final=True,
        language=LanguageCode.HINGLISH,
        confidence=0.97,
    )
    display.show_final(final)
    assert display.last_was_partial is False


def test_pipeline_callbacks():
    """Verify external callbacks are invoked when transcripts arrive."""
    partial_received = []
    final_received = []

    s = SpeechSettings()
    pipeline = SpeechInputPipeline(
        config=s,
        on_partial=lambda u: partial_received.append(u),
        on_final=lambda u: final_received.append(u),
    )

    u1 = UserUtterance(text="Hello", is_final=False, language=LanguageCode.EN)
    u2 = UserUtterance(text="Hello, how are you?", is_final=True, language=LanguageCode.EN)

    pipeline._handle_partial(u1)
    pipeline._handle_final(u2)

    assert len(partial_received) == 1
    assert partial_received[0].text == "Hello"
    assert len(final_received) == 1
    assert final_received[0].text == "Hello, how are you?"

