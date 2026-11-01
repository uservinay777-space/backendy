"""Tests validating integration contracts and Protocol runtime conformance."""

import pytest
from phantomflow.contracts.common import (
    Domain,
    LanguageCode,
    RoutingPath,
    TurnState,
    UserUtterance,
)
from phantomflow.contracts.audio_contracts import AudioChunk, GhostBargeInterruption, ISpeechPipeline
from phantomflow.contracts.conversation_contracts import IConversationEngine, IntentResult, TaskResponse
from phantomflow.speech.mock_speech import MockSpeechPipeline
from phantomflow.conversation.mock_conversation import MockConversationEngine


def test_speech_pipeline_protocol_conformance():
    """Verify MockSpeechPipeline strictly adheres to ISpeechPipeline interface."""
    mock_pipeline = MockSpeechPipeline()
    assert isinstance(mock_pipeline, ISpeechPipeline)


def test_conversation_engine_protocol_conformance():
    """Verify MockConversationEngine strictly adheres to IConversationEngine interface."""
    mock_engine = MockConversationEngine()
    assert isinstance(mock_engine, IConversationEngine)


def test_user_utterance_model():
    """Verify UserUtterance schema validation and defaults."""
    utterance = UserUtterance(
        text="Delhi se Mumbai train",
        is_final=True,
        language=LanguageCode.HINGLISH,
    )
    assert utterance.text == "Delhi se Mumbai train"
    assert utterance.is_final is True
    assert utterance.language == LanguageCode.HINGLISH
    assert utterance.confidence == 1.0


def test_audio_chunk_model():
    """Verify AudioChunk model creation."""
    chunk = AudioChunk(
        data=b"\x00\x01\x02\x03",
        sample_rate=24000,
        channels=1,
        turn_id="turn_test1",
        chunk_index=1,
        is_final=False,
    )
    assert chunk.turn_id == "turn_test1"
    assert len(chunk.data) == 4
    assert chunk.sample_rate == 24000


def test_intent_result_validation():
    """Verify IntentResult model with preferred routing."""
    intent = IntentResult(
        domain=Domain.RAILWAY,
        intent_name="check_pnr",
        confidence=0.98,
        extracted_slots={"pnr": "1234567890"},
        preferred_routing=RoutingPath.FAST,
    )
    assert intent.domain == Domain.RAILWAY
    assert intent.preferred_routing == RoutingPath.FAST
    assert intent.extracted_slots["pnr"] == "1234567890"


def test_audio_input_chunk_model():
    """Verify AudioInputChunk schema for incoming live audio frames."""
    from phantomflow.contracts.audio_contracts import AudioInputChunk

    chunk = AudioInputChunk(
        data=b"\x00\x00" * 160,
        sample_rate=16000,
        channels=1,
    )
    assert len(chunk.data) == 320
    assert chunk.sample_rate == 16000
    assert chunk.channels == 1
    assert chunk.timestamp_ns > 0


@pytest.mark.asyncio
async def test_stream_transcribe_partial_and_final():
    """Verify stream_transcribe emits partial and final transcription events with language."""
    from phantomflow.contracts.audio_contracts import AudioInputChunk

    mock_speech = MockSpeechPipeline()

    async def _audio_gen():
        # Feed enough audio frames to trigger partial followed by final
        for _ in range(15):
            yield AudioInputChunk(data=b"\x00\x00" * 160)

    utterances = []
    async for utt in mock_speech.stream_transcribe(_audio_gen()):
        utterances.append(utt)

    assert len(utterances) == 2
    # First is partial
    assert utterances[0].is_final is False
    assert utterances[0].language == LanguageCode.HINGLISH
    assert utterances[0].stability is not None
    # Second is final
    assert utterances[1].is_final is True
    assert utterances[1].language == LanguageCode.HINGLISH
    assert utterances[1].speech_end_time_ns is not None
