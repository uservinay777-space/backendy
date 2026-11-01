"""Unit tests for Sarvam Speech Pipeline and LLM Conversation Engine Adapters.

Tests contract adherence, configuration handling, SwitchSync slot resolution,
Fast Lane simulation execution, and graceful degradation when credentials are unset.
"""

import asyncio
from unittest.mock import AsyncMock, patch
import pytest

from phantomflow.contracts.audio_contracts import AudioInputChunk
from phantomflow.contracts.common import Domain, LanguageCode, RoutingPath, UserUtterance
from phantomflow.conversation.llm_conversation import (
    DOCTOR_SCHEDULE_FIXTURE,
    LLMConversationEngine,
    RAILWAY_SCHEDULE_FIXTURE,
)
from phantomflow.speech.sarvam_speech import SarvamSpeechPipeline


# =====================================================================
# 1. Sarvam Speech Pipeline Adapter Tests
# =====================================================================

def test_sarvam_speech_pipeline_unconfigured():
    pipeline = SarvamSpeechPipeline(api_key="")
    assert not pipeline.is_configured
    assert pipeline.stt_model == "saaras:v3-realtime"
    assert pipeline.tts_model == "bulbul:v4-flash"


def test_sarvam_speech_pipeline_configured():
    pipeline = SarvamSpeechPipeline(api_key="test_sarvam_key_12345")
    assert pipeline.is_configured
    assert pipeline.api_key == "test_sarvam_key_12345"


@pytest.mark.asyncio
async def test_sarvam_stream_transcribe_unconfigured_raises():
    pipeline = SarvamSpeechPipeline(api_key="")

    async def mock_audio_stream():
        yield AudioInputChunk(data=b"\x00" * 320, sample_rate=16000, channels=1)

    with pytest.raises(RuntimeError, match="SARVAM_API_KEY is not configured"):
        async for _ in pipeline.stream_transcribe(mock_audio_stream()):
            pass


@pytest.mark.asyncio
async def test_sarvam_synthesize_stream_unconfigured_raises():
    pipeline = SarvamSpeechPipeline(api_key="")

    async def mock_text_stream():
        yield "Hello world"

    with pytest.raises(RuntimeError, match="SARVAM_API_KEY is not configured for Bulbul TTS"):
        async for _ in pipeline.synthesize_stream(mock_text_stream(), turn_id="turn-1"):
            pass


@pytest.mark.asyncio
async def test_sarvam_synthesize_stream_with_mocked_tts_api():
    pipeline = SarvamSpeechPipeline(api_key="valid_key")
    assert pipeline.is_configured

    async def mock_text_stream():
        yield "Namaste. "
        yield "Aapki train confirm hai."

    mock_resp_data = {
        "audios": ["AQIDBAU="]  # base64 encoded b"\x01\x02\x03\x04\x05"
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: mock_resp_data
        mock_post.return_value = mock_response

        chunks = []
        async for chunk in pipeline.synthesize_stream(mock_text_stream(), turn_id="turn-tts-test"):
            chunks.append(chunk)

        # Chunks should include synthesized audio chunk and final delimiter chunk
        assert len(chunks) >= 2
        assert chunks[-1].is_final
        assert chunks[0].data == b"\x01\x02\x03\x04\x05"
        assert pipeline.synthesized_count >= 1


@pytest.mark.asyncio
async def test_sarvam_ghostbarge_interruption_cancels_tts():
    pipeline = SarvamSpeechPipeline(api_key="valid_key")

    await pipeline.stop_playback("turn-cancelled")

    async def mock_text_stream():
        yield "This should not be spoken."

    chunks = []
    async for chunk in pipeline.synthesize_stream(mock_text_stream(), turn_id="turn-cancelled"):
        chunks.append(chunk)

    # When turn is pre-cancelled, no audio should be synthesized or yielded
    assert len(chunks) == 0


# =====================================================================
# 2. LLM Conversation Engine Adapter Tests
# =====================================================================

def test_llm_conversation_engine_init():
    engine = LLMConversationEngine(api_key="test_llm_key", model="gpt-4o-mini")
    assert engine.is_llm_configured
    assert engine.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_llm_fast_lane_railway_intent_and_slot_extraction():
    engine = LLMConversationEngine()

    utterance = UserUtterance(
        text="Delhi se Mumbai train chahiye Saturday ko",
        language=LanguageCode.HINGLISH,
        is_final=True,
    )
    session_ctx = {"slots": {}}

    intent_res = await engine.resolve_intent(utterance, session_ctx)
    assert intent_res.domain == Domain.RAILWAY
    assert intent_res.intent_name == "search_train"
    assert intent_res.preferred_routing == RoutingPath.FAST
    assert intent_res.extracted_slots.get("origin") == "New Delhi"
    assert intent_res.extracted_slots.get("destination") == "Mumbai"
    assert intent_res.extracted_slots.get("travel_date") == "Saturday"

    # Execute task via Fast Lane deterministic fixture
    task_res = await engine.execute_fast_lane(intent_res, session_ctx)
    assert task_res is not None
    assert task_res.is_complete
    assert "Rajdhani Express" in task_res.text


@pytest.mark.asyncio
async def test_llm_switchsync_slot_correction():
    engine = LLMConversationEngine()

    # Initial session context has travel_date = Saturday
    session_ctx = {
        "slots": {
            "origin": "New Delhi",
            "destination": "Mumbai",
            "travel_date": "Saturday",
        }
    }

    # Correction utterance
    correction_utterance = UserUtterance(
        text="Actually Sunday ko jana hai",
        language=LanguageCode.HINGLISH,
        is_final=True,
    )

    intent_res = await engine.resolve_intent(correction_utterance, session_ctx)
    assert intent_res.domain == Domain.RAILWAY
    assert intent_res.intent_name == "update_date"
    assert intent_res.is_correction
    # Sunday should replace Saturday
    assert intent_res.extracted_slots.get("travel_date") == "Sunday"

    # Task execution returns updated schedule
    task_res = await engine.execute_fast_lane(intent_res, session_ctx)
    assert task_res is not None
    assert "Sunday" in task_res.text


@pytest.mark.asyncio
async def test_llm_fast_lane_doctor_appointment():
    engine = LLMConversationEngine()

    utterance = UserUtterance(
        text="Dr Sharma cardiologist appointment book kardo",
        language=LanguageCode.HINGLISH,
        is_final=True,
    )
    session_ctx = {"slots": {}}

    intent_res = await engine.resolve_intent(utterance, session_ctx)
    assert intent_res.domain == Domain.HEALTHCARE
    assert intent_res.intent_name == "book_appointment"
    assert intent_res.preferred_routing == RoutingPath.FAST
    assert intent_res.extracted_slots.get("doctor_name") == "Dr. Sharma"
    assert intent_res.extracted_slots.get("specialty") == "Cardiologist"

    task_res = await engine.execute_fast_lane(intent_res, session_ctx)
    assert task_res is not None
    assert task_res.is_complete
    assert "Dr. Sharma" in task_res.text
    assert "appointment slot" in task_res.text


@pytest.mark.asyncio
async def test_llm_smart_path_routing_for_general_query():
    engine = LLMConversationEngine()

    utterance = UserUtterance(
        text="Mere seene mein dard hai, what should I do?",
        language=LanguageCode.HINGLISH,
        is_final=True,
    )
    session_ctx = {"slots": {}}

    intent_res = await engine.resolve_intent(utterance, session_ctx)
    assert intent_res.domain == Domain.HEALTHCARE
    assert intent_res.preferred_routing == RoutingPath.SMART


@pytest.mark.asyncio
async def test_llm_stream_response_smart_path_tokens():
    engine = LLMConversationEngine(api_key="")

    utterance = UserUtterance(
        text="Mere seene mein dard hai, explain symptoms",
        language=LanguageCode.HINGLISH,
        is_final=True,
    )
    session_ctx = {"slots": {}}
    intent_res = await engine.resolve_intent(utterance, session_ctx)

    tokens = []
    async for token in engine.execute_smart_path_stream(intent_res, session_ctx):
        tokens.append(token)

    full_text = "".join(tokens)
    assert len(full_text) > 0
    assert "Cardiologist" in full_text or "paramarsh" in full_text


@pytest.mark.asyncio
async def test_llm_switchsync_doctor_correction():
    engine = LLMConversationEngine()
    session_ctx = {
        "current_domain": Domain.HEALTHCARE,
        "slots": {"doctor_name": "Dr. Sharma", "specialty": "Cardiologist"},
    }
    utterance = UserUtterance(
        text="No, change to Dr. Verma instead",
        language=LanguageCode.HINGLISH,
        is_final=True,
    )
    intent_res = await engine.resolve_intent(utterance, session_ctx)
    assert intent_res.domain == Domain.HEALTHCARE
    assert intent_res.is_correction is True
    assert intent_res.extracted_slots.get("doctor_name") == "Dr. Verma"

    task_res = await engine.execute_fast_lane(intent_res, session_ctx)
    assert task_res is not None
    assert "Dr. Verma" in task_res.text


@pytest.mark.asyncio
async def test_llm_switchsync_multislot_correction():
    engine = LLMConversationEngine()
    session_ctx = {
        "current_domain": Domain.RAILWAY,
        "slots": {"origin": "New Delhi", "destination": "Kolkata", "travel_date": "Friday"},
    }
    utterance = UserUtterance(
        text="Actually Sunday ko Mumbai jana hai",
        language=LanguageCode.HINGLISH,
        is_final=True,
    )
    intent_res = await engine.resolve_intent(utterance, session_ctx)
    assert intent_res.domain == Domain.RAILWAY
    assert intent_res.is_correction is True
    assert intent_res.extracted_slots.get("travel_date") == "Sunday"
    assert intent_res.extracted_slots.get("destination") == "Mumbai"

    task_res = await engine.execute_fast_lane(intent_res, session_ctx)
    assert task_res is not None
    assert "Sunday" in task_res.text
    assert "Mumbai" in task_res.text

