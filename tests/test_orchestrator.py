"""Tests for Central Orchestrator end-to-end workflow and routing."""

import pytest
from phantomflow.contracts.common import LanguageCode, UserUtterance
from phantomflow.core.orchestrator import CentralOrchestrator
from phantomflow.speech.mock_speech import MockSpeechPipeline
from phantomflow.conversation.mock_conversation import MockConversationEngine


@pytest.mark.asyncio
async def test_orchestrator_fast_lane_workflow():
    """Verify Fast Lane routing executes and yields audio chunks."""
    speech = MockSpeechPipeline(chunk_latency_ms=1.0)
    conv = MockConversationEngine(simulated_llm_delay_ms=2.0)
    orchestrator = CentralOrchestrator(speech_pipeline=speech, conversation_engine=conv)

    utterance = UserUtterance(
        text="Mera PNR 9876543210 ka status kya hai?",
        is_final=True,
        language=LanguageCode.HINGLISH,
    )

    chunks = []
    async for chunk in orchestrator.process_utterance("test_session_fast", utterance):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert chunks[-1].is_final is True

    session = orchestrator.session_manager.get_or_create("test_session_fast")
    assert session.slots.get("pnr") == "9876543210"
    assert session.slots.get("status") == "CONFIRMED"
    assert len(session.history) == 2  # 1 user + 1 assistant


@pytest.mark.asyncio
async def test_orchestrator_smart_path_workflow():
    """Verify Smart Path routing executes streaming tokens and produces audio."""
    speech = MockSpeechPipeline(chunk_latency_ms=1.0)
    conv = MockConversationEngine(simulated_llm_delay_ms=2.0)
    orchestrator = CentralOrchestrator(speech_pipeline=speech, conversation_engine=conv)

    utterance = UserUtterance(
        text="Please explain symptoms of migraine in detail",
        is_final=True,
        language=LanguageCode.EN,
    )

    chunks = []
    async for chunk in orchestrator.process_utterance("test_session_smart", utterance):
        chunks.append(chunk)

    assert len(chunks) > 0
    session = orchestrator.session_manager.get_or_create("test_session_smart")
    assert session.current_intent == "medical_consultation"


@pytest.mark.asyncio
async def test_orchestrator_handles_partial_utterance():
    """Verify partial utterance updates language/state and does not emit premature audio."""
    from phantomflow.contracts.common import TurnState

    speech = MockSpeechPipeline(chunk_latency_ms=1.0)
    conv = MockConversationEngine(simulated_llm_delay_ms=2.0)
    orchestrator = CentralOrchestrator(speech_pipeline=speech, conversation_engine=conv)

    partial_utt = UserUtterance(
        text="Mujhe Delhi se",
        is_final=False,
        language=LanguageCode.HINGLISH,
    )

    chunks = []
    async for chunk in orchestrator.process_utterance("sess_partial_test", partial_utt):
        chunks.append(chunk)

    # Partial utterance must yield zero audio chunks
    assert len(chunks) == 0

    session = orchestrator.session_manager.get_or_create("sess_partial_test")
    assert session.turn_state == TurnState.LISTENING
    assert session.language_context == LanguageCode.HINGLISH

    # Follow up with final utterance
    final_utt = UserUtterance(
        text="Mujhe Delhi se Mumbai train chahiye",
        is_final=True,
        language=LanguageCode.HINGLISH,
    )
    async for chunk in orchestrator.process_utterance("sess_partial_test", final_utt):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert session.turn_state == TurnState.COMPLETED
