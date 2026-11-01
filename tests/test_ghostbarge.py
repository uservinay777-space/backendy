"""Tests for GhostBarge interruption handling and cancellation (USP 2)."""

import asyncio
import time
import pytest
from phantomflow.contracts.common import LanguageCode, TurnState, UserUtterance
from phantomflow.core.orchestrator import CentralOrchestrator
from phantomflow.speech.mock_speech import MockSpeechPipeline
from phantomflow.conversation.mock_conversation import MockConversationEngine


@pytest.mark.asyncio
async def test_ghostbarge_interruption_halts_playback():
    """Verify that calling handle_interruption immediately stops audio generation and flags session."""
    # Slower chunk synthesis to give time for interruption
    speech = MockSpeechPipeline(chunk_latency_ms=20.0)
    conv = MockConversationEngine(simulated_llm_delay_ms=25.0)
    orchestrator = CentralOrchestrator(speech_pipeline=speech, conversation_engine=conv)

    session_id = "test_ghostbarge_sess"
    utterance = UserUtterance(
        text="Explain in detail all available doctors",
        is_final=True,
        language=LanguageCode.EN,
    )

    emitted_chunks = []

    async def _run_stream():
        async for chunk in orchestrator.process_utterance(session_id, utterance):
            emitted_chunks.append(chunk)
            if len(emitted_chunks) == 2:
                # User starts speaking while assistant is speaking!
                await orchestrator.handle_interruption(session_id)

    await _run_stream()

    session = orchestrator.session_manager.get_or_create(session_id)
    assert session.turn_state == TurnState.INTERRUPTED
    assert len(speech.stopped_turns) > 0
    # Chunks should have halted early (far fewer than full generation)
    assert len(emitted_chunks) <= 3


@pytest.mark.asyncio
async def test_ghostbarge_idempotent_repeated_interruptions():
    """Verify that firing multiple interruption events concurrently is idempotent and race-free."""
    speech = MockSpeechPipeline(chunk_latency_ms=10.0)
    conv = MockConversationEngine()
    orchestrator = CentralOrchestrator(speech_pipeline=speech, conversation_engine=conv)

    session_id = "test_repeat_interrupt"
    utterance = UserUtterance(
        text="Can you explain the detailed symptoms of acute bronchitis in Hindi?",
        is_final=True,
        language=LanguageCode.HINGLISH,
    )

    # Start generation in background
    chunks = []
    async def _stream():
        async for c in orchestrator.process_utterance(session_id, utterance):
            chunks.append(c)

    task = asyncio.create_task(_stream())
    await asyncio.sleep(0.015)

    # Fire 5 concurrent interruptions
    await asyncio.gather(
        orchestrator.handle_interruption(session_id),
        orchestrator.handle_interruption(session_id),
        orchestrator.handle_interruption(session_id),
        orchestrator.handle_interruption(session_id),
        orchestrator.handle_interruption(session_id),
    )

    await task

    session = orchestrator.session_manager.get_or_create(session_id)
    assert session.turn_state == TurnState.INTERRUPTED


@pytest.mark.asyncio
async def test_ghostbarge_immediate_new_turn_after_interruption():
    """Verify that a new turn can begin immediately after interruption without stale audio overlap."""
    speech = MockSpeechPipeline(chunk_latency_ms=15.0)
    conv = MockConversationEngine()
    orchestrator = CentralOrchestrator(speech_pipeline=speech, conversation_engine=conv)

    session_id = "test_turn_recovery"

    # Turn 1: Long query
    t1_chunks = []
    async def _turn_1():
        u1 = UserUtterance(text="Explain all cardiac symptoms", is_final=True, language=LanguageCode.EN)
        async for c in orchestrator.process_utterance(session_id, u1):
            t1_chunks.append(c)
            if len(t1_chunks) == 1:
                # Interrupted
                await orchestrator.handle_interruption(session_id)

    await _turn_1()
    session = orchestrator.session_manager.get_or_create(session_id)
    first_turn_id = session.active_turn_id

    # Turn 2: New user utterance immediately starts
    t2_chunks = []
    u2 = UserUtterance(text="Actually Delhi to Mumbai train", is_final=True, language=LanguageCode.HINGLISH)
    async for c in orchestrator.process_utterance(session_id, u2):
        t2_chunks.append(c)

    second_turn_id = session.active_turn_id
    assert first_turn_id != second_turn_id
    assert session.turn_state == TurnState.COMPLETED
    assert len(t2_chunks) > 0
    # Every chunk in Turn 2 must belong exclusively to second_turn_id
    assert all(c.turn_id == second_turn_id for c in t2_chunks)


@pytest.mark.asyncio
async def test_ghostbarge_cancellation_latency_under_50ms():
    """Verify that internal GhostBarge cancellation executes well within the 50ms budget."""
    speech = MockSpeechPipeline()
    conv = MockConversationEngine()
    orchestrator = CentralOrchestrator(speech_pipeline=speech, conversation_engine=conv)

    session_id = "test_cancel_latency"
    u = UserUtterance(text="Test query for timing", is_final=True, language=LanguageCode.EN)

    # Start turn
    gen = orchestrator.process_utterance(session_id, u)
    await gen.__anext__()

    t_start = time.perf_counter_ns()
    await orchestrator.handle_interruption(session_id)
    t_end = time.perf_counter_ns()

    elapsed_ms = (t_end - t_start) / 1_000_000.0
    assert elapsed_ms < 50.0  # Typically < 1ms in local memory
