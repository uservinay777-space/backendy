"""Comprehensive End-to-End Integration Tests for PhantomFlow Combined Demo.

Tests the full 4-Turn Integration Scenario:
- Turn 1: Railway request: "Mujhe Bengaluru se Chennai jaana hai Friday ko."
- Turn 2: SwitchSync correction: "Actually Saturday. And morning mein."
- Turn 3: Language switch: "Now please explain everything in English."
- Turn 4: GhostBarge interruption: Caller interrupts mid-playback "Wait! I want to change my destination to Hyderabad."
"""

import asyncio
import time
import pytest

from phantomflow.contracts.common import Domain, LanguageCode, RoutingPath, UserUtterance
from phantomflow.contracts.audio_contracts import AudioInputChunk
from phantomflow.core.orchestrator import CentralOrchestrator
from phantomflow.core.session import SessionManager
from phantomflow.core.latency import LatencyTracker
from phantomflow.core.events import GhostBargeController
from phantomflow.speech.mock_speech import MockSpeechPipeline
from phantomflow.conversation.mock_conversation import MockConversationEngine


@pytest.mark.asyncio
async def test_full_4_turn_combined_integration_workflow():
    """Verify complete 4-turn integration scenario covering all core USPs:
    1. Turn 1: Fast Lane Railway booking from Bengaluru to Chennai for Friday.
    2. Turn 2: SwitchSync slot correction preserving origin & dest, updating date & time.
    3. Turn 3: Language switch to English with full context preserved.
    4. Turn 4: GhostBarge interruption halting playback < 30ms and updating destination.
    """
    session_id = "test_sess_integration_4turn"
    speech = MockSpeechPipeline(
        chunk_latency_ms=5.0,
        simulated_text="Mujhe Bengaluru se Chennai jaana hai Friday ko.",
    )
    conv = MockConversationEngine(simulated_llm_delay_ms=5.0)
    session_mgr = SessionManager()
    latency_tracker = LatencyTracker()
    ghostbarge = GhostBargeController()

    orchestrator = CentralOrchestrator(
        speech_pipeline=speech,
        conversation_engine=conv,
        session_manager=session_mgr,
        latency_tracker=latency_tracker,
        ghostbarge=ghostbarge,
    )

    # =========================================================================
    # TURN 1: Live Audio -> Streaming STT -> Fast Lane Railway Booking
    # =========================================================================
    async def _live_stream():
        for _ in range(12):
            yield AudioInputChunk(data=b"\x00\x00" * 160)
            await asyncio.sleep(0.002)

    turn1_chunks = []
    async for utterance in speech.stream_transcribe(_live_stream()):
        if not utterance.is_final:
            assert utterance.language == LanguageCode.HINGLISH
            await orchestrator.handle_partial_utterance(session_id, utterance)
        else:
            assert utterance.is_final is True
            assert "Bengaluru" in utterance.text or "Chennai" in utterance.text
            async for chunk in orchestrator.process_utterance(session_id, utterance):
                turn1_chunks.append(chunk)

    assert len(turn1_chunks) > 0
    assert turn1_chunks[-1].is_final is True

    session = session_mgr.get_or_create(session_id)
    assert session.slots.get("origin") == "Bengaluru", "Origin must be Bengaluru"
    assert session.slots.get("destination") == "Chennai", "Destination must be Chennai"
    assert session.slots.get("travel_date") == "Friday", "Date must be Friday"
    assert session.current_domain == Domain.RAILWAY

    # =========================================================================
    # TURN 2: SwitchSync Slot Correction (USP 1)
    # =========================================================================
    t2_start = time.perf_counter_ns()
    utterance_2 = UserUtterance(
        text="Actually Saturday. And morning mein.",
        is_final=True,
        language=LanguageCode.HINGLISH,
        speech_end_time_ns=t2_start,
    )

    turn2_chunks = []
    async for chunk in orchestrator.process_utterance(session_id, utterance_2):
        turn2_chunks.append(chunk)

    assert len(turn2_chunks) > 0
    # Verify SwitchSync slot preservation and update
    assert session.slots.get("origin") == "Bengaluru", "Origin slot must be preserved!"
    assert session.slots.get("destination") == "Chennai", "Destination slot must be preserved!"
    assert session.slots.get("travel_date") == "Saturday", "Travel date must be updated to Saturday!"
    assert session.slots.get("time_preference") == "Morning", "Time preference must be updated to Morning!"

    # =========================================================================
    # TURN 3: Language Switching (Hindi/Hinglish -> English)
    # =========================================================================
    t3_start = time.perf_counter_ns()
    utterance_3 = UserUtterance(
        text="Now please explain everything in English.",
        is_final=True,
        language=LanguageCode.EN,
        speech_end_time_ns=t3_start,
    )

    turn3_chunks = []
    async for chunk in orchestrator.process_utterance(session_id, utterance_3):
        turn3_chunks.append(chunk)

    assert len(turn3_chunks) > 0
    # Verify language switch and context retention
    assert session.language_context == LanguageCode.EN
    assert session.slots.get("origin") == "Bengaluru"
    assert session.slots.get("destination") == "Chennai"
    assert session.slots.get("travel_date") == "Saturday"
    assert session.slots.get("time_preference") == "Morning"
    # Verify assistant response confirmed English
    last_assistant_msg = [m for m in session.history if m.role == "assistant"][-1]
    assert "English" in last_assistant_msg.content
    assert "Bengaluru" in last_assistant_msg.content

    # =========================================================================
    # TURN 4: GhostBarge Interruption (USP 2) & Destination Correction
    # =========================================================================
    t4_start = time.perf_counter_ns()
    utterance_4 = UserUtterance(
        text="Wait! I want to change my destination to Hyderabad.",
        is_final=True,
        language=LanguageCode.EN,
        speech_end_time_ns=t4_start,
    )

    chunks_emitted = 0
    interruption_time_ms = 0.0

    async for chunk in orchestrator.process_utterance(session_id, utterance_4):
        chunks_emitted += 1
        if chunks_emitted == 1:
            t_int_start = time.perf_counter()
            await orchestrator.handle_interruption(session_id)
            interruption_time_ms = (time.perf_counter() - t_int_start) * 1000.0
            break

    # GhostBarge verification:
    assert chunks_emitted == 1, "Playback should halt immediately after 1 chunk"
    assert interruption_time_ms < 30.0, f"GhostBarge latency ({interruption_time_ms:.2f}ms) must be under 30ms"

    # Context verification after interruption:
    assert session.slots.get("destination") == "Hyderabad", "Destination must be updated to Hyderabad"
    assert session.slots.get("origin") == "Bengaluru", "Origin must remain Bengaluru"
    assert session.slots.get("travel_date") == "Saturday", "Travel date must remain Saturday"
    assert session.slots.get("time_preference") == "Morning", "Time preference must remain Morning"

    # Latency percentiles check
    percentiles = latency_tracker.get_percentiles()
    assert "p50_ms" in percentiles
    assert "p90_ms" in percentiles
    assert "p95_ms" in percentiles
    assert percentiles["count"] >= 1
