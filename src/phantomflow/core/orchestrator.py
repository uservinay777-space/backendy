"""Central Orchestrator for PhantomFlow.

Asynchronous coordinator connecting:
- Speech Pipeline (Member 2)
- Conversation & Intent Engine (Member 3)
- Session & Slot Context (Team Lead)
- Latency Profiling (Team Lead)
- GhostBarge Cancellation (Team Lead)
"""

import asyncio
import logging
import uuid
from typing import AsyncIterator, Optional

from phantomflow.contracts.common import (
    RoutingPath,
    TurnState,
    UserUtterance,
)
from phantomflow.contracts.audio_contracts import AudioChunk, ISpeechPipeline
from phantomflow.contracts.conversation_contracts import IConversationEngine, IntentResult
from phantomflow.core.events import GhostBargeController
from phantomflow.core.latency import LatencyTracker
from phantomflow.core.session import SessionManager

logger = logging.getLogger("phantomflow.orchestrator")


class CentralOrchestrator:
    """Asynchronous Orchestrator managing real-time voice turns and routing."""

    def __init__(
        self,
        speech_pipeline: ISpeechPipeline,
        conversation_engine: IConversationEngine,
        session_manager: Optional[SessionManager] = None,
        latency_tracker: Optional[LatencyTracker] = None,
        ghostbarge: Optional[GhostBargeController] = None,
    ):
        self.speech_pipeline = speech_pipeline
        self.conversation_engine = conversation_engine
        self.session_manager = session_manager or SessionManager()
        self.latency_tracker = latency_tracker or LatencyTracker()
        self.ghostbarge = ghostbarge or GhostBargeController()

    async def handle_interruption(self, session_id: str, turn_id: Optional[str] = None) -> None:
        """GhostBarge action: cancel ongoing processing and halt audio playback."""
        session = self.session_manager.get_or_create(session_id)
        target_turn = turn_id or session.active_turn_id
        if not target_turn:
            return

        logger.info("GhostBarge interruption triggered for session %s, turn %s", session_id, target_turn)
        self.ghostbarge.interrupt_turn(target_turn, reason="caller_barge_in")
        self.session_manager.mark_interrupted(session_id, target_turn)
        await self.speech_pipeline.stop_playback(target_turn)
        self.latency_tracker.finish_turn(target_turn, interrupted=True)

    async def handle_partial_utterance(self, session_id: str, utterance: UserUtterance) -> None:
        """Handle streaming partial transcript event (is_final=False).

        Updates language context and records latency metrics without triggering premature audio playback.
        """
        session = self.session_manager.get_or_create(session_id)
        session.language_context = utterance.language
        session.turn_state = TurnState.LISTENING
        if session.active_turn_id:
            self.latency_tracker.mark_stt_partial(session.active_turn_id)
        logger.debug("Partial transcript for %s: %s (%s)", session_id, utterance.text, utterance.language)

    async def process_utterance(
        self,
        session_id: str,
        utterance: UserUtterance,
    ) -> AsyncIterator[AudioChunk]:
        """Process transcribed user speech and stream synthesized audio responses."""
        # Handle streaming partial transcript without triggering audio generation
        if not utterance.is_final:
            await self.handle_partial_utterance(session_id, utterance)
            return

        # 1. Initialize Turn and Cancellation Token
        turn_id = f"turn_{uuid.uuid4().hex[:8]}"
        cancel_token = self.ghostbarge.create_token(turn_id)
        session = self.session_manager.get_or_create(session_id)
        session.active_turn_id = turn_id
        session.turn_state = TurnState.PROCESSING

        # Record User Message
        self.session_manager.add_message(
            session_id=session_id,
            role="user",
            content=utterance.text,
            turn_id=turn_id,
            language=utterance.language,
        )

        # High-resolution latency initialization
        self.latency_tracker.start_turn(turn_id, utterance.speech_end_time_ns)
        self.latency_tracker.mark_stt_final(turn_id)

        try:
            if cancel_token.is_cancelled:
                logger.info("Turn %s cancelled before intent resolution", turn_id)
                return

            # 2. Intent Resolution & Slot Extraction (Member 3)
            context = session.to_context_dict()
            intent: IntentResult = await self.conversation_engine.resolve_intent(utterance, context)

            if cancel_token.is_cancelled:
                logger.info("Turn %s cancelled after intent resolution", turn_id)
                return

            # Update session with intent, slots, and correction flag
            self.session_manager.update_session_turn(
                session_id=session_id,
                turn_id=turn_id,
                domain=intent.domain,
                intent=intent.intent_name,
                language=intent.detected_language,
                new_slots=intent.extracted_slots,
                is_correction=intent.is_correction,
            )

            # Record routing path in latency profiler
            self.latency_tracker.mark_intent_resolved(turn_id, intent.preferred_routing.value)

            # 3. Path Selection: Fast Path vs Smart Path
            response_text_stream: AsyncIterator[str]
            self.latency_tracker.mark_task_start(turn_id)

            if intent.preferred_routing == RoutingPath.FAST:
                # Fast Path: Deterministic Python execution (< 30ms)
                fast_response = await self.conversation_engine.execute_fast_lane(
                    intent, self.session_manager.get_or_create(session_id).to_context_dict()
                )
                self.latency_tracker.mark_task_end(turn_id)

                if fast_response is not None:
                    # Update slots returned by task execution
                    if fast_response.slots_to_update:
                        self.session_manager.get_or_create(session_id).update_slots(fast_response.slots_to_update)

                    # Create single-item or chunked text generator for TTS
                    async def _single_text_gen(text: str) -> AsyncIterator[str]:
                        yield text

                    response_text_stream = _single_text_gen(fast_response.text)
                    full_response_text = fast_response.text
                else:
                    # Fallback to Smart Path if deterministic handler could not satisfy request
                    logger.info("Fast path returned None, falling back to Smart Path for turn %s", turn_id)
                    response_text_stream = self._stream_smart_path(turn_id, intent, session_id, cancel_token)
                    full_response_text = ""
            else:
                # Smart Path: LLM reasoning stream
                response_text_stream = self._stream_smart_path(turn_id, intent, session_id, cancel_token)
                full_response_text = ""

            # 4. Stream Synthesized Audio (Member 2)
            session.turn_state = TurnState.STREAMING_AUDIO
            audio_stream = self.speech_pipeline.synthesize_stream(response_text_stream, turn_id)

            first_audio_marked = False
            async for chunk in audio_stream:
                if cancel_token.is_cancelled:
                    logger.info("GhostBarge halt: Discarding remaining audio chunk for turn %s", turn_id)
                    break

                if not first_audio_marked:
                    self.latency_tracker.mark_tts_first_audio(turn_id)
                    first_audio_marked = True

                yield chunk

            if not cancel_token.is_cancelled:
                session.turn_state = TurnState.COMPLETED
                self.session_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=full_response_text or "(streamed response)",
                    turn_id=turn_id,
                )

        except asyncio.CancelledError:
            logger.info("Turn %s asyncio task cancelled", turn_id)
            self.session_manager.mark_interrupted(session_id, turn_id)
            raise
        except Exception as e:
            logger.error("Error processing turn %s: %s", turn_id, e, exc_info=True)
            session.turn_state = TurnState.FAILED
            raise
        finally:
            self.latency_tracker.finish_turn(turn_id, interrupted=cancel_token.is_cancelled)
            self.ghostbarge.cleanup_turn(turn_id)

    async def _stream_smart_path(
        self,
        turn_id: str,
        intent: IntentResult,
        session_id: str,
        cancel_token,
    ) -> AsyncIterator[str]:
        """Wrap Smart Path LLM stream with latency first-token tracking and cancellation."""
        context = self.session_manager.get_or_create(session_id).to_context_dict()
        llm_stream = self.conversation_engine.execute_smart_path_stream(intent, context)
        first_token_marked = False

        async for token in llm_stream:
            if cancel_token.is_cancelled:
                break
            if not first_token_marked:
                self.latency_tracker.mark_llm_first_token(turn_id)
                self.latency_tracker.mark_task_end(turn_id)
                first_token_marked = True
            yield token
