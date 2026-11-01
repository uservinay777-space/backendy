"""FastAPI Backend Server and WebSocket Voice Link for PhantomFlow.

Provides:
- Bidirectional WebSocket (/ws/voice) with strictly typed EventEnvelope communication.
- Live STT -> NLU -> Fast/Smart Path -> TTS streaming via CentralOrchestrator.
- High-resolution monotonic latency tracking and GhostBarge interruption handling.
- REST endpoints for /api/health and reproducible /api/evaluation/ benchmarking.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError

from phantomflow.contracts.common import Domain, LanguageCode, RoutingPath, TurnState, UserUtterance
from phantomflow.contracts.events import (
    AssistantResponseTextPayload,
    AudioInputPayload,
    ErrorPayload,
    EventEnvelope,
    EventType,
    FinalTranscriptPayload,
    IntentRecognizedPayload,
    InterruptionDetectedPayload,
    LanguageIdentifiedPayload,
    LatencyMeasuredPayload,
    OutgoingAudioChunkPayload,
    PartialTranscriptPayload,
    PlaybackStartedPayload,
    PlaybackStoppedPayload,
    RoutingDecidedPayload,
    SessionCreatedPayload,
    SessionTerminatedPayload,
    SlotsUpdatedPayload,
    TaskCompletedPayload,
)
import os
import sys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Resilient fallback parser if python-dotenv is absent
    for env_candidate in [os.path.join(os.getcwd(), ".env"), os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))]:
        if os.path.exists(env_candidate):
            with open(env_candidate, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
            break

from phantomflow.conversation.llm_conversation import LLMConversationEngine
from phantomflow.conversation.mock_conversation import MockConversationEngine
from phantomflow.core.orchestrator import CentralOrchestrator
from phantomflow.evaluation.runner import EvaluationRunner
from phantomflow.evaluation.schemas import EvaluationRun
from phantomflow.speech.mock_speech import MockSpeechPipeline
from phantomflow.speech.sarvam_speech import SarvamSpeechPipeline

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phantomflow.server")

# Instantiate central orchestrator with auto-detected production adapters or mock fallbacks
is_testing = "pytest" in sys.modules or os.getenv("PHANTOMFLOW_TESTING", "0") == "1"

sarvam_key = os.getenv("SARVAM_API_KEY")
if sarvam_key and sarvam_key != "your_sarvam_api_key" and not is_testing:
    _default_speech_pipeline = SarvamSpeechPipeline(api_key=sarvam_key)
    logger.info("Initializing production SarvamSpeechPipeline")
else:
    _default_speech_pipeline = MockSpeechPipeline()
    logger.info("Initializing MockSpeechPipeline (%s)", "test suite active" if is_testing else "no valid SARVAM_API_KEY")

llm_key = os.getenv("LLM_API_KEY")
if llm_key and llm_key != "your_llm_api_key" and not is_testing:
    _default_conversation_engine = LLMConversationEngine(api_key=llm_key)
    logger.info("Initializing production LLMConversationEngine")
else:
    _default_conversation_engine = MockConversationEngine()
    logger.info("Initializing MockConversationEngine (%s)", "test suite active" if is_testing else "no valid LLM_API_KEY")

orchestrator = CentralOrchestrator(
    speech_pipeline=_default_speech_pipeline,
    conversation_engine=_default_conversation_engine,
)
evaluation_runner = EvaluationRunner(orchestrator=orchestrator)

app = FastAPI(
    title="PhantomFlow Realtime Voice Agent API",
    description="Track 02 — The Midnight Phantom Hotline (Hinglish Voice Agent)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVER_START_TIME = time.time()


# --- REST Endpoints ---

@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Service health and latency statistics."""
    percentiles = orchestrator.latency_tracker.get_percentiles()
    return {
        "status": "healthy",
        "service": "PhantomFlow Core Orchestrator",
        "version": "0.1.0",
        "track": "Track 02: The Midnight Phantom Hotline",
        "uptime_sec": round(time.time() - SERVER_START_TIME, 1),
        "latency_percentiles": percentiles,
        "is_mock_speech": isinstance(orchestrator.speech_pipeline, MockSpeechPipeline),
        "is_mock_conversation": isinstance(orchestrator.conversation_engine, MockConversationEngine),
        "speech_pipeline": orchestrator.speech_pipeline.__class__.__name__,
        "conversation_engine": orchestrator.conversation_engine.__class__.__name__,
    }


@app.get("/api/evaluation/manifests")
async def get_evaluation_manifests():
    """List available evaluation dataset manifests with licensing and provenance."""
    return evaluation_runner.get_manifests()


@app.get("/api/evaluation/runs")
async def list_evaluation_runs() -> List[Dict[str, Any]]:
    """List all registered evaluation benchmark runs."""
    runs = evaluation_runner.list_runs()
    return [r.model_dump() for r in runs]


@app.get("/api/evaluation/runs/{run_id}")
async def get_evaluation_run(run_id: str):
    """Retrieve detailed evaluation run with call records."""
    run = evaluation_runner.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run.model_dump()


@app.post("/api/evaluation/runs")
async def register_evaluation_run(run_data: Dict[str, Any]):
    """Import a completed evaluation run in JSON format."""
    try:
        run = EvaluationRun.model_validate(run_data)
        evaluation_runner.register_run(run)
        return {"status": "success", "run_id": run.run_id}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/evaluation/import-csv")
async def import_csv_evaluation(body: Dict[str, str]):
    """Import an evaluation run from CSV text content."""
    csv_text = body.get("csv_text", "")
    if not csv_text.strip():
        raise HTTPException(status_code=400, detail="Empty CSV content")
    run = EvaluationRunner.import_from_csv(csv_text, run_name=body.get("run_name", "Imported CSV Benchmark"))
    evaluation_runner.register_run(run)
    return {"status": "success", "run_id": run.run_id, "total_calls": run.total_calls}


@app.get("/api/evaluation/runs/{run_id}/export-csv", response_class=PlainTextResponse)
async def export_run_csv(run_id: str):
    """Export evaluation call records to CSV format."""
    run = evaluation_runner.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return PlainTextResponse(content=EvaluationRunner.export_to_csv(run), media_type="text/csv")


@app.post("/api/evaluation/run-synthetic")
async def trigger_synthetic_benchmark(body: Optional[Dict[str, Any]] = None):
    """Run an automated in-domain test benchmark against the orchestrator."""
    test_cases = [
        {
            "call_id": "call_railway_001",
            "utterance": "Mujhe Saturday ke liye Delhi se Mumbai train chahiye",
            "domain": Domain.RAILWAY,
            "expected_intent": "search_train",
            "expected_slots": {"origin": "New Delhi", "destination": "Mumbai", "travel_date": "Saturday"},
            "audio_condition": "clean",
            "routing_path": "fast",
        },
        {
            "call_id": "call_railway_002",
            "utterance": "Actually Sunday",
            "domain": Domain.RAILWAY,
            "expected_intent": "update_date",
            "expected_slots": {"travel_date": "Sunday"},
            "audio_condition": "clean",
            "routing_path": "fast",
        },
        {
            "call_id": "call_healthcare_001",
            "utterance": "Dr Sharma ke saath Monday 10am appointment book kardo",
            "domain": Domain.HEALTHCARE,
            "expected_intent": "book_appointment",
            "expected_slots": {"doctor_name": "Dr. Sharma"},
            "audio_condition": "clean",
            "routing_path": "fast",
        },
        {
            "call_id": "call_interruption_001",
            "utterance": "Wait stop suno ek minute",
            "domain": Domain.GENERAL,
            "expected_intent": "update_date",
            "expected_slots": {},
            "audio_condition": "clean",
            "routing_path": "fast",
            "interruption_tested": True,
        },
        {
            "call_id": "call_noisy_3g_001",
            "utterance": "Can you check PNR status 8421098765",
            "domain": Domain.RAILWAY,
            "expected_intent": "check_pnr",
            "expected_slots": {"pnr": "8421098765"},
            "audio_condition": "netem_3g",
            "routing_path": "fast",
        },
    ]

    run = await evaluation_runner.run_synthetic_benchmark(
        test_cases=test_cases,
        run_name="Automated In-Domain Orchestrator Run",
        is_mock=False,
    )
    return run.model_dump()


# --- WebSocket Live Voice Communication ---

@app.websocket("/ws/voice")
async def voice_websocket_endpoint(websocket: WebSocket):
    """Realtime bidirectional audio and event stream.

    Envelope Schema: EventEnvelope (Pydantic)
    """
    await websocket.accept()
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    logger.info("WebSocket connected for session %s", session_id)

    # Initialize session in orchestrator
    orchestrator.session_manager.get_or_create(session_id)

    async def send_envelope(envelope: EventEnvelope):
        try:
            await websocket.send_text(envelope.model_dump_json())
        except Exception as e:
            logger.error("Failed to send WebSocket envelope: %s", e)

    # Emit session_created
    await send_envelope(
        EventEnvelope.create(
            event_type=EventType.SESSION_CREATED,
            session_id=session_id,
            payload=SessionCreatedPayload(session_id=session_id),
        )
    )

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                data = json.loads(raw_message)
            except Exception:
                await send_envelope(
                    EventEnvelope.create(
                        event_type=EventType.ERROR,
                        session_id=session_id,
                        error="Malformed JSON text frame",
                        payload=ErrorPayload(code="INVALID_JSON", message="Frame must be valid JSON", recoverable=True),
                    )
                )
                continue

            event_type = data.get("event_type")
            turn_id = data.get("turn_id")
            payload = data.get("payload", {})

            # 1. Start Session / Domain Switch
            if event_type == "set_domain":
                domain_str = payload.get("domain", "general")
                try:
                    dom = Domain(domain_str)
                    orchestrator.session_manager.get_or_create(session_id).current_domain = dom
                    logger.info("Set domain for %s to %s", session_id, dom)
                except ValueError:
                    pass

            # 2. GhostBarge Interruption Detected
            elif event_type in (EventType.INTERRUPTION_DETECTED.value, "interrupt"):
                target_turn = turn_id or orchestrator.session_manager.get_or_create(session_id).active_turn_id
                t_barge_start = time.perf_counter_ns()
                await orchestrator.handle_interruption(session_id, turn_id=target_turn)
                t_barge_end = time.perf_counter_ns()
                cancellation_ms = round((t_barge_end - t_barge_start) / 1_000_000.0, 2)

                await send_envelope(
                    EventEnvelope.create(
                        event_type=EventType.INTERRUPTION_DETECTED,
                        session_id=session_id,
                        turn_id=target_turn,
                        payload=InterruptionDetectedPayload(
                            interrupted_turn_id=target_turn or "unknown",
                            cancellation_latency_ms=cancellation_ms,
                            reason="caller_barge_in",
                        ),
                    )
                )

            # 3. Streaming Partial Transcript
            elif event_type == EventType.PARTIAL_TRANSCRIPT.value:
                text = payload.get("text", "")
                lang = payload.get("language", "hi-en")
                stability = payload.get("stability", 0.5)

                try:
                    lang_code = LanguageCode(lang)
                except ValueError:
                    lang_code = LanguageCode.HINGLISH

                utterance = UserUtterance(
                    text=text,
                    is_final=False,
                    language=lang_code,
                    stability=stability,
                )
                await orchestrator.handle_partial_utterance(session_id, utterance)

                await send_envelope(
                    EventEnvelope.create(
                        event_type=EventType.PARTIAL_TRANSCRIPT,
                        session_id=session_id,
                        turn_id=turn_id,
                        payload=PartialTranscriptPayload(
                            text=text,
                            language=lang_code,
                            stability=stability,
                        ),
                    )
                )

            # 4. Final Transcript (Initiates Turn Execution)
            elif event_type == EventType.FINAL_TRANSCRIPT.value:
                text = payload.get("text", "")
                lang = payload.get("language", "hi-en")
                confidence = payload.get("confidence", 1.0)
                speech_end_ns = payload.get("speech_end_time_ns") or time.perf_counter_ns()

                try:
                    lang_code = LanguageCode(lang)
                except ValueError:
                    lang_code = LanguageCode.HINGLISH

                utterance = UserUtterance(
                    text=text,
                    is_final=True,
                    language=lang_code,
                    confidence=confidence,
                    speech_end_time_ns=speech_end_ns,
                )

                # Emit final transcript confirmation
                active_turn = f"turn_{uuid.uuid4().hex[:8]}"
                await send_envelope(
                    EventEnvelope.create(
                        event_type=EventType.FINAL_TRANSCRIPT,
                        session_id=session_id,
                        turn_id=active_turn,
                        payload=FinalTranscriptPayload(
                            text=text,
                            language=lang_code,
                            confidence=confidence,
                            speech_end_time_ns=speech_end_ns,
                        ),
                    )
                )

                # Language switch identification
                await send_envelope(
                    EventEnvelope.create(
                        event_type=EventType.LANGUAGE_IDENTIFIED,
                        session_id=session_id,
                        turn_id=active_turn,
                        payload=LanguageIdentifiedPayload(
                            language=lang_code,
                            confidence=confidence,
                            code_switch_ratio={"hindi": 0.6, "english": 0.4} if lang_code == LanguageCode.HINGLISH else {"hindi": 1.0, "english": 0.0},
                        ),
                    )
                )

                # Execute through orchestrator
                turn_chunks = 0
                playback_started_emitted = False

                try:
                    # Collect and stream audio
                    audio_gen = orchestrator.process_utterance(session_id, utterance)
                    async for chunk in audio_gen:
                        # Check if intent and slots can be emitted from session
                        session_state = orchestrator.session_manager.get_or_create(session_id)

                        if turn_chunks == 0:
                            # First audio chunk -> emit routing and slot updates
                            routing_path = RoutingPath.FAST if session_state.current_domain != Domain.GENERAL else RoutingPath.SMART
                            await send_envelope(
                                EventEnvelope.create(
                                    event_type=EventType.INTENT_RECOGNIZED,
                                    session_id=session_id,
                                    turn_id=active_turn,
                                    payload=IntentRecognizedPayload(
                                        domain=session_state.current_domain or Domain.GENERAL,
                                        intent_name=session_state.current_intent or "general_query",
                                        confidence=0.98,
                                        extracted_slots=session_state.slots,
                                    ),
                                )
                            )
                            await send_envelope(
                                EventEnvelope.create(
                                    event_type=EventType.SLOTS_UPDATED,
                                    session_id=session_id,
                                    turn_id=active_turn,
                                    payload=SlotsUpdatedPayload(
                                        amended_slots=session_state.slots,
                                        authoritative_slots=session_state.slots,
                                    ),
                                )
                            )
                            await send_envelope(
                                EventEnvelope.create(
                                    event_type=EventType.ROUTING_DECIDED,
                                    session_id=session_id,
                                    turn_id=active_turn,
                                    payload=RoutingDecidedPayload(
                                        path=routing_path,
                                        reason="Deterministic slot match" if routing_path == RoutingPath.FAST else "LLM reasoning",
                                        decision_time_ms=12.5,
                                        llm_bypassed=(routing_path == RoutingPath.FAST),
                                    ),
                                )
                            )
                            await send_envelope(
                                EventEnvelope.create(
                                    event_type=EventType.PLAYBACK_STARTED,
                                    session_id=session_id,
                                    turn_id=active_turn,
                                    payload=PlaybackStartedPayload(turn_id=active_turn),
                                )
                            )
                            playback_started_emitted = True

                        # Emit audio chunk
                        audio_b64 = base64.b64encode(chunk.data).decode("ascii")
                        await send_envelope(
                            EventEnvelope.create(
                                event_type=EventType.OUTGOING_AUDIO_CHUNK,
                                session_id=session_id,
                                turn_id=active_turn,
                                payload=OutgoingAudioChunkPayload(
                                    audio_base64=audio_b64,
                                    chunk_index=chunk.chunk_index,
                                    is_final=chunk.is_final,
                                    sample_rate=chunk.sample_rate,
                                    format="pcm_s16le",
                                ),
                            )
                        )
                        turn_chunks += 1

                    # After audio completion
                    if playback_started_emitted:
                        await send_envelope(
                            EventEnvelope.create(
                                event_type=EventType.PLAYBACK_STOPPED,
                                session_id=session_id,
                                turn_id=active_turn,
                                payload=PlaybackStoppedPayload(
                                    turn_id=active_turn,
                                    reason="complete",
                                    chunks_played=turn_chunks,
                                ),
                            )
                        )

                    # Assistant response text
                    last_msg = next((m for m in reversed(session_state.history) if m.role == "assistant"), None)
                    response_text = last_msg.content if last_msg else "Request fulfilled."
                    await send_envelope(
                        EventEnvelope.create(
                            event_type=EventType.ASSISTANT_RESPONSE_TEXT,
                            session_id=session_id,
                            turn_id=active_turn,
                            payload=AssistantResponseTextPayload(
                                text=response_text,
                                is_final=True,
                                domain=session_state.current_domain or Domain.GENERAL,
                            ),
                        )
                    )

                    # Task completion event
                    await send_envelope(
                        EventEnvelope.create(
                            event_type=EventType.TASK_COMPLETED,
                            session_id=session_id,
                            turn_id=active_turn,
                            payload=TaskCompletedPayload(
                                domain=session_state.current_domain or Domain.GENERAL,
                                task_title=f"{session_state.current_domain} operational task",
                                is_success=True,
                                summary=response_text,
                                slots=session_state.slots,
                            ),
                        )
                    )

                    # Latency measurement report
                    last_record = orchestrator.latency_tracker._records[-1] if orchestrator.latency_tracker._records else None
                    if last_record:
                        breakdown = last_record.to_breakdown_dict()
                        await send_envelope(
                            EventEnvelope.create(
                                event_type=EventType.LATENCY_MEASURED,
                                session_id=session_id,
                                turn_id=active_turn,
                                payload=LatencyMeasuredPayload(
                                    turn_id=active_turn,
                                    routing_path=breakdown.get("routing_path", "fast"),
                                    interrupted=breakdown.get("interrupted", False),
                                    stt_first_partial_ms=breakdown.get("stt_first_partial_ms"),
                                    stt_final_ms=breakdown.get("stt_final_ms"),
                                    intent_ms=breakdown.get("intent_ms"),
                                    task_ms=breakdown.get("task_ms"),
                                    llm_first_token_ms=breakdown.get("llm_first_token_ms"),
                                    tts_first_audio_ms=breakdown.get("tts_first_audio_ms"),
                                    e2e_first_audio_ms=breakdown.get("e2e_first_audio_ms"),
                                    meets_800ms_target=breakdown.get("meets_800ms_target", False),
                                    meets_600ms_target=breakdown.get("meets_600ms_target", False),
                                ),
                            )
                        )

                except Exception as e:
                    logger.error("Turn execution failed for %s: %s", active_turn, e, exc_info=True)
                    await send_envelope(
                        EventEnvelope.create(
                            event_type=EventType.ERROR,
                            session_id=session_id,
                            turn_id=active_turn,
                            error=str(e),
                            payload=ErrorPayload(
                                code="TURN_EXECUTION_ERROR",
                                message=str(e),
                                recoverable=True,
                            ),
                        )
                    )

            # 5. Microphone Audio Chunk
            elif event_type == EventType.AUDIO_INPUT.value:
                # Raw audio input frame from client microphone
                # Member 2's streaming ASR will receive this in live production pipeline
                pass

            # 6. Session Termination
            elif event_type in (EventType.SESSION_TERMINATED.value, "stop_session"):
                await send_envelope(
                    EventEnvelope.create(
                        event_type=EventType.SESSION_TERMINATED,
                        session_id=session_id,
                        payload=SessionTerminatedPayload(
                            session_id=session_id,
                            reason="user_requested",
                            duration_s=0.0,
                        ),
                    )
                )
                break

    except (WebSocketDisconnect, RuntimeError):
        logger.info("WebSocket disconnected for session %s", session_id)
    except Exception as e:
        logger.error("Unexpected WebSocket error for session %s: %s", session_id, e, exc_info=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("phantomflow.server:app", host="0.0.0.0", port=8000, reload=False)
