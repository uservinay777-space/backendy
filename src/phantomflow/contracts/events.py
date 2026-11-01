"""Typed event envelope and schema contracts for PhantomFlow.

Defines the shared event envelope, event types, and payloads for bidirectional
communication between frontend, backend, speech pipeline, and conversation engine.
"""

from enum import Enum
import time
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from phantomflow.contracts.common import Domain, LanguageCode, RoutingPath, TurnState


class EventType(str, Enum):
    """Enumeration of all typed PhantomFlow system events."""
    # Session lifecycle
    SESSION_CREATED = "session_created"
    SESSION_TERMINATED = "session_terminated"

    # Audio input / STT stream
    AUDIO_INPUT = "audio_input"
    PARTIAL_TRANSCRIPT = "partial_transcript"
    FINAL_TRANSCRIPT = "final_transcript"
    LANGUAGE_IDENTIFIED = "language_identified"

    # NLU / Intent & Routing
    INTENT_RECOGNIZED = "intent_recognized"
    SLOTS_UPDATED = "slots_updated"
    ROUTING_DECIDED = "routing_decided"

    # Assistant Response & Speech Output
    ASSISTANT_RESPONSE_TEXT = "assistant_response_text"
    OUTGOING_AUDIO_CHUNK = "outgoing_audio_chunk"
    PLAYBACK_STARTED = "playback_started"
    PLAYBACK_STOPPED = "playback_stopped"

    # Interruption & Lifecycle
    INTERRUPTION_DETECTED = "interruption_detected"
    TASK_COMPLETED = "task_completed"

    # Telemetry & Diagnostics
    LATENCY_MEASURED = "latency_measured"
    ERROR = "error"


# --- Specific Event Payloads ---

class SessionCreatedPayload(BaseModel):
    session_id: str
    created_at_epoch_s: float = Field(default_factory=time.time)
    initial_domain: Optional[Domain] = None


class SessionTerminatedPayload(BaseModel):
    session_id: str
    reason: str = "normal"
    duration_s: float = 0.0
    total_turns: int = 0


class AudioInputPayload(BaseModel):
    """Audio frame from client microphone."""
    audio_base64: str = Field(..., description="Base64-encoded raw PCM or webm/wav bytes")
    sample_rate: int = 16000
    channels: int = 1
    format: str = "pcm_s16le"


class PartialTranscriptPayload(BaseModel):
    """Streaming intermediate ASR hypothesis."""
    text: str
    language: LanguageCode = LanguageCode.HINGLISH
    stability: float = Field(default=0.5, ge=0.0, le=1.0)


class FinalTranscriptPayload(BaseModel):
    """Finalized ASR utterance triggering turn orchestration."""
    text: str
    language: LanguageCode = LanguageCode.HINGLISH
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    speech_end_time_ns: Optional[int] = None


class LanguageIdentifiedPayload(BaseModel):
    """SwitchSync language identification update."""
    language: LanguageCode
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    code_switch_ratio: Dict[str, float] = Field(default_factory=lambda: {"hindi": 0.5, "english": 0.5})


class IntentRecognizedPayload(BaseModel):
    """NLU classification result."""
    domain: Domain
    intent_name: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    extracted_slots: Dict[str, Any] = Field(default_factory=dict)
    is_correction: bool = False


class SlotsUpdatedPayload(BaseModel):
    """Session entity slots delta and authoritative state."""
    amended_slots: Dict[str, Any] = Field(default_factory=dict)
    authoritative_slots: Dict[str, Any] = Field(default_factory=dict)
    is_correction: bool = False


class RoutingDecidedPayload(BaseModel):
    """Fast Lane / Smart Path decision."""
    path: RoutingPath
    reason: str
    decision_time_ms: float
    llm_bypassed: bool = False


class AssistantResponseTextPayload(BaseModel):
    """Streamed or complete assistant response text."""
    text: str
    is_final: bool = False
    domain: Domain = Domain.GENERAL


class OutgoingAudioChunkPayload(BaseModel):
    """Synthesized audio chunk intended for speaker output."""
    audio_base64: str = Field(..., description="Base64-encoded audio chunk bytes")
    chunk_index: int = 0
    is_final: bool = False
    sample_rate: int = 24000
    format: str = "pcm_s16le"


class PlaybackStartedPayload(BaseModel):
    turn_id: str
    first_chunk_latency_ms: Optional[float] = None


class PlaybackStoppedPayload(BaseModel):
    turn_id: str
    reason: str = "complete"  # 'complete', 'interrupted', 'error'
    chunks_played: int = 0


class InterruptionDetectedPayload(BaseModel):
    """GhostBarge event when caller speaks while assistant is responding."""
    interrupted_turn_id: str
    cancellation_latency_ms: float
    reason: str = "caller_barge_in"


class TaskCompletedPayload(BaseModel):
    """Domain task execution outcome."""
    domain: Domain
    task_title: str
    is_success: bool = True
    summary: str = ""
    slots: Dict[str, Any] = Field(default_factory=dict)


class LatencyMeasuredPayload(BaseModel):
    """High-resolution latency report for a turn."""
    turn_id: str
    routing_path: str
    interrupted: bool
    stt_first_partial_ms: Optional[float] = None
    stt_final_ms: Optional[float] = None
    intent_ms: Optional[float] = None
    task_ms: Optional[float] = None
    llm_first_token_ms: Optional[float] = None
    tts_first_audio_ms: Optional[float] = None
    e2e_first_audio_ms: Optional[float] = None
    meets_800ms_target: bool = False
    meets_600ms_target: bool = False


class ErrorPayload(BaseModel):
    """Surfaced recoverable or fatal error."""
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    recoverable: bool = True


# --- Common Event Envelope ---

class EventEnvelope(BaseModel):
    """Unified event envelope for all PhantomFlow real-time messages."""
    event_type: EventType = Field(..., description="Type of event")
    session_id: str = Field(..., description="Unique conversation session identifier")
    turn_id: Optional[str] = Field(default=None, description="Turn identifier if applicable")
    timestamp_ns: int = Field(default_factory=time.perf_counter_ns, description="Monotonic timestamp in nanoseconds")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event-specific payload dictionary")
    error: Optional[str] = Field(default=None, description="Error message if event represents an error")

    @classmethod
    def create(
        cls,
        event_type: EventType,
        session_id: str,
        turn_id: Optional[str] = None,
        payload: Optional[Union[BaseModel, Dict[str, Any]]] = None,
        error: Optional[str] = None,
    ) -> "EventEnvelope":
        """Convenience factory method for building envelopes."""
        if payload is None:
            p_dict = {}
        elif isinstance(payload, BaseModel):
            p_dict = payload.model_dump()
        else:
            p_dict = payload

        return cls(
            event_type=event_type,
            session_id=session_id,
            turn_id=turn_id,
            timestamp_ns=time.perf_counter_ns(),
            payload=p_dict,
            error=error,
        )
