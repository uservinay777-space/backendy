"""Unit tests for typed EventEnvelope and schema contracts."""

import json
import pytest
from pydantic import ValidationError

from phantomflow.contracts.common import Domain, LanguageCode, RoutingPath
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


def test_event_envelope_factory_creation():
    envelope = EventEnvelope.create(
        event_type=EventType.SESSION_CREATED,
        session_id="test_session_123",
        payload=SessionCreatedPayload(session_id="test_session_123"),
    )
    assert envelope.event_type == EventType.SESSION_CREATED
    assert envelope.session_id == "test_session_123"
    assert envelope.turn_id is None
    assert envelope.payload["session_id"] == "test_session_123"
    assert envelope.timestamp_ns > 0


def test_event_envelope_json_roundtrip():
    envelope = EventEnvelope.create(
        event_type=EventType.FINAL_TRANSCRIPT,
        session_id="sess_001",
        turn_id="turn_001",
        payload=FinalTranscriptPayload(
            text="Mujhe Delhi se Mumbai train chahiye",
            language=LanguageCode.HINGLISH,
            confidence=0.95,
        ),
    )
    json_str = envelope.model_dump_json()
    parsed = EventEnvelope.model_validate_json(json_str)

    assert parsed.event_type == EventType.FINAL_TRANSCRIPT
    assert parsed.turn_id == "turn_001"
    assert parsed.payload["text"] == "Mujhe Delhi se Mumbai train chahiye"
    assert parsed.payload["confidence"] == 0.95


def test_ghostbarge_interruption_event():
    envelope = EventEnvelope.create(
        event_type=EventType.INTERRUPTION_DETECTED,
        session_id="sess_001",
        turn_id="turn_001",
        payload=InterruptionDetectedPayload(
            interrupted_turn_id="turn_001",
            cancellation_latency_ms=45.2,
            reason="caller_barge_in",
        ),
    )
    assert envelope.event_type == EventType.INTERRUPTION_DETECTED
    assert envelope.payload["interrupted_turn_id"] == "turn_001"
    assert envelope.payload["cancellation_latency_ms"] == 45.2


def test_latency_measured_payload_contract():
    payload = LatencyMeasuredPayload(
        turn_id="turn_abc",
        routing_path="fast",
        interrupted=False,
        stt_final_ms=80.5,
        intent_ms=12.0,
        task_ms=5.0,
        tts_first_audio_ms=140.0,
        e2e_first_audio_ms=237.5,
        meets_800ms_target=True,
        meets_600ms_target=True,
    )
    assert payload.e2e_first_audio_ms == 237.5
    assert payload.meets_800ms_target is True
    assert payload.meets_600ms_target is True


def test_error_payload_handling():
    envelope = EventEnvelope.create(
        event_type=EventType.ERROR,
        session_id="sess_err",
        error="TTS synthesis timeout",
        payload=ErrorPayload(
            code="TTS_TIMEOUT",
            message="Bulbul TTS stream failed to emit audio within deadline",
            recoverable=True,
        ),
    )
    assert envelope.error == "TTS synthesis timeout"
    assert envelope.payload["code"] == "TTS_TIMEOUT"
    assert envelope.payload["recoverable"] is True


def test_event_envelope_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        EventEnvelope.model_validate({"session_id": "missing_event_type"})

    with pytest.raises(ValidationError):
        EventEnvelope.model_validate({"event_type": "session_created"})


def test_event_envelope_invalid_event_type_raises():
    with pytest.raises(ValidationError):
        EventEnvelope.model_validate({
            "event_type": "non_existent_event_type",
            "session_id": "sess_1",
            "timestamp_ns": 123456,
            "payload": {},
        })


def test_malformed_json_parsing():
    with pytest.raises(ValidationError):
        EventEnvelope.model_validate_json('{"event_type": "audio_input", "session_id": 123}')


def test_all_17_event_types_payload_schemas():
    """Verify that all 17 event types and their payloads can be constructed and serialized."""
    payloads = [
        (EventType.SESSION_CREATED, SessionCreatedPayload(session_id="s1")),
        (EventType.SESSION_TERMINATED, SessionTerminatedPayload(session_id="s1", reason="user_hangup")),
        (EventType.AUDIO_INPUT, AudioInputPayload(audio_base64="AQID")),
        (EventType.PARTIAL_TRANSCRIPT, PartialTranscriptPayload(text="partial", stability=0.6)),
        (EventType.FINAL_TRANSCRIPT, FinalTranscriptPayload(text="final sentence", confidence=0.98)),
        (EventType.LANGUAGE_IDENTIFIED, LanguageIdentifiedPayload(language=LanguageCode.HINGLISH, confidence=0.92)),
        (EventType.INTENT_RECOGNIZED, IntentRecognizedPayload(domain=Domain.RAILWAY, intent_name="search_train")),
        (EventType.SLOTS_UPDATED, SlotsUpdatedPayload(amended_slots={"date": "Sunday"}, authoritative_slots={"date": "Sunday"})),
        (EventType.ROUTING_DECIDED, RoutingDecidedPayload(path=RoutingPath.FAST, reason="low_complexity", decision_time_ms=1.2)),
        (EventType.ASSISTANT_RESPONSE_TEXT, AssistantResponseTextPayload(text="Train confirmed", is_final=True)),
        (EventType.OUTGOING_AUDIO_CHUNK, OutgoingAudioChunkPayload(audio_base64="AQIDBAU=", chunk_index=1, is_final=False)),
        (EventType.PLAYBACK_STARTED, PlaybackStartedPayload(turn_id="t1", first_chunk_latency_ms=150.2)),
        (EventType.PLAYBACK_STOPPED, PlaybackStoppedPayload(turn_id="t1", reason="complete", chunks_played=3)),
        (EventType.INTERRUPTION_DETECTED, InterruptionDetectedPayload(interrupted_turn_id="t1", cancellation_latency_ms=18.5)),
        (EventType.TASK_COMPLETED, TaskCompletedPayload(domain=Domain.RAILWAY, task_title="Train Booking", is_success=True)),
        (EventType.LATENCY_MEASURED, LatencyMeasuredPayload(turn_id="t1", routing_path="fast", interrupted=False, e2e_first_audio_ms=120.0)),
        (EventType.ERROR, ErrorPayload(code="AUDIO_ERR", message="Audio buffer dropped", recoverable=True)),
    ]

    assert len(payloads) == 17

    for event_type, payload in payloads:
        envelope = EventEnvelope.create(
            event_type=event_type,
            session_id="validation_session",
            turn_id="validation_turn",
            payload=payload,
        )
        assert envelope.event_type == event_type
        assert envelope.session_id == "validation_session"
        json_repr = envelope.model_dump_json()
        assert len(json_repr) > 0
        deserialized = EventEnvelope.model_validate_json(json_repr)
        assert deserialized.event_type == event_type
