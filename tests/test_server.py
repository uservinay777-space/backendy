"""Integration tests for FastAPI server and WebSocket voice link."""

import json
import pytest
from starlette.testclient import TestClient

from phantomflow.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "PhantomFlow Core Orchestrator"
    assert "latency_percentiles" in data
    assert "speech_pipeline" in data
    assert "conversation_engine" in data


def test_evaluation_manifests(client):
    response = client.get("/api/evaluation/manifests")
    assert response.status_code == 200
    manifests = response.json()
    assert len(manifests) >= 5
    dataset_ids = [m["dataset_id"] for m in manifests]
    assert "mucs-2021" in dataset_ids
    assert "hinglish-top" in dataset_ids
    assert "musan-noise" in dataset_ids


def test_run_synthetic_benchmark_and_export(client):
    response = client.post("/api/evaluation/run-synthetic")
    assert response.status_code == 200
    run = response.json()
    assert "run_id" in run
    assert run["total_calls"] > 0
    assert run["overall_tcr"] > 0.0

    run_id = run["run_id"]

    # Retrieve run
    get_resp = client.get(f"/api/evaluation/runs/{run_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["run_id"] == run_id

    # Export CSV
    csv_resp = client.get(f"/api/evaluation/runs/{run_id}/export-csv")
    assert csv_resp.status_code == 200
    assert "call_id,dataset_name" in csv_resp.text

    # Import CSV
    import_resp = client.post(
        "/api/evaluation/import-csv",
        json={"csv_text": csv_resp.text, "run_name": "Test Benchmark"},
    )
    assert import_resp.status_code == 200
    imported = import_resp.json()
    assert imported["total_calls"] > 0


def test_websocket_voice_turn_flow(client):
    with client.websocket_connect("/ws/voice") as websocket:
        # 1. Expect session_created envelope
        msg1 = websocket.receive_json()
        assert msg1["event_type"] == "session_created"
        session_id = msg1["session_id"]
        assert session_id.startswith("session_")

        # 2. Send final_transcript
        websocket.send_json({
            "event_type": "final_transcript",
            "session_id": session_id,
            "payload": {
                "text": "Mujhe Saturday ke liye Delhi se Mumbai train chahiye",
                "language": "hi-en",
                "confidence": 0.98,
            },
        })

        # Collect event stream until task_completed
        event_types = []
        for _ in range(15):
            msg = websocket.receive_json()
            event_types.append(msg["event_type"])
            if msg["event_type"] == "latency_measured":
                break

        assert "final_transcript" in event_types
        assert "language_identified" in event_types
        assert "intent_recognized" in event_types
        assert "slots_updated" in event_types
        assert "routing_decided" in event_types
        assert "outgoing_audio_chunk" in event_types
        assert "playback_started" in event_types
        assert "playback_stopped" in event_types
        assert "assistant_response_text" in event_types
        assert "task_completed" in event_types
        assert "latency_measured" in event_types


def test_websocket_interruption_barge_in(client):
    with client.websocket_connect("/ws/voice") as websocket:
        msg1 = websocket.receive_json()
        session_id = msg1["session_id"]

        # Send interrupt event
        websocket.send_json({
            "event_type": "interruption_detected",
            "session_id": session_id,
            "payload": {
                "reason": "caller_barge_in",
            },
        })

        resp = websocket.receive_json()
        assert resp["event_type"] == "interruption_detected"
        assert resp["payload"]["reason"] == "caller_barge_in"


def test_websocket_malformed_json_error_handling(client):
    with client.websocket_connect("/ws/voice") as websocket:
        # Receive session_created
        msg1 = websocket.receive_json()
        assert msg1["event_type"] == "session_created"

        # Send invalid non-JSON frame
        websocket.send_text("THIS IS NOT JSON")

        err_msg = websocket.receive_json()
        assert err_msg["event_type"] == "error"
        assert err_msg["payload"]["code"] == "INVALID_JSON"
        assert err_msg["payload"]["recoverable"] is True


def test_websocket_concurrent_sessions_isolation(client):
    with client.websocket_connect("/ws/voice") as ws1, client.websocket_connect("/ws/voice") as ws2:
        m1 = ws1.receive_json()
        m2 = ws2.receive_json()
        assert m1["session_id"] != m2["session_id"]

        # Turn on session 1: Railway query
        ws1.send_json({
            "event_type": "final_transcript",
            "session_id": m1["session_id"],
            "payload": {"text": "Delhi to Mumbai train", "language": "hi-en"},
        })

        # Turn on session 2: Doctor query
        ws2.send_json({
            "event_type": "final_transcript",
            "session_id": m2["session_id"],
            "payload": {"text": "Dr Sharma cardiologist appointment", "language": "hi-en"},
        })

        # Check that session 1 events only have session 1 ID
        resp1 = ws1.receive_json()
        assert resp1["session_id"] == m1["session_id"]

        # Check that session 2 events only have session 2 ID
        resp2 = ws2.receive_json()
        assert resp2["session_id"] == m2["session_id"]


def test_websocket_session_termination(client):
    with client.websocket_connect("/ws/voice") as websocket:
        msg1 = websocket.receive_json()
        session_id = msg1["session_id"]

        websocket.send_json({
            "event_type": "session_terminated",
            "session_id": session_id,
            "payload": {"reason": "user_hangup"},
        })

        term_resp = websocket.receive_json()
        assert term_resp["event_type"] == "session_terminated"
        assert term_resp["payload"]["reason"] == "user_requested"
