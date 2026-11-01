"""Tests for Session State and SwitchSync slot preservation (USP 1)."""

import pytest
from phantomflow.contracts.common import Domain, LanguageCode, TurnState
from phantomflow.core.session import SessionManager


def test_session_creation_and_defaults():
    mgr = SessionManager()
    session = mgr.get_or_create("s_1")
    assert session.session_id == "s_1"
    assert session.slots == {}
    assert session.turn_state == TurnState.IDLE


def test_switchsync_slot_retention_on_correction():
    """USP 1 Verification: Updating a single slot must preserve all existing entity slots."""
    mgr = SessionManager()
    session_id = "s_test_switchsync"

    # Turn 1: User provides origin, destination, and travel date
    mgr.update_session_turn(
        session_id=session_id,
        turn_id="turn_1",
        domain=Domain.RAILWAY,
        intent="search_train",
        language=LanguageCode.HINGLISH,
        new_slots={"origin": "New Delhi", "destination": "Mumbai", "travel_date": "Saturday"},
        is_correction=False,
    )

    session = mgr.get_or_create(session_id)
    assert session.slots == {
        "origin": "New Delhi",
        "destination": "Mumbai",
        "travel_date": "Saturday",
    }

    # Turn 2: User interrupts or clarifies: "Actually Sunday"
    # Only "travel_date" should be updated; "origin" and "destination" MUST remain intact!
    mgr.update_session_turn(
        session_id=session_id,
        turn_id="turn_2",
        domain=Domain.RAILWAY,
        intent="update_date",
        language=LanguageCode.HINGLISH,
        new_slots={"travel_date": "Sunday"},
        is_correction=True,
    )

    updated_session = mgr.get_or_create(session_id)
    assert updated_session.slots["travel_date"] == "Sunday"
    assert updated_session.slots["origin"] == "New Delhi"
    assert updated_session.slots["destination"] == "Mumbai"


def test_conversation_history_retention():
    mgr = SessionManager()
    session_id = "s_history"

    mgr.add_message(session_id, role="user", content="Hello", turn_id="t1")
    mgr.add_message(session_id, role="assistant", content="Namaste", turn_id="t1")

    session = mgr.get_or_create(session_id)
    assert len(session.history) == 2
    assert session.history[0].role == "user"
    assert session.history[1].role == "assistant"

    context_dict = session.to_context_dict()
    assert len(context_dict["recent_history"]) == 2
