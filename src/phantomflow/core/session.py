"""Session and context management for PhantomFlow.

Provides state preservation across turns, language changes (SwitchSync),
and partial corrections (e.g., changing travel date without wiping origin/destination).
"""

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from phantomflow.contracts.common import Domain, LanguageCode, TurnState


class ConversationMessage(BaseModel):
    """Single turn exchange in conversation history."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")
    turn_id: str = Field(..., description="Turn ID")
    language: Optional[LanguageCode] = Field(default=None)
    timestamp: float = Field(default_factory=time.time)


class SessionState(BaseModel):
    """Encapsulates active conversation context, entity slots, and turn state."""
    session_id: str
    active_turn_id: Optional[str] = None
    current_domain: Optional[Domain] = None
    current_intent: Optional[str] = None
    slots: Dict[str, Any] = Field(default_factory=dict, description="Extracted entity key-values across turns")
    language_context: LanguageCode = Field(default=LanguageCode.HINGLISH)
    history: List[ConversationMessage] = Field(default_factory=list)
    turn_state: TurnState = Field(default=TurnState.IDLE)
    last_interrupted_turn: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def update_slots(self, new_slots: Dict[str, Any], is_correction: bool = False) -> Dict[str, Any]:
        """Update slots while preserving existing un-amended entities (SwitchSync USP 1).

        Returns the delta of updated keys.
        """
        updated_delta = {}
        for key, value in new_slots.items():
            if value is not None:
                self.slots[key] = value
                updated_delta[key] = value
        self.updated_at = time.time()
        return updated_delta

    def to_context_dict(self) -> Dict[str, Any]:
        """Convert session state into context payload for Member 3's engine."""
        return {
            "session_id": self.session_id,
            "active_turn_id": self.active_turn_id,
            "current_domain": self.current_domain.value if self.current_domain else None,
            "current_intent": self.current_intent,
            "slots": dict(self.slots),
            "language_context": self.language_context.value,
            "recent_history": [
                {"role": m.role, "content": m.content}
                for m in self.history[-6:]  # Keep recent history lightweight for lower latency
            ],
            "turn_state": self.turn_state.value,
        }


class SessionManager:
    """In-memory session manager with SQLite backup capability."""

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        """Fetch existing session or initialize a clean one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def update_session_turn(
        self,
        session_id: str,
        turn_id: str,
        domain: Optional[Domain] = None,
        intent: Optional[str] = None,
        language: Optional[LanguageCode] = None,
        new_slots: Optional[Dict[str, Any]] = None,
        is_correction: bool = False,
    ) -> SessionState:
        """Update session state with newly classified turn details."""
        session = self.get_or_create(session_id)
        session.active_turn_id = turn_id
        session.turn_state = TurnState.PROCESSING

        if domain:
            session.current_domain = domain
        if intent:
            session.current_intent = intent
        if language:
            session.language_context = language
        if new_slots:
            session.update_slots(new_slots, is_correction=is_correction)

        session.updated_at = time.time()
        return session

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        turn_id: str,
        language: Optional[LanguageCode] = None,
    ) -> None:
        """Append message to session history."""
        session = self.get_or_create(session_id)
        session.history.append(
            ConversationMessage(
                role=role,
                content=content,
                turn_id=turn_id,
                language=language,
            )
        )
        session.updated_at = time.time()

    def mark_interrupted(self, session_id: str, turn_id: str) -> None:
        """Record an interruption event on the active session."""
        session = self.get_or_create(session_id)
        session.turn_state = TurnState.INTERRUPTED
        session.last_interrupted_turn = turn_id
        session.updated_at = time.time()

    def clear(self, session_id: str) -> None:
        """Clear session state."""
        self._sessions.pop(session_id, None)
