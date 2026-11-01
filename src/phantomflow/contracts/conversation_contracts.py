"""Integration contracts for AI Conversation & Task Engine (Member 3).

Defines the exact interface, data schemas, and callback signatures that Member 3
implements for Intent Recognition, Railway/Doctor task handlers, and Fast/Smart paths.
"""

from abc import abstractmethod
from typing import Any, AsyncIterator, Dict, Optional, Protocol, runtime_checkable
from pydantic import BaseModel, Field

from phantomflow.contracts.common import Domain, LanguageCode, RoutingPath, UserUtterance


class IntentResult(BaseModel):
    """Result of intent classification and slot extraction from user utterance."""
    domain: Domain = Field(..., description="Target functional domain")
    intent_name: str = Field(..., description="Canonical intent name (e.g. check_pnr, book_appointment)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Classification confidence")
    extracted_slots: Dict[str, Any] = Field(default_factory=dict, description="Extracted entity key-value pairs")
    is_correction: bool = Field(default=False, description="True if utterance amends/corrects a previously provided slot")
    preferred_routing: RoutingPath = Field(default=RoutingPath.FAST, description="Recommended path based on query complexity")
    detected_language: LanguageCode = Field(default=LanguageCode.HINGLISH, description="Utterance language")


class TaskResponse(BaseModel):
    """Structured response from a deterministic Fast Lane task execution."""
    text: str = Field(..., description="Spoken textual response intended for TTS synthesis")
    slots_to_update: Dict[str, Any] = Field(default_factory=dict, description="New or amended session slot values")
    domain: Domain = Field(default=Domain.GENERAL, description="Domain that handled the task")
    is_complete: bool = Field(default=True, description="True if the active user request has been fulfilled")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Auxiliary debugging or business data")


@runtime_checkable
class IConversationEngine(Protocol):
    """Contract implemented by Member 3 (AI Conversation & Backend Engineer)."""

    @abstractmethod
    async def resolve_intent(
        self,
        utterance: UserUtterance,
        session_context: Dict[str, Any],
    ) -> IntentResult:
        """Analyze user utterance in the context of current session state.

        Must support SwitchSync Hinglish code-switching, identify corrections,
        and indicate whether the query is suitable for the Fast Path.
        """
        ...

    @abstractmethod
    async def execute_fast_lane(
        self,
        intent: IntentResult,
        session_context: Dict[str, Any],
    ) -> Optional[TaskResponse]:
        """Execute deterministic Python logic for standard railway/doctor queries.

        Must return None if the query cannot be handled deterministically,
        prompting the orchestrator to route to the Smart Path.
        Target execution latency: < 30ms.
        """
        ...

    @abstractmethod
    async def execute_smart_path_stream(
        self,
        intent: IntentResult,
        session_context: Dict[str, Any],
    ) -> AsyncIterator[str]:
        """Stream response tokens from LLM reasoning when deep NLU is required.

        Yields text chunks as they arrive from the streaming LLM to allow early TTS.
        """
        ...
