"""Common types, enums, and data models for PhantomFlow."""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class LanguageCode(str, Enum):
    """Supported language codes for SwitchSync."""
    HI = "hi"
    EN = "en"
    HINGLISH = "hi-en"
    AUTO = "auto"


class Domain(str, Enum):
    """Supported functional domains."""
    RAILWAY = "railway"
    HEALTHCARE = "healthcare"
    GENERAL = "general"


class RoutingPath(str, Enum):
    """Phantom Fast Lane routing paths."""
    FAST = "fast"    # Deterministic Python logic, sample data, cached answers (target < 50ms)
    SMART = "smart"  # LLM reasoning, complex intent resolution, code-mixed NLU


class TurnState(str, Enum):
    """Lifecycle state of an orchestration turn."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    STREAMING_AUDIO = "streaming_audio"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


class UserUtterance(BaseModel):
    """Transcribed user utterance from Speech Pipeline (Member 2)."""
    text: str = Field(..., description="Transcript text (English, Hindi, or Hinglish)")
    is_final: bool = Field(..., description="True if final transcript; False if partial streaming transcript")
    language: LanguageCode = Field(default=LanguageCode.HINGLISH, description="Detected language code")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Transcription confidence")
    speech_start_time_ns: Optional[int] = Field(default=None, description="Monotonic timestamp when speech started")
    speech_end_time_ns: Optional[int] = Field(default=None, description="Monotonic timestamp when speech ended (VAD endpoint)")
    stability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Stability score for partial transcripts")

