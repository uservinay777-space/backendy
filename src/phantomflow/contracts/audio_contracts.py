"""Integration contracts for Speech & Audio Pipeline (Member 2).

Defines the exact interface, callback signatures, and data contracts that Member 2
implements for LiveKit / Sarvam Saaras STT and Bulbul Flash TTS.
"""

import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, Optional, Protocol, runtime_checkable
from pydantic import BaseModel, Field

from phantomflow.contracts.common import UserUtterance


class AudioInputChunk(BaseModel):
    """Incoming live audio frame from microphone or WebRTC audio track (Member 2)."""
    data: bytes = Field(..., description="Raw PCM audio bytes")
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz (default 16kHz for STT)")
    channels: int = Field(default=1, description="Number of audio channels (mono = 1)")
    timestamp_ns: int = Field(default_factory=time.perf_counter_ns, description="Monotonic capture timestamp (ns)")


class AudioChunk(BaseModel):
    """Streaming audio buffer output from TTS (Member 2)."""
    data: bytes = Field(..., description="Raw PCM or encoded audio chunk bytes")
    sample_rate: int = Field(default=24000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of channels (mono = 1)")
    turn_id: str = Field(..., description="Turn ID to verify validity against active turn")
    chunk_index: int = Field(default=0, description="Incremental chunk sequence index")
    is_final: bool = Field(default=False, description="True if this is the last audio chunk of the turn")


class GhostBargeInterruption(BaseModel):
    """GhostBarge event emitted when user speech is detected during agent response."""
    interrupted_turn_id: str = Field(..., description="The turn ID that was interrupted")
    interruption_time_ns: int = Field(..., description="Monotonic timestamp (ns) when interruption was detected")
    audio_offset_ms: Optional[float] = Field(default=None, description="Playback position when interrupted")


@runtime_checkable
class ISpeechPipeline(Protocol):
    """Contract implemented by Member 2 (Speech & Audio Engineer)."""

    @abstractmethod
    async def stream_transcribe(
        self,
        audio_stream: AsyncIterator[AudioInputChunk],
    ) -> AsyncIterator[UserUtterance]:
        """Stream live audio chunks to STT (Sarvam Saaras Realtime) and yield transcribed utterances.

        Must yield both partial (is_final=False) and final (is_final=True) utterances
        with detected language (LanguageCode).
        """
        ...

    @abstractmethod
    async def synthesize_stream(
        self,
        text_generator: AsyncIterator[str],
        turn_id: str,
    ) -> AsyncIterator[AudioChunk]:
        """Stream synthesized audio chunks from a streaming text token generator.

        Must emit chunks as fast as possible to minimize time-to-first-audio (target < 250ms).
        Must cease emitting if turn_id has been cancelled.
        """
        ...

    @abstractmethod
    async def stop_playback(self, turn_id: str) -> None:
        """GhostBarge action: Immediately cancel and flush speaker audio playback buffer."""
        ...
