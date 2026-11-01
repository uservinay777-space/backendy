"""Mock Speech & Audio Pipeline for integration testing and development (Member 2 stub).

Conforms to ISpeechPipeline protocol.
"""

import asyncio
import logging
import time
from typing import AsyncIterator, List, Optional
from phantomflow.contracts.common import LanguageCode, UserUtterance
from phantomflow.contracts.audio_contracts import AudioChunk, AudioInputChunk, ISpeechPipeline

logger = logging.getLogger("phantomflow.speech.mock")


class MockSpeechPipeline(ISpeechPipeline):
    """Mock implementation of Member 2's Speech & Audio Pipeline."""

    def __init__(self, chunk_latency_ms: float = 10.0, simulated_text: Optional[str] = None):
        self.chunk_latency_ms = chunk_latency_ms
        self.simulated_text = simulated_text or "Mujhe Saturday ke liye Delhi se Mumbai train chahiye"
        self.stopped_turns: List[str] = []
        self.synthesized_chunks_count: int = 0
        self.transcribed_utterances_count: int = 0

    async def stream_transcribe(
        self,
        audio_stream: AsyncIterator[AudioInputChunk],
    ) -> AsyncIterator[UserUtterance]:
        """Simulate real-time streaming STT from incoming live audio chunks."""
        total_bytes = 0
        partial_emitted = False

        async for chunk in audio_stream:
            total_bytes += len(chunk.data)

            # Emit early partial transcript after receiving initial audio (~100ms)
            if not partial_emitted and total_bytes >= 3200:
                partial_emitted = True
                self.transcribed_utterances_count += 1
                words = self.simulated_text.split()
                partial_text = " ".join(words[:min(4, len(words))])
                yield UserUtterance(
                    text=partial_text,
                    is_final=False,
                    language=LanguageCode.HINGLISH,
                    confidence=0.88,
                    stability=0.75,
                    speech_start_time_ns=chunk.timestamp_ns,
                )

        # Emit final transcript once incoming audio stream closes/endpoints
        self.transcribed_utterances_count += 1
        yield UserUtterance(
            text=self.simulated_text,
            is_final=True,
            language=LanguageCode.HINGLISH,
            confidence=0.96,
            stability=1.0,
            speech_end_time_ns=time.perf_counter_ns(),
        )

    async def synthesize_stream(
        self,
        text_generator: AsyncIterator[str],
        turn_id: str,
    ) -> AsyncIterator[AudioChunk]:
        """Convert incoming text stream into simulated PCM audio chunks."""
        chunk_idx = 0
        async for token in text_generator:
            if turn_id in self.stopped_turns:
                logger.info("MockSpeechPipeline: Turn %s was stopped, halting synthesis", turn_id)
                break

            # Simulate fast TTS chunk synthesis latency
            if self.chunk_latency_ms > 0:
                await asyncio.sleep(self.chunk_latency_ms / 1000.0)

            chunk_idx += 1
            self.synthesized_chunks_count += 1
            # Produce mock 16-bit PCM bytes (silence or placeholder)
            mock_pcm = b"\x00\x00" * 240  # 10ms of 24kHz 16-bit mono
            yield AudioChunk(
                data=mock_pcm,
                sample_rate=24000,
                channels=1,
                turn_id=turn_id,
                chunk_index=chunk_idx,
                is_final=False,
            )

        # Final delimiter chunk if not cancelled
        if turn_id not in self.stopped_turns:
            yield AudioChunk(
                data=b"",
                sample_rate=24000,
                channels=1,
                turn_id=turn_id,
                chunk_index=chunk_idx + 1,
                is_final=True,
            )

    async def stop_playback(self, turn_id: str) -> None:
        """GhostBarge signal: Immediately halt playback."""
        logger.info("MockSpeechPipeline: stop_playback called for turn %s", turn_id)
        if turn_id not in self.stopped_turns:
            self.stopped_turns.append(turn_id)
