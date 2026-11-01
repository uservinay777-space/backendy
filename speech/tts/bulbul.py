"""Sarvam Bulbul Text-to-Speech (TTS) Integration for PhantomFlow.

Uses the official livekit-plugins-sarvam TTS integration to stream audio responses
in real-time for Hindi, English, and conversational Hinglish.

Configured with:
- model="bulbul:v3"
- speaker="shubh" (or configurable voice)
- speech_sample_rate=22050 Hz
- streaming audio generation for low time-to-first-audio (TTFB)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional, Union
import aiohttp
import sounddevice as sd

from livekit import rtc
from livekit.agents import utils
from livekit.plugins import sarvam

from phantomflow.contracts.audio_contracts import AudioChunk
from speech.config.settings import SpeechSettings, settings

logger = logging.getLogger("phantomflow.speech.tts.bulbul")


@dataclass
class TTSMetrics:
    """Detailed performance and latency telemetry for a TTS synthesis turn."""

    text: str
    target_language: str
    speaker: str
    request_started_at: float = field(default_factory=time.perf_counter)
    first_chunk_received_at: Optional[float] = None
    first_chunk_played_at: Optional[float] = None
    generation_completed_at: Optional[float] = None
    playback_completed_at: Optional[float] = None
    total_chunks: int = 0
    total_bytes: int = 0
    sample_rate: int = 22050

    @property
    def time_to_first_audio_ms(self) -> Optional[float]:
        """Latency from synthesis request start to arrival of first audio chunk (ms)."""
        if self.first_chunk_received_at is not None:
            return (self.first_chunk_received_at - self.request_started_at) * 1000.0
        return None

    @property
    def time_to_first_playback_ms(self) -> Optional[float]:
        """Latency from synthesis request start to first audio playback (ms)."""
        if self.first_chunk_played_at is not None:
            return (self.first_chunk_played_at - self.request_started_at) * 1000.0
        return None

    @property
    def total_generation_time_ms(self) -> Optional[float]:
        """Total time to generate all audio chunks from Sarvam (ms)."""
        if self.generation_completed_at is not None:
            return (self.generation_completed_at - self.request_started_at) * 1000.0
        return None

    @property
    def total_audio_duration_ms(self) -> float:
        """Calculated duration of the generated audio based on PCM byte length (ms)."""
        # 16-bit mono PCM = 2 bytes per sample
        num_samples = self.total_bytes / 2.0
        if self.sample_rate > 0:
            return (num_samples / self.sample_rate) * 1000.0
        return 0.0

    def format_summary(self) -> str:
        """Format metrics into a clean human-readable string."""
        ttfb = f"{self.time_to_first_audio_ms:.1f}ms" if self.time_to_first_audio_ms else "N/A"
        first_play = f"{self.time_to_first_playback_ms:.1f}ms" if self.time_to_first_playback_ms else "N/A"
        total_gen = f"{self.total_generation_time_ms:.1f}ms" if self.total_generation_time_ms else "N/A"
        return (
            f"[TTS Metrics] TTFB: {ttfb} | First Play: {first_play} | "
            f"Gen Time: {total_gen} | Audio Dur: {self.total_audio_duration_ms:.0f}ms | "
            f"Chunks: {self.total_chunks} ({self.total_bytes} bytes)"
        )


def detect_target_language(text: str, fallback_language: str = "hi-IN") -> str:
    """Determine the optimal target language code for Sarvam Bulbul.

    - Devanagari text -> 'hi-IN'
    - Pure English text -> 'en-IN'
    - Hinglish / Mixed text -> 'hi-IN' (recommended for Indian phonetic prosody)
    """
    if not text or not text.strip():
        return fallback_language

    # Check for Devanagari Unicode range (\u0900 - \u097F)
    if re.search(r"[\u0900-\u097F]", text):
        return "hi-IN"

    # Common Hindi / Hinglish keywords in Roman script
    hinglish_markers = {
        "kya", "hai", "hain", "ko", "se", "mein", "me", "jaana", "chahiye", "kar",
        "raha", "rahi", "sakte", "sakta", "ho", "hoon", "aap", "aapko", "tum",
        "kaise", "kab", "kahan", "kitna", "namaste", "shukriya", "madad", "kal",
        "aaj", "zaroor", "theek", "accha", "bahut", "dena", "lena", "booking"
    }

    words = {w.lower().strip(".,?!\"'") for w in text.split()}
    if words.intersection(hinglish_markers):
        return "hi-IN"

    # Default to English if no Hindi/Hinglish indicators found
    return "en-IN"


class LocalAudioPlayer:
    """Non-blocking audio player using sounddevice for local speaker playback."""

    def __init__(self, sample_rate: int = 22050, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream: Optional[sd.RawOutputStream] = None
        self._is_active = False

    def _ensure_stream(self) -> sd.RawOutputStream:
        if self._stream is None or self._stream.closed:
            self._stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
            )
            self._stream.start()
            self._is_active = True
        return self._stream

    def play_chunk(self, data: bytes) -> None:
        """Write raw PCM bytes to output stream."""
        if not data:
            return
        stream = self._ensure_stream()
        stream.write(data)

    def close(self) -> None:
        """Stop and close the output stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug("Error closing local audio output: %s", e)
            self._stream = None
            self._is_active = False


class SarvamBulbulTTS:
    """Production Sarvam Bulbul TTS client wrapped for PhantomFlow."""

    def __init__(
        self,
        config: Optional[SpeechSettings] = None,
        model: Optional[str] = None,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        sample_rate: Optional[int] = None,
        pace: Optional[float] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.config = config or settings
        self.api_key = api_key or self.config.sarvam_api_key
        self.model = model or self.config.tts_model
        self.speaker = speaker or self.config.tts_speaker
        self.language = language or self.config.tts_language
        self.sample_rate = sample_rate or self.config.tts_sample_rate
        self.pace = pace or self.config.tts_pace

        self._tts_instances: dict[str, sarvam.TTS] = {}
        self._http_session: Optional[aiohttp.ClientSession] = None
        # Playback lock to prevent overlapping voice responses
        self._playback_lock = asyncio.Lock()
        self._stopped_turns: List[str] = []

    def _get_or_create_tts(self, target_lang: str, speaker_name: str) -> sarvam.TTS:
        """Get or initialize a cached sarvam.TTS instance for a specific language and speaker."""
        if not self.api_key:
            raise ValueError(
                "SARVAM_API_KEY is not set. Please set SARVAM_API_KEY in your environment or .env file."
            )

        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()

        cache_key = f"{self.model}_{target_lang}_{speaker_name}_{self.sample_rate}"
        if cache_key not in self._tts_instances:
            logger.info(
                "Creating Sarvam Bulbul TTS instance: model=%s, speaker=%s, language=%s, rate=%d",
                self.model,
                speaker_name,
                target_lang,
                self.sample_rate,
            )
            self._tts_instances[cache_key] = sarvam.TTS(
                target_language_code=target_lang,
                model=self.model,
                speaker=speaker_name,
                speech_sample_rate=self.sample_rate,
                num_channels=1,
                pace=self.pace,
                api_key=self.api_key,
                http_session=self._http_session,
            )

        return self._tts_instances[cache_key]

    async def aclose(self) -> None:
        """Clean up all TTS instances and sessions."""
        for tts_inst in self._tts_instances.values():
            try:
                await tts_inst.aclose()
            except Exception as e:
                logger.debug("Error closing TTS instance: %s", e)
        self._tts_instances.clear()

        if self._http_session is not None and not self._http_session.closed:
            try:
                await self._http_session.close()
            except Exception as e:
                logger.debug("Error closing HTTP session: %s", e)
            self._http_session = None

    async def stream_synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        metrics: Optional[TTSMetrics] = None,
    ) -> AsyncIterator[rtc.AudioFrame]:
        """Stream generated audio chunks as LiveKit rtc.AudioFrame objects.

        - Begins yielding audio chunks as fast as received (streaming).
        - Handles API connection errors and bad requests gracefully.
        - Records exact latency timestamps in the provided metrics object.
        """
        if not text or not text.strip():
            logger.warning("Empty or whitespace text passed to stream_synthesize; skipping.")
            return

        target_lang = language or detect_target_language(text, fallback_language=self.language)
        active_speaker = speaker or self.speaker

        if metrics is None:
            metrics = TTSMetrics(
                text=text,
                target_language=target_lang,
                speaker=active_speaker,
                sample_rate=self.sample_rate,
            )

        tts_inst = self._get_or_create_tts(target_lang=target_lang, speaker_name=active_speaker)

        chunk_count = 0
        total_bytes = 0

        try:
            # ChunkedStream yields SynthesizedAudio events with AudioFrame frames
            chunked_stream = tts_inst.synthesize(text.strip())

            async for event in chunked_stream:
                now = time.perf_counter()
                if metrics.first_chunk_received_at is None:
                    metrics.first_chunk_received_at = now

                frame: rtc.AudioFrame = event.frame
                chunk_count += 1
                total_bytes += len(frame.data)

                metrics.total_chunks = chunk_count
                metrics.total_bytes = total_bytes

                yield frame

            metrics.generation_completed_at = time.perf_counter()

        except asyncio.CancelledError:
            logger.info("TTS stream_synthesize cancelled")
            raise
        except Exception as exc:
            logger.error("Error during Sarvam Bulbul TTS synthesis: %s", exc)
            raise

    async def synthesize_stream(
        self,
        text_generator: AsyncIterator[str],
        turn_id: str,
    ) -> AsyncIterator[AudioChunk]:
        """Conforms to ISpeechPipeline.synthesize_stream for PhantomFlow Orchestrator."""
        chunk_idx = 0
        accumulated_text: List[str] = []

        async for token in text_generator:
            if turn_id in self._stopped_turns:
                logger.info("Turn %s was interrupted/stopped, halting Bulbul TTS", turn_id)
                break
            accumulated_text.append(token)

        full_text = "".join(accumulated_text).strip()
        if not full_text or turn_id in self._stopped_turns:
            return

        async for frame in self.stream_synthesize(full_text):
            if turn_id in self._stopped_turns:
                break
            chunk_idx += 1
            yield AudioChunk(
                data=bytes(frame.data),
                sample_rate=frame.sample_rate,
                channels=frame.num_channels,
                turn_id=turn_id,
                chunk_index=chunk_idx,
                is_final=False,
            )

        if turn_id not in self._stopped_turns:
            yield AudioChunk(
                data=b"",
                sample_rate=self.sample_rate,
                channels=1,
                turn_id=turn_id,
                chunk_index=chunk_idx + 1,
                is_final=True,
            )

    async def stop_playback(self, turn_id: str) -> None:
        """GhostBarge Action: Immediately cancel and invalidate active playback."""
        if turn_id not in self._stopped_turns:
            self._stopped_turns.append(turn_id)
        logger.info("Bulbul TTS stopped playback for turn: %s", turn_id)

    async def play_response(
        self,
        text: str,
        livekit_source: Optional[rtc.AudioSource] = None,
        local_player: Optional[LocalAudioPlayer] = None,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
    ) -> TTSMetrics:
        """Synthesize text and play audio chunks as they arrive, avoiding playback overlap."""
        async with self._playback_lock:
            target_lang = language or detect_target_language(text, fallback_language=self.language)
            active_speaker = speaker or self.speaker

            metrics = TTSMetrics(
                text=text,
                target_language=target_lang,
                speaker=active_speaker,
                sample_rate=self.sample_rate,
            )

            async for frame in self.stream_synthesize(
                text=text,
                language=target_lang,
                speaker=active_speaker,
                metrics=metrics,
            ):
                if metrics.first_chunk_played_at is None:
                    metrics.first_chunk_played_at = time.perf_counter()

                # Stream to LiveKit room audio track
                if livekit_source is not None:
                    await livekit_source.capture_frame(frame)

                # Play to local speaker (non-blocking)
                if local_player is not None:
                    local_player.play_chunk(bytes(frame.data))

            metrics.playback_completed_at = time.perf_counter()
            return metrics


async def generate_speech(
    text: str,
    language: str = "hi-IN",
    speaker: str = "shubh",
    sample_rate: int = 22050,
    config: Optional[SpeechSettings] = None,
) -> AsyncIterator[rtc.AudioFrame]:
    """Reusable standalone function providing a modular interface for speech generation."""
    tts_client = SarvamBulbulTTS(
        config=config,
        language=language,
        speaker=speaker,
        sample_rate=sample_rate,
    )
    try:
        async for frame in tts_client.stream_synthesize(text=text, language=language, speaker=speaker):
            yield frame
    finally:
        await tts_client.aclose()

