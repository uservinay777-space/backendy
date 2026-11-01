"""TTS Manager with Automatic Fallback (Primary: ElevenLabs, Backup: Sarvam Bulbul).

Orchestrates speech synthesis by prioritizing ElevenLabs (eleven_flash_v2_5) for
natural, high-fidelity conversational speech and Hinglish, while automatically
failing over to Sarvam Bulbul (bulbul:v3) upon eligible API or connection errors.

Uses the official LiveKit agents FallbackAdapter to manage provider availability,
automatic failover, and telemetry.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional, Union
import aiohttp

from livekit import rtc
from livekit.agents.tts import AvailabilityChangedEvent, FallbackAdapter, TTS

from phantomflow.contracts.audio_contracts import AudioChunk
from speech.audio.playback import LiveKitAudioOutputTrack, LocalAudioPlayer
from speech.config.settings import SpeechSettings, settings
from speech.tts.elevenlabs_tts import create_elevenlabs_tts
from speech.tts.sarvam_tts import create_sarvam_tts

logger = logging.getLogger("phantomflow.speech.tts.manager")


@dataclass
class TTSMetrics:
    """Detailed performance and latency telemetry for a TTS synthesis turn."""

    text: str
    target_language: str
    provider: str
    speaker_or_voice: str
    request_started_at: float = field(default_factory=time.perf_counter)
    first_chunk_received_at: Optional[float] = None
    first_chunk_played_at: Optional[float] = None
    generation_completed_at: Optional[float] = None
    playback_completed_at: Optional[float] = None
    fallback_occurred: bool = False
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
        """Total time to generate all audio chunks from provider (ms)."""
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
        fallback_tag = " [FALLBACK TRIGGERED: Sarvam Bulbul]" if self.fallback_occurred else ""
        return (
            f"[TTS Provider: {self.provider}{fallback_tag}] TTFB: {ttfb} | First Play: {first_play} | "
            f"Gen Time: {total_gen} | Audio Dur: {self.total_audio_duration_ms:.0f}ms | "
            f"Chunks: {self.total_chunks} ({self.total_bytes} bytes)"
        )


def detect_target_language(text: str, fallback_language: str = "hi-IN") -> str:
    """Determine the optimal target language code.

    - Devanagari text -> 'hi-IN'
    - Pure English text -> 'en-IN'
    - Hinglish / Mixed text -> 'hi-IN'
    """
    if not text or not text.strip():
        return fallback_language

    if re.search(r"[\u0900-\u097F]", text):
        return "hi-IN"

    hinglish_markers = {
        "kya", "hai", "hain", "ko", "se", "mein", "me", "jaana", "chahiye", "kar",
        "raha", "rahi", "sakte", "sakta", "ho", "hoon", "aap", "aapko", "tum",
        "kaise", "kab", "kahan", "kitna", "namaste", "shukriya", "madad", "kal",
        "aaj", "zaroor", "theek", "accha", "bahut", "dena", "lena", "booking",
        "bilkul", "batata", "details"
    }

    words = {w.lower().strip(".,?!\"'") for w in text.split()}
    if words.intersection(hinglish_markers):
        return "hi-IN"

    return "en-IN"


class TTSManager:
    """Manages Primary (ElevenLabs) and Backup (Sarvam Bulbul) TTS with LiveKit FallbackAdapter."""

    def __init__(
        self,
        config: Optional[SpeechSettings] = None,
        enable_fallback: Optional[bool] = None,
    ) -> None:
        self.config = config or settings
        self.enable_fallback = (
            enable_fallback if enable_fallback is not None else self.config.tts_fallback_enabled
        )

        self._http_session: Optional[aiohttp.ClientSession] = None
        self._playback_lock = asyncio.Lock()
        self._stopped_turns: List[str] = []

        # Active providers
        self.primary_tts: Optional[TTS] = None
        self.backup_tts: Optional[TTS] = None
        self.active_adapter: Optional[TTS] = None

        # Telemetry & State Tracking
        self.last_provider_used: str = "Unknown"
        self.primary_available: bool = True
        self.backup_available: bool = True
        self.total_requests: int = 0
        self.total_fallbacks: int = 0

        self._initialized = False

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    def initialize_providers(
        self,
        target_language: str = "hi-IN",
    ) -> TTS:
        """Initialize ElevenLabs (Primary) and Sarvam Bulbul (Backup) and wire into FallbackAdapter."""
        session = self._ensure_session()
        providers: List[TTS] = []

        # 1. Primary: ElevenLabs (Flash v2.5)
        if self.config.is_elevenlabs_configured:
            try:
                el_lang = "hi" if target_language.startswith("hi") else "en"
                self.primary_tts = create_elevenlabs_tts(
                    config=self.config,
                    language=el_lang,
                    http_session=session,
                )
                providers.append(self.primary_tts)
            except Exception as e:
                logger.error("Failed to initialize ElevenLabs Primary TTS: %s", e)

        # 2. Backup: Sarvam Bulbul (v3)
        if self.config.is_sarvam_configured:
            try:
                self.backup_tts = create_sarvam_tts(
                    config=self.config,
                    target_language=target_language,
                    sample_rate=22050,
                    http_session=session,
                )
                if self.enable_fallback or not providers:
                    providers.append(self.backup_tts)
            except Exception as e:
                logger.error("Failed to initialize Sarvam Backup TTS: %s", e)

        if not providers:
            # If neither has real credentials in dev/test, instantiate stubs or raise error on run
            if self.config.eleven_api_key or self.config.sarvam_api_key:
                raise ValueError("Could not initialize any TTS provider. Check API keys.")
            # Default placeholder initialization for inspection
            logger.warning("No TTS API keys provided in environment. Running in unconfigured state.")
            self.primary_tts = create_elevenlabs_tts(config=self.config, api_key="placeholder_key", http_session=session)
            providers.append(self.primary_tts)

        # 3. Create FallbackAdapter when multiple providers are available
        if len(providers) > 1 and self.enable_fallback:
            logger.info("Configuring LiveKit FallbackAdapter: Primary=%s, Backup=%s", providers[0].label, providers[1].label)
            self.active_adapter = FallbackAdapter(
                providers,
                max_retry_per_tts=1,
                sample_rate=22050,
            )

            @self.active_adapter.on("tts_availability_changed")
            def on_availability_changed(ev: AvailabilityChangedEvent):
                if ev.tts == self.primary_tts:
                    self.primary_available = ev.available
                    if not ev.available:
                        self.total_fallbacks += 1
                        logger.warning(
                            "⚠️ [FALLBACK EVENT] Primary TTS (%s) became unavailable! Failing over to Sarvam Bulbul.",
                            ev.tts.label,
                        )
                    else:
                        logger.info(
                            "✅ [RECOVERY EVENT] Primary TTS (%s) recovered and is now available.",
                            ev.tts.label,
                        )
                elif ev.tts == self.backup_tts:
                    self.backup_available = ev.available
                    logger.info("Backup TTS (%s) availability: %s", ev.tts.label, ev.available)

        else:
            self.active_adapter = providers[0]

        self._initialized = True
        return self.active_adapter

    async def aclose(self) -> None:
        """Clean up active providers and HTTP sessions."""
        if self.active_adapter is not None:
            try:
                await self.active_adapter.aclose()
            except Exception as e:
                logger.debug("Error closing TTS adapter: %s", e)
            self.active_adapter = None

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
        metrics: Optional[TTSMetrics] = None,
    ) -> AsyncIterator[rtc.AudioFrame]:
        """Stream generated audio chunks using ElevenLabs with automatic Sarvam fallback.

        - Streams audio chunks as LiveKit AudioFrame objects as fast as received.
        - Automatically retries and fails over if ElevenLabs errors before audio is pushed.
        - Records exact TTFB and generation metrics.
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to stream_synthesize; skipping.")
            return

        target_lang = language or detect_target_language(text, fallback_language=self.config.tts_language)

        if not self._initialized or self.active_adapter is None:
            self.initialize_providers(target_language=target_lang)

        active_tts = self.active_adapter
        if active_tts is None:
            raise RuntimeError("No active TTS engine initialized.")

        # Determine expected provider label
        expected_provider = (
            "ElevenLabs"
            if self.primary_available and self.config.is_elevenlabs_configured
            else "Sarvam Bulbul"
        )

        if metrics is None:
            metrics = TTSMetrics(
                text=text,
                target_language=target_lang,
                provider=expected_provider,
                speaker_or_voice=self.config.elevenlabs_voice_id or self.config.tts_speaker,
                sample_rate=active_tts.sample_rate or 22050,
            )

        chunk_count = 0
        total_bytes = 0
        audio_started = False

        self.total_requests += 1

        try:
            chunked_stream = active_tts.synthesize(text.strip())

            async for event in chunked_stream:
                now = time.perf_counter()
                if metrics.first_chunk_received_at is None:
                    metrics.first_chunk_received_at = now
                    audio_started = True

                    # Verify which provider actually served the audio
                    if isinstance(active_tts, FallbackAdapter):
                        actual_inst = getattr(active_tts, "_active_instance", None)
                        if actual_inst == self.backup_tts:
                            metrics.provider = "Sarvam Bulbul"
                            metrics.fallback_occurred = True
                            self.last_provider_used = "Sarvam Bulbul"
                        else:
                            metrics.provider = "ElevenLabs"
                            self.last_provider_used = "ElevenLabs"
                    else:
                        metrics.provider = active_tts.label or "ElevenLabs"
                        self.last_provider_used = metrics.provider

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
            # Safe mid-utterance handling: if error occurred AFTER audio was already pushed,
            # we do not attempt mid-utterance retry from the beginning to prevent duplicate words.
            if audio_started:
                logger.error(
                    "TTS streaming interrupted mid-utterance after %d chunks emitted. "
                    "Halting to prevent duplicate audio replay: %s",
                    chunk_count,
                    exc,
                )
            else:
                logger.error("TTS synthesis error before audio start: %s", exc)
            raise

    async def play_response(
        self,
        text: str,
        livekit_track: Optional[Union[rtc.AudioSource, LiveKitAudioOutputTrack]] = None,
        local_player: Optional[LocalAudioPlayer] = None,
        language: Optional[str] = None,
    ) -> TTSMetrics:
        """Synthesize text and play audio chunks as they arrive, avoiding playback overlap."""
        async with self._playback_lock:
            target_lang = language or detect_target_language(text, fallback_language=self.config.tts_language)

            metrics = TTSMetrics(
                text=text,
                target_language=target_lang,
                provider="ElevenLabs",
                speaker_or_voice=self.config.elevenlabs_voice_id or self.config.tts_speaker,
                sample_rate=22050,
            )

            async for frame in self.stream_synthesize(
                text=text,
                language=target_lang,
                metrics=metrics,
            ):
                if metrics.first_chunk_played_at is None:
                    metrics.first_chunk_played_at = time.perf_counter()

                # Stream to LiveKit room audio track
                if livekit_track is not None:
                    if isinstance(livekit_track, LiveKitAudioOutputTrack):
                        await livekit_track.push_frame(frame)
                    elif isinstance(livekit_track, rtc.AudioSource):
                        await livekit_track.capture_frame(frame)

                # Play to local speaker (non-blocking)
                if local_player is not None:
                    local_player.play_chunk(bytes(frame.data), sample_rate=frame.sample_rate)

            metrics.playback_completed_at = time.perf_counter()
            return metrics

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
                logger.info("Turn %s was interrupted/stopped, halting TTS", turn_id)
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
                sample_rate=22050,
                channels=1,
                turn_id=turn_id,
                chunk_index=chunk_idx + 1,
                is_final=True,
            )

    async def stop_playback(self, turn_id: str) -> None:
        """GhostBarge Action: Immediately cancel and invalidate active playback."""
        if turn_id not in self._stopped_turns:
            self._stopped_turns.append(turn_id)
        logger.info("TTSManager stopped playback for turn: %s", turn_id)


async def generate_speech(
    text: str,
    language: str = "auto",
    config: Optional[SpeechSettings] = None,
) -> AsyncIterator[rtc.AudioFrame]:
    """Reusable standalone function providing a modular interface for speech generation."""
    manager = TTSManager(config=config)
    try:
        async for frame in manager.stream_synthesize(text=text, language=language):
            yield frame
    finally:
        await manager.aclose()

