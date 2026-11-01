"""Sarvam Saaras Realtime Speech-to-Text Integration.

Uses the official livekit-plugins-sarvam STTRealtime integration to stream live audio
over WebSockets and receive both partial (interim) and final transcripts in real-time.

Configured with:
- language="auto" (Dynamic Indic/English/Hinglish language identification)
- stream_type="fast" (Ultra-low latency partial transcription)
- mode="codemix" (Specialized for conversational Indian code-mixing / Hinglish)
- endpointing="vad" (Server-side VAD turn detection)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, Callable, Optional, Union
import aiohttp

from livekit import rtc
from livekit.agents import stt, utils
from livekit.plugins import sarvam

from phantomflow.contracts.audio_contracts import AudioInputChunk
from phantomflow.contracts.common import LanguageCode, UserUtterance
from speech.config.settings import SpeechSettings, settings

logger = logging.getLogger("phantomflow.speech.stt.saaras")


def normalize_to_language_code(sarvam_lang: Optional[str]) -> LanguageCode:
    """Normalize Sarvam BCP-47 / language tag into PhantomFlow LanguageCode enum."""
    if not sarvam_lang:
        return LanguageCode.HINGLISH

    lang_lower = str(sarvam_lang).lower().strip()
    if lang_lower in ("hi", "hi-in", "hindi"):
        return LanguageCode.HI
    elif lang_lower in ("en", "en-in", "en-us", "english"):
        return LanguageCode.EN
    elif lang_lower in ("hi-en", "codemix", "hinglish"):
        return LanguageCode.HINGLISH
    elif lang_lower in ("auto", "unknown"):
        return LanguageCode.AUTO
    # Default to HINGLISH for code-mixed Indian contexts
    return LanguageCode.HINGLISH


class SarvamSaarasSTT:
    """Production Sarvam Saaras Realtime STT client wrapped for PhantomFlow."""

    def __init__(
        self,
        config: Optional[SpeechSettings] = None,
        language: Optional[str] = None,
        stream_type: Optional[str] = None,
        mode: Optional[str] = None,
        endpointing: Optional[str] = None,
        sample_rate: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.config = config or settings
        self.api_key = api_key or self.config.sarvam_api_key
        self.language = language or self.config.stt_language
        self.stream_type = stream_type or self.config.stt_stream_type
        self.mode = mode or self.config.stt_mode
        self.endpointing = endpointing or self.config.stt_endpointing
        self.sample_rate = sample_rate or self.config.stt_sample_rate

        self._stt_instance: Optional[sarvam.STTRealtime] = None
        self._http_session: Optional[aiohttp.ClientSession] = None

    def _ensure_stt_instance(self) -> sarvam.STTRealtime:
        """Create and validate the livekit-plugins-sarvam STTRealtime instance."""
        if not self.api_key:
            raise ValueError(
                "SARVAM_API_KEY is not set. Please set the SARVAM_API_KEY environment variable "
                "or define it in your .env file."
            )

        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()

        if self._stt_instance is None:
            logger.info(
                "Initializing Sarvam STTRealtime: language=%s, stream_type=%s, mode=%s, endpointing=%s, sample_rate=%d",
                self.language,
                self.stream_type,
                self.mode,
                self.endpointing,
                self.sample_rate,
            )
            self._stt_instance = sarvam.STTRealtime(
                language=self.language,
                stream_type=self.stream_type,
                mode=self.mode,
                endpointing=self.endpointing,
                encoding="linear16",
                sample_rate=self.sample_rate,
                api_key=self.api_key,
                http_session=self._http_session,
            )

        return self._stt_instance

    async def aclose(self) -> None:
        """Close underlying STT instance and HTTP session."""
        if self._stt_instance is not None:
            try:
                await self._stt_instance.aclose()
            except Exception as e:
                logger.debug("Error closing Sarvam STT instance: %s", e)
            self._stt_instance = None

        if self._http_session is not None and not self._http_session.closed:
            try:
                await self._http_session.close()
            except Exception as e:
                logger.debug("Error closing HTTP session: %s", e)
            self._http_session = None

    async def stream_transcribe(
        self,
        audio_stream: AsyncIterator[Union[AudioInputChunk, rtc.AudioFrame, bytes]],
        on_partial: Optional[Callable[[UserUtterance], None]] = None,
        on_final: Optional[Callable[[UserUtterance], None]] = None,
    ) -> AsyncIterator[UserUtterance]:
        """Stream live audio frames/chunks to Sarvam Saaras Realtime STT and yield utterances.

        Conforms directly to ISpeechPipeline.stream_transcribe contract:
        - Yields partial UserUtterance (is_final=False) for early latency tracking
        - Yields final UserUtterance (is_final=True) when speech turn endpoints
        - Attaches detected language (LanguageCode) and confidence score
        """
        stt_client = self._ensure_stt_instance()
        speech_stream = stt_client.stream()

        # Shared queue to receive parsed UserUtterance events from the STT stream
        utterance_queue: asyncio.Queue[Optional[UserUtterance]] = asyncio.Queue()
        stop_event = asyncio.Event()

        # Task 1: Feed audio frames into Sarvam's WebSocket stream
        async def _audio_feeder():
            try:
                async for chunk in audio_stream:
                    if stop_event.is_set():
                        break

                    frame: Optional[rtc.AudioFrame] = None

                    if isinstance(chunk, rtc.AudioFrame):
                        frame = chunk
                    elif isinstance(chunk, AudioInputChunk):
                        samples = len(chunk.data) // 2
                        frame = rtc.AudioFrame(
                            data=chunk.data,
                            sample_rate=chunk.sample_rate,
                            num_channels=chunk.channels,
                            samples_per_channel=samples,
                        )
                    elif isinstance(chunk, (bytes, bytearray)):
                        samples = len(chunk) // 2
                        frame = rtc.AudioFrame(
                            data=bytes(chunk),
                            sample_rate=self.sample_rate,
                            num_channels=1,
                            samples_per_channel=samples,
                        )

                    if frame is not None:
                        speech_stream.push_frame(frame)

                # Signal end of incoming audio
                speech_stream.end_input()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("Error pushing audio frame to Sarvam STT stream: %s", exc)
                stop_event.set()

        # Task 2: Consume transcription events emitted by Sarvam STT stream
        async def _event_consumer():
            try:
                async for event in speech_stream:
                    if stop_event.is_set():
                        break

                    if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                        # Partial transcript event
                        for alt in event.alternatives:
                            text = alt.text.strip()
                            if not text:
                                continue
                            lang = normalize_to_language_code(getattr(alt, "language", None))
                            utterance = UserUtterance(
                                text=text,
                                is_final=False,
                                language=lang,
                                confidence=float(getattr(alt, "confidence", 0.85) or 0.85),
                                stability=0.75,
                                speech_start_time_ns=time.perf_counter_ns(),
                            )
                            if on_partial:
                                try:
                                    on_partial(utterance)
                                except Exception as cb_err:
                                    logger.warning("Error in on_partial callback: %s", cb_err)
                            await utterance_queue.put(utterance)

                    elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        # Final transcript event
                        for alt in event.alternatives:
                            text = alt.text.strip()
                            if not text:
                                continue
                            lang = normalize_to_language_code(getattr(alt, "language", None))
                            utterance = UserUtterance(
                                text=text,
                                is_final=True,
                                language=lang,
                                confidence=float(getattr(alt, "confidence", 0.95) or 0.95),
                                stability=1.0,
                                speech_end_time_ns=time.perf_counter_ns(),
                            )
                            if on_final:
                                try:
                                    on_final(utterance)
                                except Exception as cb_err:
                                    logger.warning("Error in on_final callback: %s", cb_err)
                            await utterance_queue.put(utterance)

                    elif event.type == stt.SpeechEventType.END_OF_SPEECH:
                        logger.debug("Sarvam STT reported END_OF_SPEECH boundary")

            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("Error receiving events from Sarvam STT: %s", exc)
            finally:
                # Sentinel to end the queue
                await utterance_queue.put(None)

        feeder_task = asyncio.create_task(_audio_feeder())
        consumer_task = asyncio.create_task(_event_consumer())

        try:
            while True:
                utterance = await utterance_queue.get()
                if utterance is None:
                    break
                yield utterance
        finally:
            stop_event.set()
            feeder_task.cancel()
            consumer_task.cancel()
            await asyncio.gather(feeder_task, consumer_task, return_exceptions=True)
            await speech_stream.aclose()

