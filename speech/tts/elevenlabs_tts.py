"""Primary Text-to-Speech integration using ElevenLabs (Flash v2.5).

Uses the official livekit-plugins-elevenlabs plugin for natural-sounding,
low-latency conversational speech with multilingual and Hinglish support.

Configuration:
- Provider: ElevenLabs (Primary)
- Model: eleven_flash_v2_5
- Voice ID: Configured via ELEVENLABS_VOICE_ID
- API Key: Configured via ELEVEN_API_KEY (or ELEVENLABS_API_KEY)
- Streaming: low-latency streaming audio generation (streaming_latency=3, auto_mode=True)
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional
import aiohttp

from livekit import rtc
from livekit.plugins import elevenlabs as el

from speech.config.settings import SpeechSettings, settings

logger = logging.getLogger("phantomflow.speech.tts.elevenlabs")


def create_elevenlabs_tts(
    config: Optional[SpeechSettings] = None,
    api_key: Optional[str] = None,
    voice_id: Optional[str] = None,
    model: Optional[str] = None,
    language: Optional[str] = None,
    auto_mode: Optional[bool] = None,
    streaming_latency: Optional[int] = None,
    http_session: Optional[aiohttp.ClientSession] = None,
) -> el.TTS:
    """Factory creating an official livekit-plugins-elevenlabs TTS instance."""
    cfg = config or settings
    resolved_key = api_key or cfg.eleven_api_key
    if not resolved_key:
        raise ValueError(
            "ELEVEN_API_KEY (or ELEVENLABS_API_KEY) is not set. Please set ELEVEN_API_KEY in your .env file."
        )

    v_id = voice_id or cfg.elevenlabs_voice_id
    if not v_id or v_id.lower() == "default":
        resolved_voice_id = el.DEFAULT_VOICE_ID
    else:
        resolved_voice_id = v_id
    resolved_model = model or cfg.elevenlabs_model
    resolved_auto_mode = auto_mode if auto_mode is not None else cfg.elevenlabs_auto_mode
    resolved_latency = streaming_latency if streaming_latency is not None else cfg.elevenlabs_streaming_latency

    logger.info(
        "Initializing ElevenLabs Primary TTS: model=%s, voice_id=%s, auto_mode=%s, latency=%d",
        resolved_model,
        resolved_voice_id,
        resolved_auto_mode,
        resolved_latency,
    )

    kwargs = {
        "api_key": resolved_key,
        "voice_id": resolved_voice_id,
        "model": resolved_model,
        "auto_mode": resolved_auto_mode,
        "streaming_latency": resolved_latency,
    }
    if http_session is not None:
        kwargs["http_session"] = http_session
    if language is not None:
        kwargs["language"] = language

    return el.TTS(**kwargs)


class ElevenLabsTTSClient:
    """High-level client wrapper around ElevenLabs TTS for standalone streaming."""

    def __init__(
        self,
        config: Optional[SpeechSettings] = None,
        voice_id: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.config = config or settings
        self.api_key = api_key or self.config.eleven_api_key
        self.voice_id = voice_id or self.config.elevenlabs_voice_id
        self.model = model or self.config.elevenlabs_model
        self._tts_instance: Optional[el.TTS] = None
        self._http_session: Optional[aiohttp.ClientSession] = None

    def _ensure_instance(self, language: Optional[str] = None) -> el.TTS:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()

        if self._tts_instance is None:
            self._tts_instance = create_elevenlabs_tts(
                config=self.config,
                api_key=self.api_key,
                voice_id=self.voice_id,
                model=self.model,
                language=language,
                http_session=self._http_session,
            )
        return self._tts_instance

    async def stream_synthesize(
        self,
        text: str,
        language: Optional[str] = None,
    ) -> AsyncIterator[rtc.AudioFrame]:
        """Stream synthesized speech chunks as LiveKit AudioFrame objects."""
        if not text or not text.strip():
            return

        tts_inst = self._ensure_instance(language=language)
        chunked_stream = tts_inst.synthesize(text.strip())

        async for event in chunked_stream:
            yield event.frame

    async def aclose(self) -> None:
        """Close TTS instance and underlying HTTP session."""
        if self._tts_instance is not None:
            try:
                await self._tts_instance.aclose()
            except Exception as e:
                logger.debug("Error closing ElevenLabs TTS: %s", e)
            self._tts_instance = None

        if self._http_session is not None and not self._http_session.closed:
            try:
                await self._http_session.close()
            except Exception as e:
                logger.debug("Error closing HTTP session: %s", e)
            self._http_session = None

