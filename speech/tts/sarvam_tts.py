"""Backup Text-to-Speech integration using Sarvam Bulbul (v3).

Used as the secondary fallback provider when ElevenLabs encounters an error,
connection failure, or rate limit.

Configuration:
- Provider: Sarvam Bulbul (Backup)
- Model: bulbul:v3
- Speaker: shubh
- API Key: Configured via SARVAM_API_KEY
- Speech Sample Rate: 22050 Hz
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Optional
import aiohttp

from livekit import rtc
from livekit.plugins import sarvam as sp

from speech.config.settings import SpeechSettings, settings

logger = logging.getLogger("phantomflow.speech.tts.sarvam")


def create_sarvam_tts(
    config: Optional[SpeechSettings] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    speaker: Optional[str] = None,
    target_language: Optional[str] = None,
    sample_rate: Optional[int] = None,
    pace: Optional[float] = None,
    http_session: Optional[aiohttp.ClientSession] = None,
) -> sp.TTS:
    """Factory creating an official livekit-plugins-sarvam TTS instance."""
    cfg = config or settings
    resolved_key = api_key or cfg.sarvam_api_key
    if not resolved_key:
        raise ValueError(
            "SARVAM_API_KEY is not set. Please set SARVAM_API_KEY in your .env file."
        )

    resolved_model = model or cfg.tts_model
    resolved_speaker = speaker or cfg.tts_speaker
    if resolved_speaker == "shubh" and "v4" in str(resolved_model):
        resolved_speaker = "shubh_hi_customer"
    resolved_lang = target_language or cfg.tts_language
    resolved_rate = sample_rate or cfg.tts_sample_rate
    resolved_pace = pace if pace is not None else cfg.tts_pace

    logger.info(
        "Initializing Sarvam Bulbul Backup TTS: model=%s, speaker=%s, lang=%s, sample_rate=%d",
        resolved_model,
        resolved_speaker,
        resolved_lang,
        resolved_rate,
    )

    kwargs = {
        "api_key": resolved_key,
        "model": resolved_model,
        "speaker": resolved_speaker,
        "target_language_code": resolved_lang,
        "speech_sample_rate": resolved_rate,
        "num_channels": 1,
        "pace": resolved_pace,
    }
    if http_session is not None:
        kwargs["http_session"] = http_session

    return sp.TTS(**kwargs)


class SarvamTTSClient:
    """High-level client wrapper around Sarvam Bulbul TTS."""

    def __init__(
        self,
        config: Optional[SpeechSettings] = None,
        model: Optional[str] = None,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        sample_rate: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.config = config or settings
        self.api_key = api_key or self.config.sarvam_api_key
        self.model = model or self.config.tts_model
        self.speaker = speaker or self.config.tts_speaker
        self.language = language or self.config.tts_language
        self.sample_rate = sample_rate or self.config.tts_sample_rate
        self._tts_instance: Optional[sp.TTS] = None
        self._http_session: Optional[aiohttp.ClientSession] = None

    def _ensure_instance(self, language: Optional[str] = None) -> sp.TTS:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()

        lang = language or self.language
        if self._tts_instance is None or self._tts_instance._opts.target_language_code != lang:
            self._tts_instance = create_sarvam_tts(
                config=self.config,
                api_key=self.api_key,
                model=self.model,
                speaker=self.speaker,
                target_language=lang,
                sample_rate=self.sample_rate,
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
                logger.debug("Error closing Sarvam TTS: %s", e)
            self._tts_instance = None

        if self._http_session is not None and not self._http_session.closed:
            try:
                await self._http_session.close()
            except Exception as e:
                logger.debug("Error closing HTTP session: %s", e)
            self._http_session = None

