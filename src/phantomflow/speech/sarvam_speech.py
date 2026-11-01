"""Production adapter for Sarvam AI Speech-to-Text and Text-to-Speech APIs.

Implements the ISpeechPipeline protocol for:
- Sarvam Saaras Realtime streaming STT (wss://api.sarvam.ai/speech-to-text-realtime/ws)
- Sarvam Bulbul Flash streaming TTS (wss://api.sarvam.ai/text-to-speech/ws or HTTP)
"""

import asyncio
import base64
import json
import logging
import os
import time
from typing import AsyncIterator, List, Optional, Set
import urllib.parse

from phantomflow.contracts.common import LanguageCode, UserUtterance
from phantomflow.contracts.audio_contracts import AudioChunk, AudioInputChunk, ISpeechPipeline

logger = logging.getLogger("phantomflow.speech.sarvam")


class SarvamSpeechPipeline(ISpeechPipeline):
    """Production Speech Pipeline connecting to Sarvam AI APIs.

    Follows Member 2's contract with high-resolution latency tracking,
    graceful degradation, and GhostBarge cancellation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        stt_model: str = "saaras:v3-realtime",
        stt_language_code: str = "hi-IN",
        tts_model: str = "bulbul:v4-flash",
        tts_speaker: str = "shubh_hi_customer",
        tts_sample_rate: int = 24000,
        request_timeout_s: float = 8.0,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("SARVAM_API_KEY", "")
        self.stt_model = os.getenv("SARVAM_STT_MODEL", stt_model)
        self.stt_language_code = os.getenv("SARVAM_STT_LANGUAGE_CODE", stt_language_code)
        self.tts_model = os.getenv("SARVAM_TTS_MODEL", tts_model)
        raw_speaker = os.getenv("SARVAM_TTS_VOICE_ID", os.getenv("SARVAM_TTS_SPEAKER", os.getenv("SARVAM_SPEAKER", tts_speaker)))
        if raw_speaker == "shubh" and "v4" in self.tts_model:
            self.tts_speaker = "shubh_hi_customer"
        else:
            self.tts_speaker = raw_speaker
        self.tts_sample_rate = tts_sample_rate
        self.request_timeout_s = request_timeout_s

        self._stopped_turns: Set[str] = set()
        self.transcribed_count = 0
        self.synthesized_count = 0

    @property
    def is_configured(self) -> bool:
        """Check whether a valid Sarvam API key is available in the environment."""
        return bool(self.api_key and self.api_key != "your_sarvam_api_key")

    async def stream_transcribe(
        self,
        audio_stream: AsyncIterator[AudioInputChunk],
    ) -> AsyncIterator[UserUtterance]:
        """Stream microphone PCM chunks to Sarvam Saaras Realtime STT.

        If SARVAM_API_KEY is not configured or network fails, raises RuntimeError
        or logs clear diagnostic error without exposing secrets.
        """
        if not self.is_configured:
            logger.warning("SarvamSpeechPipeline: SARVAM_API_KEY is unconfigured. Fallback required.")
            raise RuntimeError(
                "SARVAM_API_KEY is not configured. Please supply a valid Sarvam AI API key "
                "in .env or use MockSpeechPipeline for local development."
            )

        try:
            import websockets
        except ImportError:
            raise RuntimeError("websockets package is required for Sarvam streaming STT.")

        query_params = urllib.parse.urlencode({
            "model": self.stt_model,
            "language_code": self.stt_language_code,
            "stream_type": "fast",
        })
        ws_url = f"wss://api.sarvam.ai/speech-to-text-realtime/ws?{query_params}"
        headers = {"Api-Subscription-Key": self.api_key}

        try:
            async with websockets.connect(
                ws_url,
                additional_headers=headers,
                open_timeout=self.request_timeout_s,
            ) as ws:
                logger.info("Connected to Sarvam Saaras Realtime STT (%s)", self.stt_model)

                async def _send_audio():
                    try:
                        async for chunk in audio_stream:
                            if ws.closed:
                                break
                            # Send binary PCM audio frame directly
                            await ws.send(chunk.data)
                    except Exception as exc:
                        logger.debug("Sarvam STT audio send loop ended: %s", exc)

                send_task = asyncio.create_task(_send_audio())

                try:
                    while True:
                        try:
                            msg_raw = await asyncio.wait_for(ws.recv(), timeout=self.request_timeout_s)
                        except asyncio.TimeoutError:
                            logger.info("Sarvam STT idle timeout reached")
                            break

                        data = json.loads(msg_raw) if isinstance(msg_raw, str) else {}
                        nested_data = data.get("data", {}) if isinstance(data.get("data"), dict) else {}
                        text = (data.get("transcript") or nested_data.get("transcript") or data.get("text") or "").strip()
                        is_final = bool(data.get("is_final") or (data.get("type") == "final") or nested_data.get("is_final", False))
                        confidence = float(data.get("confidence") or nested_data.get("confidence") or 0.95)
                        stability = float(data.get("stability", 0.7 if not is_final else 1.0))
                        detected_lang_str = str(data.get("language_code") or nested_data.get("language_code") or self.stt_language_code)

                        lang_code = LanguageCode.HINGLISH
                        if "hi" in detected_lang_str and "en" in detected_lang_str:
                            lang_code = LanguageCode.HINGLISH
                        elif "hi" in detected_lang_str:
                            lang_code = LanguageCode.HI
                        elif "en" in detected_lang_str:
                            lang_code = LanguageCode.EN

                        if text:
                            self.transcribed_count += 1
                            yield UserUtterance(
                                text=text,
                                is_final=is_final,
                                language=lang_code,
                                confidence=confidence,
                                stability=stability,
                                speech_end_time_ns=time.perf_counter_ns() if is_final else None,
                            )

                        if is_final:
                            break

                finally:
                    send_task.cancel()
                    try:
                        await send_task
                    except asyncio.CancelledError:
                        pass

        except Exception as e:
            logger.error("Sarvam STT stream error: %s", e)
            raise

    async def synthesize_stream(
        self,
        text_generator: AsyncIterator[str],
        turn_id: str,
    ) -> AsyncIterator[AudioChunk]:
        """Stream text tokens to Sarvam Bulbul Flash and yield synthesized AudioChunks.

        Ceases output immediately if turn_id is halted by GhostBarge.
        """
        if not self.is_configured:
            logger.warning("SarvamSpeechPipeline: SARVAM_API_KEY is unconfigured for TTS.")
            raise RuntimeError(
                "SARVAM_API_KEY is not configured for Bulbul TTS. "
                "Supply credentials or use MockSpeechPipeline."
            )

        # Buffer incoming tokens to sentence or phrase boundaries for natural prosody
        text_buffer: List[str] = []
        chunk_index = 0

        async for token in text_generator:
            if turn_id in self._stopped_turns:
                logger.info("GhostBarge: Turn %s was cancelled during Bulbul synthesis", turn_id)
                break

            text_buffer.append(token)
            accumulated = "".join(text_buffer)

            # Synthesize on clause delimiters (. , ? ! or > 80 chars)
            if any(accumulated.endswith(punct) for punct in (".", ",", "?", "!", "\n")) or len(accumulated) >= 80:
                text_to_speak = accumulated.strip()
                text_buffer.clear()

                if not text_to_speak:
                    continue

                chunks = await self._call_sarvam_tts(text_to_speak, turn_id, chunk_index)
                for chunk in chunks:
                    if turn_id in self._stopped_turns:
                        break
                    chunk_index += 1
                    self.synthesized_count += 1
                    yield chunk

        # Flush remaining text
        if text_buffer and turn_id not in self._stopped_turns:
            remaining_text = "".join(text_buffer).strip()
            if remaining_text:
                chunks = await self._call_sarvam_tts(remaining_text, turn_id, chunk_index)
                for chunk in chunks:
                    if turn_id in self._stopped_turns:
                        break
                    chunk_index += 1
                    self.synthesized_count += 1
                    yield chunk

        # Final delimiter chunk
        if turn_id not in self._stopped_turns:
            yield AudioChunk(
                data=b"",
                sample_rate=self.tts_sample_rate,
                channels=1,
                turn_id=turn_id,
                chunk_index=chunk_index + 1,
                is_final=True,
            )

    async def _call_sarvam_tts(
        self,
        text: str,
        turn_id: str,
        starting_chunk_index: int,
    ) -> List[AudioChunk]:
        """Invoke Sarvam Bulbul REST API with audio chunks returned."""
        import httpx

        url = "https://api.sarvam.ai/text-to-speech"
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "language_code": self.stt_language_code,
            "target_language_code": self.stt_language_code,
            "speaker": self.tts_speaker,
            "model": self.tts_model,
            "speech_sample_rate": self.tts_sample_rate,
            "output_audio_codec": "linear16",
            "enable_preprocessing": True,
        }

        async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                logger.warning("Sarvam TTS error: HTTP %d: %s. Handing off to client voice engine.", resp.status_code, resp.text)
                return []

            data = resp.json()
            audios = data.get("audios", [])
            chunks: List[AudioChunk] = []

            for idx, audio_b64 in enumerate(audios):
                raw_bytes = base64.b64decode(audio_b64)
                chunks.append(
                    AudioChunk(
                        data=raw_bytes,
                        sample_rate=self.tts_sample_rate,
                        channels=1,
                        turn_id=turn_id,
                        chunk_index=starting_chunk_index + idx,
                        is_final=False,
                    )
                )
            return chunks

    async def stop_playback(self, turn_id: str) -> None:
        """Immediately flag turn as stopped for GhostBarge interruption."""
        logger.info("SarvamSpeechPipeline: stop_playback invoked for turn %s", turn_id)
        self._stopped_turns.add(turn_id)
