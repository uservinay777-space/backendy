"""Microphone audio capture and LiveKit WebRTC audio publisher.

Captures live audio from the physical microphone asynchronously without blocking
the asyncio event loop, and streams audio frames either directly or via LiveKit WebRTC.
"""

from __future__ import annotations

import asyncio
import logging
import math
import struct
import time
from typing import AsyncIterator, List, Optional
import sounddevice as sd

from livekit import rtc
from phantomflow.contracts.audio_contracts import AudioInputChunk
from speech.config.settings import SpeechSettings, settings

logger = logging.getLogger("phantomflow.speech.audio.microphone")


class MicrophoneCapture:
    """Non-blocking microphone audio capturer using sounddevice."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 20,
        device_index: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.device_index = device_index

        # 16-bit PCM = 2 bytes per sample
        self.bytes_per_sample = 2
        # Number of samples in one chunk (e.g., 16000 * 0.02 = 320 samples)
        self.samples_per_chunk = int(self.sample_rate * (self.chunk_duration_ms / 1000.0))
        # Total bytes per chunk (e.g., 320 * 2 = 640 bytes for mono)
        self.bytes_per_chunk = self.samples_per_chunk * self.channels * self.bytes_per_sample

        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stream: Optional[sd.RawInputStream] = None
        self._is_running = False

    @classmethod
    def list_input_devices(cls) -> List[dict]:
        """List all available microphone audio input devices."""
        devices = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                devices.append({
                    "index": idx,
                    "name": dev.get("name"),
                    "channels": dev.get("max_input_channels"),
                    "sample_rate": dev.get("default_samplerate"),
                })
        return devices

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio thread callback called by PortAudio/sounddevice."""
        if status:
            logger.debug("Microphone audio status: %s", status)
        if self._is_running and self._loop is not None and not self._loop.is_closed():
            raw_bytes = bytes(indata)
            self._loop.call_soon_threadsafe(self._queue.put_nowait, raw_bytes)

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Start capturing audio from the microphone."""
        if self._is_running:
            return

        self._loop = loop or asyncio.get_running_loop()
        self._is_running = True

        logger.info(
            "Starting microphone capture: %d Hz, %d ch, %d ms chunk (%d samples/chunk), device=%s",
            self.sample_rate,
            self.channels,
            self.chunk_duration_ms,
            self.samples_per_chunk,
            self.device_index if self.device_index is not None else "default",
        )

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.samples_per_chunk,
            device=self.device_index,
            channels=self.channels,
            dtype="int16",
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop capturing audio and clean up the audio stream."""
        if not self._is_running:
            return

        self._is_running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug("Error stopping microphone stream: %s", e)
            self._stream = None

        logger.info("Microphone capture stopped")

    async def __aenter__(self) -> MicrophoneCapture:
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    async def stream_pcm(self) -> AsyncIterator[bytes]:
        """Asynchronously yield raw PCM audio chunks."""
        while self._is_running:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                yield chunk
            except asyncio.TimeoutError:
                continue

    async def stream_audio_input_chunks(self) -> AsyncIterator[AudioInputChunk]:
        """Asynchronously yield PhantomFlow AudioInputChunk objects."""
        while self._is_running:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                yield AudioInputChunk(
                    data=chunk,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    timestamp_ns=time.perf_counter_ns(),
                )
            except asyncio.TimeoutError:
                continue

    async def stream_audio_frames(self) -> AsyncIterator[rtc.AudioFrame]:
        """Asynchronously yield LiveKit rtc.AudioFrame objects."""
        while self._is_running:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                samples = len(chunk) // (self.channels * self.bytes_per_sample)
                yield rtc.AudioFrame(
                    data=chunk,
                    sample_rate=self.sample_rate,
                    num_channels=self.channels,
                    samples_per_channel=samples,
                )
            except asyncio.TimeoutError:
                continue


class LiveKitMicrophonePublisher:
    """Publishes microphone audio to a LiveKit room as a WebRTC audio track."""

    def __init__(
        self,
        config: Optional[SpeechSettings] = None,
        identity: str = "user-microphone",
        name: str = "Microphone Speaker",
    ) -> None:
        self.config = config or settings
        self.identity = identity
        self.name = name
        self.room: Optional[rtc.Room] = None
        self.source: Optional[rtc.AudioSource] = None
        self.track: Optional[rtc.LocalAudioTrack] = None
        self.mic: Optional[MicrophoneCapture] = None
        self._publish_task: Optional[asyncio.Task] = None

    async def connect_and_publish(self) -> None:
        """Connect to LiveKit room and begin publishing microphone audio."""
        if not self.config.is_livekit_configured:
            raise ValueError(
                "LiveKit credentials not configured. Please set LIVEKIT_URL, "
                "LIVEKIT_API_KEY, and LIVEKIT_API_SECRET in your .env file."
            )

        token = self.config.generate_livekit_token(
            participant_identity=self.identity,
            participant_name=self.name,
            can_publish=True,
            can_subscribe=False,
        )

        logger.info("Connecting microphone participant '%s' to LiveKit...", self.identity)
        self.room = rtc.Room()
        await self.room.connect(self.config.livekit_url, token)
        logger.info("Microphone participant connected to room: %s", self.room.name)

        # Create AudioSource matching the microphone sample rate
        sample_rate = self.config.audio_sample_rate
        channels = self.config.audio_channels
        self.source = rtc.AudioSource(sample_rate, channels)
        self.track = rtc.LocalAudioTrack.create_audio_track("microphone", self.source)

        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        publication = await self.room.local_participant.publish_track(self.track, options)
        logger.info("Published microphone audio track to LiveKit room: %s", publication.sid)

        # Start microphone capture
        self.mic = MicrophoneCapture(
            sample_rate=sample_rate,
            channels=channels,
            chunk_duration_ms=self.config.audio_chunk_duration_ms,
            device_index=self.config.audio_device_index,
        )
        self.mic.start()

        # Pump captured frames into LiveKit AudioSource
        async def _pump_frames():
            try:
                async for frame in self.mic.stream_audio_frames():
                    await self.source.capture_frame(frame)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error("Error publishing microphone frame to LiveKit: %s", e)

        self._publish_task = asyncio.create_task(_pump_frames())

    async def stop(self) -> None:
        """Stop publishing and disconnect from the LiveKit room."""
        if self._publish_task is not None:
            self._publish_task.cancel()
            try:
                await self._publish_task
            except asyncio.CancelledError:
                pass
            self._publish_task = None

        if self.mic is not None:
            self.mic.stop()
            self.mic = None

        if self.room is not None:
            try:
                await self.room.disconnect()
            except Exception as e:
                logger.debug("Error disconnecting room: %s", e)
            self.room = None

        logger.info("LiveKit microphone publisher stopped")


def generate_synthetic_pcm(
    frequency_hz: float = 440.0,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
) -> bytes:
    """Generate synthetic sine wave 16-bit mono PCM bytes for unit testing."""
    num_samples = int(sample_rate * duration_s)
    pcm_bytes = bytearray()
    max_amp = 32767.0 * amplitude

    for i in range(num_samples):
        val = int(max_amp * math.sin(2.0 * math.pi * frequency_hz * (i / sample_rate)))
        pcm_bytes.extend(struct.pack("<h", val))

    return bytes(pcm_bytes)

