"""Audio playback mechanisms for local speaker and LiveKit WebRTC tracks.

Provides:
- LocalAudioPlayer: Non-blocking speaker output using sounddevice.
- LiveKitAudioOutputTrack: LocalAudioTrack publisher for LiveKit room participants.
- Concurrency control to prevent overlapping voice playback.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional
import sounddevice as sd

from livekit import rtc
from speech.config.settings import SpeechSettings, settings

logger = logging.getLogger("phantomflow.speech.audio.playback")


class LocalAudioPlayer:
    """Non-blocking audio player using sounddevice for physical local speaker playback."""

    def __init__(self, sample_rate: int = 22050, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream: Optional[sd.RawOutputStream] = None
        self._is_active = False

    def _ensure_stream(self, sample_rate: int) -> sd.RawOutputStream:
        # Re-initialize stream if sample rate changed (e.g. 24kHz ElevenLabs vs 22050Hz Sarvam)
        if self._stream is not None and not self._stream.closed:
            if self.sample_rate != sample_rate:
                self.close()

        if self._stream is None or self._stream.closed:
            self.sample_rate = sample_rate
            logger.debug("Opening local audio output stream at %d Hz", self.sample_rate)
            self._stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
            )
            self._stream.start()
            self._is_active = True
        return self._stream

    def play_chunk(self, data: bytes, sample_rate: Optional[int] = None) -> None:
        """Write raw 16-bit PCM bytes to local speakers."""
        if not data:
            return
        sr = sample_rate or self.sample_rate
        stream = self._ensure_stream(sr)
        stream.write(data)

    def close(self) -> None:
        """Stop and close the speaker output stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug("Error closing local speaker stream: %s", e)
            self._stream = None
            self._is_active = False


class LiveKitAudioOutputTrack:
    """Publishes synthesized speech to a LiveKit room as a WebRTC audio track."""

    def __init__(
        self,
        room: rtc.Room,
        track_name: str = "assistant-voice",
        sample_rate: int = 22050,
        channels: int = 1,
    ) -> None:
        self.room = room
        self.track_name = track_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.source: Optional[rtc.AudioSource] = None
        self.track: Optional[rtc.LocalAudioTrack] = None
        self.publication: Optional[rtc.LocalTrackPublication] = None

    async def initialize_and_publish(self) -> None:
        """Create the AudioSource and LocalAudioTrack, and publish to the room."""
        self.source = rtc.AudioSource(self.sample_rate, self.channels)
        self.track = rtc.LocalAudioTrack.create_audio_track(self.track_name, self.source)

        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        self.publication = await self.room.local_participant.publish_track(self.track, options)
        logger.info("Published assistant speech track '%s' to LiveKit room", self.track_name)

    async def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Push an AudioFrame into the LiveKit audio track."""
        if self.source is not None:
            await self.source.capture_frame(frame)

    async def close(self) -> None:
        """Unpublish the track and clean up."""
        if self.track is not None and self.room.local_participant is not None:
            try:
                if self.publication and self.publication.sid:
                    await self.room.local_participant.unpublish_track(self.publication.sid)
            except Exception as e:
                logger.debug("Error unpublishing speech track: %s", e)
            self.track = None
            self.source = None
            self.publication = None

