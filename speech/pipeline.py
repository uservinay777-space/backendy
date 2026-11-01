"""Speech Input & Voice Output Pipeline for PhantomFlow (Role 02 — Speech & Audio Engineer).

Implements the complete real-time speech interaction pipeline:
Microphone → LiveKit WebRTC → Sarvam Saaras STT → Demo Response → Sarvam Bulbul TTS → LiveKit / Local Playback

Supports:
- LiveKit Room Mode: Ingests mic audio track, transcribes via Saaras STT, synthesizes response with Bulbul TTS, and plays audio track back into the LiveKit room.
- Direct Mode: Ingests local microphone, transcribes, synthesizes with Bulbul TTS, and plays audio through local speakers.
- Real-time partial & final transcripts with language detection and latency metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from typing import AsyncIterator, Callable, Optional

from livekit import rtc

from phantomflow.contracts.common import UserUtterance
from phantomflow.contracts.audio_contracts import AudioChunk, AudioInputChunk, ISpeechPipeline
from speech.audio.microphone import LiveKitMicrophonePublisher, MicrophoneCapture
from speech.audio.playback import LiveKitAudioOutputTrack, LocalAudioPlayer
from speech.config.settings import SpeechSettings, settings
from speech.stt.saaras import SarvamSaarasSTT
from speech.tts.bulbul import SarvamBulbulTTS
from speech.tts.tts_manager import TTSManager, TTSMetrics, detect_target_language

try:
    from speech.asr_experiment.livekit_adapter import LocalWhisperSTT
except ImportError:
    LocalWhisperSTT = None

# Setup clean console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phantomflow.speech.pipeline")


def get_predefined_response(user_text: str) -> str:
    """Provide a simple contextual demonstration response for Milestone 2 testing."""
    t = user_text.lower()
    if "bengaluru" in t and "chennai" in t:
        return "Saturday ke liye Bengaluru se Chennai trains available hain. Kya main timings check karoon?"
    elif "bengaluru" in t or "jaana" in t or "train" in t:
        return "Sure! Aapko kab jaana hai?"
    elif "appointment" in t or "doctor" in t or "hospital" in t:
        return "Haan zaroor! Kal morning 10 AM ka slot available hai. Kya main book kar doon?"
    elif "hello" in t or "how are you" in t:
        return "Hello! I am doing well. How can I help you today?"
    elif "namaste" in t or "नमस्ते" in user_text:
        return "नमस्ते! मैं आपकी क्या मदद कर सकता हूँ?"
    else:
        return f"Maine sun liya: {user_text}. Main aapki ticket ya appointment booking mein madad kar sakta hoon."


class TranscriptDisplay:
    """Terminal UI display for real-time streaming transcripts and assistant responses."""

    def __init__(self) -> None:
        self.last_was_partial = False

    def show_partial(self, utterance: UserUtterance) -> None:
        """Display an interim partial transcript (yellow/cyan) without pushing a new line."""
        lang_str = utterance.language.value if hasattr(utterance.language, "value") else str(utterance.language)
        sys.stdout.write(
            f"\r\033[K\033[33m⏳ [PARTIAL] \033[36m({lang_str})\033[33m \"{utterance.text}\"\033[0m"
        )
        sys.stdout.flush()
        self.last_was_partial = True

    def show_final(self, utterance: UserUtterance) -> None:
        """Display a committed final transcript (bold green) with newline and metrics."""
        lang_str = utterance.language.value if hasattr(utterance.language, "value") else str(utterance.language)
        conf_pct = int(utterance.confidence * 100)
        prefix = "\n" if self.last_was_partial else ""
        sys.stdout.write(
            f"{prefix}\033[1;32m✅ [FINAL]   \033[36m({lang_str})\033[0m \033[35m[{conf_pct}%]\033[0m "
            f"\033[1;37m\"{utterance.text}\"\033[0m\n"
        )
        sys.stdout.flush()
        self.last_was_partial = False

    def show_assistant_response(self, text: str, lang: str) -> None:
        """Display the assistant's outgoing spoken response."""
        prefix = "\n" if self.last_was_partial else ""
        sys.stdout.write(
            f"{prefix}\033[1;36m🤖 [ASSISTANT] \033[35m({lang})\033[1;36m \"{text}\"\033[0m\n"
        )
        sys.stdout.flush()
        self.last_was_partial = False

    def show_tts_metrics(self, metrics: TTSMetrics) -> None:
        """Display performance telemetry for the TTS generation."""
        sys.stdout.write(f"\033[90m   ⚡ {metrics.format_summary()}\033[0m\n")
        sys.stdout.flush()

    def show_info(self, message: str) -> None:
        """Print an informational status line."""
        prefix = "\n" if self.last_was_partial else ""
        sys.stdout.write(f"{prefix}\033[34mℹ️  {message}\033[0m\n")
        sys.stdout.flush()
        self.last_was_partial = False

    def show_error(self, message: str) -> None:
        """Print an error alert."""
        prefix = "\n" if self.last_was_partial else ""
        sys.stdout.write(f"{prefix}\033[1;31m❌ [ERROR] {message}\033[0m\n")
        sys.stdout.flush()
        self.last_was_partial = False


class SpeechInputPipeline:
    """Orchestrates Microphone, LiveKit, Sarvam Saaras STT, and Sarvam Bulbul TTS."""

    def __init__(
        self,
        config: Optional[SpeechSettings] = None,
        on_partial: Optional[Callable[[UserUtterance], None]] = None,
        on_final: Optional[Callable[[UserUtterance], None]] = None,
        enable_tts: bool = True,
    ) -> None:
        self.config = config or settings
        self.display = TranscriptDisplay()
        self.enable_tts = enable_tts

        # External callbacks for integration with Orchestrator (Member 1)
        self._external_on_partial = on_partial
        self._external_on_final = on_final

        # Initialize STT engine (Sarvam Saaras if configured, otherwise Local Whisper ASR)
        if self.config.is_sarvam_configured:
            self.stt = SarvamSaarasSTT(
                config=self.config,
                language=self.config.stt_language,
                stream_type=self.config.stt_stream_type,
                mode=self.config.stt_mode,
                endpointing=self.config.stt_endpointing,
                sample_rate=self.config.stt_sample_rate,
                api_key=self.config.sarvam_api_key,
            )
            self.stt_provider_name = "Sarvam Saaras Realtime STT (Cloud)"
        elif LocalWhisperSTT is not None:
            self.stt = LocalWhisperSTT(sample_rate=self.config.audio_sample_rate)
            self.stt_provider_name = "PhantomFlow Local Whisper ASR (Apple Silicon / On-Device)"
        else:
            self.stt = SarvamSaarasSTT(
                config=self.config,
                api_key=self.config.sarvam_api_key,
            )
            self.stt_provider_name = "Sarvam Saaras Realtime STT"

        # Initialize TTS Manager (Primary: ElevenLabs, Backup: Sarvam Bulbul)
        self.tts_manager = TTSManager(config=self.config)
        self.tts = SarvamBulbulTTS(
            config=self.config,
            model=self.config.tts_model,
            speaker=self.config.tts_speaker,
            language=self.config.tts_language,
            sample_rate=self.config.tts_sample_rate,
            pace=self.config.tts_pace,
            api_key=self.config.sarvam_api_key,
        )

        self.local_player: Optional[LocalAudioPlayer] = (
            LocalAudioPlayer(sample_rate=self.config.tts_sample_rate, channels=1)
            if self.config.tts_enable_local_playback
            else None
        )

        self.mic_publisher: Optional[LiveKitMicrophonePublisher] = None
        self.worker_room: Optional[rtc.Room] = None
        self.speaker_source: Optional[rtc.AudioSource] = None
        self.speaker_track: Optional[rtc.LocalAudioTrack] = None
        self._stop_event = asyncio.Event()

    def _handle_partial(self, utterance: UserUtterance) -> None:
        """Internal callback for partial transcripts."""
        self.display.show_partial(utterance)
        if self._external_on_partial:
            try:
                self._external_on_partial(utterance)
            except Exception as e:
                logger.warning("External on_partial error: %s", e)

    def _handle_final(self, utterance: UserUtterance) -> None:
        """Internal callback for final transcripts."""
        self.display.show_final(utterance)
        if self._external_on_final:
            try:
                self._external_on_final(utterance)
            except Exception as e:
                logger.warning("External on_final error: %s", e)

        # Synthesize and speak a contextual response
        if self.enable_tts:
            try:
                loop = asyncio.get_running_loop()
                response_text = get_predefined_response(utterance.text)
                loop.create_task(self._synthesize_and_play_response(response_text))
            except RuntimeError:
                # No running event loop (e.g. running in synchronous unit test environment)
                pass

    async def _synthesize_and_play_response(self, response_text: str) -> None:
        """Asynchronously stream TTS response to LiveKit audio track and local speakers."""
        try:
            target_lang = detect_target_language(response_text, fallback_language=self.config.tts_language)
            self.display.show_assistant_response(response_text, lang=target_lang)

            if self.config.is_elevenlabs_configured:
                try:
                    # Primary TTS: ElevenLabs
                    metrics = await self.tts_manager.play_response(
                        text=response_text,
                        livekit_track=self.speaker_source,
                        local_player=self.local_player,
                        language=target_lang,
                    )
                    self.display.show_tts_metrics(metrics)
                    return
                except Exception as exc:
                    logger.warning("Primary TTS (ElevenLabs) error: %s. Falling back to backup/local voice...", exc)

            if self.config.is_sarvam_configured:
                try:
                    # Backup TTS: Sarvam Bulbul
                    metrics = await self.tts.play_response(
                        text=response_text,
                        livekit_source=self.speaker_source,
                        local_player=self.local_player,
                        language=target_lang,
                        speaker=self.config.tts_speaker,
                    )
                    self.display.show_tts_metrics(metrics)
                    return
                except Exception as exc:
                    logger.warning("Backup TTS (Sarvam Bulbul) error: %s. Falling back to On-Device...", exc)

            # On-Device speech synthesis (macOS Core Audio)
            voice = "Lekha" if target_lang.startswith("hi") else "Rishi"
            proc = await asyncio.create_subprocess_exec(
                "say", "-v", voice, response_text,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception as exc:
            logger.error("Error synthesizing or playing TTS response: %s", exc)
            self.display.show_error(f"TTS synthesis failure: {exc}")

    # --- ISpeechPipeline Protocol Implementation ---

    async def stream_transcribe(
        self,
        audio_stream: AsyncIterator[AudioInputChunk],
    ) -> AsyncIterator[UserUtterance]:
        """Stream live audio chunks to STT and yield transcribed utterances (ISpeechPipeline)."""
        async for utt in self.stt.stream_transcribe(
            audio_stream,
            on_partial=self._handle_partial,
            on_final=self._handle_final,
        ):
            yield utt

    async def synthesize_stream(
        self,
        text_generator: AsyncIterator[str],
        turn_id: str,
    ) -> AsyncIterator[AudioChunk]:
        """Stream synthesized audio chunks from a streaming text token generator (ISpeechPipeline)."""
        async for chunk in self.tts_manager.synthesize_stream(text_generator, turn_id):
            yield chunk

    async def stop_playback(self, turn_id: str) -> None:
        """GhostBarge action: Immediately cancel and flush speaker audio playback buffer (ISpeechPipeline)."""
        await self.tts_manager.stop_playback(turn_id)

    # --- Pipeline Execution Modes ---

    async def run_livekit_pipeline(self, room_name: Optional[str] = None) -> None:
        """Run the end-to-end LiveKit WebRTC pipeline:
        Microphone → LiveKit Room → Sarvam Saaras STT → Predefined Response → Sarvam Bulbul TTS → LiveKit Speaker Track.
        """
        if not self.config.is_livekit_configured:
            raise ValueError(
                "LiveKit credentials not configured in environment. "
                "Set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET in your .env file, "
                "or run in direct mic mode with `--mode direct`."
            )

        target_room = room_name or self.config.livekit_room_name
        self.display.show_info(f"Connecting to LiveKit room: {target_room}")

        # 1. Connect the voice worker to the LiveKit room (can subscribe to caller audio AND publish assistant voice)
        worker_token = self.config.generate_livekit_token(
            participant_identity="phantomflow-voice-worker",
            participant_name="PhantomFlow Audio Worker",
            room_name=target_room,
            can_publish=True,
            can_subscribe=True,
        )

        self.worker_room = rtc.Room()
        await self.worker_room.connect(self.config.livekit_url, worker_token)
        self.display.show_info(f"Voice worker connected to room '{target_room}'")

        # Create outgoing LiveKit audio track for Bulbul TTS playback
        self.speaker_source = rtc.AudioSource(self.config.tts_sample_rate, 1)
        self.speaker_track = rtc.LocalAudioTrack.create_audio_track("assistant-voice", self.speaker_source)
        pub_options = rtc.TrackPublishOptions()
        pub_options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await self.worker_room.local_participant.publish_track(self.speaker_track, pub_options)
        self.display.show_info("Assistant TTS audio track published to LiveKit room.")

        # 2. Queue for incoming audio frames received from remote participants
        audio_frame_queue: asyncio.Queue[rtc.AudioFrame] = asyncio.Queue()

        async def _consume_remote_track(track: rtc.Track):
            self.display.show_info(f"Subscribed to remote audio track: {track.sid}. Starting STT stream...")
            audio_stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
            try:
                async for audio_event in audio_stream:
                    if self._stop_event.is_set():
                        break
                    await audio_frame_queue.put(audio_event.frame)
            except Exception as e:
                logger.error("Error reading audio stream: %s", e)

        @self.worker_room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_AUDIO and participant.identity != "phantomflow-voice-worker":
                logger.info("Audio track subscribed from caller %s", participant.identity)
                asyncio.create_task(_consume_remote_track(track))

        for participant in self.worker_room.remote_participants.values():
            if participant.identity == "phantomflow-voice-worker":
                continue
            for pub in participant.track_publications.values():
                if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                    asyncio.create_task(_consume_remote_track(pub.track))

        # 3. Connect microphone publisher to the same room
        self.display.show_info("Starting microphone and publishing to LiveKit room...")
        self.mic_publisher = LiveKitMicrophonePublisher(
            config=self.config,
            identity="user-microphone",
            name="User Microphone",
        )
        await self.mic_publisher.connect_and_publish()
        self.display.show_info("Microphone track published. Speak into your microphone now!")

        # 4. Async generator to feed queued frames to Sarvam
        async def _frame_generator():
            while not self._stop_event.is_set():
                try:
                    frame = await asyncio.wait_for(audio_frame_queue.get(), timeout=0.1)
                    yield frame
                except asyncio.TimeoutError:
                    continue

        # 5. Transcribe in real-time
        try:
            async for _ in self.stt.stream_transcribe(
                _frame_generator(),
                on_partial=self._handle_partial,
                on_final=self._handle_final,
            ):
                pass
        finally:
            await self.stop()

    async def run_direct_pipeline(self) -> None:
        """Run direct microphone to STT & TTS pipeline:
        Microphone → STT (Saaras / Local Whisper) → Response → TTS (ElevenLabs / Bulbul / Local Voice) → Speaker.
        """
        self.display.show_info("Initializing microphone capture (16kHz 16-bit mono)...")
        mic = MicrophoneCapture(
            sample_rate=self.config.audio_sample_rate,
            channels=self.config.audio_channels,
            chunk_duration_ms=self.config.audio_chunk_duration_ms,
            device_index=self.config.audio_device_index,
        )
        mic.start()

        tts_info = (
            "ElevenLabs Flash v2.5"
            if self.config.is_elevenlabs_configured
            else (f"Bulbul ({self.config.tts_model})" if self.config.is_sarvam_configured else "Native Speech Synthesis (macOS Core Audio)")
        )

        if self.config.is_sarvam_configured:
            self.display.show_info(
                f"Active Engine: STT Saaras ({self.config.stt_language}, {self.config.stt_mode}) → TTS {tts_info}"
            )
        else:
            self.display.show_info(
                f"Active Engine: {self.stt_provider_name} → TTS {tts_info}"
            )
            if not self.config.is_sarvam_configured:
                self.display.show_info(
                    "Tip: Add SARVAM_API_KEY to your .env file to enable Cloud Saaras STT."
                )

        self.display.show_info("Listening... Speak in Hindi, English, or Hinglish! (Press Ctrl+C to stop)")

        try:
            while not self._stop_event.is_set():
                try:
                    async for _ in self.stt.stream_transcribe(
                        mic.stream_audio_frames(),
                        on_partial=self._handle_partial,
                        on_final=self._handle_final,
                    ):
                        if self._stop_event.is_set():
                            break
                except Exception as stream_err:
                    if self._stop_event.is_set():
                        break
                    logger.info("STT session ended or timed out (%s). Reconnecting live session...", stream_err)
                    await asyncio.sleep(0.5)
        finally:
            mic.stop()
            await self.stop()

    async def stop(self) -> None:
        """Stop all pipeline tasks and cleanly disconnect connections."""
        self._stop_event.set()

        if self.mic_publisher is not None:
            await self.mic_publisher.stop()
            self.mic_publisher = None

        if self.worker_room is not None:
            try:
                await self.worker_room.disconnect()
            except Exception as e:
                logger.debug("Error disconnecting worker room: %s", e)
            self.worker_room = None

        if self.local_player is not None:
            self.local_player.close()

        if hasattr(self.stt, "aclose"):
            await self.stt.aclose()
        if hasattr(self.tts, "aclose"):
            await self.tts.aclose()
        if hasattr(self, "tts_manager") and hasattr(self.tts_manager, "aclose"):
            await self.tts_manager.aclose()
        self.display.show_info("Pipeline stopped cleanly.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhantomFlow Speech & Audio Pipeline (STT Saaras + TTS Bulbul)"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "livekit", "direct"],
        default="auto",
        help="Pipeline mode: 'livekit' (WebRTC room), 'direct' (local mic to Sarvam), or 'auto'",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Sarvam STT language code: 'auto' (default), 'hi-IN', 'en-IN'",
    )
    parser.add_argument(
        "--stream-type",
        default="fast",
        choices=["fast", "balanced", "simulated"],
        help="Sarvam STT stream type: 'fast' (default for low latency) or 'balanced'",
    )
    parser.add_argument(
        "--mode-type",
        default="codemix",
        choices=["codemix", "transcribe", "translate", "verbatim", "translit"],
        help="Sarvam STT mode: 'codemix' (default for Hinglish), 'transcribe', etc.",
    )
    parser.add_argument(
        "--tts-speaker",
        default=None,
        help="Sarvam Bulbul TTS speaker: 'shubh' (default), 'anushka', 'vidya', etc.",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable automatic TTS spoken responses",
    )
    parser.add_argument(
        "--room",
        default=None,
        help="LiveKit room name override (default: phantomflow-room)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # Apply CLI overrides to global settings
    if args.language:
        settings.stt_language = args.language
    if args.stream_type:
        settings.stt_stream_type = args.stream_type
    if args.mode_type:
        settings.stt_mode = args.mode_type
    if args.tts_speaker:
        settings.tts_speaker = args.tts_speaker
    if args.room:
        settings.livekit_room_name = args.room

    settings.print_diagnostics()

    selected_mode = args.mode
    if selected_mode == "auto":
        selected_mode = "livekit" if settings.is_livekit_configured else "direct"

    pipeline = SpeechInputPipeline(config=settings, enable_tts=not args.no_tts)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(pipeline.stop()))
        except NotImplementedError:
            pass

    try:
        if selected_mode == "livekit":
            print("🚀 Launching LiveKit WebRTC Audio Pipeline (STT + TTS)...")
            await pipeline.run_livekit_pipeline(room_name=args.room)
        else:
            print("🎤 Launching Direct Microphone Pipeline (STT + TTS)...")
            await pipeline.run_direct_pipeline()
    except KeyboardInterrupt:
        print("\nStopping on user interruption...")
    except ValueError as val_err:
        pipeline.display.show_error(str(val_err))
        print("\n👉 Please configure the missing variables in your `.env` file:")
        print("   SARVAM_API_KEY=your_sarvam_api_key")
        print("   LIVEKIT_URL=wss://your-livekit-server.com")
        print("   LIVEKIT_API_KEY=your_livekit_api_key")
        print("   LIVEKIT_API_SECRET=your_livekit_api_secret\n")
    except Exception as exc:
        pipeline.display.show_error(f"Unexpected pipeline error: {exc}")
        logger.exception("Pipeline failure")
    finally:
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(main())
