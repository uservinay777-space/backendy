# PhantomFlow — Team Integration Guide & Developer Contracts

This document provides clear integration instructions for **Member 2** and **Member 3**.

---

## 1. Guide for Member 2 — Speech & Audio Engineer

**Branch:** `feature/speech-pipeline`  
**Location:** `src/phantomflow/speech/`  

### Your Responsibilities
1. Set up LiveKit Agents audio room ingestion.
2. Integrate **Sarvam Saaras Realtime STT** for low-latency streaming Hinglish transcription.
3. Integrate **Sarvam Bulbul Flash** for fast streaming TTS.
4. Implement Voice Activity Detection (VAD) and GhostBarge interruption handling.

### The Contract You Must Implement
You must implement the `ISpeechPipeline` protocol located in `phantomflow.contracts.audio_contracts`:

```python
from typing import AsyncIterator
from phantomflow.contracts.common import UserUtterance
from phantomflow.contracts.audio_contracts import AudioChunk, AudioInputChunk, ISpeechPipeline

class SpeechPipeline(ISpeechPipeline):
    """Production implementation of Speech & Audio Pipeline."""

    async def stream_transcribe(
        self,
        audio_stream: AsyncIterator[AudioInputChunk],
    ) -> AsyncIterator[UserUtterance]:
        """Stream live audio chunks to STT (Sarvam Saaras Realtime) and yield transcribed utterances.

        - Consume incoming live PCM AudioInputChunk frames.
        - Send chunks to Sarvam Saaras Realtime WebSocket.
        - Yield partial UserUtterance (is_final=False) for early latency tracking & language detection.
        - Yield final UserUtterance (is_final=True) when VAD endpoints the speech turn.
        - Always attach detected LanguageCode (HI, EN, or HINGLISH).
        """
        ...

    async def synthesize_stream(
        self,
        text_generator: AsyncIterator[str],
        turn_id: str,
    ) -> AsyncIterator[AudioChunk]:
        """Stream synthesized audio chunks from Sarvam Bulbul Flash.

        - Consume incoming text tokens from text_generator immediately.
        - Send chunks to Sarvam Bulbul WebSocket streaming endpoint.
        - Yield AudioChunk objects as soon as first audio bytes return.
        - Check if turn_id has been stopped; abort if stopped.
        """
        ...

    async def stop_playback(self, turn_id: str) -> None:
        """GhostBarge Action: Immediately flush speaker buffer and cancel active TTS stream."""
        ...
```

### Feeding Transcripts to the Orchestrator
When Sarvam Saaras delivers partial or final transcripts via `stream_transcribe`:
```python
# Member 2's audio worker loop:
async for utterance in speech_pipeline.stream_transcribe(live_audio_stream):
    if not utterance.is_final:
        # Delivers partial transcript to orchestrator for early latency tracking & language context
        await orchestrator.handle_partial_utterance(session_id, utterance)
    else:
        # Delivers final transcript to orchestrator and streams synthesized audio to room
        async for audio_chunk in orchestrator.process_utterance(session_id, utterance):
            await livekit_room.local_participant.publish_audio(audio_chunk.data)
```

### Triggering GhostBarge on User Interruption
When your VAD detects user voice during assistant playback:
```python
# Call orchestrator interruption handler immediately (< 20ms)
await orchestrator.handle_interruption(session_id)
```

---

## 2. Guide for Member 3 — AI Conversation & Backend Engineer

**Branch:** `feature/conversation-engine`  
**Location:** `src/phantomflow/conversation/`  

### Your Responsibilities
1. Implement Hinglish Intent Recognition & Slot Extraction supporting **SwitchSync**.
2. Implement **Phantom Fast Lane** deterministic handlers for standard Railway and Doctor queries.
3. Implement **Smart Path** streaming LLM reasoning for complex/unstructured requests.

### The Contract You Must Implement
You must implement the `IConversationEngine` protocol located in `phantomflow.contracts.conversation_contracts`:

```python
from typing import Any, AsyncIterator, Dict, Optional
from phantomflow.contracts.common import Domain, RoutingPath, UserUtterance
from phantomflow.contracts.conversation_contracts import (
    IConversationEngine,
    IntentResult,
    TaskResponse,
)

class ConversationEngine(IConversationEngine):
    """Production implementation of Conversation & Task Engine."""

    async def resolve_intent(
        self,
        utterance: UserUtterance,
        session_context: Dict[str, Any],
    ) -> IntentResult:
        """Classify user intent and extract slots from Hinglish text.

        - Detect domain (Domain.RAILWAY, Domain.HEALTHCARE, Domain.GENERAL).
        - Extract slots (e.g. {'pnr': '1234567890'} or {'travel_date': 'Sunday'}).
        - Set is_correction=True if caller amended previous info ('Actually Sunday').
        - Set preferred_routing=RoutingPath.FAST if predictable (< 30ms).
        - Set preferred_routing=RoutingPath.SMART if complex reasoning is needed.
        """
        ...

    async def execute_fast_lane(
        self,
        intent: IntentResult,
        session_context: Dict[str, Any],
    ) -> Optional[TaskResponse]:
        """Execute deterministic Python logic without an LLM.

        - Return TaskResponse with spoken text and slots_to_update.
        - Return None if query requires LLM fallback.
        - Target latency: < 30ms.
        """
        ...

    async def execute_smart_path_stream(
        self,
        intent: IntentResult,
        session_context: Dict[str, Any],
    ) -> AsyncIterator[str]:
        """Stream tokens from LLM API with prompt and session history."""
        ...
```

---

## 3. How to Validate Locally

Run the complete test suite:
```bash
source .venv/bin/activate
pytest -v
```

Run the end-to-end integration demo:
```bash
python -m phantomflow.main
```
