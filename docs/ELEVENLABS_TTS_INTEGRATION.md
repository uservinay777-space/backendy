# PHANTOMFLOW — ElevenLabs Text-to-Speech (TTS) Integration Guide

**Owner:** Senior Backend & Voice AI Engineer (Member 3)  
**Status:** Implemented & Verified  
**Default Port:** `5001`  

---

## 1. Overview & Architectural Role

PHANTOMFLOW is a real-time Hinglish voice assistant designed for Indian Railway queries. The system pipeline follows a strict unidirectional stream:

```
User Microphone 
     │
     ▼
Streaming STT (Saaras / Web Speech)
     │
     ▼
Conversation Engine (Intent & Entity Parsing, SwitchSync Context)
     │
     ▼
Generated Response Text
     │
     ▼
Speech Adapter (ElevenLabs Streaming TTS)
     │
     ▼
Audio Playback & LiveKit Stream (Speaker)
```

### Key Architectural Constraints
1. **Decoupled Speech Adapter:** The Conversation Engine does **not** directly depend on ElevenLabs. Any future TTS provider (e.g., Sarvam Bulbul Flash, Cartesia, or OpenAI TTS) can be swapped seamlessly without touching dialogue management.
2. **GhostBarge Interruption:** When the caller speaks while the assistant is talking, audio playback and upstream network synthesis must be halted immediately (< 5ms) to suppress stale audio.
3. **Low Latency (< 800ms Target):** Audio is generated as a continuous stream of chunks using `optimize_streaming_latency=3` over ElevenLabs' `eleven_multilingual_v2` model.

---

## 2. Speech Adapter Interface & Contracts

The Speech Adapter is located at [`backend/src/speech/speechAdapter.js`](file:///Users/vinayraj/Desktop/phantom/backend/src/speech/speechAdapter.js) and exposes two core asynchronous methods:

### A. Speech Synthesis (`generateSpeech`)
```javascript
const speechAdapter = require('./speech/speechAdapter');

const result = await speechAdapter.generateSpeech({
  sessionId: 'session_user_001',
  turnId: 'turn_101',
  text: 'Bengaluru se Chennai Shatabdi Express subah 6:00 baje depart hoti hai.',
  language: 'hinglish',
  voiceId: '21m00Tcm4TlvDq8ikWAM', // Optional override
  modelId: 'eleven_multilingual_v2', // Optional override
});

console.log(result.success);                   // true
console.log(result.audioBuffer);               // Buffer (MP3 or WAV)
console.log(result.audioMimeType);             // 'audio/mpeg' | 'audio/wav'
console.log(result.metrics.timeToFirstAudioChunkMs); // Time to first audio chunk
console.log(result.metrics.totalAudioGenerationMs);  // Total synthesis time
```

### B. Streaming Synthesis (`generateSpeechStream`)
For WebSockets, LiveKit audio sources, or chunked HTTP streaming:
```javascript
for await (const chunk of speechAdapter.generateSpeechStream({
  sessionId: 'session_user_001',
  turnId: 'turn_101',
  text: 'Aapka train platform number 4 par aayega.',
  language: 'hinglish',
})) {
  // Push chunk.data (Buffer) directly to audio output stream
  console.log(`Chunk ${chunk.chunkIndex}: ${chunk.data.length} bytes`);
}
```

### C. GhostBarge Interruption (`cancelSpeech`)
When the user speaks over the assistant:
```javascript
speechAdapter.cancelSpeech({
  sessionId: 'session_user_001',
  turnId: 'turn_101',
});
```
This triggers [`backend/src/speech/audioBufferManager.js`](file:///Users/vinayraj/Desktop/phantom/backend/src/speech/audioBufferManager.js), which invokes `abortController.abort()` on the active HTTP stream, deletes the turn from the buffer registry, and ensures no subsequent audio chunks are forwarded to the playback device.

---

## 3. Secure Environment Configuration

All credentials and model configurations are read from environment variables and never exposed to the client or checked into version control.

Configured in [`backend/src/config/config.js`](file:///Users/vinayraj/Desktop/phantom/backend/src/config/config.js):

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `TTS_PROVIDER` | `elevenlabs` | Active TTS provider (`elevenlabs` or `mock`). |
| `ELEVENLABS_API_KEY` | *(empty string)* | Secure API Key from ElevenLabs dashboard. |
| `ELEVENLABS_VOICE_ID` | `21m00Tcm4TlvDq8ikWAM` | Voice ID (e.g. Rachel / custom bilingual voice). |
| `ELEVENLABS_MODEL_ID` | `eleven_multilingual_v2` | Multilingual model with native Hindi & English code-mixing. |
| `ELEVENLABS_OPTIMIZE_LATENCY` | `3` | Aggressive streaming latency optimization (0 to 4). |
| `ELEVENLABS_STABILITY` | `0.5` | Voice stability parameter. |
| `ELEVENLABS_SIMILARITY` | `0.75` | Voice similarity boost parameter. |
| `TTS_TIMEOUT_MS` | `5000` | Abort timeout if ElevenLabs API stalls. |

### Setup Example (`backend/.env`):
```bash
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=xi_api_key_xxxxxxxxxxxxxxxxxxxx
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
ELEVENLABS_OPTIMIZE_LATENCY=3
```

---

## 4. Latency Optimization & Profiling

### Metrics Measured:
1. **TTS API Latency (`ttsApiLatencyMs`):** Duration from dispatching the HTTPS request to ElevenLabs until HTTP response headers and status `200 OK` are received.
2. **Time to First Audio Chunk (`timeToFirstAudioChunkMs`):** Duration from request dispatch until the first byte of decodable audio arrives from the stream.
3. **Total Audio Generation Time (`totalAudioGenerationMs`):** Duration until the audio stream terminates (`EOF`).
4. **Audio Playback Start Time (`audioPlaybackStartTime`):** Client-side timestamp when Web Audio API / HTML5 Audio begins emitting sound.

> [!IMPORTANT]
> **Engineering Latency Note:**
> Do **not** claim the voice pipeline achieves the sub-800ms target based solely on TTS latency!  
> Total End-to-End Latency consists of:  
> $\text{Latency}_{\text{E2E}} = \text{STT}_{\text{final}} + \text{NLU / Fast Lane} + \text{LLM}_{\text{first token}} + \text{TTS}_{\text{first chunk}} + \text{Network jitter}$.  
> ElevenLabs streaming with `optimize_streaming_latency=3` achieves a first audio chunk within ~150–250ms under typical conditions, leaving ample budget for STT and Fast Lane (< 30ms).

---

## 5. Resilient Fallback & Error Handling

To guarantee that the voice assistant never fails silently or throws an uncaught exception:
1. **Missing or Invalid API Key:** If `ELEVENLABS_API_KEY` is not set or ElevenLabs returns HTTP 401/429/500, the system automatically falls back to native high-quality speech synthesis via macOS `say` (using Indian English voice `Aman` or Hindi voice `Lekha`) transcoded to 16-bit PCM WAV.
2. **Synthetic PCM Generator:** If running in containerized Linux environments without audio daemons, a valid 44-byte RIFF/WAVE tone generator synthesizes valid audio buffers so the browser/LiveKit player never crashes.
3. **Turn ID Tracking:** Each request is uniquely identified by `turnId`. Audio chunks arriving after a turn has been cancelled are discarded before buffering or playing.

---

## 6. HTTP REST API Endpoints

Mounted in [`backend/src/server.js`](file:///Users/vinayraj/Desktop/phantom/backend/src/server.js) under `/api/speech`:

### `POST /api/speech/synthesize`
Generates speech audio for a text response.

**Request:**
```json
{
  "sessionId": "session_demo_01",
  "turnId": "turn_1730000000000",
  "text": "Bengaluru se Delhi Rajdhani Express kal subah 8 baje ravana hogi.",
  "language": "hinglish"
}
```

**JSON Response (default):**
```json
{
  "success": true,
  "sessionId": "session_demo_01",
  "turnId": "turn_1730000000000",
  "audioBase64": "//uQZ...",
  "audioMimeType": "audio/mpeg",
  "interrupted": false,
  "metrics": {
    "ttsApiLatencyMs": 42.1,
    "timeToFirstAudioChunkMs": 182.4,
    "totalAudioGenerationMs": 350.2,
    "totalBytes": 16384,
    "chunkCount": 8
  }
}
```
*(Query param `?format=audio` directly streams raw binary audio with `Transfer-Encoding: chunked`).*

### `POST /api/speech/cancel`
Halts in-flight audio generation immediately (GhostBarge).

**Request:**
```json
{
  "sessionId": "session_demo_01",
  "turnId": "turn_1730000000000"
}
```

**Response:**
```json
{
  "success": true,
  "cancelled": true,
  "turnId": "turn_1730000000000",
  "sessionId": "session_demo_01",
  "timestamp": 1730421000000
}
```

### `GET /api/speech/telemetry`
Returns rolling metrics of the last 20 audio synthesis events.

---

## 7. Running & Testing Locally

### Run Automated Unit & Integration Tests:
```bash
cd backend
npm test
```
*Executes all 26 tests across Conversation Engine, REST Routes, and Speech Adapter.*

### Run Local Development Server:
```bash
cd backend
npm run dev
```
The server will start at:
- **Web UI & Voice Console:** [http://localhost:5001/](http://localhost:5001/)
- **Health Check:** [http://localhost:5001/health](http://localhost:5001/health)

### Testing in the Browser:
1. Open [http://localhost:5001/](http://localhost:5001/) in Google Chrome or any modern browser.
2. In the **Audio Engine** dropdown, choose:
   - **ElevenLabs Cloud (Default):** Calls `/api/speech/synthesize` to generate streaming speech with precise latency metrics.
   - **Browser Web Speech (Local Fallback):** Direct local synthesis.
3. Click **"Test Voice (ElevenLabs)"** to synthesize a sample Hinglish railway response.
4. Click **"Halt Audio (GhostBarge)"** during playback to verify instant interruption.
