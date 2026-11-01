# PhantomFlow — Member 3 (AI Conversation & Railway Engine) API Contract

**Owner:** Member 3 — AI Conversation & Backend Engineer  
**Service:** Node.js / Express Conversation Engine  
**Default Base URL:** `http://localhost:5001`  
**Primary TTS:** Sarvam AI — Bulbul v4 Flash (`bulbul:v4-flash`)  
**Optional TTS:** ElevenLabs Flash v2.5 (`eleven_flash_v2_5`)  
**Railway Data:** RailKit SDK (`railkit` via https://railkit.in)  

---

## 1. Overview & Architectural Role

Member 3 delivers the **AI Conversation, Railway Service & Voice Engine** for PHANTOMFLOW:
1. **Hinglish Intent Recognition (10 Intents):** `railway_search` (`TRAIN_SEARCH`), `train_status` (`TRAIN_STATUS`), `pnr_status` (`PNR_STATUS`), `train_info` (`TRAIN_INFO`), `station_status` (`STATION_STATUS`), `fare_lookup` (`FARE_LOOKUP`), `seat_availability` (`SEAT_AVAILABILITY`), `timetable` (`TIMETABLE`), `railway_information` (`GENERAL_RAILWAY_QUERY`), and `unknown` (`UNKNOWN`).
2. **Entity Extraction & Canonical Station Normalization:** Extracts `origin`, `destination`, `date`, `time`, `trainNumber`, `pnr`, `travelClass`, and `quota` across English, Hindi (Devanagari), and Hinglish with automatic station code resolution (e.g. Bangalore -> `SBC`, New Delhi -> `NDLS`).
3. **SwitchSync Context Retention:** Non-destructive slot merging across multi-turn dialogues (e.g., updating travel date to "Sunday" retains origin and destination).
4. **Phantom Fast Lane (< 30ms):** Direct deterministic dispatch to verified railway operations without LLM overhead.
5. **Real Railway Data via RailKit SDK:** Live PNR status, live train tracking, seat availability, timetables, and fares. **Never fabricates fake railway data in production.**
6. **Smart Path (LLM Fallback):** Safe, anti-hallucination conversational reasoning for open-ended or ambiguous questions.
7. **Natural Voice TTS (Sarvam AI & ElevenLabs):** Pre-synthesis text normalization with GhostBarge interruption support.

---

## 2. HTTP REST Endpoints

### A. Process Message Turn
**Endpoint:** `POST /api/conversation/message`  
**Description:** Consumes a user utterance text with a session ID, processes context and routing, invokes the verified railway service, and returns the response with separated railway facts and latency metrics.

#### Request Headers
```http
Content-Type: application/json
```

#### Request Payload
```json
{
  "sessionId": "session_001",
  "text": "Train 12301 ka running status kya hai?"
}
```

#### Success Response (`200 OK`)
```json
{
  "success": true,
  "sessionId": "session_001",
  "intent": "train_status",
  "language": "hinglish",
  "entities": {
    "origin": null,
    "destination": null,
    "date": null,
    "time": null,
    "trainNumber": "12301",
    "pnr": null,
    "travelClass": null,
    "quota": null
  },
  "missingFields": [],
  "processingPath": "fast",
  "response": "Train 12301 (Rajdhani Express) abhi Kanpur ke paas hai aur right time par chal rahi hai.",
  "responseText": "Train 12301 (Rajdhani Express) abhi Kanpur ke paas hai aur right time par chal rahi hai.",
  "railwayData": {
    "source": "railkit",
    "isLive": true,
    "retrievedAt": "2026-11-01T04:00:00.000Z",
    "data": {
      "trainNumber": "12301",
      "trainName": "Howrah Rajdhani Express",
      "currentLocation": "Kanpur Central",
      "delayMinutes": 0
    }
  },
  "metadata": {
    "language": "hinglish",
    "provider": "sarvam"
  },
  "latencyMs": 14.8
}
```

---

## 3. RailKit Indian Railway Operations

All real railway data operations are executed inside `src/services/railwayService.js`:

| Operation | SDK Method | Description |
| :--- | :--- | :--- |
| **PNR Status** | `checkPNRStatus(pnr)` | 10-digit PNR booking status, coach, berth, and chart status. |
| **Train Info** | `getTrainInfo(trainNumber)` | 5-digit train route, schedule, and operating days. |
| **Live Tracking** | `trackTrain(trainNumber, date)` | Real-time ordered timeline, delays, and current location (`isLive: true`). |
| **Upcoming at Station**| `liveAtStation(stationCode, hours)` | Upcoming trains at a station in the next 2, 4, or 8 hours. |
| **Train Search** | `searchTrainBetweenStations(...)`| Direct trains between source and destination stations. |
| **Seat Availability** | `getAvailability(...)` | Real-time seat availability by coach (`3A`, `2A`, `SL`, etc.) and quota. |
| **Fare Lookup** | `fareLookup(...)` | Total fare and breakdown (base fare, superfast, GST). |
| **Station Info** | `stationByCode(stationCode)` | Station name, division, and coordinates. |
| **Station Search** | `stationsByName(name)` | Matches station codes by partial name. |
| **Timetable** | `trainTimetableAtStation(...)` | Station schedule for today, yesterday, or tomorrow. |

### Standard Data Envelope
```json
{
  "success": true,
  "data": {},
  "source": "railkit",
  "isLive": true,
  "retrievedAt": "2026-11-01T04:00:00.000Z"
}
```

### Standard Error Envelope (When Unconfigured or Failed)
```json
{
  "success": false,
  "error": {
    "code": "RAILWAY_API_UNCONFIGURED",
    "message": "Railway service key is not configured. Please set RAILKIT_API_KEY in .env to retrieve verified railway information."
  },
  "source": "railkit",
  "retrievedAt": "2026-11-01T04:00:00.000Z"
}
```

---

## 4. Voice TTS Engine (Sarvam AI & ElevenLabs)

### A. JavaScript Speech Adapter Interface
Located at [`src/speech/speechAdapter.js`](file:///Users/vinayraj/Desktop/phantom/backend/src/speech/speechAdapter.js):

```javascript
const speechAdapter = require('./speech/speechAdapter');

// 1. Generate Voice Audio (routes to Sarvam AI or ElevenLabs)
const result = await speechAdapter.generateSpeech({
  sessionId: 'session_001',
  turnId: 'turn_101',
  text: 'Train number 12301 subah 6 baje New Delhi se depart hogi.',
  language: 'hi-IN',
});

// 2. Real-Time Streaming Audio Chunks
for await (const chunk of speechAdapter.generateSpeechStream({ sessionId, turnId, text })) {
  liveKitAudioSource.write(chunk.data);
}

// 3. GhostBarge User Interruption (Cancels active turn immediately in < 5ms)
speechAdapter.cancelSpeech({ sessionId, turnId: 'turn_101' });
```

### B. Text Normalization (`src/services/tts/textNormalizer.js`)
Pre-processes spoken text prior to TTS:
- 5-digit train numbers (e.g. `12301` -> `1 2 3 0 1`)
- 10-digit PNRs (e.g. `1234567890` -> `1 2 3 4 5 6 7 8 9 0`)
- Currency: `₹1250` -> `1250 Rupees`
- Times: `06:00 AM` -> `subah 6 baje`, `02:30 PM` -> `dopahar 2:30`
- Delays: `20m late` -> `20 minutes late`

---

## 5. HTTP Speech Endpoints

### 1. `POST /api/speech/synthesize`
Generates base64 audio and microsecond telemetry (or streams raw audio with `?format=audio`).

### 2. `POST /api/speech/cancel`
Aborts in-flight audio generation for GhostBarge barge-in.

### 3. `GET /api/speech/telemetry`
Returns rolling metrics: `timeToFirstAudioChunkMs`, `totalAudioGenerationMs`, and cancellation logs.
