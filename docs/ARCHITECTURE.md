# PhantomFlow — System Architecture & Integration Specification

**Hackathon:** MSOT Halloween Night Hackathon 2026  
**Track:** 02 — The Midnight Phantom Hotline  
**Target Latency:** Primary: **< 800 ms** | Midnight Curse: **< 600 ms**  

---

## 1. System Topology & Ownership Boundaries

PhantomFlow is architected with strict decoupling between speech hardware/streaming I/O, conversational reasoning, and the asynchronous orchestration pipeline:

```
                  +-------------------------------------------------------------+
                  |                 CALLER (Microphone / Speaker)               |
                  +-------------------------------------------------------------+
                                     | (Audio In)               ^ (Audio Out)
                                     v                          |
+---------------------------------------------------------------------------------------------------------+
| MEMBER 2: Speech & Audio Pipeline (feature/speech-pipeline)                                             |
|                                                                                                         |
|  [ LiveKit Audio Room / WebRTC Ingestion ]                                                              |
|        |                                                                                                |
|        v                                                                                                |
|  [ Silero / LiveKit VAD ] -----> (User Barge-In Signal) ------> [ GhostBarge Halt Event ]               |
|        |                                                                 |                              |
|        v                                                                 v                              |
|  [ Sarvam Saaras Realtime STT (Streaming Transcripts) ]         [ Sarvam Bulbul Flash TTS Playback ]    |
+---------------------------------------------------------------------------------------------------------+
                                     | UserUtterance (Final/Partial)       ^ AudioChunk Stream
                                     v                                     |
+---------------------------------------------------------------------------------------------------------+
| TEAM LEAD: Central Orchestrator & Integration (feature/orchestrator)                                    |
|                                                                                                         |
|  +---------------------------------------------------------------------------------------------------+  |
|  | [ Session & Slot State (SwitchSync) ]       | [ Latency Profiler (time.perf_counter_ns) ]          |  |
|  | - Preserves intact entities across turns    | - Tracks STT, Intent, Task, First Audio              |  |
|  | - Handles mid-sentence language switching   | - Computes p50, p90, p95 benchmarks                  |  |
|  +---------------------------------------------------------------------------------------------------+  |
|                                                                                                         |
|  +---------------------------------------------------------------------------------------------------+  |
|  | [ GhostBarge Cancellation Controller ]                                                            |  |
|  | - Holds per-turn CancellationToken; halts task and suppresses stale chunks on barge-in (< 30ms)   |  |
|  +---------------------------------------------------------------------------------------------------+  |
|                                     |                                                                   |
|                                     v Routing Decision                                                  |
|                        +---------------------------+                                                    |
|                        | Fast Path vs Smart Path?  |                                                    |
|                        +---------------------------+                                                    |
|                           /                     \                                                       |
|                          / (Predictable query)   \ (Complex / Ambiguous)                                |
|                         v                         v                                                     |
+---------------------------------------------------------------------------------------------------------+
| MEMBER 3: AI Conversation & Task Engine (feature/conversation-engine)                                   |
|                                                                                                         |
|  [ Intent Classification & NLU ]                                                                        |
|  - SwitchSync Hinglish token parsing & slot extraction                                                  |
|                                                                                                         |
|  [ FAST LANE ] (< 30ms)                             [ SMART PATH ] (< 400ms TTFT)                       |
|  - Deterministic Python execution                   - Low-latency Streaming LLM                         |
|  - Railway: PNR status, Train schedules             - Complex NLU, symptoms, open consultation          |
|  - Healthcare: Slot booking, Rescheduling           - Yields token stream for early TTS start           |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Component Ownership Matrix

| Component | Owner | Git Branch | Core Interfaces |
| :--- | :--- | :--- | :--- |
| **Central Orchestrator** | **Team Lead** | `feature/orchestrator` | `CentralOrchestrator`, `process_utterance()` |
| **Integration Contracts** | **Team Lead** | `feature/orchestrator` | `src/phantomflow/contracts/` |
| **Session Management** | **Team Lead** | `feature/orchestrator` | `SessionManager`, `SessionState` |
| **Latency Instrumentation** | **Team Lead** | `feature/orchestrator` | `LatencyTracker`, `TurnLatencyRecord` |
| **Interruption Controller**| **Team Lead** | `feature/orchestrator` | `GhostBargeController`, `CancellationToken` |
| **Speech Pipeline (STT/TTS)** | **Member 2** | `feature/speech-pipeline` | `ISpeechPipeline`, Sarvam Saaras & Bulbul |
| **VAD & Audio Interruption** | **Member 2** | `feature/speech-pipeline` | LiveKit VAD, `stop_playback()` |
| **Intent & NLU Engine** | **Member 3** | `feature/conversation-engine` | `IConversationEngine.resolve_intent()` |
| **Fast Lane Task Logic** | **Member 3** | `feature/conversation-engine` | `IConversationEngine.execute_fast_lane()` |
| **Smart Path LLM Engine**| **Member 3** | `feature/conversation-engine` | `IConversationEngine.execute_smart_path_stream()` |

---

## 3. The Three Core USPs

### USP 1: SwitchSync (Adaptive Hinglish & Slot Retention)
- **Problem:** Callers fluidly switch between Hindi and English mid-sentence and frequently correct single details (e.g., *"Mujhe Saturday ke liye train chahiye... actually Sunday"*).
- **Architecture Solution:**
  1. Member 2 passes `UserUtterance` tagged with `LanguageCode.HINGLISH`.
  2. Member 3 extracts entities (`origin`, `destination`, `travel_date`) and flags `is_correction=True`.
  3. `SessionManager.update_slots()` applies a non-destructive merge: updates `travel_date` to `Sunday` while keeping `origin="New Delhi"` and `destination="Mumbai"` completely intact.

### USP 2: GhostBarge (Real-Time Voice Interruption)
- **Problem:** Outdated assistant responses playing over user speech destroy conversational naturalness and increase turn latency.
- **Architecture Solution:**
  1. As soon as Member 2's VAD detects user speech while assistant audio is emitting, it invokes `orchestrator.handle_interruption(session_id)`.
  2. `GhostBargeController` trips the active `CancellationToken` and cancels the running asyncio task.
  3. `speech_pipeline.stop_playback(turn_id)` is awaited immediately to clear hardware audio buffers (< 30ms).
  4. The orchestrator suppresses any lingering audio chunks from being yielded.

### USP 3: Phantom Fast Lane (Dual-Path Latency Architecture)
- **Problem:** Round-tripping every request to an LLM adds 400–700ms of latency, making the 600ms Midnight Curse impossible.
- **Architecture Solution:**
  - **Fast Path:** Deterministic Python logic for predictable queries (PNR lookup, train status, slot confirmations, cancellations). Executes in **< 15ms**.
  - **Smart Path:** Invoked only when queries are ambiguous, open-ended, or require medical reasoning. Streams tokens to Bulbul Flash immediately.

---

## 4. Latency Budget & Measurement Methodology

### Target Latency Budget (End-of-Speech to First Playable Audio)

| Stage | Fast Path Budget | Smart Path Budget | Midnight Curse Target |
| :--- | :--- | :--- | :--- |
| **1. VAD & Speech Endpointing** | 100 ms | 100 ms | 80 ms |
| **2. Saaras Realtime STT Final** | 150 ms | 150 ms | 120 ms |
| **3. Intent Classification & Routing** | 15 ms | 40 ms | 15 ms |
| **4. Task / Fast Lane Execution** | 15 ms | — | 10 ms |
| **5. LLM First-Token Latency (TTFT)** | — | 250 ms | 200 ms |
| **6. Bulbul Flash First Audio Chunk** | 120 ms | 120 ms | 100 ms |
| **Total End-to-End Latency** | **~400 ms** (Target < 800ms) | **~660 ms** (Target < 800ms) | **< 600 ms** |

### Timing Implementation
- All measurements utilize `time.perf_counter_ns()`.
- Zero synthetic or fabricated metrics: `LatencyTracker` logs exact millisecond breakdowns for every turn and calculates statistical percentiles (`p50`, `p90`, `p95`).
