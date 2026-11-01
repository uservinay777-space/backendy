# PhantomFlow — UI/UX Design System & Architectural Guidelines

**Product:** The Midnight Phantom Hotline  
**Concept:** *"Midnight Signal Room"* — A high-stakes, retro-futuristic voice operations console inspired by Indian railway central interlocking cabins, night-shift radio communication terminals, and vintage aerospace broadcasting equipment.

---

## 1. Distinctive Design Philosophy (Four Vectors)

In alignment with the `distinctive-frontend` methodology, PhantomFlow avoids generic "AI SaaS" templates (purple gradients, generic chatbot bubbles, safe Inter/Roboto fonts, flat gray cards).

### Vector 1: Typography (Extreme Weight Hierarchy)
- **Display / Headers:** `Space Grotesk` (Weight: 800–900). Tight tracking (`-0.03em`), all-caps technical telemetry styling.
- **Body & Dialogue:** `DM Sans` (Weight: 300–400). High readability for English, Hindi, and Hinglish transcripts.
- **Technical & Latency Metrics:** `IBM Plex Mono` (Weight: 400–600). Strict tabular numbers, precision millisecond displays.

### Vector 2: Color & Theme (Midnight Signal Console)
Harmonious, high-contrast, military/industrial night-shift palette:
- **Base Obsidian:** `#0A0F0D` (Deep night moss background)
- **Console Secondary:** `#121916` (Card and frame backdrop)
- **Terminal Panel:** `#18221E` (Active control surface)
- **Panel Border:** `#28362F` (Industrial bezel lines)
- **Primary Text:** `#F2EFE9` (Warm antique eggshell)
- **Muted Metadata:** `#8D9C92` (Subdued technical labels)
- **Phosphor Green:** `#5BE38C` (Listening state, confirmed booking, verified PNR)
- **Signal Amber:** `#F4A242` (Processing state, Fast Lane routing, warning)
- **Telegraph Teal:** `#59C3C3` (Railway domain, playback active)
- **Signal Scarlet:** `#F2545B` (GhostBarge interruption halt, connection error)

### Vector 3: Motion & Choreography
- **Staggered Console Boot:** Panels power on with sequential entrance animations (`stagger-1` to `stagger-5`).
- **Reactive Waveform:** Real-time canvas waveform visualizer pulsating dynamically based on speech state.
- **GhostBarge Snap Reaction:** When an interruption occurs, the audio waveform instantly collapses, emitting a sharp scarlet pulse and transition into caller-capture mode (< 30ms).
- **Reduced Motion:** Fully complies with `prefers-reduced-motion` media queries.

### Vector 4: Atmospheric Depth & Backgrounds
- **Delicate CRT/Console Grid:** `linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px)` at 40px grid spacing.
- **Layered Illumination:** Ambient radial glows originating behind the central microphone transmitter.
- **Procedural Grain:** Subtle micro-texture overlay eliminating sterile flat digital surfaces.

---

## 2. Voice State Machine

```
              +-----------------------------------+
              |               IDLE                |
              +-----------------------------------+
                                | (Push to Talk / VAD Start)
                                v
              +-----------------------------------+
              |             LISTENING             | <-----+
              +-----------------------------------+       |
                                | (VAD Endpoint Final)    |
                                v                         |
              +-----------------------------------+       |
              |            PROCESSING             |       |
              +-----------------------------------+       |
                                | (First Audio Chunk)     |
                                v                         | (Caller Barge-In)
              +-----------------------------------+       |
              |             SPEAKING              |       |
              +-----------------------------------+       |
                                |                         |
                 (Interruption) |-------------------------+
                                v
              +-----------------------------------+
              |            INTERRUPTED            |
              | (GhostBarge Halt in < 30ms)       |
              +-----------------------------------+
```

---

## 3. Core Console Components

1. **Header & Status Bar:**
   - Identity: `PHANTOMFLOW // SIGNAL-02`
   - Real-time session identifier, network latency ping, and simulation mode toggles.
2. **Main Voice Transmitter (Centerpiece):**
   - High-precision audio canvas waveform with frequency bar simulations.
   - Large physical-feel rotary mic toggle button with glowing telemetry ring.
   - Real-time state indicators (IDLE, LISTENING, PROCESSING, SPEAKING, INTERRUPTED).
3. **Live Hinglish Transcript Stream:**
   - Dual-column or chat log with distinct speaker tags (`OPERATOR`, `PHANTOM-AI`).
   - Partial streaming transcripts in italicized amber; final transcripts in bold eggshell.
   - Language tags (`HI`, `EN`, `HI-EN`).
4. **SwitchSync Language-Switching Visualisation:**
   - Real-time code-switching detection meter (`EN ➔ HI-EN` with confidence rating).
   - Dual-language token ratio bar (e.g. Hindi 58% vs English 42%).
   - Preserved vs amended slot diff highlights (`[PRESERVED]` tags on context-retained keys like `origin` and `destination`; `[AMENDED ⚡]` on modified keys like `travel_date`).
5. **GhostBarge Telemetry:**
   - Visual interruption counter and timestamp delta.
   - Shows cancellation latency from user voice trigger to speaker mute (< 20ms benchmark).
   - Instant audio dump with zero overlap guarantee.
6. **Phantom Fast Lane Dual-Engine Router:**
   - Dynamic architectural path visualization:
     - **Fast Lane (Deterministic)**: Rule/regex match, IRCTC cache query, < 25ms execution, bypasses LLM (0 tokens).
     - **Smart Path (Generative)**: Sarvam streaming LLM reasoning for complex/unstructured queries (< 600ms).
   - Shows decision execution time, trigger rule, and active pipeline indicator.
7. **Latency Profiler & Benchmark Dashboard:**
   - Live stage-by-stage waterfall chart: `STT First Partial`, `STT Final`, `Intent`, `Fast/Smart Task`, and `Bulbul TTS First Audio`.
   - Distinctive 800ms (Primary Target) and 600ms (Midnight Curse) threshold lines.
   - P50, P90, P95 statistical aggregations and sample counters.
8. **Operations Dossier & Session History:**
   - Domain toggle: **IRCTC Railway Dispatch** (Train 12952 Rajdhani, token #SG-904) vs **Doctor Emergency Triage** (Dr. Rajesh Sharma, Cardiology, token #DOC-402).
   - Live call session timers, turn counters, and night-shift call history table with status indicators (`RESOLVED`, `INTERRUPTED`, `DEGRADED`).
9. **Audio & Network Telemetry:**
   - Real-time RTT latency, jitter, packet loss percentage, and 16kHz PCM audio ingest status.
   - Interactive 3G degraded network simulation toggle.
10. **Evaluation Lab Dashboard (Section G):**
    - Recorded-call evaluation progress (42 of 50 target volunteer calls evaluated).
    - Domain Task Completion Rates: Railway (91.7%), Doctor Triage (88.9%), Overall (90.5%).
    - End-to-end Latency metrics: p50 (342ms), p95 (540ms), % calls < 800ms (95.2%), % calls < 600ms (88.1%).
    - Clean vs MUSAN-noisy comparative telemetry bar (Clean 92.8% vs Noisy 85.7%).
    - Interruption robustness success rate (95.2% halted < 20ms and redirected).
    - MUCS 2021 1-hour Hindi-English ASR WER side score (14.8%).
    - Interactive filters for Clean/Noisy, Interrupted, Railway/Doctor, and Network conditions.
    - Explicit provenance tags: `[DEMO BENCHMARK (SIMULATED)]`, `[ACTUAL LAB RUN]`, `[NOT TESTED]`.
11. **Dataset Information & Provenance Panel (Section H):**
    - Compact manifest card displaying dataset name, official source, primary purpose, modality, size, license, split, and attribution requirements for MUCS 2021, Hinglish-TOP, Datameet Railways, MUSAN, Lahaja, ASR-EN-HI-400H, and Volunteer Call Set.

---

## 4. Demo Scenarios

- **Scenario 1:** Normal Hinglish Railway Query (*"Mujhe Bengaluru se Chennai jaana hai Saturday ko"*) — Fast Lane deterministic resolution.
- **Scenario 2:** SwitchSync Mid-Sentence Correction (*"Actually Sunday morning mein"*) — Origin/dest preserved; date amended across language switch.
- **Scenario 3:** GhostBarge Real-Time Interruption (*"Wait, Sunday nahi Monday!"*) — Instant audio cancellation in < 20ms with VAD flush.
- **Scenario 4:** Fast Lane Deterministic Resolution (*"Mera PNR 1234567890 ka status check karke batao"*) — Deterministic 4ms lookup, total 295ms.
- **Scenario 5:** Weak Network / Packet Loss Simulation — 280ms RTT with adaptive jitter compensation.
- **Scenario 6:** Doctor Emergency Triage (*"Mujhe Dr. Sharma ke paas emergency appointment chahiye chest pain ke liye"*) — Clinical triage, Smart Path reasoning, token #DOC-402.

---

## 5. Team Ownership & Integration Contracts

- **Harshini (Me):** System Architect, Core Orchestration, UI/UX Console, Latency Monitoring, and Contract Definitions (`src/phantomflow/core`, `frontend/`).
- **Srishti (Member 2):** Speech & Audio Engineer (`feature/speech-pipeline`). Plugs real STT/TTS streams into `IVoiceClient.onAmplitude`, `onTranscript`, and `GhostBargeEvent`.
- **Vinay (Member 3):** AI Conversation & Backend Engineer (`feature/conversation-engine`). Plugs real intent recognition, SwitchSync slot resolution, and Fast Lane / Smart Path into `IVoiceClient.onSlotsUpdate` and `onRoutingDecision`.

