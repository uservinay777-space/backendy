# PhantomFlow — Evaluation Protocol & Benchmarking Methodology

**Track 02:** The Midnight Phantom Hotline  
**Competition:** MSOT Halloween Night Hackathon 2026  
**Guiding Principle:** "Build PhantomFlow around measurable performance. The UI must make the actual technical achievement visible to judges."

---

## 1. Overview of Evaluation Framework

PhantomFlow is evaluated across **six formal metrics** designed to measure conversational fidelity, low-latency execution, and operational resilience under real-world Indian telecommunication conditions (code-switching, noise, packet drop, and caller interruptions).

```
                        +---------------------------------------+
                        |        PHANTOMFLOW BENCHMARK          |
                        +---------------------------------------+
                                            |
         +-----------------+----------------+-----------------+-----------------+
         |                 |                                  |                 |
         v                 v                                  v                 v
   [METRIC 1]        [METRIC 2]                         [METRIC 3 & 4]    [METRIC 5 & 6]
 Task Completion    End-to-End Latency                  Resilience        Robustness
   Rate (TCR)     (p50/p95, <800ms, <600ms)          (3G Net & MUSAN)   (GhostBarge & WER)
```

---

## 2. The Six Evaluation Metrics

### METRIC 1 — Task Completion Rate (TCR)
* **Definition:** The percentage of evaluation calls where the user's operational goal is executed with 100% semantic and database accuracy.
* **Formula:**
  $$\text{TCR} = \left( \frac{N_{\text{successful}}}{N_{\text{total\_valid}}} \right) \times 100\%$$
* **Validation Criteria by Domain:**
  * **Railway Interlocking:**
    1. Correct Intent detected (`TRAIN_SCHEDULE_QUERY` or `PNR_STATUS`).
    2. Correct origin station code and destination station code resolved.
    3. Correct travel date interpretation (including mid-sentence SwitchSync corrections).
    4. Deterministic query execution against Datameet static reference.
    5. Correct spoken confirmation response generated.
  * **Doctor Appointments:**
    1. Correct Intent detected (`BOOK_APPOINTMENT` or `EMERGENCY_TRIAGE`).
    2. Correct clinical specialty and physician name resolved.
    3. Accurate date and time slot extracted.
    4. Availability slot lookup completed against mock hospital schedule.
    5. Booking confirmation token issued to caller.
* **Strict Rule:** A fluent, polite conversational response alone does **NOT** count as a completed task if slot extraction or database lookup failed.

---

### METRIC 2 — End-to-End Latency
* **Definition:** Time elapsed from the exact moment the caller ceases speaking (VAD speech endpoint) to the emission of the first playable audio byte of the assistant's speech.
* **Formula:**
  $$T_{\text{E2E}} = T_{\text{First Playable Audio Byte}} - T_{\text{VAD Speech Endpoint}}$$
* **Component Breakdown Waterfall:**
  1. $T_{\text{STT\_Partial}}$: First partial transcript emission.
  2. $T_{\text{STT\_Final}}$: Final speech-to-text endpoint token.
  3. $T_{\text{Intent}}$: Semantic parsing and intent resolution.
  4. $T_{\text{Task}}$: Task execution (Fast Lane deterministic lookup vs Smart Path reasoning).
  5. $T_{\text{LLM\_TTFT}}$: Large language model time-to-first-token (when Smart Path is triggered).
  6. $T_{\text{TTS\_First\_Audio}}$: Text-to-speech first audio chunk synthesis (Bulbul TTS).
* **Evaluation Targets:**
  * **Round 1 Target:** $T_{\text{E2E}} < 800\text{ ms}$ on $\ge 90\%$ of calls.
  * **Midnight Challenge Target:** $T_{\text{E2E}} < 600\text{ ms}$ on Fast Lane queries.
* **Statistical Deliverables:** $\text{p50}$, $\text{p90}$, $\text{p95}$, and percentage meeting the 800ms and 600ms thresholds.
* **Measurement Standard:** High-resolution monotonic timestamps (`performance.now()` in browser, `time.monotonic()` in Python).

---

### METRIC 3 — 3G-Like Network Robustness
* **Objective:** Verify operational stability under high-latency, packet-loss-heavy mobile networks typical of rural Indian railways and night-shift connections.
* **Network Profile Specification:**
  * **Round Trip Time (RTT):** $280\text{ ms} \pm 40\text{ ms}$ jitter.
  * **Packet Loss:** $5.2\%$ random drop.
  * **Bandwidth Cap:** $384\text{ kbps}$ symmetrical.
* **Platform-Specific Shaping Methodology:**
  * *Development Machine (macOS):* Since macOS does not support Linux `tc netem`, native simulation uses the application-level jitter/loss injector in `IVoiceClient.toggleWeakNetwork()`.
  * *Linux Container / CI Environment:* Configured with `tc netem`:
    ```bash
    tc qdisc add dev eth0 root netem delay 140ms 20ms distribution normal loss 5.2%
    ```
* **Success Metric:** Task Completion Rate degradation must not exceed $12\%$ compared to optimal fiber conditions.

---

### METRIC 4 — Noise Robustness (MUSAN Injection)
* **Objective:** Assess speech understanding and task completion under chaotic ambient environments (railway station platform chatter, babble, train horn rumble).
* **Methodology:**
  * Test calls are evaluated in paired conditions: **Clean** vs **Noise-Augmented**.
  * Noise injection uses a curated 200MB subset of **OpenSLR MUSAN (SLR17)**.
  * Noise tracks are mixed with clean volunteer recordings at calibrated Signal-to-Noise Ratios (SNR):
    * $\text{SNR} = 15\text{ dB}$ (Moderate platform chatter).
    * $\text{SNR} = 10\text{ dB}$ (High background noise / train arrival).
* **Reporting:** Clean and noisy results are reported side-by-side.

---

### METRIC 5 — Interruption Robustness (GhostBarge)
* **Objective:** Ensure seamless handling when a caller interrupts ("barges in") while the assistant is mid-utterance.
* **Verification Checklist:**
  1. **Interruption Detection:** Audio activity detected via client/server VAD during `SPEAKING` state.
  2. **Cancellation Latency:** Audio playback buffer immediately flushed in $< 20\text{ ms}$.
  3. **Ingest Buffer Purge:** Outdated assistant speech is canceled; downstream TTS generation is aborted to save compute and bandwidth.
  4. **Context Retention:** Existing multi-turn slots are preserved in SwitchSync; only the newly corrected information is updated.
  5. **Task Completion:** The post-interruption instruction is successfully completed.
* **Formula:**
  $$\text{Interruption Success Rate} = \left( \frac{N_{\text{interrupted\_handled}}}{N_{\text{total\_interrupted}}} \right) \times 100\%$$

---

### METRIC 6 — ASR Word Error Rate (MUCS 2021 Side Metric)
* **Objective:** Benchmark the raw acoustic transcript accuracy of the streaming ASR on Hindi-English code-switched speech.
* **Dataset:** Official test split of **OpenSLR SLR104 (MUCS 2021)** evaluated on a curated, reproducible 1-hour subset.
* **Formula:**
  $$\text{WER} = \left( \frac{S + D + I}{N_{\text{ref}}} \right) \times 100\%$$
  Where $S = \text{Substitutions}$, $D = \text{Deletions}$, $I = \text{Insertions}$, $N_{\text{ref}} = \text{Total Reference Words}$.
* **Text Normalization Rules:** Lowercase conversion, punctuation stripping, consistent Devanagari numerals to Hindi words.
* **Strict Reporting Boundary:** MUCS WER is an acoustic side-metric. It is **never** averaged with task completion or latency metrics.

---

## 3. Benchmark Reporting Standards & Provenance Tagging

To ensure scientific honesty and transparency before judges, every evaluation record and metric displayed in the PhantomFlow UI is tagged with its empirical provenance:

| Provenance Tag | Visual Badge | Definition |
| :--- | :--- | :--- |
| **ACTUAL LAB RUN** | `[ACTUAL LAB RUN]` | Empirically verified metric measured against real volunteer recordings or live streaming API. |
| **DEMO BENCHMARK** | `[DEMO BENCHMARK (SIMULATED)]` | High-fidelity simulated evaluation data used during active development prior to final batch runs. |
| **NOT TESTED** | `[NOT TESTED / PENDING]` | Benchmark category scheduled for execution but not yet completed. |

*Fabrication of production metrics without explicit labeling is strictly forbidden.*
