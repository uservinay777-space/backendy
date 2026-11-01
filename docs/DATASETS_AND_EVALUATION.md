# PhantomFlow — Dataset Manifest & Data Strategy

**Track 02:** The Midnight Phantom Hotline  
**Competition:** MSOT Halloween Night Hackathon 2026  
**System Strategy:** Pretrained Streaming ASR + MUCS 2021 + Hinglish-TOP Semantic Parsing + Real Volunteer Call Evaluation  

---

## 1. Executive Summary & Data Philosophy

PhantomFlow does **NOT** train an Automatic Speech Recognition (ASR) foundation model from scratch. Building and training a production-grade 16 kHz code-switched ASR model within hackathon constraints is inefficient and misaligned with real-time latency objectives.

Instead, our architecture combines:
1. **Pretrained High-Throughput Streaming ASR** (Sarvam Saaras Realtime / Whisper-Streaming via clean replaceable adapters).
2. **Standardized ASR Benchmarking** on OpenSLR MUCS 2021 (SLR104) test split.
3. **Semantic Query & Slot Understanding** grounded in Google Research Hinglish-TOP.
4. **Static Railway Interlocking Data** grounded in Datameet Railways (CC0).
5. **Empirical Task-Completion Evaluation** on an authentic 30–50 volunteer recorded call corpus covering both railway operations and emergency doctor appointments.

---

## 2. Dataset Manifest

| Dataset Name | Official Source / Identifier | Modality | Size / Volume | Licence | Primary Purpose | Project Integration Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MUCS 2021** | OpenSLR SLR104 ([openslr.org/104](https://www.openslr.org/104/)) | 16 kHz Audio + Text | ~89.86h Train, ~5.18h Test | CC BY-SA 4.0 | ASR Code-Switched Word Error Rate (WER) Benchmark | Evaluation subset curated (1-hour reproducible test slice) |
| **Hinglish-TOP** | Google Research ([github.com/google-research-datasets/Hinglish-TOP-Dataset](https://github.com/google-research-datasets/Hinglish-TOP-Dataset)) | Text (Hierarchical Parses) | ~10K Human, ~170K Synthetic | Apache 2.0 | Intent Recognition, Slot Extraction, Code-Switch Grammar | Semantic grammar mapping implemented for Railway & Doctor tasks |
| **Datameet Railways** | Datameet Community ([github.com/datameet/railways](https://github.com/datameet/railways)) | JSON Tabular | ~8,400+ stations, 11,000+ trains | CC0 (Public Domain) | Static reference for routes, stations, and schedules | Integrated as static lookup; clearly labeled non-live reference |
| **MUSAN** | OpenSLR SLR17 ([openslr.org/17](https://www.openslr.org/17/)) | 16 kHz Audio (Noise/Babble) | ~109 hours total | CC BY 4.0 | Controlled background noise injection (speech, music, babble) | Curated 200MB noise profile subset for SNR 10dB/15dB tests |
| **Volunteer Call Set** | Internal PhantomFlow Lab Corpus | 16 kHz Audio + JSON Meta | 42 recorded calls (target: 30–50) | Proprietary / Consented | Primary Task Completion Rate & Latency Benchmark | Active evaluation corpus in Evaluation Lab |
| **Lahaja (Optional)** | AI4Bharat ([huggingface.co/datasets/ai4bharat/Lahaja](https://huggingface.co/datasets/ai4bharat/Lahaja)) | Audio (Accents) | 12.5h (132 speakers, 83 districts) | CC BY-NC 4.0 (Gated) | Hindi regional accent robustness testing | Optional auxiliary benchmark; terms accepted |
| **ASR-EN-HI-400H (Optional)** | fhai50032 ([huggingface.co/datasets/fhai50032/asr-en-hi-400h](https://huggingface.co/datasets/fhai50032/asr-en-hi-400h)) | Audio | ~366.7h mixed | Multi-Source (Check row-level terms) | Experimental acoustic adaptation | Stored in manifest only; not downloaded into Git |

---

## 3. Dataset Specifications & Usage Protocols

### A. MUCS 2021 (Primary ASR Code-Switched Benchmark)
* **Source:** OpenSLR SLR104 (Multilingual and Code-Switching ASR Challenge for Indic Languages).
* **Domain:** Technical tutorial lectures recorded in mixed Hindi and English.
* **Licence:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
* **Usage in PhantomFlow:**
  * Evaluate streaming ASR transcript fidelity using Word Error Rate (WER).
  * Evaluated on a **reproducible 1-hour subset** of the official test split.
  * **Strict Isolation Guardrail:** The test subset is NEVER used for fine-tuning, training, or prompt engineering.
  * **Score Independence:** MUCS WER is reported as an independent speech recognition metric and is NEVER averaged with or substituted for end-to-end task completion rate.

### B. Hinglish-TOP (Semantic Parsing & Slot Extraction)
* **Source:** Google Research Hinglish-TOP (Task Oriented Parsing) Dataset.
* **Structure:**
  * Human-annotated split (~10K utterances with paired compositional trees).
  * Synthetic augmentations (~170K utterances across navigation, messaging, and event scheduling).
* **Licence:** Apache 2.0.
* **Usage in PhantomFlow:**
  * Semantic intent recognition and slot extraction for mixed-language queries.
  * **Application Intent Mapping:**
    * **Railway Domain:** `[IN:GET_TRAIN_SCHEDULE]`, `[SL:ORIGIN]`, `[SL:DESTINATION]`, `[SL:TRAVEL_DATE]`, `[SL:TRAIN_NUMBER]`, `[SL:BERTH_CLASS]`, `[SL:PNR]`.
    * **Doctor Appointments:** `[IN:BOOK_APPOINTMENT]`, `[SL:PHYSICIAN_NAME]`, `[SL:SPECIALTY]`, `[SL:APPOINTMENT_DATE]`, `[SL:PREFERRED_TIME]`, `[SL:SEVERITY_LEVEL]`.
  * **Important:** Hinglish-TOP is purely a text semantic resource. It contains NO audio. It does not contain out-of-the-box IRCTC railway routes; we mapped application-specific schemas using its compositional annotation style.

### C. Pretrained ASR Streaming Service
* **Primary Integration:** Sarvam Saaras Realtime WebSocket API (streaming 16 kHz PCM chunks).
* **Fallback / Alternate Adapter:** Whisper-Streaming / FastConformer via modular `ISpeechPipeline` contract.
* **Requirements:**
  * Continuous chunk streaming (< 100ms chunk latency).
  * Real-time partial hypothesis emission for live UI transcripts.
  * Robust punctuation, Devanagari/Latin mixed-script transcription, and code-switch boundary stability.
  * Clean adapter isolation: Swapping the ASR provider requires updating only the speech adapter, leaving orchestrator and UI untouched.

### D. Datameet Railways (Static Timetable Reference)
* **Source:** Datameet Community Indian Railways Data Repository.
* **Files Used:** `trains.json`, `stations.json`, `schedules.json`.
* **Licence:** CC0 1.0 Universal (Public Domain Dedication).
* **Usage:** Deterministic station identification (e.g., `SBC` = Bangalore City, `MAS` = Chennai Central), train numbers (e.g., `12951/12952` Rajdhani Express), and scheduled transit hours.
* **Truth in Advertising:** This is static reference data. The UI and backend explicitly display: `STATIC REFERENCE SCHEDULE — NOT LIVE PNR RUNNING STATUS`.

### E. MUSAN (Controlled Noise Robustness)
* **Source:** OpenSLR SLR17 (MUSAN: A Music, Speech, and Noise Corpus).
* **Licence:** CC BY 4.0.
* **Usage:** Audio degradation testing at Signal-to-Noise Ratios (SNR) of 15 dB and 10 dB.
* **Noise Categories:** Technical babble (background passenger chatter), ambient station rumble, and acoustic reverberation.

---

## 4. Volunteer Recorded Call Evaluation Corpus (30–50 Calls)

Public benchmarks do not reflect realistic multi-turn Indian voice operations with mid-sentence corrections (*"Wait, Sunday nahi Monday!"*) or simultaneous background noise.

We created an empirical test suite of **42 volunteer-recorded calls** (scaling to 50):
* **Informed Consent:** All volunteers provided documented verbal/written consent for hackathon research evaluation. Zero personally identifiable information (PII) is collected; all PNRs, patient names, and addresses are purely fictional.
* **Audio Characteristics:** 16 kHz, single-channel WAV/Opus, natural phone microphone captures.
* **Speech Diversity:** Native Hindi speakers, native South-Indian speakers speaking English/Hindi, balanced genders, natural disfluencies (hesitations, pauses, self-corrections).

### Test Call Schema

```json
{
  "call_id": "PF-CALL-019",
  "domain": "railway",
  "language_pattern": "hi-en",
  "volunteer_consent": true,
  "audio_clean_path": "audio/eval/clean/PF-CALL-019.wav",
  "audio_noisy_path": "audio/eval/musan_15db/PF-CALL-019.wav",
  "reference_transcript": "Mujhe Bengaluru se Chennai train check karni hai Saturday morning actually Sunday morning mein",
  "intended_request": "Book morning train from Bangalore to Chennai on Sunday",
  "expected_intent": "TRAIN_SCHEDULE_QUERY",
  "expected_slots": {
    "origin": "Bengaluru (SBC)",
    "destination": "Chennai (MAS)",
    "travel_date": "Sunday, Morning"
  },
  "switchsync_markers": [
    {
      "token_index": 7,
      "detected_switch": "hi -> en",
      "preserved_slots": ["origin", "destination"],
      "amended_slots": ["travel_date"]
    }
  ],
  "interruption_markers": {
    "has_interruption": true,
    "interruption_timestamp_sec": 4.2,
    "barge_in_text": "Sunday nahi Monday!",
    "expected_recovery_slot": "Monday"
  },
  "network_condition": "OPTIMAL",
  "noise_condition": "CLEAN"
}
```

---

## 5. Data Hygiene & Security Guardrails

1. **Large Binary Files Stay Outside Git:** Audio files (`.tar.gz`, `.wav`, `.mp3`) are excluded via `.gitignore`. Only checksums, manifests, and synthetic fixtures are committed.
2. **Train/Test Leakage Elimination:** The 1-hour MUCS test subset and the 42 volunteer test calls are strictly write-protected from any model training scripts.
3. **Reproducibility:** All evaluation runs use fixed pseudo-random seeds (`seed=2026`) and explicit pipeline configuration hashes.
