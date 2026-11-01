# ASR-EN-HI-400H Experimental Local Fine-Tuning Dataset

Official Hugging Face Repository: [fhai50032/asr-en-hi-400h](https://huggingface.co/datasets/fhai50032/asr-en-hi-400h)  
License Designation: **`other` (Multi-source compilation)**  
Format: **Apache Parquet with 16 kHz FLAC audio**  
Dataset Scale: **~366.7 hours, ~198,893 audio samples, ~21.6 GB**

---

## 1. Overview & Source Composition

The ASR-EN-HI-400H compilation aggregates 9 public speech datasets covering English, Hindi, and Hinglish code-switching:

1. **IndicVoices Hindi:** Crowdsourced multi-dialectal conversational Hindi (AI4Bharat, CC BY 4.0).
2. **Kathbath Hindi:** Clean read Hindi prompts across diverse districts (AI4Bharat, CC BY 4.0).
3. **Vaani Hindi:** Spoken dialectal corpus from Indian districts (IISc/Google, CC BY 4.0).
4. **Hinglish Conversational:** Code-switched conversational phrases (CC BY-SA 4.0).
5. **AMI English:** Multiparty meeting speech (CC BY 4.0).
6. **The People's Speech:** Diverse conversational English (MLCommons, CC BY-SA 4.0).
7. **VoxPopuli English:** Parliamentary speech recordings (Meta AI, CC0).
8. **Lahaja Hindi:** Multi-dialectal Hindi accent recordings (AI4Bharat, CC BY-NC-SA 4.0).
9. **Svarah English:** Indian-accented English speech (IIT Bombay, Research Only).

---

## 2. Benchmark Leakage Prevention Policy

> [!IMPORTANT]
> The upstream dataset card identifies **Lahaja** and **Svarah** as public evaluation benchmarks. Furthermore, PhantomFlow actively evaluates **Lahaja (Milestone 5)** and **MUCS 2021 (Milestone 3)**.

To maintain strict scientific integrity and avoid benchmark contamination:
- **Zero Training on Lahaja:** Every sample originating from Lahaja is strictly filtered out prior to training.
- **Zero Training on Svarah:** Svarah recordings are filtered out.
- **Zero Training on MUCS Test Set:** No official MUCS 2021 test audio is ever used during training.
- **Independent Baseline:** All reported Lahaja and MUCS benchmark numbers reflect genuine out-of-domain evaluation without prior exposure.

Full source-by-source eligibility and attribution mapping is maintained in [`source_licenses.json`](./source_licenses.json).

---

## 3. Directory Layout

```
speech/asr_experiment/
├── dataset/
│   ├── README.md               # Dataset overview and leakage policy (this file)
│   ├── source_licenses.json    # Upstream source licenses & leakage eligibility map
│   └── manifest.json           # Filtered training/validation manifest
├── preprocessing/
│   ├── __init__.py
│   ├── load_dataset.py         # Leakage-aware dataset loader & Parquet streamer
│   ├── audio_preprocessing.py  # 16kHz PCM feature extraction & Mel-spectrograms
│   └── transcript_cleaning.py  # Bilingual text cleaning & Unicode normalization
├── training/
│   ├── __init__.py
│   ├── config.yaml             # Training hyperparameters & hardware constraints
│   ├── dataset_split.py        # Speaker-aware Train/Val/Test partitioning
│   └── train.py                # Fine-tuning loop (LoRA/PEFT parameter-efficient)
├── evaluation/
│   ├── __init__.py
│   ├── evaluate_local.py       # Local ASR inference & evaluation
│   └── compare_models.py       # 3-way comparison (Saaras vs. Baseline vs. Fine-tuned)
├── checkpoints/                # Saved model weights and training checkpoints
└── reports/                    # Experiment reports, training curves, and latency metrics
    ├── training_history.csv
    ├── evaluation_results.json
    ├── model_comparison.csv
    ├── experiment_config.yaml
    └── experiment_report.md
```

