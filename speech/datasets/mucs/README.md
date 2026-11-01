# MUCS 2021 — Multilingual Code-Switching Speech Recognition Dataset

Official OpenSLR Repository: [OpenSLR 104](https://openslr.org/104/)  
Dataset: **Multilingual Code-Switching Speech Recognition (MUCS 2021)**  
Licence: **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**

---

## 1. Overview & Attribution

MUCS 2021 was developed for the Interspeech 2021 Special Session on Multilingual Code-Switching Speech Recognition. The dataset consists of code-switched conversational speech across Indian language pairs:
- **Hindi-English (`hi-en`)**
- Bengali-English (`bn-en`)

### Official Citation & Attribution

```bibtex
@inproceedings{diwan2021multilingual,
  title={Multilingual and Code-Switching ASR Challenges for Low Resource Indian Languages},
  author={Diwan, Anuj and Thomas, Preethi and Jyothi, Preethi and others},
  booktitle={Proc. Interspeech 2021},
  pages={2446--2450},
  year={2021},
  doi={10.21437/Interspeech.2021-1440}
}
```

*Attribution Notice:* This dataset is made available under the CC BY-SA 4.0 licence. In accordance with the license terms, any distribution or modified versions must retain proper attribution and share-alike licensing.

---

## 2. Dataset Split & Hackathon 1-Hour Test Subset Policy

Per PhantomFlow hackathon evaluation policy:
- **Test Set Protection:** The official MUCS test set is used **strictly for evaluation/benchmarking**. It is **never** used for model training, prompt tuning, or parameter optimization.
- **1-Hour Reproducible Test Subset:** A representative 1-hour subset (~3,600 seconds) of Hindi-English conversational speech has been selected and frozen in `manifest.json`.
- **Side-Score Benchmark:** Used as the ground truth benchmark for evaluating the Sarvam Saaras Realtime STT model.

---

## 3. Directory Layout

```
speech/datasets/mucs/
├── README.md               # This documentation and attribution file
├── manifest.json           # Frozen metadata and reference transcripts for the 1-hour test subset
├── download_mucs.py        # Automated downloader and extractor for OpenSLR 104 test set
└── audio/                  # WAV audio files (16kHz, 16-bit mono PCM)
    ├── mucs_test_hien_0001.wav
    ├── mucs_test_hien_0002.wav
    └── ...
```

---

## 4. Downloading the Official OpenSLR 104 Test Subset

To download the official test subset from OpenSLR 104:

### Method A: Automated Download Script
Run the helper script from the workspace:
```bash
python -m speech.datasets.mucs.download_mucs --subset test
```

### Method B: Manual Download via curl / wget
```bash
# Navigate to the dataset directory
cd speech/datasets/mucs

# Download test audio from OpenSLR 104 mirror
curl -L -O https://www.openslr.org/resources/104/Hindi-English-test.zip
# or: wget https://www.openslr.org/resources/104/Hindi-English-test.zip

# Unzip audio into the audio/ folder
unzip -j Hindi-English-test.zip "*/audio/*" -d audio/
```

---

## 5. Running the ASR Evaluation Benchmark

Execute the evaluation benchmark using the real Sarvam Saaras API:

```bash
# Set your Sarvam API Key
export SARVAM_API_KEY="your_actual_sarvam_api_key"

# Run the MUCS 2021 benchmark
python -m speech.evaluation.evaluate_mucs

# Or run from speech directory:
PYTHONPATH=speech python -m evaluation.evaluate_mucs
```

### Options:
- `--limit <N>`: Process only the first N samples (useful for quick spot checks).
- `--concurrency <N>`: Maximum parallel requests to Sarvam API (default: `1` for rate-limit protection).
- `--resume / --no-resume`: Resumes from previous checkpoint without re-querying completed samples.
- `--output-dir <path>`: Directory to save CSV and JSON reports (default: `speech/reports/mucs_results`).

