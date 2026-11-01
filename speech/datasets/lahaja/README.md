# AI4Bharat Lahaja — Hindi Accent & Regional Dialect Dataset

Official Hugging Face Repository: [ai4bharat/Lahaja](https://huggingface.co/datasets/ai4bharat/Lahaja)  
Official GitHub Repository: [AI4Bharat/Lahaja](https://github.com/AI4Bharat/Lahaja)  
Licence: **AI4Bharat Research License / CC BY-NC-SA 4.0 (Research & Non-Commercial Evaluation Only)**  
Authors: AI4Bharat (IIT Madras)

---

## 1. Overview & Attribution

Lahaja is a comprehensive multi-dialectal Hindi speech dataset curated to benchmark ASR acoustic robustness across regional accents, native language substrates, and speaking scenarios.

### Dataset Specifications:
- **Total Duration:** ~12.5 hours of recorded Hindi speech.
- **Speakers:** 132 diverse native speakers from 83 administrative districts across India.
- **Speaking Scenarios:** Both **Read Speech** (scripted sentence prompts) and **Extempore Speech** (spontaneous conversational discourse).
- **Acoustic Diversity:** Native language backgrounds including Awadhi, Bhojpuri, Maithili, Magahi, Braj, Marwari, Bundeli, Malvi, Haryanvi, Nagpuri, and Standard Hindi.

### Official Citation & Attribution

```bibtex
@inproceedings{ai4bharat2023lahaja,
  title={Lahaja: A Multidialectal Speech Dataset for Indian Languages},
  author={AI4Bharat},
  booktitle={Interspeech / AI4Bharat Research},
  year={2023}
}
```

---

## 2. Gated Access & Authentication Procedure

The official Lahaja dataset on Hugging Face is **gated**. Prior to downloading the complete archive, users must authenticate:

### Step 1: Accept Dataset Access on Hugging Face
1. Log in to [Hugging Face](https://huggingface.co).
2. Visit the dataset card: [https://huggingface.co/datasets/ai4bharat/Lahaja](https://huggingface.co/datasets/ai4bharat/Lahaja).
3. Review the access conditions and submit the access request form.

### Step 2: Configure Your HF_TOKEN
Generate a User Access Token with read permissions at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then export it:

```bash
export HF_TOKEN="hf_your_actual_token_here"
```

Or add it to your `.env` file:
```ini
HF_TOKEN=hf_your_actual_token_here
```

> **Security Notice:** The token is read strictly from `HF_TOKEN`. It is never hardcoded, committed to Git, or exposed in output reports.

---

## 3. License Verification & Terms Notice

- **Upstream License Terms:** The dataset is designated for academic research and evaluation purposes.
- **Evaluation-Only Guarantee:** In accordance with the hackathon policy, Lahaja is used **strictly as an evaluation benchmark**. No models in PhantomFlow are trained, fine-tuned, or adapted on Lahaja speech recordings.
- **Discrepancy Check:** Please verify that your usage complies with the specific terms accepted during the Hugging Face gate sign-up. Redistribution of raw audio files outside authorized research platforms is prohibited.

---

## 4. Metadata Schema & Reference Transcript Selection

Each record in `meta-data-lahaja.csv` and `manifest.json` provides:

| Column | Type | Description |
| :--- | :--- | :--- |
| `fname` | `str` | Audio filename (e.g. `lhj_up_lko_001.wav`) |
| `verbatim` | `str` | Exact transcribed speech as spoken, including regional colloquialisms |
| `normalized` | `str` | **Primary reference transcript** with standardized spellings and numbers |
| `text` | `str` | Cleaned text field |
| `scenario` | `str` | `read` (scripted) or `extempore` (spontaneous) |
| `native_state` | `str` | Speaker's home state (e.g., Uttar Pradesh, Bihar, Rajasthan) |
| `native_district` | `str` | Speaker's home district (e.g., Lucknow, Varanasi, Patna) |
| `native_language` | `str` | Speaker's primary mother tongue (e.g., Bhojpuri, Awadhi, Maithili) |
| `sp_id` | `str` | Pseudonymous speaker identifier (e.g., `spk_up_001`) |
| `duration` | `float`| Recording length in seconds |

*Policy on Reference Text:* Per evaluation requirements, the upstream `normalized` field is used as the primary reference for WER and CER calculation. The `verbatim` field is retained in records for linguistic auditing.

---

## 5. Pilot Mode vs. Full Dataset Benchmark

1. **Pilot Mode (Default, 35 Utterances):**
   - Packaged in `speech/datasets/lahaja/audio/` and `manifest.json`.
   - Balanced across 8 states and 22 speakers.
   - Runs out-of-the-box without requiring 1.42 GB download.

2. **Full Dataset Benchmark (~1.42 GB):**
   - Download the full dataset using `download_lahaja.py`:
   ```bash
   python -m speech.datasets.lahaja.download_lahaja --split test
   ```

