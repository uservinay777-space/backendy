# AI4Bharat Lahaja Accent Robustness Evaluation Summary Report

**Project:** PhantomFlow (Role 02 — Speech & Audio Engineering)  
**Model Under Test:** Sarvam Saaras Realtime STT (`language='hi-IN'`)  
**Evaluation Dataset:** AI4Bharat Lahaja (Hindi Multi-Dialectal Speech)  
**Licence:** AI4Bharat Research License / CC BY-NC-SA 4.0  
**Timestamp:** 2026-11-01 02:31:41  
**Minimum Sample Threshold:** 3 samples per regional cohort  

---

## 1. Executive Summary Table

| Evaluation Group | Samples | Speakers | Corpus WER | Corpus CER | Macro WER | Success Rate | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Dataset** | **35** | **22** | **0.79%** | **0.18%** | 0.86% | 100.0% | 10.9ms |

### Speaking Scenario Comparison

| Scenario | Samples | Speakers | Corpus WER | Corpus CER | Macro WER | Success Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Extempore Speech** | 16 | 16 | **1.86%** | **0.46%** | 1.88% | 100.0% |
| **Read Speech** | 19 | 19 | **0.00%** | **0.00%** | 0.00% | 100.0% |

### Regional State-wise Performance

| State | Samples | Speakers | Corpus WER | Corpus CER | Status / Sample Size |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Uttar Pradesh** | 7 | 4 | **0.00%** | **0.00%** | Sufficient Data |
| **Bihar** | 6 | 4 | **1.49%** | **0.37%** | Sufficient Data |
| **Madhya Pradesh** | 5 | 3 | **0.00%** | **0.00%** | Sufficient Data |
| **Rajasthan** | 4 | 3 | **2.56%** | **0.56%** | Sufficient Data |
| **Chhattisgarh** | 4 | 2 | **0.00%** | **0.00%** | Sufficient Data |
| **Delhi** | 3 | 2 | **0.00%** | **0.00%** | Sufficient Data |
| **Haryana** | 3 | 2 | **3.23%** | **0.74%** | Sufficient Data |
| **Jharkhand** | 3 | 2 | **0.00%** | **0.00%** | Sufficient Data |

### Native Mother Tongue / Dialectal Substrate Performance

| Native Language | Samples | Speakers | Corpus WER | Corpus CER | Status / Sample Size |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Hindi** | 7 | 4 | **0.00%** | **0.00%** | Sufficient Data |
| **Chhattisgarhi** | 4 | 2 | **0.00%** | **0.00%** | Sufficient Data |
| **Bhojpuri** | 3 | 2 | **2.94%** | **0.77%** | Sufficient Data |
| **Magahi** | 3 | 2 | **0.00%** | **0.00%** | Sufficient Data |
| **Marwari** | 3 | 2 | **3.33%** | **0.70%** | Sufficient Data |
| **Haryanvi** | 3 | 2 | **3.23%** | **0.74%** | Sufficient Data |
| **Awadhi** | 2 | 1 | **0.00%** | **0.00%** | Low Sample Size (< 3) |
| **Maithili** | 2 | 1 | **0.00%** | **0.00%** | Low Sample Size (< 3) |
| **Malvi** | 2 | 1 | **0.00%** | **0.00%** | Low Sample Size (< 3) |
| **Nagpuri** | 2 | 1 | **0.00%** | **0.00%** | Low Sample Size (< 3) |
| **Braj** | 1 | 1 | **0.00%** | **0.00%** | Low Sample Size (< 3) |
| **Bundeli** | 1 | 1 | **0.00%** | **0.00%** | Low Sample Size (< 3) |
| **Punjabi** | 1 | 1 | **0.00%** | **0.00%** | Low Sample Size (< 3) |
| **Khortha** | 1 | 1 | **0.00%** | **0.00%** | Low Sample Size (< 3) |

---

## 2. Key Acoustic & Dialectal Insights

1. **Read vs. Extempore Speech:**
   - Read speech exhibits lower WER due to standard grammar and predictable prosody.
   - Extempore speech displays natural pauses, discourse markers, and regional inflections.
2. **Regional Accent Robustness:**
   - Central Hindi belt states (Uttar Pradesh, Delhi, Madhya Pradesh) demonstrate robust phonetic recognition.
   - Dialectal substrate influence (Bhojpuri, Maithili, Awadhi) is accommodated by Sarvam Saaras Indic acoustic models.
3. **Sample Size Guidance:** Groups with sample counts below the minimum threshold are flagged to avoid over-interpreting individual speaker idiosyncrasies.

---

## 3. Generated Artifact Files
- Predictions CSV: `speech/reports/lahaja_results/predictions.csv`
- Regional Analysis CSV: `speech/reports/lahaja_results/regional_analysis.csv`
- Scenario Comparison CSV: `speech/reports/lahaja_results/scenario_comparison.csv`
- Complete Metrics JSON: `speech/reports/lahaja_results/metrics.json`
- Summary Plaintext: `speech/reports/lahaja_results/summary_report.txt`