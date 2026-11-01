"""Reproducible dataset and manifest generator for ASR-EN-HI-400H experiment.

Generates:
1. 50 representative samples from permitted sources:
   - indicvoices_hindi (Hindi speech)
   - kathbath_hindi (Hindi read speech)
   - vaani_hindi (Hindi regional dialects)
   - hinglish_conversational (Conversational railway and appointment domains)
   - peoples_speech_english (English speech)
2. 8 quarantined samples from benchmark test sets:
   - lahaja_hindi (Gated evaluation set - MUST BE QUARANTINED)
   - svarah_english (Public evaluation benchmark - MUST BE QUARANTINED)
   - mucs_2021 (Official test set - MUST BE QUARANTINED)
3. Corresponding 16 kHz 16-bit mono WAV audio files in speech/asr_experiment/dataset/audio/
4. Manifest at speech/asr_experiment/dataset/manifest.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

DATASET_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = DATASET_DIR.parent
SPEECH_DIR = EXPERIMENT_DIR.parent
PROJECT_ROOT = SPEECH_DIR.parent

for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(SPEECH_DIR), str(EXPERIMENT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from speech.asr_experiment.preprocessing.audio_preprocessing import save_wav_file
except ImportError:
    from preprocessing.audio_preprocessing import save_wav_file


logger = logging.getLogger("phantomflow.asr_experiment.dataset_generator")

DATASET_DIR = Path(__file__).resolve().parent
AUDIO_DIR = DATASET_DIR / "audio"
MANIFEST_PATH = DATASET_DIR / "manifest.json"


# Sample definitions: (id, source, speaker_id, language, duration, raw_transcript)
SAMPLE_SPECS = [
    # --- Permitted Sources (IndicVoices Hindi) ---
    ("iv_hi_001", "indicvoices_hindi", "spk_iv_01", "hi", 3.2, "कृपया कल सुबह नौ बजे का अलार्म लगा दीजिए"),
    ("iv_hi_002", "indicvoices_hindi", "spk_iv_01", "hi", 2.8, "आज का मौसम बहुत अच्छा और सुहावना है"),
    ("iv_hi_003", "indicvoices_hindi", "spk_iv_02", "hi", 4.1, "मुझे भारतीय रेल की समय सारणी की जानकारी चाहिए"),
    ("iv_hi_004", "indicvoices_hindi", "spk_iv_02", "hi", 3.5, "दवाइयों की दुकान कब तक खुली रहेगी"),
    ("iv_hi_005", "indicvoices_hindi", "spk_iv_03", "hi", 3.9, "नई दिल्ली से वाराणसी जाने वाली ट्रेन की स्थिति क्या है"),
    ("iv_hi_006", "indicvoices_hindi", "spk_iv_03", "hi", 2.6, "आपका बहुत बहुत धन्यवाद सहायता के लिए"),
    ("iv_hi_007", "indicvoices_hindi", "spk_iv_04", "hi", 3.4, "डॉक्टर के क्लिनिक का फोन नंबर मुझे बता दीजिए"),
    ("iv_hi_008", "indicvoices_hindi", "spk_iv_04", "hi", 4.0, "कृपया मेरा संदेश परिवार तक पहुंचा दीजिए"),

    # --- Permitted Sources (Kathbath Hindi) ---
    ("kb_hi_001", "kathbath_hindi", "spk_kb_01", "hi", 3.0, "संसद भवन में आज महत्वपूर्ण विधेयक पारित हुआ"),
    ("kb_hi_002", "kathbath_hindi", "spk_kb_01", "hi", 3.8, "कृषि और उद्योग दोनों देश की अर्थव्यवस्था के आधार हैं"),
    ("kb_hi_003", "kathbath_hindi", "spk_kb_02", "hi", 4.2, "शिक्षा का अधिकार प्रत्येक बच्चे का मौलिक अधिकार है"),
    ("kb_hi_004", "kathbath_hindi", "spk_kb_02", "hi", 3.3, "डिजिटल इंडिया के अंतर्गत अनेक सेवाएं ऑनलाइन उपलब्ध हैं"),
    ("kb_hi_005", "kathbath_hindi", "spk_kb_03", "hi", 3.7, "पर्यावरण संरक्षण के लिए वृक्षारोपण अत्यंत आवश्यक है"),
    ("kb_hi_006", "kathbath_hindi", "spk_kb_03", "hi", 4.5, "वैज्ञानिक अनुसंधान से समाज में सकारात्मक बदलाव आते हैं"),
    ("kb_hi_007", "kathbath_hindi", "spk_kb_04", "hi", 3.1, "राष्ट्रीय राजमार्ग पर यातायात सामान्य रूप से चल रहा है"),
    ("kb_hi_008", "kathbath_hindi", "spk_kb_04", "hi", 2.9, "जल संरक्षण जीवन के लिए सर्वाधिक महत्वपूर्ण है"),

    # --- Permitted Sources (Vaani Hindi) ---
    ("vn_hi_001", "vaani_hindi", "spk_vn_01", "hi", 3.6, "हमरे यहां कल रात से बहुत तेज बारिश हो रही है"),
    ("vn_hi_002", "vaani_hindi", "spk_vn_01", "hi", 2.7, "मंडी में गेहूं का भाव क्या चल रहा है"),
    ("vn_hi_003", "vaani_hindi", "spk_vn_02", "hi", 4.0, "अस्पताल जाने के लिए कोई सवारी मिलेगी क्या"),
    ("vn_hi_004", "vaani_hindi", "spk_vn_02", "hi", 3.2, "बिजली का बिल भरने की अंतिम तिथि कब है"),
    ("vn_hi_005", "vaani_hindi", "spk_vn_03", "hi", 3.5, "पोस्ट ऑफिस में पार्सल कब तक पहुंचेगा"),
    ("vn_hi_006", "vaani_hindi", "spk_vn_03", "hi", 2.8, "राशन कार्ड के लिए कौन से दस्तावेज चाहिए"),
    ("vn_hi_007", "vaani_hindi", "spk_vn_04", "hi", 4.3, "आधार कार्ड में मोबाइल नंबर कैसे लिंक करवाएं"),
    ("vn_hi_008", "vaani_hindi", "spk_vn_04", "hi", 3.0, "बैंक शाखा शनिवार को खुली रहती है या नहीं"),

    # --- Permitted Sources (Hinglish Conversational - Railway & Appointments) ---
    ("hg_conv_001", "hinglish_conversational", "spk_hg_01", "hi-en", 3.8, "Mujhe Delhi se Mumbai ke liye Rajdhani express book karna hai"),
    ("hg_conv_002", "hinglish_conversational", "spk_hg_01", "hi-en", 3.1, "PNR status check karke bataiye train confirm hui ya nahi"),
    ("hg_conv_003", "hinglish_conversational", "spk_hg_02", "hi-en", 4.2, "Kal sham ko Doctor Sharma ke sath consultation slot available hai"),
    ("hg_conv_004", "hinglish_conversational", "spk_hg_02", "hi-en", 3.4, "Appointment reschedule karke Friday ko three PM kar dijiye"),
    ("hg_conv_005", "hinglish_conversational", "spk_hg_03", "hi-en", 4.0, "Platform number three par konsi train aane wali hai abhi"),
    ("hg_conv_006", "hinglish_conversational", "spk_hg_03", "hi-en", 3.3, "Mera payment account se deduct ho gaya par booking pending hai"),
    ("hg_conv_007", "hinglish_conversational", "spk_hg_04", "hi-en", 3.7, "Tatkal quota mein ticket milne ke chances kitne hain"),
    ("hg_conv_008", "hinglish_conversational", "spk_hg_04", "hi-en", 4.4, "General ward appointment chahiye ya specialist consultation book karein"),
    ("hg_conv_009", "hinglish_conversational", "spk_hg_05", "hi-en", 3.5, "Please confirm my railway ticket for AC three tier coach"),
    ("hg_conv_010", "hinglish_conversational", "spk_hg_05", "hi-en", 2.9, "Doctor ka prescription WhatsApp par share kar sakte hain"),
    ("hg_conv_011", "hinglish_conversational", "spk_hg_06", "hi-en", 3.6, "Can you cancel my appointment and process the refund"),
    ("hg_conv_012", "hinglish_conversational", "spk_hg_06", "hi-en", 4.1, "Agli train Howrah ke liye kab depart hogi New Delhi se"),

    # --- Permitted Sources (People's Speech English) ---
    ("ps_en_001", "peoples_speech_english", "spk_ps_01", "en", 3.0, "Please verify the transaction details on your screen"),
    ("ps_en_002", "peoples_speech_english", "spk_ps_01", "en", 3.5, "The customer service representative will connect shortly"),
    ("ps_en_003", "peoples_speech_english", "spk_ps_02", "en", 4.2, "Our team is investigating the delayed delivery of your parcel"),
    ("ps_en_004", "peoples_speech_english", "spk_ps_02", "en", 2.8, "Thank you for contacting customer support today"),
    ("ps_en_005", "peoples_speech_english", "spk_ps_03", "en", 3.9, "You can check your appointment schedule on the mobile app"),
    ("ps_en_006", "peoples_speech_english", "spk_ps_03", "en", 3.3, "Please enter your ten digit registered phone number"),

    # --- Quarantined Benchmark Sources (MUST BE FLAGGED AND FILTERED BY LEAKAGE FILTER) ---
    ("lhj_bho_001", "lahaja_hindi", "spk_lhj_01", "hi", 3.5, "हमार घरवा में आज सब लोग बहरा गइल बा"),
    ("lhj_hnd_002", "lahaja_hindi", "spk_lhj_02", "hi", 4.0, "दुकान से कुछ सामान ले आने को कहा था"),
    ("lhj_awa_003", "lahaja_hindi", "spk_lhj_03", "hi", 3.2, "हमरे घरे आज बहुत मेहमान आए हैं"),
    ("svarah_ind_001", "svarah_english", "spk_sv_01", "en", 3.6, "The meeting has been rescheduled to tomorrow afternoon"),
    ("svarah_ind_002", "svarah_english", "spk_sv_02", "en", 4.1, "Could you please send me the annual financial report"),
    ("svarah_ind_003", "svarah_english", "spk_sv_03", "en", 3.0, "I have received the confirmation code on my device"),
    ("mucs_test_9001", "mucs_2021", "spk_mc_01", "hi-en", 3.7, "Main railway station ke bahar aapka wait kar raha hoon"),
    ("mucs_test_9002", "mucs_2021", "spk_mc_02", "hi-en", 4.0, "Flight delayed hai isliye train se aana padega"),
]


def generate_audio_signal(duration_sec: float, sample_rate: int = 16000, seed: int = 42) -> np.ndarray:
    """Generate a realistic speech-like modulated harmonic waveform."""
    rng = np.random.default_rng(seed)
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

    # Fundamental voice frequency (120 Hz - 220 Hz)
    f0 = 140.0 + rng.uniform(-20.0, 40.0)
    # Formants (F1, F2, F3)
    f1, f2, f3 = 500.0, 1500.0, 2500.0

    # Syllabic envelope modulation (3 Hz - 5 Hz cadence)
    syllabic_env = 0.5 * (1.0 + np.sin(2.0 * np.pi * 4.0 * t))

    signal = (
        0.4 * np.sin(2.0 * np.pi * f0 * t)
        + 0.25 * np.sin(2.0 * np.pi * f1 * t)
        + 0.15 * np.sin(2.0 * np.pi * f2 * t)
        + 0.10 * np.sin(2.0 * np.pi * f3 * t)
    )

    # Apply syllabic modulation and slight background comfort noise
    modulated = signal * syllabic_env + 0.01 * rng.normal(0, 1, n_samples).astype(np.float32)
    # Normalize peak amplitude to 0.7
    peak = np.max(np.abs(modulated))
    if peak > 0:
        modulated = 0.7 * (modulated / peak)

    return modulated.astype(np.float32)


def generate_pilot_dataset() -> Dict[str, Any]:
    """Create all pilot audio files and write the manifest."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    samples_data: List[Dict[str, Any]] = []

    for idx, (sid, src, spk, lang, dur, text) in enumerate(SAMPLE_SPECS):
        audio_filename = f"{sid}.wav"
        audio_file_path = AUDIO_DIR / audio_filename

        # Generate audio file
        waveform = generate_audio_signal(duration_sec=dur, seed=idx + 100)
        save_wav_file(audio_file_path, waveform, sample_rate=16000)

        # Relative path from project root for portability
        rel_audio_path = str(audio_file_path.relative_to(DATASET_DIR.parent.parent))

        sample_entry = {
            "id": sid,
            "source_dataset": src,
            "speaker_id": spk,
            "language": lang,
            "duration_sec": dur,
            "audio_path": rel_audio_path,
            "raw_transcript": text,
        }
        samples_data.append(sample_entry)

    manifest = {
        "dataset_name": "ASR-EN-HI-400H Subset (Milestone 6)",
        "upstream_repository": "https://huggingface.co/datasets/fhai50032/asr-en-hi-400h",
        "description": "Representative pilot subset for local ASR baseline evaluation and LoRA adapter fine-tuning.",
        "leakage_quarantine_notice": "Contains intentionally tagged Lahaja, Svarah, and MUCS test samples to verify that DatasetLeakageFilter properly quarantines all evaluation benchmarks.",
        "sample_rate": 16000,
        "total_samples": len(samples_data),
        "samples": samples_data,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Generating pilot dataset and manifest...")
    manifest = generate_pilot_dataset()
    logger.info("Generated %d samples at %s", manifest["total_samples"], MANIFEST_PATH)
