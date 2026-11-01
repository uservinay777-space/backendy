# MUSAN: A Corpus of Music, Speech, and Noise

Official OpenSLR Repository: [OpenSLR 17](https://openslr.org/17/)  
Licence: **Creative Commons CC0 (Public Domain)**  
Authors: David Snyder, Guoguo Chen, Daniel Povey (Interspeech 2015)

---

## 1. Overview & Attribution

MUSAN is an acoustic noise corpus designed for training and evaluating speech recognition (ASR) and speaker identification models under realistic environmental noise conditions.

### Official Citation & Attribution

```bibtex
@article{snyder2015musan,
  title={MUSAN: A Corpus of Music, Speech, and Noise},
  author={Snyder, David and Chen, Guoguo and Povey, Daniel},
  journal={arXiv preprint arXiv:1510.08484},
  year={2015}
}
```

*Attribution Notice:* MUSAN is dedicated to the public domain under CC0 1.0 Universal.

---

## 2. Noise Categories

The PhantomFlow noise robustness benchmark evaluates three distinct acoustic categories:

1. **`noise/` (Technical & Ambient Noise):**
   - HVAC/Office air conditioning hum, street traffic, automobile interior rumble, white/pink noise.
   - Used to simulate caller environments on moving transport (train platforms, autorickshaws, streets).

2. **`music/` (Musical Backgrounds):**
   - Instrumental piano, acoustic guitar, western/classical melodies.
   - Used to simulate hold music, radios, or retail store background music.

3. **`babble/` (Multi-Speaker Babble):**
   - Overlapping cocktail party speech chatter, café murmur, crowded waiting rooms.
   - The most challenging condition for ASR because background speech shares spectral characteristics with the caller.

---

## 3. Directory Layout

```
speech/datasets/musan/
├── README.md               # This documentation and attribution file
├── download_musan.py       # Selective downloader for OpenSLR 17 audio subsets
├── raw/                    # Raw noise sources (16kHz 16-bit PCM mono)
│   ├── noise/
│   │   ├── traffic_street_01.wav
│   │   └── ambient_office_fan_01.wav
│   ├── music/
│   │   └── instrumental_piano_01.wav
│   └── babble/
│       └── cocktail_babble_01.wav
└── processed/              # Mixed noisy speech outputs generated during evaluation
    ├── mucs_test_hien_0001_noise_15dB.wav
    ├── mucs_test_hien_0001_music_10dB.wav
    └── mucs_test_hien_0001_babble_5dB.wav
```

---

## 4. Signal-to-Noise Ratio (SNR) Policy

The evaluation tests each clean utterance across three standard SNR levels:
- **15 dB SNR (Mild Noise):** Standard home or office setting with soft background murmur.
- **10 dB SNR (Moderate Noise):** Walking on a city street, nearby traffic, moving train coach.
- **5 dB SNR (Heavy Noise / Stress Test):** Loud train station platform, crowded café, noisy road.

The mixing formula uses standard signal power scaling:
$$SNR_{dB} = 10 \log_{10}\left(\frac{P_{speech}}{P_{noise}}\right)$$

---

## 5. Selective Download Script

To download additional raw noise archives from OpenSLR 17 without pulling the full 11 GB corpus:

```bash
python -m speech.datasets.musan.download_musan --category noise
```

