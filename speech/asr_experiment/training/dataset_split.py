"""Speaker-aware Train/Validation/Test dataset splitting for local ASR experimentation.

Features:
- Prevents speaker overlap between training and validation sets
- Stratifies by language (Hindi, English, Hinglish) and source dataset
- Deterministic and reproducible using configurable random seed
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

try:
    from speech.asr_experiment.preprocessing.load_dataset import ASRSample
except ImportError:
    from preprocessing.load_dataset import ASRSample


def split_dataset_speaker_aware(
    samples: Sequence[ASRSample],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
) -> Tuple[List[ASRSample], List[ASRSample], List[ASRSample]]:
    """Partition ASR samples into Train, Validation, and Test sets without speaker overlap.

    Args:
        samples: Sequence of clean ASRSample objects.
        train_ratio: Fraction for training (default 0.80).
        val_ratio: Fraction for validation (default 0.10).
        test_ratio: Fraction for testing (default 0.10).
        random_seed: Seed for reproducible partitioning.

    Returns:
        Tuple of (train_samples, val_samples, test_samples).
    """
    if not samples:
        return [], [], []

    rng = random.Random(random_seed)

    # 1. Group samples by speaker_id
    speaker_groups: Dict[str, List[ASRSample]] = defaultdict(list)
    unknown_samples: List[ASRSample] = []

    for s in samples:
        spk = s.speaker_id.strip()
        if spk and spk.lower() not in ("unknown", "spk", "none", ""):
            speaker_groups[spk].append(s)
        else:
            unknown_samples.append(s)

    train_samples: List[ASRSample] = []
    val_samples: List[ASRSample] = []
    test_samples: List[ASRSample] = []

    # 2. Partition known speakers
    speakers = list(speaker_groups.keys())
    rng.shuffle(speakers)

    total_known = sum(len(v) for v in speaker_groups.values())
    target_train = int(total_known * train_ratio)
    target_val = int(total_known * val_ratio)

    curr_train = 0
    curr_val = 0

    for spk in speakers:
        spk_samples = speaker_groups[spk]
        count = len(spk_samples)
        if curr_train + count <= target_train or (curr_train == 0):
            train_samples.extend(spk_samples)
            curr_train += count
        elif curr_val + count <= target_val:
            val_samples.extend(spk_samples)
            curr_val += count
        else:
            test_samples.extend(spk_samples)

    # 3. Partition unknown samples individually
    rng.shuffle(unknown_samples)
    u_total = len(unknown_samples)
    u_train_end = int(u_total * train_ratio)
    u_val_end = u_train_end + int(u_total * val_ratio)

    train_samples.extend(unknown_samples[:u_train_end])
    val_samples.extend(unknown_samples[u_train_end:u_val_end])
    test_samples.extend(unknown_samples[u_val_end:])

    # Shuffle final splits
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    return train_samples, val_samples, test_samples

