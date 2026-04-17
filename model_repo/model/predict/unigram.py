"""Base unigram log-probability distribution.

Computed once at import time from the train corpus as a prior. All later
layers add log-biases on top of this base.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

_TRAIN = Path(__file__).resolve().parents[3] / "corpus" / "train.txt"


def _unigram_logprobs() -> list[float]:
    text = _TRAIN.read_text(encoding="utf-8")
    counts = [0] * VOCAB_SIZE
    for ch in text:
        counts[VOCAB_INDEX[ch]] += 1
    total = sum(counts)
    return [math.log(c / total) for c in counts]


UNIGRAM_LOGPROBS: list[float] = _unigram_logprobs()
