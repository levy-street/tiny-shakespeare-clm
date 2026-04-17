"""The `predict` half of the two-function model API.

`predict(state)` returns a list of natural-log-probabilities over VOCAB.
Exponentiated, the vector must be a valid probability distribution: every
entry in [0, 1] and the sum in [0.999, 1.001]. The harness enforces this
on every call.

Baseline behavior: return unigram log-probabilities computed once from the
train corpus at import time. This is the reference distribution; any
state-conditional logic the optimizer adds must beat it on BPC.

When the logic grows beyond what fits comfortably in one file, split this
into a `predict/` directory module (e.g. `predict/base.py`,
`predict/bigram.py`, `predict/compose.py`) with `predict` re-exported from
`predict/__init__.py`.
"""

from __future__ import annotations

import math
from pathlib import Path

from .state import ModelState
from .vocab import VOCAB_INDEX, VOCAB_SIZE

_TRAIN = Path(__file__).resolve().parents[2] / "corpus" / "train.txt"


def _unigram_logprobs() -> list[float]:
    text = _TRAIN.read_text(encoding="utf-8")
    counts = [0] * VOCAB_SIZE
    for ch in text:
        counts[VOCAB_INDEX[ch]] += 1
    total = sum(counts)
    return [math.log(c / total) for c in counts]


_UNIGRAM_LOGPROBS: list[float] = _unigram_logprobs()


def predict(state: ModelState) -> list[float]:
    return list(_UNIGRAM_LOGPROBS)
