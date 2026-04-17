"""Vocabulary — the token space the model predicts over.

VOCAB is the canonical ordering; token id i corresponds to VOCAB[i]. Built
once at import from the train corpus so that every character the model can
ever see has a reserved id.
"""

from __future__ import annotations

from pathlib import Path

_TRAIN = Path(__file__).resolve().parents[2] / "corpus" / "train.txt"


def _build_vocab() -> list[str]:
    text = _TRAIN.read_text(encoding="utf-8")
    return sorted(set(text))


VOCAB: list[str] = _build_vocab()
VOCAB_SIZE: int = len(VOCAB)
VOCAB_INDEX: dict[str, int] = {ch: i for i, ch in enumerate(VOCAB)}
