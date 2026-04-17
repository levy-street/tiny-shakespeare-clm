"""Word-start bigram bias layer.

At letter_run_len == 1 (we just emitted the first letter of a new word),
bias the second letter given the first. This specializes the bigram
layer for the word-start context, where letter transition distributions
differ substantially from mid-word.

Keys are single lowercase letters (the first letter of the word).
Values are dicts of second-letter → log-bias. Based on prior knowledge
of common English/Shakespearean word beginnings.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# first-letter (lowercase) -> { second-letter: bias }
_SB: dict[str, dict[str, float]] = {
    "t": {"h": 2.0, "o": 1.4, "r": 1.0, "e": 0.9, "a": 0.8, "i": 0.7,
          "u": 0.7, "w": 0.9, "y": 0.3, "'": 0.4},
    "a": {"n": 1.8, "t": 1.0, "s": 1.0, "r": 0.8, "l": 0.9, "b": 0.8,
          "g": 0.8, "f": 0.7, "d": 0.7, "m": 0.7, "c": 0.5, "p": 0.7,
          "w": 0.6, "u": 0.5, "i": 0.5, "h": 0.5, "v": 0.4, "o": 0.3,
          "e": 0.3, "y": 0.4, "x": -0.5, "z": -0.5},
    "w": {"h": 2.5, "i": 1.4, "e": 1.5, "o": 1.2, "a": 1.1, "r": 0.6,
          "y": 0.4, "u": -0.2},
    "s": {"h": 1.5, "o": 1.2, "e": 1.1, "t": 1.4, "i": 1.1, "p": 0.9,
          "u": 0.7, "a": 0.8, "l": 0.7, "c": 0.8, "w": 0.6, "m": 0.6,
          "k": 0.4, "n": 0.4, "q": 0.2, "y": 0.3, "'": 0.3},
    "h": {"a": 1.5, "e": 1.8, "i": 1.6, "o": 1.3, "u": 0.7, "y": 0.3,
          "'": 0.3, "r": -0.3},
    "b": {"e": 1.5, "r": 1.0, "l": 1.0, "u": 1.0, "o": 0.9, "a": 0.9,
          "i": 0.7, "y": 0.6, "'": 0.3, "h": -0.3},
    "c": {"o": 1.5, "a": 1.2, "h": 1.3, "l": 1.0, "u": 0.9, "r": 1.1,
          "e": 0.8, "i": 0.7, "y": 0.4, "'": 0.2},
    "m": {"a": 1.4, "e": 1.5, "o": 1.3, "i": 1.0, "u": 0.8, "y": 1.4,
          "r": 0.3, "'": 0.5},
    "p": {"r": 1.4, "l": 1.1, "a": 1.2, "o": 1.1, "e": 1.1, "i": 1.0,
          "u": 0.8, "h": 0.6, "y": 0.3, "s": 0.3},
    "n": {"o": 1.5, "e": 1.2, "a": 1.0, "i": 0.8, "u": 0.6, "y": 0.3,
          "'": 0.3},
    "f": {"o": 1.5, "a": 1.3, "e": 1.0, "r": 1.4, "l": 1.0, "i": 1.0,
          "u": 0.7, "y": 0.3, "'": 0.2},
    "d": {"o": 1.3, "e": 1.5, "i": 1.3, "a": 1.0, "u": 0.7, "r": 1.0,
          "w": 0.5, "y": 0.4, "'": 0.4},
    "l": {"o": 1.4, "i": 1.3, "a": 1.3, "e": 1.3, "u": 0.5, "y": 0.4,
          "'": 0.3},
    "r": {"e": 1.6, "a": 1.2, "o": 1.3, "i": 1.0, "u": 0.8, "y": 0.3,
          "h": -0.5, "'": 0.2},
    "y": {"o": 1.5, "e": 1.3, "i": 0.5, "a": 0.5, "'": 0.3},
    "g": {"o": 1.4, "e": 1.2, "r": 1.2, "a": 1.2, "i": 0.9, "u": 0.7,
          "l": 1.0, "h": 0.5, "y": 0.2, "'": 0.2},
    "u": {"n": 1.8, "p": 1.4, "s": 0.7, "t": 0.5, "r": 0.3, "'": 0.2,
          "a": 0.3, "e": 0.3, "i": 0.3, "o": 0.3},
    "v": {"i": 1.2, "e": 1.3, "o": 1.0, "a": 1.0, "u": 0.3, "r": 0.2,
          "y": 0.2},
    "e": {"n": 1.2, "v": 1.0, "x": 1.0, "a": 1.0, "r": 0.8, "m": 0.7,
          "l": 0.7, "s": 0.7, "y": 0.5, "t": 0.5, "d": 0.4, "q": 0.3,
          "'": 0.3, "f": 0.3},
    "o": {"f": 2.0, "n": 1.5, "u": 1.3, "v": 1.0, "r": 0.9, "p": 0.7,
          "b": 0.5, "t": 0.4, "w": 0.3, "'": 0.6, "a": 0.2, "e": 0.2,
          "i": 0.2, "o": 0.2},
    "i": {"n": 1.6, "s": 1.4, "t": 1.2, "f": 1.1, "'": 0.9, "m": 0.7,
          "l": 0.5, "d": 0.3, "r": 0.2, "a": 0.3, "e": 0.3, "o": 0.3,
          "u": 0.3},
    "k": {"n": 1.3, "i": 1.2, "e": 0.8, "a": 0.6, "y": 0.2, "'": 0.2},
    "j": {"o": 1.0, "u": 1.0, "e": 0.8, "a": 0.8, "i": 0.7},
    "q": {"u": 3.0, "'": 0.2},
    "z": {"e": 0.5, "a": 0.3},
    "x": {},
}


def _build_vectors() -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    lowers = "abcdefghijklmnopqrstuvwxyz"
    for first_letter, entries in _SB.items():
        vec = [0.0] * VOCAB_SIZE
        # Negative default for letters not listed — unusual word-start
        # bigrams (lf, df, sz, rb, mb, kd, pn...) get mildly penalized.
        for target in lowers:
            if target not in entries:
                vec[VOCAB_INDEX[target]] = -9.0
        for nxt, bias in entries.items():
            if nxt in VOCAB_INDEX:
                vec[VOCAB_INDEX[nxt]] = bias
                if nxt.isalpha() and nxt.lower() == nxt:
                    up = nxt.upper()
                    if up in VOCAB_INDEX:
                        vec[VOCAB_INDEX[up]] = bias * 0.3
        out[first_letter] = vec
    return out


STARTBIGRAM_BIAS: dict[str, list[float]] = _build_vectors()


# Global scale (lowered at integration tune time)
_GLOBAL_SCALE = 1.4


def startbigram_bias(first_letter: str) -> list[float] | None:
    if not first_letter or not first_letter.isalpha():
        return None
    key = first_letter.lower()
    v = STARTBIGRAM_BIAS.get(key)
    if v is None:
        return None
    return [x * _GLOBAL_SCALE for x in v]
