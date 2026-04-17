"""Letter-level bigram bias layer.

Hand-coded English-letter continuations: after a given lowercase letter
(uppercase is lowercased first), which next letters are strongly expected?
All biases are derived from prior knowledge of English spelling and
Shakespearean phonology — not from corpus counts.

Applied only when `state.last_char` is an ASCII letter. The biases are
additive log-space adjustments on top of the context-class layer; they
boost specific next-letter tokens (VOCAB letters a–z) to reflect common
bigrams: qu, th, he, er, in, an, nd, re, ed, es, ou, st, ing, etc.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# Per-previous-letter, per-next-letter bias (a strong bump means "this
# continuation is very likely compared to the average letter in this
# context"; a weak one means "plausible continuation"). Missing entries
# default to 0 bias; very unlikely continuations get a small negative
# bump applied uniformly below.
_B: dict[str, dict[str, float]] = {
    "a": {"n": 2.4, "r": 1.8, "t": 1.9, "l": 1.7, "s": 1.5, "d": 1.6,
          "c": 1.2, "m": 1.2, "b": 0.8, "g": 1.1, "p": 1.0, "v": 1.0,
          "i": 1.1, "y": 0.9, "k": 0.8, "w": 0.7, "u": 0.5, "e": 0.5,
          "f": 0.5, "h": -0.5, "j": -2.0, "q": -3.0, "x": -2.0, "z": -2.0,
          "o": -0.5},
    "b": {"e": 2.4, "o": 1.8, "y": 1.6, "u": 1.5, "l": 1.6, "r": 1.4,
          "a": 1.3, "i": 1.0, "s": 0.8, "t": 0.4, "j": -2.0, "q": -3.0,
          "x": -2.5, "z": -2.0},
    "c": {"h": 2.7, "o": 1.8, "e": 1.7, "a": 1.6, "t": 1.5, "l": 1.5,
          "r": 1.4, "k": 1.2, "i": 1.3, "u": 1.0, "y": 0.6, "j": -2.0,
          "q": -3.0, "w": -1.5, "x": -2.5, "z": -2.0},
    "d": {"e": 2.3, "i": 1.8, "o": 1.5, "s": 1.4, " ": 0.0, "a": 1.2,
          "u": 0.9, "r": 0.9, "y": 0.9, "g": 0.3, "d": 0.2, "l": 0.3,
          "j": -2.0, "q": -3.0, "x": -2.5, "z": -2.0},
    "e": {"r": 2.2, "n": 1.9, "s": 2.0, "d": 1.8, "a": 1.4, "l": 1.5,
          "v": 1.4, "t": 1.3, "m": 1.1, "e": 1.0, "c": 1.0, "w": 0.9,
          "x": 0.3, "i": 0.8, "p": 0.8, "f": 0.6, "g": 0.6, "h": -0.3,
          "j": -2.0, "q": -3.0, "z": -1.5, "k": -0.5, "o": -0.5, "u": -0.3},
    "f": {"o": 2.0, "e": 1.6, "a": 1.5, "r": 1.5, "i": 1.4, "u": 1.3,
          "l": 1.2, "t": 0.8, "f": 1.4, "y": 0.6, "j": -2.0, "q": -3.0,
          "x": -2.5, "z": -2.0},
    "g": {"o": 1.7, "e": 1.9, "h": 1.8, "r": 1.5, "a": 1.4, "i": 1.3,
          "u": 1.1, "l": 1.0, "s": 0.8, "y": 0.6, "n": 0.6, "j": -2.0,
          "q": -3.0, "x": -2.5, "z": -2.0},
    "h": {"e": 2.8, "a": 2.2, "i": 1.9, "o": 1.8, "u": 1.0, "y": 0.9,
          "r": 0.5, "t": 0.3, "s": 0.2, "j": -2.0, "q": -3.0, "x": -2.5,
          "z": -2.0, "h": -1.0, "l": -0.5, "w": -1.0},
    "i": {"n": 2.4, "s": 2.0, "t": 2.1, "c": 1.8, "l": 1.7, "d": 1.6,
          "o": 1.4, "e": 1.4, "v": 1.3, "r": 1.5, "g": 1.4, "m": 1.2,
          "f": 0.9, "a": 0.8, "p": 1.0, "b": 0.8, "k": 0.5, "h": -0.5,
          "q": -3.0, "j": -2.0, "x": -1.0, "z": -1.0, "u": -0.5, "y": -0.5,
          "w": -0.5},
    "j": {"u": 2.2, "o": 1.8, "e": 1.6, "a": 1.5, "i": 0.5, "y": -0.5},
    "k": {"e": 2.0, "i": 1.5, "n": 1.6, "s": 1.2, " ": 0.0, "y": 1.0,
          "l": 0.5, "j": -2.0, "q": -3.0, "x": -2.5, "z": -2.0, "h": 0.5,
          "f": 0.2, "a": 0.5, "o": 0.2, "u": 0.0},
    "l": {"e": 2.3, "y": 2.0, "l": 1.9, "i": 1.6, "o": 1.6, "a": 1.6,
          "d": 1.4, "s": 1.0, "t": 0.6, "f": 0.8, "m": 0.4, "p": 0.3,
          "j": -2.0, "q": -3.0, "x": -2.5, "z": -2.0, "u": 0.5},
    "m": {"e": 2.2, "a": 1.8, "o": 1.8, "i": 1.5, "y": 1.6, "p": 1.4,
          "u": 1.2, "b": 1.2, "s": 0.8, "m": 1.0, "n": 0.4, "j": -2.0,
          "q": -3.0, "x": -2.5, "z": -2.0},
    "n": {"d": 2.4, "t": 2.2, "g": 2.0, "e": 1.9, "s": 1.6, "o": 1.4,
          "c": 1.3, "i": 1.2, "a": 1.1, "k": 1.0, "l": 0.3, "y": 0.7,
          "n": 1.0, "f": 0.5, "j": -2.0, "q": -3.0, "x": -2.5, "z": -2.0,
          "r": -0.5, "h": -0.5, "w": -0.5},
    "o": {"n": 2.0, "r": 2.1, "u": 2.0, "f": 1.7, "m": 1.6, "t": 1.6,
          "v": 1.4, "s": 1.3, "l": 1.3, "w": 1.2, "o": 1.3, "d": 1.1,
          "p": 1.0, "c": 1.0, "b": 0.9, "k": 0.9, "i": 0.9, "e": 0.6,
          "g": 0.5, "a": 0.3, "j": -2.0, "q": -3.0, "x": -2.0, "z": -2.0,
          "h": -0.3, "y": -0.3},
    "p": {"e": 2.0, "o": 1.8, "a": 1.7, "r": 1.9, "l": 1.7, "i": 1.4,
          "u": 1.2, "h": 1.3, "p": 1.4, "t": 0.5, "s": 0.4, "y": 0.6,
          "j": -2.0, "q": -3.0, "x": -2.5, "z": -2.0},
    "q": {"u": 5.0, "a": -3.0, "e": -3.0, "i": -3.0, "o": -3.0, "r": -3.0,
          "s": -3.0, "t": -3.0, "n": -3.0, "l": -3.0, "d": -3.0, "c": -3.0,
          "b": -3.0, "f": -3.0, "g": -3.0, "h": -3.0, "j": -3.0, "k": -3.0,
          "m": -3.0, "p": -3.0, "v": -3.0, "w": -3.0, "x": -3.0, "y": -3.0,
          "z": -3.0, "q": -3.0},
    "r": {"e": 2.3, "s": 1.7, "i": 1.6, "o": 1.6, "a": 1.6, "t": 1.5,
          "d": 1.2, "n": 1.2, "y": 1.2, "m": 1.0, "u": 1.0, "k": 0.8,
          "l": 0.6, "v": 0.5, "g": 0.4, "b": 0.3, "c": 0.3, "p": 0.3,
          "r": 0.5, "f": 0.2, "h": -0.2, "j": -2.0, "q": -3.0, "w": -1.0,
          "x": -2.5, "z": -2.0},
    "s": {"t": 2.2, "e": 2.0, " ": 0.0, "p": 1.5, "h": 1.7, "s": 1.8,
          "i": 1.3, "o": 1.3, "a": 1.2, "u": 1.1, "c": 1.0, "l": 0.5,
          "m": 0.4, "w": 0.4, "y": 0.3, "k": 0.4, "n": 0.3, "j": -2.0,
          "q": -3.0, "x": -2.5, "z": -2.0},
    "t": {"h": 2.9, "o": 2.0, "i": 1.8, "e": 1.8, "r": 1.7, "y": 1.5,
          "a": 1.4, "s": 1.3, "u": 1.0, "t": 1.3, "l": 0.7, "w": 0.6,
          "c": 0.4, "n": 0.3, "f": 0.3, "j": -2.0, "q": -3.0, "x": -2.5,
          "z": -2.0},
    "u": {"n": 1.8, "r": 1.9, "s": 1.8, "t": 1.8, "l": 1.7, "p": 1.5,
          "c": 1.4, "m": 1.3, "i": 1.2, "d": 1.2, "b": 1.1, "g": 1.0,
          "e": 0.8, "f": 0.7, "a": 0.6, "o": 0.3, "k": 0.3, "v": 0.2,
          "j": -2.0, "q": -3.0, "x": -1.0, "z": -1.5, "y": -0.3, "h": -0.3,
          "u": -0.5, "w": -0.5},
    "v": {"e": 2.6, "i": 1.8, "o": 1.4, "a": 1.4, "y": 0.5, "j": -2.0,
          "q": -3.0, "r": -1.5, "l": -1.5, "s": -1.5, "t": -1.5, "n": -1.5,
          "d": -1.5, "c": -1.5, "b": -1.5, "f": -1.5, "g": -1.5, "h": -1.5,
          "k": -1.5, "m": -1.5, "p": -1.5, "u": -0.5, "w": -1.5, "x": -2.5,
          "z": -2.0},
    "w": {"h": 2.0, "e": 2.0, "a": 1.8, "i": 1.6, "o": 1.7, "n": 1.3,
          "s": 0.9, "r": 0.6, "l": 0.3, "y": 0.5, "j": -2.0, "q": -3.0,
          "x": -2.5, "z": -2.0, "t": -0.5, "d": -0.5, "g": -0.5,
          "k": -0.5, "m": -0.5, "p": -0.5, "u": -0.3, "w": -1.0,
          "f": -1.0, "c": -0.5, "v": -1.0, "b": -1.0},
    "x": {"e": 1.0, "i": 0.6, "t": 0.6, "p": 0.5, "c": 0.5, "y": -0.5,
          "q": -3.0, "j": -2.0, "z": -2.0},
    "y": {"o": 1.8, "e": 1.5, " ": 0.0, "s": 1.2, "t": 0.8, "m": 0.6,
          "l": 0.5, "a": 0.3, "n": 0.5, "i": 0.3, "r": 0.3, "d": 0.3,
          "c": 0.3, "j": -2.0, "q": -3.0, "x": -2.0, "z": -2.0},
    "z": {"e": 1.6, "a": 1.0, "i": 0.8, "o": 0.6, "y": 0.5, "u": 0.3,
          "j": -2.0, "q": -3.0, "x": -2.0, "z": -1.0},
}


_GLOBAL_SCALE = 0.6


def _build_bias_vectors() -> dict[str, list[float]]:
    """For each previous letter (a–z), produce a VOCAB_SIZE-length bias
    vector nudging likely/unlikely next letters. Only letter-target indices
    are non-zero; all other positions are 0.
    """
    out: dict[str, list[float]] = {}
    lowers = "abcdefghijklmnopqrstuvwxyz"
    for prev in lowers:
        vec = [0.0] * VOCAB_SIZE
        entries = _B.get(prev, {})
        # Default tiny nudge for letters not explicitly listed.
        for target in lowers:
            if target not in entries:
                # Slight negative — the listed ones are the common ones.
                vec[VOCAB_INDEX[target]] = -0.2 * _GLOBAL_SCALE
        for target, bias in entries.items():
            if target in VOCAB_INDEX and len(target) == 1 and target.isalpha():
                vec[VOCAB_INDEX[target]] = bias * _GLOBAL_SCALE
                # also apply half-bias to the uppercase counterpart
                up = target.upper()
                if up in VOCAB_INDEX:
                    vec[VOCAB_INDEX[up]] = bias * 0.5 * _GLOBAL_SCALE
        out[prev] = vec
    return out


BIGRAM_BIAS_VECTORS: dict[str, list[float]] = _build_bias_vectors()


def bigram_bias(last_char: str) -> list[float] | None:
    """Return a VOCAB_SIZE-length bias vector or None if no bigram applies."""
    if not last_char:
        return None
    key = last_char.lower()
    return BIGRAM_BIAS_VECTORS.get(key)
