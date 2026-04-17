"""Start-of-word bias layer.

When the previous character is a space or a (single) newline — i.e. we
are starting a new word — hand-coded biases nudge the distribution
toward letters that commonly begin English/Shakespearean words: t (the,
to, that), a (and, a, as, at), w (which, when, with, we), h (he, his,
have, hath), o, i, s, b, c, m, p, n, f, d, l, r, y, g, u, v, k.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# Log-bias for each letter that commonly starts a word.
_START_BOOST: dict[str, float] = {
    "t": 1.8,
    "a": 1.6,
    "w": 1.5,
    "h": 1.4,  # he, his, have, hath
    "o": 1.3,
    "i": 1.3,
    "s": 1.2,
    "b": 1.1,
    "c": 1.1,
    "m": 1.1,  # my, me, must, make
    "p": 1.0,
    "n": 0.9,
    "f": 1.0,  # for, from, father
    "d": 0.9,
    "l": 0.8,  # lord, love, let
    "r": 0.7,
    "y": 0.9,  # you, your, yet
    "g": 0.7,  # good, go
    "u": 0.6,  # upon, us
    "v": 0.4,
    "k": 0.3,
    "e": 0.5,
    "j": -0.5,
    "q": -0.5,
    "x": -2.0,
    "z": -2.0,
}


def _build() -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch, bias in _START_BOOST.items():
        vec[VOCAB_INDEX[ch]] = bias
        up = ch.upper()
        if up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] = bias * 0.8
    return vec


START_BIAS: list[float] = _build()
