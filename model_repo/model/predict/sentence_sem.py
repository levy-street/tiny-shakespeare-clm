"""Sentence-scoped semantic-field word-start bias.

Complements `predict/noun_class.py` (a POS-gated per-step bias) by
applying a SENTENCE-WIDE semantic tilt: once the sentence has
introduced two nouns of the same coarse class, every subsequent
word-start gets a mild tilt toward letters that open words in that
class.

The effect is structural, not local: a sentence talking of hearts
and tongues has its whole word-start distribution nudged toward
body/speech letters; a sentence talking of crowns and kings gets
nudged toward royalty letters. Small magnitudes — the point is a
weak but pervasive coherence pressure, not a content choice.

Gates:
  * `speaker_label_state == 0`       (not inside speaker label)
  * `sentence_sem_strength >= 2`      (field is LOCKED)
  * `letter_run_len == 0`             (currently between words)
  * `chars_since_sentence_end >= 2`   (past sentence start —
       sentence start has its own heavy bias)
  * `words_in_sentence < 14`          (long sentence = field dilutes)

Magnitudes are significantly smaller than the per-step noun-class
bias because this fires at EVERY word start inside a locked sentence,
whereas the per-step layer fires only after a small set of POS tags.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# Mirror the class ids (kept local to avoid state-module import cycles).
NC_KINSHIP = 1
NC_ROYALTY = 2
NC_BODY = 3
NC_EMOTION = 4
NC_NATURE = 5
NC_ABSTRACT = 6
NC_WEAPON = 7
NC_PLACE = 8
NC_TIME = 9
NC_CREATURE = 10
NC_DIVINE = 11


# Per-class first-letter tilts. Content-word starter letters only.
# These are SMALLER than predict/noun_class.py magnitudes by design —
# this layer fires every word-start, so its magnitude compounds.

_KINSHIP = {
    "m": 0.10, "b": 0.09, "s": 0.09, "f": 0.08, "d": 0.07,
    "c": 0.07, "l": 0.07, "p": 0.06, "y": 0.06, "h": 0.05,
    "w": 0.06, "k": 0.05, "g": 0.05,
}

_ROYALTY = {
    "k": 0.10, "l": 0.09, "p": 0.09, "c": 0.09, "r": 0.09,
    "m": 0.09, "g": 0.08, "h": 0.07, "n": 0.07, "s": 0.07,
    "t": 0.07, "d": 0.06, "q": 0.06, "e": 0.05, "f": 0.05,
}

_BODY = {
    "h": 0.11, "b": 0.10, "e": 0.08, "f": 0.08, "l": 0.08,
    "t": 0.08, "p": 0.07, "c": 0.07, "w": 0.07, "s": 0.06,
    "m": 0.06, "n": 0.05, "r": 0.05, "a": 0.06,
}

_EMOTION = {
    "l": 0.10, "s": 0.09, "g": 0.09, "h": 0.08, "f": 0.08,
    "d": 0.08, "w": 0.07, "b": 0.07, "p": 0.06, "r": 0.06,
    "t": 0.06, "c": 0.06, "m": 0.06, "a": 0.05, "j": 0.04,
}

_NATURE = {
    "s": 0.10, "w": 0.10, "f": 0.09, "m": 0.09, "b": 0.08,
    "l": 0.08, "r": 0.08, "d": 0.07, "n": 0.07, "c": 0.07,
    "t": 0.07, "h": 0.07, "g": 0.06, "p": 0.06, "o": 0.05,
    "e": 0.05,
}

_ABSTRACT = {
    "t": 0.09, "s": 0.08, "h": 0.08, "d": 0.07, "g": 0.07,
    "p": 0.07, "v": 0.07, "l": 0.07, "n": 0.07, "f": 0.07,
    "e": 0.07, "m": 0.06, "w": 0.06, "c": 0.06, "i": 0.05,
    "r": 0.05,
}

_WEAPON = {
    "s": 0.11, "b": 0.10, "f": 0.09, "w": 0.09, "d": 0.08,
    "a": 0.08, "c": 0.07, "k": 0.07, "l": 0.06, "t": 0.06,
    "p": 0.06, "r": 0.06, "h": 0.06, "m": 0.05, "g": 0.05,
}

_PLACE = {
    "c": 0.10, "t": 0.09, "h": 0.09, "r": 0.08, "p": 0.08,
    "s": 0.07, "g": 0.07, "a": 0.07, "d": 0.07, "f": 0.07,
    "l": 0.07, "o": 0.06, "n": 0.06, "w": 0.06, "b": 0.06,
    "m": 0.05, "e": 0.05,
}

_TIME = {
    "h": 0.09, "d": 0.09, "n": 0.09, "y": 0.08, "m": 0.08,
    "a": 0.08, "e": 0.07, "l": 0.07, "t": 0.07, "s": 0.06,
    "w": 0.06, "p": 0.05, "o": 0.05, "f": 0.04,
}

_CREATURE = {
    "h": 0.09, "w": 0.09, "b": 0.08, "s": 0.08, "l": 0.07,
    "f": 0.07, "c": 0.07, "d": 0.07, "r": 0.06, "p": 0.06,
    "e": 0.05, "t": 0.05, "o": 0.05, "g": 0.05,
}

_DIVINE = {
    "g": 0.10, "h": 0.10, "s": 0.09, "b": 0.08, "d": 0.07,
    "e": 0.08, "p": 0.07, "a": 0.07, "f": 0.07, "l": 0.06,
    "r": 0.06, "t": 0.06, "m": 0.06, "c": 0.05, "w": 0.05,
}


def _build_vec(src: dict[str, float]) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch, w in src.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w
        up = VOCAB_INDEX.get(ch.upper())
        if up is not None:
            vec[up] += w * 0.5
    return vec


_CLASS_VECS: dict[int, list[float]] = {
    NC_KINSHIP: _build_vec(_KINSHIP),
    NC_ROYALTY: _build_vec(_ROYALTY),
    NC_BODY: _build_vec(_BODY),
    NC_EMOTION: _build_vec(_EMOTION),
    NC_NATURE: _build_vec(_NATURE),
    NC_ABSTRACT: _build_vec(_ABSTRACT),
    NC_WEAPON: _build_vec(_WEAPON),
    NC_PLACE: _build_vec(_PLACE),
    NC_TIME: _build_vec(_TIME),
    NC_CREATURE: _build_vec(_CREATURE),
    NC_DIVINE: _build_vec(_DIVINE),
}


# Strength scaling: weak prime (strength==1: single in-class noun)
# through hard lock (>=3). Graded so BPC-neutrally rare strong-lock
# gets strong tilt while common weak-prime gets minimal tilt.
_STRENGTH_SCALE: dict[int, float] = {
    1: 0.35,
    2: 1.00,
    3: 1.35,
}


def sentence_sem_bias(
    sentence_sem_field: int,
    sentence_sem_strength: int,
    speaker_label_state: int,
    letter_run_len: int,
    chars_since_sentence_end: int,
    words_in_sentence: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if sentence_sem_strength < 1:
        return None
    if letter_run_len != 0:
        return None
    if chars_since_sentence_end < 2:
        return None
    if words_in_sentence >= 14:
        return None
    vec = _CLASS_VECS.get(sentence_sem_field)
    if vec is None:
        return None
    scale = _STRENGTH_SCALE.get(sentence_sem_strength, 1.0)
    return [v * scale for v in vec]
