"""Semantic-class coherence bias at word-start, gated on context.

Fires ONLY when:
  - We're at a word-start outside a speaker label,
  - last_noun_class != 0 (a semantic field is primed),
  - noun_class_age <= 5 (not faded),
  - last_word_pos is one of {PREPOSITION, POSSESSIVE, ARTICLE,
    CONJUNCTION} — the exact positions where the next word is
    likely a content word whose semantic compatibility matters.

Inside those gates, emit a small first-letter bias tilting toward
letters opening words that are semantically compatible with the
primed class. No suppression of out-of-class letters (previous
attempts showed that suppressing function-word starters hurt BPC
more than biasing in-class content letters helped).

Classes: see state/noun_classes.py.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


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

# POS tag ids (mirror pipeline/pos.py).
POS_ARTICLE = 1
POS_POSSESSIVE = 3
POS_PREPOSITION = 4
POS_CONJUNCTION = 5

_GATED_POS: frozenset[int] = frozenset({
    POS_ARTICLE, POS_POSSESSIVE, POS_PREPOSITION, POS_CONJUNCTION
})


# Per-class preferred starter letters. Conservative values —
# content-word starters only, no attempt to list every letter. These
# tilt the first letter of the likely content noun/adjective/verb
# after "throne OF ___" / "my ROYAL ___" etc.

_KINSHIP = {
    "m": 0.42, "b": 0.38, "s": 0.38, "f": 0.35,
    "d": 0.32, "c": 0.30, "l": 0.30, "p": 0.25,
    "y": 0.25, "o": 0.22, "g": 0.25, "h": 0.22,
    "w": 0.25, "k": 0.22,
}

_ROYALTY = {
    "k": 0.42, "l": 0.40, "p": 0.38, "c": 0.38, "r": 0.38,
    "m": 0.38, "g": 0.35, "h": 0.32, "n": 0.30, "s": 0.30,
    "t": 0.30, "d": 0.28, "q": 0.28, "e": 0.25, "f": 0.22,
}

_BODY = {
    "h": 0.42, "b": 0.38, "e": 0.32, "f": 0.32,
    "l": 0.30, "t": 0.30, "p": 0.30, "c": 0.28,
    "w": 0.28, "a": 0.26, "s": 0.26, "d": 0.25,
    "m": 0.24, "n": 0.22, "r": 0.22, "g": 0.22,
}

_EMOTION = {
    "l": 0.38, "s": 0.36, "g": 0.34, "h": 0.34, "f": 0.32,
    "d": 0.32, "w": 0.30, "b": 0.30, "p": 0.28, "r": 0.28,
    "t": 0.28, "c": 0.26, "m": 0.26, "a": 0.24, "j": 0.20,
}

_NATURE = {
    "s": 0.42, "w": 0.40, "f": 0.38, "m": 0.35, "b": 0.35,
    "l": 0.34, "r": 0.34, "d": 0.32, "n": 0.30, "c": 0.30,
    "t": 0.30, "h": 0.28, "g": 0.28, "p": 0.26, "o": 0.24,
    "e": 0.24,
}

_ABSTRACT = {
    "t": 0.38, "s": 0.35, "h": 0.35, "d": 0.32, "g": 0.32,
    "p": 0.32, "v": 0.32, "l": 0.30, "n": 0.30, "f": 0.30,
    "e": 0.30, "m": 0.28, "w": 0.28, "c": 0.28, "i": 0.24,
    "r": 0.22, "j": 0.22,
}

_WEAPON = {
    "s": 0.45, "b": 0.42, "f": 0.40, "w": 0.38, "d": 0.36,
    "a": 0.36, "c": 0.32, "k": 0.30, "l": 0.28, "t": 0.28,
    "p": 0.28, "r": 0.28, "h": 0.26, "m": 0.24, "g": 0.22,
}

_PLACE = {
    "c": 0.40, "t": 0.38, "h": 0.38, "r": 0.34, "p": 0.34,
    "s": 0.32, "g": 0.32, "a": 0.30, "d": 0.30, "f": 0.30,
    "l": 0.30, "o": 0.28, "n": 0.26, "w": 0.26, "b": 0.26,
    "m": 0.24, "e": 0.24,
}

_TIME = {
    "h": 0.38, "d": 0.36, "n": 0.36, "y": 0.34, "m": 0.32,
    "a": 0.32, "e": 0.32, "l": 0.30, "t": 0.30, "s": 0.28,
    "w": 0.26, "p": 0.26, "o": 0.24, "f": 0.22,
}

_CREATURE = {
    "h": 0.38, "w": 0.36, "b": 0.34, "s": 0.34, "l": 0.32,
    "f": 0.32, "c": 0.28, "d": 0.28, "r": 0.28, "p": 0.24,
    "e": 0.24, "t": 0.24, "o": 0.20, "g": 0.20,
}

_DIVINE = {
    "g": 0.42, "h": 0.40, "s": 0.36, "b": 0.32, "d": 0.30,
    "e": 0.32, "p": 0.30, "a": 0.30, "f": 0.30, "l": 0.28,
    "r": 0.28, "t": 0.28, "m": 0.28, "c": 0.26, "w": 0.22,
    "o": 0.24,
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


# Age decay: strongest right after the noun, fades out by age 5.
_AGE_SCALE = {0: 0.22, 1: 0.18, 2: 0.14, 3: 0.10, 4: 0.06, 5: 0.03}


def noun_class_bias(
    last_noun_class: int,
    noun_class_age: int,
    last_word_pos: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a word-start bias, or None when gate fails."""
    if speaker_label_state != 0:
        return None
    if last_noun_class == 0:
        return None
    if last_word_pos not in _GATED_POS:
        return None
    vec = _CLASS_VECS.get(last_noun_class)
    if vec is None:
        return None
    scale = _AGE_SCALE.get(noun_class_age)
    if scale is None:
        return None
    return [v * scale for v in vec]
