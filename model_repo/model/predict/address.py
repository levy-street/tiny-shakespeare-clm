"""Second-person addressing-register bias.

Reads `state.addressing_register` — a rolling scalar in [-3, +3]
tracking whether the current speech is in thou-register (+) or
you-register (-). Shakespeare characters strongly tend to lock
into one register within a turn:

  thou / thee / thy / thine / thyself  + their archaic verb agreement
  you / your / yours / yourself / ye   + modern verb agreement

This layer nudges the NEXT word's first letter and, mid-word, the
continuation letter of 2nd-person pronouns and their typical
companions (articles/possessives/vocative particles that commonly
collocate with each register).

Applied at word-start outside speaker labels. Strength scales with
|addressing_register|.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# First-letter bumps for the two registers. The core 2nd-person
# pronouns start with t (thou/thee/thy/thine/thyself) or y (you/
# your/yours/yourself/ye). We bump those letters when the register
# is clearly established.
_THOU_START_STRENGTH = 0.40  # per unit of register (capped)
_YOU_START_STRENGTH = 0.40

# Mid-word continuation: when the buffer is a prefix of a register-
# specific pronoun, push the next letter toward completion of the
# matching pronoun.
_MIDWORD_STRENGTH = 0.9


def address_start_bias(register: float) -> list[float] | None:
    """Return a word-start first-letter bias vector given the current
    addressing_register scalar. Returns None if register magnitude is
    too small to act on.
    """
    if register > 0.5:
        # Thou-register.
        strength = min(abs(register), 2.5) * _THOU_START_STRENGTH
        vec = [0.0] * VOCAB_SIZE
        # Boost 't' and 'T' (thou/thee/thy/thine).
        if "t" in VOCAB_INDEX:
            vec[VOCAB_INDEX["t"]] += strength
        if "T" in VOCAB_INDEX:
            vec[VOCAB_INDEX["T"]] += strength * 0.6
        # Penalize 'y' / 'Y' (you/your/ye) — the mutually exclusive
        # register.
        if "y" in VOCAB_INDEX:
            vec[VOCAB_INDEX["y"]] -= strength * 0.8
        if "Y" in VOCAB_INDEX:
            vec[VOCAB_INDEX["Y"]] -= strength * 0.5
        return vec
    if register < -0.5:
        strength = min(abs(register), 2.5) * _YOU_START_STRENGTH
        vec = [0.0] * VOCAB_SIZE
        if "y" in VOCAB_INDEX:
            vec[VOCAB_INDEX["y"]] += strength
        if "Y" in VOCAB_INDEX:
            vec[VOCAB_INDEX["Y"]] += strength * 0.6
        # Modest penalty on 't' to steer against thou-forms. Keep
        # this smaller since 't' leads many non-pronoun words (the,
        # to, this, that, etc.) and we don't want to suppress them.
        if "t" in VOCAB_INDEX:
            vec[VOCAB_INDEX["t"]] -= strength * 0.25
        return vec
    return None


# Mid-word: the buffer is a prefix of a register pronoun; push
# toward its next letter. Uses plain prefix checks against small
# sets rather than any data-derived table.
_THOU_WORDS = ("thou", "thee", "thy", "thine", "thyself")
_YOU_WORDS = ("you", "your", "yours", "yourself", "ye")


def address_midword_bias(
    register: float, buffer: str
) -> list[float] | None:
    if not buffer:
        return None
    if register > 0.5:
        words = _THOU_WORDS
        strength = min(abs(register), 2.5) * _MIDWORD_STRENGTH
    elif register < -0.5:
        words = _YOU_WORDS
        strength = min(abs(register), 2.5) * _MIDWORD_STRENGTH
    else:
        return None
    # Which register-pronouns does the buffer prefix?
    nexts: dict[str, float] = {}
    for w in words:
        if len(w) > len(buffer) and w.startswith(buffer):
            nxt = w[len(buffer)]
            nexts[nxt] = nexts.get(nxt, 0.0) + 1.0
    if not nexts:
        return None
    vec = [0.0] * VOCAB_SIZE
    n = max(1, len(nexts))
    per = strength / n ** 0.5
    for ch, _c in nexts.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += per
    return vec
