"""Antithesis-state bias layer.

Reads `state.antithesis_state` and its companion counters (maintained
by pipeline/antithesis.py) and, at word-start, biases the first letter
of the next word according to which half of the contrast we're in.

Three behaviors:
  1. State OPENER_SEEN (an opener like "not"/"neither" has fired and
     no pivot has yet arrived): when words_since_opener >= 2, boost
     the pivot-opener letters "b" (but), "o" (or), "n" (nor), "t"
     (than), "y" (yet), "e" (else) at word-start to nudge toward
     completing the contrast. The boost grows with distance since
     opener (the contrast becomes overdue).
  2. State PIVOTED (we're in the complement half): the complement
     tends to echo the structure of the first half; and when
     words_since_pivot >= 3, sentence-end punctuation (. ? !) is
     elevated slightly — the antithesis is played out.
  3. State NONE or buffer not at word-start: no effect.

This is a structural-flow signal, not a lexical signal. It's always
small in magnitude (0.05-0.25) because it stacks on top of many
richer biases.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Pivot-opener first letters when we're OPENER_SEEN and waiting.
_PIVOT_LETTERS: dict[str, float] = {
    "b": 0.30,   # but
    "o": 0.18,   # or
    "n": 0.18,   # nor
    "t": 0.15,   # than
    "y": 0.12,   # yet
    "e": 0.10,   # else
}

_CAPITAL_SCALE = 0.55


def antithesis_pivot_bias(words_since_opener: int) -> list[float]:
    """At word-start while antithesis_state == OPENER_SEEN, boost the
    pivot-opener letters. Boost scales 0 -> 1 over words_since_opener
    2 through 5, so the pressure builds as the contrast becomes overdue.
    """
    if words_since_opener < 2:
        return [0.0] * VOCAB_SIZE
    # Ramp: 2 -> 0.35, 3 -> 0.65, 4 -> 0.85, 5+ -> 1.0
    if words_since_opener >= 5:
        ramp = 1.0
    elif words_since_opener == 4:
        ramp = 0.85
    elif words_since_opener == 3:
        ramp = 0.65
    else:
        ramp = 0.35
    vec = [0.0] * VOCAB_SIZE
    for ch, lean in _PIVOT_LETTERS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = ramp * lean
        up = ch.upper()
        if up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] = ramp * lean * _CAPITAL_SCALE
    return vec


def antithesis_closure_bias(
    words_since_pivot: int, letter_run_len: int
) -> list[float]:
    """When antithesis_state == PIVOTED and we've had 3+ complement
    words, gently elevate sentence-end punctuation at word-close.

    Only applies when letter_run_len == 0 immediately after a space
    following a word — i.e., between-word positions. We elevate
    ".", "!", and "?" and also a weak boost for "," (clause close).
    """
    if words_since_pivot < 3:
        return [0.0] * VOCAB_SIZE
    # Only fire at between-word positions (letter_run_len == 0 AND
    # we're not mid-word; our predict caller will gate this).
    vec = [0.0] * VOCAB_SIZE
    # Ramp: 3 -> 0.4, 4 -> 0.7, 5+ -> 1.0
    if words_since_pivot >= 5:
        ramp = 1.0
    elif words_since_pivot == 4:
        ramp = 0.7
    else:
        ramp = 0.4
    # Note: these are end-of-sentence markers, not word-initial letters.
    # They apply at post-word positions (just after a space would be a
    # fresh word); the caller gates it. In practice this boost fires
    # at the character position AFTER the complement's final word, so
    # the space->. transition is gently favored.
    for ch, bump in (
        (".", 0.10),
        ("!", 0.05),
        ("?", 0.04),
    ):
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += ramp * bump
    return vec
