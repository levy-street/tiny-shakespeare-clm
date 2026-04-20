"""Predict layer — opener-specific pivot-letter bias.

The existing antithesis_pivot_bias applies a generic "any pivot
letter" bias (b/o/n/t/y/e) when ANT_OPENER is active. That's helpful
but coarse. English correlative coordinators have SPECIFIC pairing:

  neither ___ nor       (n-opener projects "n")
  either  ___ or        (o-opener projects "o")
  both    ___ and       (a-opener projects "a")
  more    ___ than      (t-opener projects "t")
  less    ___ than      (t-opener projects "t")
  not     ___ but       (b-opener projects "b")
  whether ___ or        (o-opener projects "o")
  though  ___ yet       (y-opener projects "y")
  although___ yet       (y-opener projects "y")
  rather  ___ than      (t-opener projects "t")

When antithesis_opener_type is set (set by pipeline/antithesis.py),
bias the SPECIFIC paired-pivot first letter sharply, rather than the
generic pivot bias.

Gate:
  - speaker_label_state == 0
  - letter_run_len == 0 (word-start)
  - antithesis_opener_type != 0
  - antithesis_state == ANT_OPENER (waiting for pivot)
  - antithesis_words_since_opener >= 2 (the complement is expected)

The bias grows with distance since opener: the further we've gone
without pivoting, the stronger the nudge toward the paired word.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


ANT_OPENER = 1


# Opener-type → (paired pivot-word first letter, pivot-word itself).
# The pivot-word is used to also bias subsequent letters once we
# commit to the first letter — e.g., if opener type is NEITHER and
# next letter is "n", expect "o" as letter 2 (nor).
_PAIRED_FIRST: dict[int, str] = {
    1: "n",   # NEITHER → nor
    2: "o",   # EITHER  → or
    3: "a",   # BOTH    → and
    4: "t",   # MORE_LESS → than
    5: "b",   # NOT     → but
    6: "o",   # WHETHER → or
    7: "y",   # THOUGH  → yet
    8: "t",   # RATHER  → than
}


def antithesis_pair_bias(
    antithesis_state: int,
    antithesis_opener_type: int,
    antithesis_words_since_opener: int,
    letter_run_len: int,
    speaker_label_state: int,
    last_char: str,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if antithesis_state != ANT_OPENER:
        return None
    if letter_run_len != 0:
        return None
    if antithesis_opener_type == 0:
        return None
    if antithesis_words_since_opener < 2:
        return None
    # Only fire right after a space — opener-pair words are mid-sentence,
    # not at sentence start.
    if last_char != " ":
        return None

    paired = _PAIRED_FIRST.get(antithesis_opener_type)
    if paired is None:
        return None

    # Ramp: 2 → 0.40, 3 → 0.70, 4 → 0.95, 5+ → 1.10.
    w = antithesis_words_since_opener
    if w >= 5:
        ramp = 1.10
    elif w == 4:
        ramp = 0.95
    elif w == 3:
        ramp = 0.70
    else:
        ramp = 0.40

    vec = [0.0] * VOCAB_SIZE
    idx = VOCAB_INDEX.get(paired)
    if idx is not None:
        vec[idx] += ramp
    # Capital too (start of sentence pivot is rare but possible).
    up = paired.upper()
    up_idx = VOCAB_INDEX.get(up)
    if up_idx is not None:
        vec[up_idx] += ramp * 0.30

    return vec
