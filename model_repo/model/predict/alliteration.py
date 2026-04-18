"""Within-line alliteration word-start bias.

Reads `line_alliteration_letter` and `line_alliteration_run` from
state. When the run is >= 2 (two content words on this line have
already started with the same letter), bias the next word's first
letter toward the same letter — both the lowercase form (mid-line
content word) and its uppercase form (line-start / sentence-start).

Escalation with run depth: run=2 nudges gently (since two-word
"alliteration" is common by chance); run=3+ is strong signal
Shakespeare is rhetorically committing.

Decays at very long runs to avoid a single letter dominating the
rest of the line.

Activated only:
  - Outside speaker-label territory (state 0)
  - When the alliteration letter is actually set
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def alliteration_start_bias(
    alliteration_letter: str,
    alliteration_run: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if not alliteration_letter:
        return None
    if alliteration_run < 2:
        return None

    # Scale: run=2 -> 0.35, run=3 -> 0.70, run=4 -> 0.85, run=5+ -> 0.65
    # (taper: after 4+ matching words, a break is plausible too).
    if alliteration_run == 2:
        sc = 0.12
    elif alliteration_run == 3:
        sc = 0.55
    elif alliteration_run == 4:
        sc = 0.75
    else:
        sc = 0.55

    vec = [0.0] * VOCAB_SIZE
    ch = alliteration_letter
    if ch in VOCAB_INDEX:
        vec[VOCAB_INDEX[ch]] += sc
    up = ch.upper()
    if up != ch and up in VOCAB_INDEX:
        vec[VOCAB_INDEX[up]] += sc * 0.6
    return vec
