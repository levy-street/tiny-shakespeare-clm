"""Predict layer — apodosis-opener bias.

Reads `state.conditional_mode` (set by pipeline/conditional.py). When
the sentence is in MODE_APODOSIS (the protasis-closing comma has just
fired, and we're about to write the main clause), bias the FIRST
letter of the next word toward apodosis-opener letters.

Apodosis openers in English / EME:
  - Subject pronouns:   I, thou, he, she, we, they, ye, you, it  → i/t/h/s/w/y
  - Modals:             shall, will, must, may, can              → s/w/m/c
  - Bare imperatives:   go, come, hear, speak, tell, stay, look  → g/c/h/s/t/l
  - Adverbs:            then, so, therefore, hence               → t/s/h
  - Existential:        there, here                              → t/h

Also mildly biases toward CAPITAL at the very first letter, because
the apodosis often begins on a new line (couplet break, dramatic
pause) — not always, but often enough that caps should be elevated.

Gate:
  - conditional_mode == 2 (APODOSIS)
  - letter_run_len == 0 (word-start)
  - speaker_label_state == 0
  - last_char == " " (after the protasis-closing comma + space)

No corpus statistics — the opener inventory comes from prior knowledge
of English conditional/concessive syntax.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


MODE_APODOSIS = 2


# Apodosis-opener letter weights (lowercase).
_OPENER_LETTERS: dict[str, float] = {
    "i": 0.80,   # I (subj pron)
    "t": 0.95,   # thou, they, thee, the (weak), then, therefore
    "h": 0.80,   # he, her, his, here, hear, hold
    "s": 0.75,   # she, shall, should, so, speak, say, stay
    "w": 0.75,   # we, will, would, what, would (main-clause WH)
    "y": 0.50,   # you, ye, yet
    "m": 0.55,   # my, me, must, may, meet, mark
    "c": 0.45,   # can, come, cry, call
    "g": 0.35,   # go, give, grant, good
    "l": 0.40,   # let, look, leave, love
    "r": 0.25,   # rise, run (imperative)
    "p": 0.30,   # pray, put, pass
    "d": 0.35,   # do, die, draw
    "b": 0.30,   # bid, be, bring
    "f": 0.25,   # forbear, find
    "n": 0.35,   # now, never, no
    "o": 0.35,   # our, O (interjection), on, or
    "a": 0.40,   # and, as, all, a (det), art
    "e": 0.20,   # ever, eye
    "k": 0.20,   # keep, know
    "v": 0.10,
    "j": 0.05,
    "q": 0.05,
    "u": 0.15,   # upon, under, unto
}


# Capital-letter boosts. Apodosis-opening is often mid-sentence (after
# comma) so caps are NOT dramatically elevated — just a light touch
# beyond what sentence-start already provides.
_OPENER_CAPS: dict[str, float] = {
    "I": 0.25,   # I (pronoun) — dominant after comma
    "T": 0.20,
    "H": 0.15,
    "S": 0.15,
    "W": 0.15,
    "Y": 0.10,
    "O": 0.10,   # O (interjection, less common mid-sentence)
}


# Overall scale. Decays with conditional_age — the apodosis bias is
# strongest at word 0 (the very first apodosis word) and weakens
# quickly after.
_MAX_SCALE = 1.00


def apodosis_opener_bias(
    conditional_mode: int,
    conditional_age: int,
    letter_run_len: int,
    speaker_label_state: int,
    last_char: str,
    consecutive_newlines: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if conditional_mode != MODE_APODOSIS:
        return None
    if letter_run_len != 0:
        return None
    # Only fire on the VERY FIRST word of the apodosis.
    if conditional_age != 0:
        return None
    # Require recent space (protasis-closing comma + space = ", ").
    if last_char != " ":
        return None
    # At speaker-label open skip.
    if consecutive_newlines >= 2:
        return None

    scale = _MAX_SCALE

    vec = [0.0] * VOCAB_SIZE
    for ch, w in _OPENER_LETTERS.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
    for ch, w in _OPENER_CAPS.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale

    return vec
