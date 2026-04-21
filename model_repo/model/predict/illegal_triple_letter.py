"""Illegal triple-same-letter penalty.

No English word has three of the same letter in a row. "eee", "ooo",
"aaa", "lll", "ssss" etc. are all invalid (the closest real case is
"bookkeeper" but that has a morphological boundary, not in our
character-level model).

This layer inspects the last two characters of `word_buffer`: if both
are the same letter, penalize emitting that letter a third time.

Complements:
  - letter_repeat_penalty (count-based, penalty 0.35 at count=3 —
    too gentle to fully block a 3rd same letter, and doesn't focus
    on the ADJACENT triple specifically)
  - illegal_vowel_double / illegal_consonant_double (block illegal
    2nd-letter doublings, but allow legal doubles like "ee", "oo",
    "ll", "ss" — this layer makes sure the allowed doubles don't
    then extend to triples)

Gates:
  * letter_run_len >= 2 (need at least 2 chars in buffer)
  * last 2 chars of word_buffer are the same letter (case-insensitive)
  * speaker_label_state == 0

No corpus statistics — English orthographic rule.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# A uniform strong penalty — no English word continues a doubled
# letter with a third same letter. Penalty applies to both cases.
_PENALTY = -4.0


def illegal_triple_letter_bias(
    word_buffer: str,
    letter_run_len: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if letter_run_len < 2:
        return None
    if len(word_buffer) < 2:
        return None
    a = word_buffer[-1]
    b = word_buffer[-2]
    # Both must be ASCII letters.
    if not (("a" <= a.lower() <= "z") and ("a" <= b.lower() <= "z")):
        return None
    if a.lower() != b.lower():
        return None
    # We've got a doubled letter at the end of the buffer. The next
    # letter must not be the same.
    ch = a.lower()
    vec = [0.0] * VOCAB_SIZE
    if ch in VOCAB_INDEX:
        vec[VOCAB_INDEX[ch]] += _PENALTY
    up = ch.upper()
    if up in VOCAB_INDEX:
        vec[VOCAB_INDEX[up]] += _PENALTY
    return vec
