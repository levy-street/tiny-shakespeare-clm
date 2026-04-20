"""Predict layer — intra-sentence clause-parallelism echo bias.

Reads `prev_clause_opener_letter` and `clauses_in_sentence_index`
(set by pipeline/clause_parallel.py). At the start of a new clause
within a sentence (post-comma/semicolon + space, about to write the
first word of the next clause), nudge the first letter of that word
toward the SAME first letter as the previous clause's opener.

Fires only when:
  - speaker_label_state == 0
  - letter_run_len == 0 (word-start)
  - last_char == " " (post-space)
  - chars_since_comma <= 2 OR prev_char was ";" (just past a clause break)
  - prev_clause_opener_letter is non-empty
  - clauses_in_sentence_index >= 1 (we've crossed at least one break)

The bias GROWS with clauses_in_sentence_index: the more clauses
have already echoed, the stronger the pressure to continue the
parallel (Shakespeare's "I came, I saw, I conquered" effect).

No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def clause_parallel_echo_bias(
    prev_clause_opener_letter: str,
    clauses_in_sentence_index: int,
    letter_run_len: int,
    speaker_label_state: int,
    last_char: str,
    prev_char: str,
    chars_since_comma: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None
    if not prev_clause_opener_letter:
        return None
    if clauses_in_sentence_index < 1:
        return None
    if last_char != " ":
        return None
    # Must be immediately post-comma or semicolon. chars_since_comma
    # counts chars since the last comma; after ", " it's 2. Allow a
    # small window (", " and "; ") only.
    # prev_char is the char before last_char — so prev_char must be
    # "," or ";".
    if prev_char not in (",", ";"):
        return None

    # Ramp: stronger when we're deep into the parallel cascade.
    idx = clauses_in_sentence_index
    if idx >= 3:
        ramp = 1.30
    elif idx == 2:
        ramp = 0.95
    else:  # idx == 1
        ramp = 0.50

    letter = prev_clause_opener_letter.lower()
    if not letter.isalpha():
        return None

    vec = [0.0] * VOCAB_SIZE
    lo_idx = VOCAB_INDEX.get(letter)
    if lo_idx is not None:
        vec[lo_idx] += ramp
    up = letter.upper()
    up_idx = VOCAB_INDEX.get(up)
    if up_idx is not None:
        vec[up_idx] += ramp * 0.45

    return vec
