"""Clause-rhythm comma-pressure bias.

Reads `state.chars_since_comma` (maintained by pipeline/linguistic.py)
and biases toward the comma token at word-end when the current
comma-less run has grown long.

Shakespeare's prose and verse both feature a pervasive comma cadence
— phrases run 3-15 chars between commas on average, with clause
breaks (",", ";", ".") far more frequent per-character than in modern
prose. A long comma-less run inside a non-opening sentence is
unusual; the text-texture wants a breath.

This layer fires only at word-end positions (letter_run_len >= 2,
on_word_trie, word_buffer is a complete form) and only when we're
not in a speaker label, not at sentence-start, and not inside a
subordinate clause opener (where the comma has just fired).

The bias is small and escalates with pause length:
  chars_since_comma <  20: no bias
  chars_since_comma == 20: +0.05 on ","
  chars_since_comma == 25: +0.12 on ","
  chars_since_comma == 30: +0.22 on ","
  chars_since_comma >= 35: +0.35 on ","

Scale is gentle — existing comma biases (in context.py char-class,
in line_break, in list_structure) carry most of the signal. This
layer is a flow-tier rhythm nudge, not a hard mandate.

All weights from prior knowledge of Shakespearean comma density —
no corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def _build_vec(csc: int) -> list[float] | None:
    if csc < 20:
        return None
    if csc < 22:
        bias = 0.05
    elif csc < 26:
        bias = 0.10
    elif csc < 30:
        bias = 0.18
    elif csc < 35:
        bias = 0.28
    else:
        bias = 0.35
    vec = [0.0] * VOCAB_SIZE
    if "," in VOCAB_INDEX:
        vec[VOCAB_INDEX[","]] += bias
    if ";" in VOCAB_INDEX:
        vec[VOCAB_INDEX[";"]] += bias * 0.25
    return vec


# Precompute for 0..60.
_VECS: list[list[float] | None] = [_build_vec(c) for c in range(61)]


def clause_rhythm_comma_bias(
    chars_since_comma: int,
    chars_since_sentence_end: int,
    word_buffer: str,
    on_word_trie: bool,
    letter_run_len: int,
    speaker_label_state: int,
    has_seen_complete: bool,
    letters_past_complete: int,
) -> list[float] | None:
    """Nudge toward comma at word-end when the pause has grown long.

    Fires only at clean word-end positions where a comma is plausible:
    on-trie complete word, past sentence-start buffer, not in
    speaker label, not drifting past a prior complete form.
    Returns None when not applicable.
    """
    if speaker_label_state != 0:
        return None
    if letter_run_len < 2:
        return None
    if not on_word_trie:
        return None
    if not has_seen_complete:
        return None
    if letters_past_complete != 0:
        # Not sitting at a clean word-ending boundary.
        return None
    if not word_buffer:
        return None
    if chars_since_sentence_end < 15:
        # Too close to sentence start — let the sentence grow.
        return None
    c = min(chars_since_comma, 60)
    return _VECS[c]
