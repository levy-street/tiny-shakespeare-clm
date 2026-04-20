"""Word-level orthographic integrity tracking.

Updates two fields:

  letters_since_apostrophe — distance (in letters) since the last
      apostrophe in the current word_buffer. 0 if no apostrophe yet
      in this word; 1 immediately after emitting an apostrophe
      (the NEXT char will be 'letter at position 1 after apos'),
      then 2, 3, ... on subsequent letters. Resets when the word
      ends (space/newline/punct).

  had_apostrophe_this_word — boolean shadow: True once we've seen
      an apostrophe in the current word_buffer, False before.

Runs AFTER update_linguistic (which sets word_buffer, letter_run_len,
last_char, last_char_class). The semantics are derivable from
word_buffer alone, but a precomputed integer is cheap and lets
predict layers read it without reparsing.

Semantics of "letters_since_apostrophe" for the NEXT prediction:
  * 0 — we have NOT yet emitted an apostrophe in this word; or the
        current word_buffer is empty. Apostrophe-elision bias does
        not fire.
  * 1 — the LAST char in word_buffer is an apostrophe. The NEXT char
        is the first letter right after the apostrophe. This is the
        tight-bias position (bias toward s/d/t/l/r/v/e/n/m).
  * 2 — one letter has been emitted after the apostrophe. Second-
        position bias is softer (e.g., 'l' → 'l' for "'ll", 'r' →
        'e' for "'re", 'v' → 'e' for "'ve").
  * 3+ — further into post-apostrophe, bias fades out.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB
from .linguistic import LOWER_CONS, LOWER_VOWEL, UPPER


def update_word_cap_apos(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]
    cls = state.last_char_class  # just set by update_linguistic
    wb = state.word_buffer

    # Empty buffer (word ended, or between words) → reset both fields.
    if not wb:
        updates: dict[str, object] = {}
        if state.letters_since_apostrophe != 0:
            updates["letters_since_apostrophe"] = 0
        if state.had_apostrophe_this_word:
            updates["had_apostrophe_this_word"] = False
        if updates:
            return state.model_copy(update=updates)
        return state

    # Within the current word. Decide transition.
    is_letter = cls in (UPPER, LOWER_VOWEL, LOWER_CONS)
    is_apos = ch == "'"

    had = state.had_apostrophe_this_word
    lsa = state.letters_since_apostrophe

    if is_apos:
        # Just emitted an apostrophe. Next prediction is at position 1
        # after the apostrophe.
        new_had = True
        new_lsa = 1
    elif is_letter and had:
        # Another letter after a previous apostrophe: advance.
        new_had = True
        new_lsa = lsa + 1 if lsa >= 1 else 1
    elif is_letter:
        # Ordinary letter, no apostrophe yet in this word.
        new_had = False
        new_lsa = 0
    else:
        # Non-letter, non-apostrophe while buffer non-empty: shouldn't
        # normally happen (linguistic resets buffer on non-letter
        # non-apos), but be defensive.
        new_had = had
        new_lsa = lsa

    updates = {}
    if new_had != state.had_apostrophe_this_word:
        updates["had_apostrophe_this_word"] = new_had
    if new_lsa != state.letters_since_apostrophe:
        updates["letters_since_apostrophe"] = new_lsa
    if not updates:
        return state
    return state.model_copy(update=updates)
