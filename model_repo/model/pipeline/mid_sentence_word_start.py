"""Tier 2 — mid-sentence word-start gate.

Maintains `mid_sentence_word_start`: True exactly when we are at a
word-start position that is mid-sentence (i.e., NOT the first word
of the sentence, NOT a verse-line-start, NOT a post-speaker-label
position), where Shakespeare would normally emit a LOWERCASE letter
(unless the word is a proper noun or vocative I/O).

Conditions for True:
  * letter_run_len == 0 (we're about to start a new word)
  * last_char_class == SPACE (the preceding char was a space)
  * cap_required_mode == NONE (no structural cap-required signal)
  * words_in_sentence >= 1 (this is not the first word of the
    sentence)
  * speaker_label_state == 0 (not inside a speaker label)

Must run AFTER `update_cap_required` so that cap_required_mode is
already set.
"""

from __future__ import annotations

from ..state import ModelState
from .cap_required import NONE as CAP_NONE
from .linguistic import SPACE


def update_mid_sentence_word_start(state: ModelState, token_id: int) -> ModelState:
    new_val = (
        state.letter_run_len == 0
        and state.last_char_class == SPACE
        and state.cap_required_mode == CAP_NONE
        and state.words_in_sentence >= 1
        and state.speaker_label_state == 0
    )
    if new_val == state.mid_sentence_word_start:
        return state
    return state.model_copy(update={"mid_sentence_word_start": new_val})
