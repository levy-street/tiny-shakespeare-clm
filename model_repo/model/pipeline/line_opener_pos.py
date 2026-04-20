"""Line-opener POS pattern memory.

Maintains `recent_line_opener_pos` — a rolling tuple of up to 4 POS
tags of the FIRST word of each of the most recent completed lines
(most-recent LAST).

Depends on:
  - `update_linguistic`  — sets `just_finished_word`
  - `update_pos`         — sets `last_word_pos` for the completed word
  - `update_anaphora`    — increments `words_completed_on_line` from
                           0 → 1 on the first word of a line

This stage must run AFTER update_anaphora (so we see the incremented
counter) and AFTER update_pos (so last_word_pos reflects the
just-completed word).

Reset on speaker-turn change (consecutive_newlines >= 2).

Cross-turn reset rationale: each speaker's anaphoric rhythm is
local to their own turn. A new speaker breaks the prior POS
pattern; a new opener POS starts a fresh memory.
"""

from __future__ import annotations

from ..state import ModelState

_MAX_OPENERS = 4


def update_line_opener_pos(state: ModelState, token_id: int) -> ModelState:
    # Reset on speaker-turn change.
    if state.consecutive_newlines >= 2:
        if state.recent_line_opener_pos:
            return state.model_copy(update={"recent_line_opener_pos": ()})
        return state

    # Only act at the moment a first-word-of-line completes.
    # update_anaphora increments words_completed_on_line BEFORE this
    # stage runs; after that increment the counter is 1 iff this was
    # the first word on the line.
    if not state.just_finished_word:
        return state
    if state.words_completed_on_line != 1:
        return state

    # Append the POS of the word just completed.
    pos = state.last_word_pos
    new_tuple = state.recent_line_opener_pos + (pos,)
    if len(new_tuple) > _MAX_OPENERS:
        new_tuple = new_tuple[-_MAX_OPENERS:]
    if new_tuple == state.recent_line_opener_pos:
        return state
    return state.model_copy(update={"recent_line_opener_pos": new_tuple})
