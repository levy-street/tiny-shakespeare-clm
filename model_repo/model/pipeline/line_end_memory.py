"""Line-terminal word memory.

Pipeline stage that maintains `recent_line_end_words` — a rolling
tuple (up to 4, most-recent first) of completed-word forms that ended
verse-plausible lines.

This is the epistrophe-axis mirror of `recent_line_starters`:
  - recent_line_starters  : word-identity memory of line BEGINNINGS
  - recent_line_end_words : word-identity memory of line ENDINGS

Consumed by predict/line_end_echo.py to bias toward repeating a
line-ender at the ending position of an upcoming verse line (a
rhetorical closing-word recurrence pattern Shakespeare uses).

Capture rule — on the newline token that closes a verse-plausible line:
  - consecutive_newlines == 1 (not blank-line turn boundary)
  - 1 <= prev_line_length <= 80 (not an oversize prose run)
  - prev_char_class != PUNCT_MID (':') — speaker labels don't count
  - last_completed_word is non-empty and all-lowercase a-z
    (skip proper nouns, numerals, apostrophe-tail forms)

Reset on speaker-turn change (consecutive_newlines >= 2 → clear).

Runs late in the pipeline so prev_line_length / prev_char_class /
last_completed_word reflect the just-closed line.

No corpus statistics — the axis is structural (lines have ends just
like they have starts), no tallying involved.
"""

from __future__ import annotations

from ..state import ModelState
from .linguistic import PUNCT_MID

_MAX = 4


def update_line_end_memory(state: ModelState, token_id: int) -> ModelState:
    # Reset on speaker-turn change.
    if state.consecutive_newlines >= 2:
        if state.recent_line_end_words == ():
            return state
        return state.model_copy(update={"recent_line_end_words": ()})

    # Only fire at a verse-plausible line-closing newline.
    if state.last_char != "\n":
        return state
    if state.consecutive_newlines != 1:
        return state
    if not (1 <= state.prev_line_length <= 80):
        return state
    # Speaker labels end in ':' — don't include them.
    if state.prev_char_class == PUNCT_MID:
        return state

    w = state.last_completed_word
    if not w:
        return state
    # Require pure a-z (skip proper nouns, numerals, apostrophe-suffix forms).
    if not all("a" <= c <= "z" for c in w):
        return state
    if len(w) < 2:
        return state

    new_tuple = (w,) + state.recent_line_end_words
    if len(new_tuple) > _MAX:
        new_tuple = new_tuple[:_MAX]
    if new_tuple == state.recent_line_end_words:
        return state
    return state.model_copy(update={"recent_line_end_words": new_tuple})
