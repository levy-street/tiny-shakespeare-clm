"""Tier 2 — line-starter anaphora tracking.

Reads post-linguistic state. Maintains:

  words_completed_on_line  (reset on newline)
  recent_line_starters     (rolling tuple of last 3 first-words-of-line)

We identify the "first word of the line" as the first word whose
completion we observe after a newline boundary. Concretely, when
`just_finished_word` fires and `words_completed_on_line` was 0
just before this increment, the completed word is the first word
of this line — we push it onto `recent_line_starters`.

Consumed by predict/anaphora.py at line-start word positions.
"""

from __future__ import annotations

from ..state import ModelState

_MAX_STARTERS = 3
_MIN_STARTER_LEN = 1


def update_anaphora(state: ModelState, token_id: int) -> ModelState:
    last_char = state.last_char

    # Reset per-line counter at newline boundary.
    if last_char == "\n":
        return state.model_copy(
            update={
                "words_completed_on_line": 0,
            }
        )

    if not state.just_finished_word:
        return state

    # A word just completed. Increment per-line counter.
    prev_count = state.words_completed_on_line
    new_count = prev_count + 1

    update: dict = {"words_completed_on_line": new_count}

    # If this was the first word of the line, record it.
    if prev_count == 0:
        w = state.last_completed_word
        if w and len(w) >= _MIN_STARTER_LEN:
            starters = state.recent_line_starters + (w,)
            if len(starters) > _MAX_STARTERS:
                starters = starters[-_MAX_STARTERS:]
            update["recent_line_starters"] = starters

    return state.model_copy(update=update)
