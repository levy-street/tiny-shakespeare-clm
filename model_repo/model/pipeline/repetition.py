"""Short-range word-repetition memory.

Maintains `state.recent_clause_words`: a rolling tuple (up to 6) of the
most-recently-completed word forms (lowercased), most-recent first.
Reset on strong boundaries:

  - sentence-ending punctuation (. ? !)
  - speaker-turn change (consecutive_newlines >= 2)

The predict layer reads this at word-start to apply a growing penalty
against the first letter of words that have already appeared in the
current clause — counteracting the echo-loop pathology ("there there
there", "hear hear hear") that emerges when content_repeat_bias and
letter n-grams both pull toward a word already said.

This is different from `content_words` (which is a POS-filtered rolling
content memory for topical coherence) — here we want ALL words tracked,
because echo loops hit function words too ("there", "here", "now").
"""

from __future__ import annotations

from ..state import ModelState


_MAX_LEN = 6


def update_repetition(state: ModelState, token_id: int) -> ModelState:
    # Reset on speaker-turn change.
    if state.consecutive_newlines >= 2:
        if state.recent_clause_words == ():
            return state
        return state.model_copy(update={"recent_clause_words": ()})

    # Reset on sentence-ending punctuation (last_char is the token we
    # just advanced through).
    if state.last_char in (".", "?", "!"):
        if state.recent_clause_words == ():
            return state
        return state.model_copy(update={"recent_clause_words": ()})

    # Append on word completion.
    if state.just_finished_word and state.last_completed_word:
        w = state.last_completed_word
        new_tuple = (w,) + state.recent_clause_words
        if len(new_tuple) > _MAX_LEN:
            new_tuple = new_tuple[:_MAX_LEN]
        if new_tuple == state.recent_clause_words:
            return state
        return state.model_copy(update={"recent_clause_words": new_tuple})

    return state
