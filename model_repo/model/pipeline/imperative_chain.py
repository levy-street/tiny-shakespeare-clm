"""Imperative-chain counter — tracks consecutive IMPER sentence runs.

Maintains `state.imperative_chain_count`:
  - Increments on sentence-end when the closing sentence_type was
    SENT_IMPER (i.e., the sentence was classified as imperative).
  - Resets to 0 on any sentence-end where the type was NOT imperative.
  - Resets to 0 on speaker-turn boundary (\\n\\n).

Runs AFTER update_sentence (which clears sentence_type and writes
prev_sentence_type at PUNCT_END). We read prev_sentence_type at the
exact token where update_sentence would have saved it — the
punctuation character — so we must run after it. We only increment
on the *same* punctuation event that update_sentence processes.

We detect the punctuation event by checking:
  - last_char is sentence-end punct (., ?, !)
  - prev_sentence_type just changed (we can't trivially detect this
    without prior state, but we can read prev_sentence_type after
    update_sentence has run — that's our new saved-type tag). This
    tells us the TYPE of the sentence that just closed.

Since update_sentence sets prev_sentence_type on every PUNCT_END, we
increment/reset on every PUNCT_END where last_char is a sentence-end
mark. This triggers exactly once per sentence close.
"""

from __future__ import annotations

from ..state import ModelState
from .sentence import SENT_IMPER


def update_imperative_chain(state: ModelState, token_id: int) -> ModelState:
    # Speaker-turn boundary: reset the chain.
    if state.last_char == "\n" and state.consecutive_newlines >= 2:
        if state.imperative_chain_count != 0:
            return state.model_copy(update={"imperative_chain_count": 0})
        return state

    # Sentence-end: update chain based on what type just closed.
    # update_sentence runs earlier in the pipeline and at PUNCT_END
    # sets state.prev_sentence_type to the closing sentence's type.
    # We detect the punctuation event by last_char being . ? ! and
    # fire exactly once per punct (this is how sentence.py itself
    # detects the event).
    if state.last_char in (".", "?", "!"):
        if state.prev_sentence_type == SENT_IMPER:
            new_count = min(state.imperative_chain_count + 1, 5)
        else:
            new_count = 0
        if new_count != state.imperative_chain_count:
            return state.model_copy(
                update={"imperative_chain_count": new_count}
            )

    return state
