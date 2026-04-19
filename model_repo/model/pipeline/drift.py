"""Tier 2/3: scene-drift detector.

Maintains `state.drift_streak` — number of CONSECUTIVE words that
completed off the word-trie (i.e., last_completed_word is not a member
of COMPLETE_WORDS). Resets to 0 on a trie-hit word-completion or on
a speaker-turn boundary (\\n\\n).

This is a structural quality signal: per-letter off-trie tracking
(letters_off_trie, offtrie_depart_pos) resets each word, so the model
has no memory that the PREVIOUS word was also gibberish. Real
Shakespeare rarely strings two off-trie words together; runaway
letter-ngram gibberish does exactly that.

Must run AFTER update_basic_counters so `just_finished_word`,
`last_completed_word`, and `consecutive_newlines` are fresh.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB
from ..predict.word_trie import COMPLETE_WORDS

_MAX_STREAK = 8


def update_drift(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-turn boundary: fresh slate.
    if ch == "\n" and state.consecutive_newlines >= 2:
        if state.drift_streak != 0:
            return state.model_copy(update={"drift_streak": 0})
        return state

    # Only update on word completion.
    if not state.just_finished_word or not state.last_completed_word:
        return state

    lcw = state.last_completed_word
    # Normalize: strip leading/trailing apostrophe (same convention as
    # other pipeline stages) so "'tis" / "tis" / "o'er" are compared
    # cleanly against COMPLETE_WORDS.
    lookup = lcw
    if lookup and lookup.startswith("'"):
        lookup = lookup[1:]
    if lookup and lookup.endswith("'"):
        lookup = lookup[:-1]

    # Also check case-folded form — COMPLETE_WORDS holds lowercased
    # entries for most words but preserves capitalized forms (proper
    # nouns, sentence-starters) separately.
    on_trie = (
        lookup in COMPLETE_WORDS
        or lookup.lower() in COMPLETE_WORDS
    )

    if on_trie:
        if state.drift_streak != 0:
            return state.model_copy(update={"drift_streak": 0})
        return state
    else:
        new_streak = state.drift_streak + 1
        if new_streak > _MAX_STREAK:
            new_streak = _MAX_STREAK
        if new_streak != state.drift_streak:
            return state.model_copy(update={"drift_streak": new_streak})
        return state
