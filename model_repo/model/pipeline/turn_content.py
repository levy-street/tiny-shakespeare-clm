"""Turn-scoped content-word echo cache.

Maintains `state.turn_content_cache`: a rolling tuple (up to 10) of
DISTINCT content words (NOUN / VERB / VERB_ING / VERB_ED / ADJECTIVE
/ ADVERB / PROPER_NOUN) emitted in the current speaker turn. Most
recent first.

Reset triggers:
  - Speaker-turn boundary (consecutive_newlines >= 2)

Update rule:
  - On word completion (just_finished_word) outside speaker-label,
    when the completed word is >=3 chars and its POS tag is in the
    content-word set, prepend it to the cache if not already present
    (case-lowered). If already present, MOVE it to the front (most
    recent) and keep the cache at its cap.

Why turn-scoped: Shakespearean speeches circle back to the same
thematic words ('honour, honour, honour'; 'blood, blood, blood').
A TURN-level cache captures this pattern without polluting across
speaker changes (where a new speaker's themes begin fresh). The
global `content_words` field (capped at 4, not turn-reset) is
complementary — it catches the few-word-back topical cluster used
by topic_midword_bias, while this cache captures the whole thematic
spine of a single speech.
"""

from __future__ import annotations

from ..state import ModelState
from .pos import (
    POS_ADJECTIVE,
    POS_ADVERB,
    POS_NOUN,
    POS_PROPER_NOUN,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
)

_CONTENT_POS = frozenset({
    POS_NOUN,
    POS_VERB,
    POS_VERB_ING,
    POS_VERB_ED,
    POS_ADJECTIVE,
    POS_ADVERB,
    POS_PROPER_NOUN,
})

_MAX_CACHE = 10


def update_turn_content(state: ModelState, token_id: int) -> ModelState:
    # Reset on turn boundary.
    if state.consecutive_newlines >= 2:
        if state.turn_content_cache == ():
            return state
        return state.model_copy(update={"turn_content_cache": ()})

    # Don't touch inside speaker-label territory.
    if state.speaker_label_state != 0:
        return state

    # Update only at word completion — read freshly-set last_word_pos
    # (since pipeline/pos.py runs BEFORE turn-content in the chain;
    # see pipeline/__init__.py). Use last_completed_word.
    if not state.just_finished_word:
        return state

    w = state.last_completed_word
    if not w or len(w) < 3:
        return state
    if state.last_word_pos not in _CONTENT_POS:
        return state

    # Build new cache: move w to front, dedupe (case-insensitive since
    # last_completed_word is already lowercased).
    wl = w
    current = state.turn_content_cache
    if current and current[0] == wl:
        return state  # already at front
    new_cache: tuple[str, ...] = (wl,) + tuple(
        x for x in current if x != wl
    )
    if len(new_cache) > _MAX_CACHE:
        new_cache = new_cache[:_MAX_CACHE]
    if new_cache == current:
        return state
    return state.model_copy(update={"turn_content_cache": new_cache})
