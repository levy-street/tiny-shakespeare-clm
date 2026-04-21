"""Pipeline stage — content-word streak length tracker.

A content-word streak is a run of consecutive completed words of any
content-class category (NOUN, PROPER_NOUN, VERB, VERB_ED, VERB_ING,
ADJECTIVE, ADVERB) without a function word OR a mid-clause punctuation
intervening. Function classes (ARTICLE, POSSESSIVE, PRONOUN,
PREPOSITION, CONJUNCTION, WH, MODAL, AUX_VERB, NEGATION, INTERJECTION,
NUMBER) reset the counter to 0. Any mid-clause punctuation (comma,
semicolon, colon, dash) also resets.

This is a targeted fix for the sample-quality failure mode where the
model produces strings of 3+ content words in a row with no function
word binding them — e.g. "the last noon Kinsmen", "swift death blood
rain crown". Real Shakespeare almost never has >= 3 bare content words
in succession; a preposition, conjunction, determiner, pronoun, or
clausal pause usually interleaves after 2.

This is the mirror / complement of `function_word_chain` — together
they catch both directions of grammatical breakdown.

Reset on sentence-end punctuation, mid-clause punctuation, and
speaker-turn boundary. Also paused inside speaker-label territory.

Runs AFTER `update_pos` (which sets last_word_pos) on just_finished_word.

No corpus statistics — the classification comes from POS tags set by
upstream stages.
"""

from __future__ import annotations

from ..state import ModelState
from .linguistic import PUNCT_END, PUNCT_MID
from .pos import (
    POS_ADJECTIVE,
    POS_ADVERB,
    POS_ARTICLE,
    POS_AUX_VERB,
    POS_CONJUNCTION,
    POS_INTERJECTION,
    POS_MODAL,
    POS_NEGATION,
    POS_NOUN,
    POS_NUMBER,
    POS_POSSESSIVE,
    POS_PREPOSITION,
    POS_PRONOUN,
    POS_PROPER_NOUN,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
    POS_WH,
)


_CONTENT_POS = frozenset({
    POS_NOUN,
    POS_PROPER_NOUN,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
    POS_ADJECTIVE,
    POS_ADVERB,
})

_FUNCTION_POS = frozenset({
    POS_ARTICLE,
    POS_POSSESSIVE,
    POS_PRONOUN,
    POS_PREPOSITION,
    POS_CONJUNCTION,
    POS_WH,
    POS_MODAL,
    POS_AUX_VERB,
    POS_NEGATION,
    POS_INTERJECTION,
    POS_NUMBER,
})


def update_content_word_streak(state: ModelState, token_id: int) -> ModelState:
    # Reset at sentence-terminator.
    if state.last_char_class == PUNCT_END:
        if state.content_word_streak != 0:
            return state.model_copy(update={"content_word_streak": 0})
        return state

    # Reset at mid-clause punctuation (comma/semicolon/colon/dash).
    # Pileups are most dangerous within a single clause; the comma in
    # "noon, drymudrted" does close the streak in real text, so we let
    # it reset here too.
    if state.last_char_class == PUNCT_MID:
        if state.content_word_streak != 0:
            return state.model_copy(update={"content_word_streak": 0})
        return state

    # Reset on speaker-turn boundary (blank line).
    if state.consecutive_newlines >= 2:
        if state.content_word_streak != 0:
            return state.model_copy(update={"content_word_streak": 0})
        return state

    # Inside speaker-label territory, pause tracking.
    if state.speaker_label_state != 0:
        if state.content_word_streak != 0:
            return state.model_copy(update={"content_word_streak": 0})
        return state

    # Only re-evaluate on word completion.
    if not state.just_finished_word:
        return state

    tag = state.last_word_pos

    cur = state.content_word_streak
    if tag in _CONTENT_POS:
        # Cap at 8 to keep the field bounded.
        new_len = min(cur + 1, 8)
    elif tag in _FUNCTION_POS:
        new_len = 0
    else:
        # Unknown / other: leave unchanged.
        return state

    if new_len == cur:
        return state
    return state.model_copy(update={"content_word_streak": new_len})
