"""Pipeline stage — function-word chain length tracker.

A function-word chain is a run of consecutive completed words of any
function-class category (ARTICLE, POSSESSIVE, PRONOUN, PREPOSITION,
CONJUNCTION, WH, MODAL, AUX_VERB, NEGATION, INTERJECTION, NUMBER)
without a content word intervening. Content classes (NOUN, PROPER_NOUN,
VERB family, ADJECTIVE, ADVERB) reset the counter to 0.

This is a targeted fix for the sample-quality failure mode where the
model produces strings of 3+ function words in a row with no content
binding them, e.g. "of your to and you" — which is ungrammatical in
English / Shakespeare. Real Shakespeare almost never has >= 3 function
words in succession without a content head (the only common exceptions
are fixed phrases like "out of the", "in spite of", etc., which are
still capped at 3 and end with a determiner/preposition that expects
a noun).

Reset on sentence-end punctuation and speaker-turn boundary.

Runs AFTER `update_pos` (which sets last_word_pos) on just_finished_word.

No corpus statistics — the classification comes from POS tags set by
upstream stages.
"""

from __future__ import annotations

from ..state import ModelState
from .linguistic import PUNCT_END
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

_CONTENT_POS = frozenset({
    POS_NOUN,
    POS_PROPER_NOUN,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
    POS_ADJECTIVE,
    POS_ADVERB,
})


def update_function_word_chain(state: ModelState, token_id: int) -> ModelState:
    # Reset at sentence-terminator.
    if state.last_char_class == PUNCT_END:
        if state.function_word_chain_len != 0:
            return state.model_copy(update={"function_word_chain_len": 0})
        return state

    # Reset on speaker-turn boundary.
    if state.consecutive_newlines >= 2:
        if state.function_word_chain_len != 0:
            return state.model_copy(update={"function_word_chain_len": 0})
        return state

    # Inside speaker-label territory, pause tracking.
    if state.speaker_label_state != 0:
        if state.function_word_chain_len != 0:
            return state.model_copy(update={"function_word_chain_len": 0})
        return state

    # Only re-evaluate on word completion.
    if not state.just_finished_word:
        return state

    tag = state.last_word_pos

    cur = state.function_word_chain_len
    if tag in _FUNCTION_POS:
        # Cap at 8 to keep the field bounded.
        new_len = min(cur + 1, 8)
    elif tag in _CONTENT_POS:
        new_len = 0
    else:
        # Unknown / other: leave unchanged.
        return state

    if new_len == cur:
        return state
    return state.model_copy(update={"function_word_chain_len": new_len})
