"""Pipeline stage — noun-phrase slot FSM.

Targeted fix for the reality-check failure:
    "the man little evilly to and when"
    "A sicily fits, hee deep the resent measure"

After a determiner or within a noun-phrase, the model does not
constrain the NEXT word's POS class — it freely follows "the" with
"man" (OK), then "little" (adjective in wrong slot), then "evilly"
(adverb with no verb to modify).

This stage maintains a 4-state FSM over completed-word POS tags:

    0 NEUTRAL  — sentence-start, post-verb, post-terminator, post-
                 conjunction. Any next category OK.
    1 POST_DET — just saw article/possessive/wh-determiner. Next OPEN-
                 class word should be ADJECTIVE or NOUN.
    2 POST_ADJ — inside an NP after an adjective. Next should be ADJ
                 or NOUN.
    3 POST_NOUN — head noun of NP is complete. Next should be PREP /
                 VERB / CONJ / terminator; NOT another determiner or
                 adjective (would start a new, ungrammatical bare NP).

Runs AFTER `pipeline/pos.py` (which sets last_word_pos) on every
just_finished_word. Resets on PUNCT_END, on speaker-turn change, and
on entering a speaker label.
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


SLOT_NEUTRAL = 0
SLOT_POST_DET = 1
SLOT_POST_ADJ = 2
SLOT_POST_NOUN = 3


# POS classes that BEHAVE AS a noun head in this FSM (singular/plural
# nouns, proper nouns, pronouns — pronouns head their own NP).
_NOUN_HEADS = frozenset({
    POS_NOUN,
    POS_PROPER_NOUN,
    POS_PRONOUN,
})
# POS classes that open an NP by demanding a following noun.
_DETERMINERS = frozenset({
    POS_ARTICLE,
    POS_POSSESSIVE,
})
# POS classes that reset the slot to NEUTRAL (matrix verbs, fresh
# clausal glue).
_VERB_LIKE = frozenset({
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
    POS_AUX_VERB,
    POS_MODAL,
})
_CLAUSE_BREAK = frozenset({
    POS_CONJUNCTION,
    POS_PREPOSITION,  # ends current NP and opens a PP whose object is new
})
# Transparent — don't change slot (don't add info, don't close NP).
# Interjections, negations, WH (in attributive position), adverbs
# inside a run are too variable to be reliable — safer to treat as
# transparent and keep the slot.
_SLOT_TRANSPARENT = frozenset({
    POS_INTERJECTION,
    POS_NEGATION,
    POS_ADVERB,
    POS_WH,
    POS_NUMBER,
})


def update_phrase_slot(state: ModelState, token_id: int) -> ModelState:
    # Reset inside speaker-label territory.
    if state.speaker_label_state != 0:
        if state.phrase_slot != 0 or state.phrase_slot_len != 0:
            return state.model_copy(update={
                "phrase_slot": 0,
                "phrase_slot_len": 0,
            })
        return state

    # Reset at sentence-terminator.
    if state.last_char_class == PUNCT_END:
        if state.phrase_slot != 0 or state.phrase_slot_len != 0:
            return state.model_copy(update={
                "phrase_slot": 0,
                "phrase_slot_len": 0,
            })
        return state

    # Only re-evaluate on word-completion. pos.py runs before us on
    # just_finished_word, so state.last_word_pos is fresh.
    if not state.just_finished_word:
        return state

    tag = state.last_word_pos
    cur = state.phrase_slot
    cur_len = state.phrase_slot_len

    if tag in _DETERMINERS:
        new_slot = SLOT_POST_DET
        new_len = 1
    elif tag == POS_ADJECTIVE:
        # If we're inside an NP (POST_DET / POST_ADJ), progress.
        # Otherwise start a bare adjective sequence — still mark POST_ADJ
        # because some Shakespeare phrases do open with a bare adj
        # ("Fair Hermia" / "Dear lord").
        if cur in (SLOT_POST_DET, SLOT_POST_ADJ):
            new_slot = SLOT_POST_ADJ
            new_len = cur_len + 1
        else:
            new_slot = SLOT_POST_ADJ
            new_len = 1
    elif tag in _NOUN_HEADS:
        # Head noun completes the NP — move to POST_NOUN.
        new_slot = SLOT_POST_NOUN
        new_len = 1
    elif tag in _VERB_LIKE:
        # Verb closes NP and opens VP; back to NEUTRAL.
        new_slot = SLOT_NEUTRAL
        new_len = 0
    elif tag in _CLAUSE_BREAK:
        # Preposition/conjunction — reset to NEUTRAL (next word is the
        # start of a new PP/clause).
        new_slot = SLOT_NEUTRAL
        new_len = 0
    elif tag in _SLOT_TRANSPARENT:
        # Keep slot; bump len.
        new_slot = cur
        new_len = cur_len + 1 if cur != SLOT_NEUTRAL else 0
    else:
        # Unknown / other: leave slot unchanged but bump len (avoids
        # infinite-slot persistence on a chain of UNKs).
        new_slot = cur
        new_len = cur_len + 1 if cur != SLOT_NEUTRAL else 0

    if new_slot == cur and new_len == cur_len:
        return state
    return state.model_copy(update={
        "phrase_slot": new_slot,
        "phrase_slot_len": new_len,
    })
