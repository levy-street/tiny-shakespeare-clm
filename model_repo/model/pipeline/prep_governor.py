"""Prep-governor FSM — block preposition→preposition / prep→verb patterns.

Maintains `prep_block_active`:
  True when the IMMEDIATELY PRECEDING completed word was a preposition
  and no clause/sentence/turn boundary has intervened. Consumed by
  `predict/prep_governor.py`.

Transitions at word completion (pos = state.last_word_pos):
  - PREPOSITION completes                    → prep_block_active = True
  - any content word after (NOUN/PRON/PROPN/ADJ/VERB/AUX/MODAL/
    VERB_ING/VERB_ED/NUMBER/WH/ADVERB)       → prep_block_active = False
  - CONJUNCTION / INTERJECTION / NEGATION   → prep_block_active = False

Also reset on:
  - sentence-end punctuation (. ! ?)
  - clause punctuation (, ; :) outside speaker labels
  - \\n-on-\\n speaker-turn boundary

Distinct from `np_open`: np_open is True after ARTICLE/POSSESSIVE too,
and it tracks "head noun expected"; prep_block_active is ONLY True
right after a PREPOSITION, and its consumer blocks the specific
prep→prep / prep→prep-starting-word pattern.

No corpus statistics.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB
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
    POS_UNKNOWN,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
    POS_WH,
)


# Any completed word that isn't a preposition clears the flag.
_CLEAR_POS: frozenset[int] = frozenset({
    POS_NOUN, POS_PRONOUN, POS_PROPER_NOUN,
    POS_ADJECTIVE, POS_ADVERB, POS_ARTICLE, POS_POSSESSIVE,
    POS_VERB, POS_AUX_VERB, POS_MODAL, POS_VERB_ING, POS_VERB_ED,
    POS_CONJUNCTION, POS_INTERJECTION, POS_NEGATION,
    POS_NUMBER, POS_WH, POS_UNKNOWN,
})


def update_prep_governor(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    active = state.prep_block_active

    # Sentence-end punctuation resets.
    if ch in ".?!":
        if active:
            return state.model_copy(update={"prep_block_active": False})
        return state

    # Speaker-turn boundary: reset.
    if state.consecutive_newlines >= 2 and ch == "\n":
        if active:
            return state.model_copy(update={"prep_block_active": False})
        return state

    # Clause-break punctuation (, ; :) outside speaker labels resets.
    if ch in ",;:" and state.speaker_label_state == 0:
        if active:
            return state.model_copy(update={"prep_block_active": False})
        return state

    # Word-completion transitions.
    if state.just_finished_word and state.last_completed_word:
        pos = state.last_word_pos
        if pos == POS_PREPOSITION:
            if not active:
                return state.model_copy(update={"prep_block_active": True})
            return state
        elif pos in _CLEAR_POS:
            if active:
                return state.model_copy(update={"prep_block_active": False})
            return state
        # else: POS_UNKNOWN or unhandled — leave unchanged.

    return state
