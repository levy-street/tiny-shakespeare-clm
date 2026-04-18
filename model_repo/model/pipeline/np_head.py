"""Tier 2 — NP-head-expectation state machine.

Runs after `update_pos` and `update_clause_slot` so that
`state.last_word_pos` reflects the word that just completed. Maintains
two fields:

  - np_open: True iff the most recent NP-opener (article, possessive,
    or preposition) has no head noun yet.
  - np_wait_words: words elapsed since np_open became True.

Transitions at word completion (pos = state.last_word_pos):
  - ARTICLE / POSSESSIVE / PREPOSITION → np_open=True,  wait=0
  - NOUN / PROPER_NOUN / PRONOUN       → np_open=False, wait=0 (resolved)
  - VERB / AUX_VERB / MODAL / VERB_ING / VERB_ED → np_open=False, wait=0 (abandoned)
  - CONJUNCTION / INTERJECTION / NEGATION → np_open=False, wait=0 (abandoned)
  - ADJECTIVE / ADVERB / NUMBER / WH / UNKNOWN → if np_open, wait+=1

Also reset on sentence-ending punctuation and on speaker-turn boundary.
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

_OPEN_POS: frozenset[int] = frozenset({POS_ARTICLE, POS_POSSESSIVE, POS_PREPOSITION})
_RESOLVE_POS: frozenset[int] = frozenset({POS_NOUN, POS_PROPER_NOUN, POS_PRONOUN})
_ABANDON_POS: frozenset[int] = frozenset({
    POS_VERB, POS_AUX_VERB, POS_MODAL, POS_VERB_ING, POS_VERB_ED,
    POS_CONJUNCTION, POS_INTERJECTION, POS_NEGATION,
})
# Pre-head modifiers — these extend the wait without resolving.
_MODIFIER_POS: frozenset[int] = frozenset({
    POS_ADJECTIVE, POS_ADVERB, POS_NUMBER, POS_WH, POS_UNKNOWN,
})


def update_np_head(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    np_open = state.np_open
    wait = state.np_wait_words

    # Sentence-end punctuation resets NP state.
    if ch in ".?!":
        if np_open or wait != 0:
            return state.model_copy(
                update={"np_open": False, "np_wait_words": 0}
            )
        return state

    # Speaker-turn boundary: reset NP state.
    if state.consecutive_newlines >= 2 and ch == "\n":
        if np_open or wait != 0:
            return state.model_copy(
                update={"np_open": False, "np_wait_words": 0}
            )
        return state

    # Clausal break (, ; :) closes any open NP — a clause break
    # rarely appears mid-NP.
    if ch in ",;:" and state.speaker_label_state == 0:
        if np_open or wait != 0:
            return state.model_copy(
                update={"np_open": False, "np_wait_words": 0}
            )
        return state

    # Word-completion transitions.
    if state.just_finished_word and state.last_completed_word:
        pos = state.last_word_pos
        if pos in _OPEN_POS:
            np_open = True
            wait = 0
        elif np_open and pos in _RESOLVE_POS:
            np_open = False
            wait = 0
        elif np_open and pos in _ABANDON_POS:
            np_open = False
            wait = 0
        elif np_open and pos in _MODIFIER_POS:
            wait = min(wait + 1, 5)
        # else: np_open False and pos is modifier — no change.

    if np_open != state.np_open or wait != state.np_wait_words:
        return state.model_copy(
            update={"np_open": np_open, "np_wait_words": wait}
        )
    return state
