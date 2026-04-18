"""Clause-slot state machine pipeline stage.

Maintains `state.clause_slot`, a coarse syntactic-position tracker for
the current clause. The slot progresses through FRESH → HAS_SUBJ →
HAS_VERB → POST_OBJ as completed words fill each role. Transitions
are driven by POS of last_completed_word (computed by pipeline/pos.py,
which runs before this stage) and by clause-break / sentence-end
punctuation from the character stream.

The point is to give the predict layer a real syntactic prior:
after seeing a subject, a verb is expected; after a verb, an object-
like element is expected; after an object, the clause can close.
The existing n-gram layers see only local character/word history;
they have no notion of "what role have we filled yet?".

Slots:
  0 FRESH      — sentence start, post-sentence-end, or post-
                 clausal-break when no slot is filled yet.
  1 HAS_SUBJ   — we've seen a plausible subject element.
  2 HAS_VERB   — we've seen an auxiliary/modal/main verb after a
                 subject.
  3 POST_OBJ   — we've seen an object/complement after a verb.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB
from .pos import (
    POS_ADJECTIVE,
    POS_ARTICLE,
    POS_AUX_VERB,
    POS_CONJUNCTION,
    POS_INTERJECTION,
    POS_MODAL,
    POS_NOUN,
    POS_POSSESSIVE,
    POS_PRONOUN,
    POS_PROPER_NOUN,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
    POS_WH,
)

SLOT_FRESH = 0
SLOT_HAS_SUBJ = 1
SLOT_HAS_VERB = 2
SLOT_POST_OBJ = 3


# POS tags that mark a subject-role fill (move FRESH → HAS_SUBJ).
_SUBJ_POS: frozenset[int] = frozenset({
    POS_PRONOUN, POS_PROPER_NOUN, POS_NOUN, POS_WH,
    POS_INTERJECTION,  # O / Alas — still progresses the slot.
})
# POS tags that mark a verb-role fill (HAS_SUBJ → HAS_VERB).
_VERB_POS: frozenset[int] = frozenset({
    POS_AUX_VERB, POS_MODAL, POS_VERB, POS_VERB_ING, POS_VERB_ED,
})
# POS tags that mark an object/complement fill (HAS_VERB → POST_OBJ).
_OBJ_POS: frozenset[int] = frozenset({
    POS_NOUN, POS_PRONOUN, POS_PROPER_NOUN, POS_ADJECTIVE,
    POS_VERB_ING, POS_VERB_ED,
})


def update_clause_slot(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]
    slot = state.clause_slot
    wsv = state.words_since_verb

    # Sentence-ending punctuation resets to FRESH and zeroes the
    # words-since-verb counter (fresh clause begins after).
    if ch in ".?!":
        return state.model_copy(
            update={"clause_slot": SLOT_FRESH, "words_since_verb": 0}
        )

    # Clausal break resets slot; keep wsv running (a sub-clause
    # hasn't started yet).
    if ch in ",;:" and state.speaker_label_state == 0:
        if slot != SLOT_FRESH:
            return state.model_copy(update={"clause_slot": SLOT_FRESH})
        return state

    # When a word just completed, progress the slot machine and
    # update words_since_verb.
    if state.just_finished_word and state.last_completed_word:
        pos = state.last_word_pos
        if slot == SLOT_FRESH:
            if pos in _SUBJ_POS:
                slot = SLOT_HAS_SUBJ
        elif slot == SLOT_HAS_SUBJ:
            if pos in _VERB_POS:
                slot = SLOT_HAS_VERB
        elif slot == SLOT_HAS_VERB:
            if pos in _OBJ_POS:
                slot = SLOT_POST_OBJ
        elif slot == SLOT_POST_OBJ:
            if pos == POS_CONJUNCTION:
                slot = SLOT_FRESH

        # words_since_verb: reset on verb-ish word; else increment.
        if pos in _VERB_POS:
            wsv = 0
        else:
            wsv = min(wsv + 1, 15)

    updates = {}
    if slot != state.clause_slot:
        updates["clause_slot"] = slot
    if wsv != state.words_since_verb:
        updates["words_since_verb"] = wsv
    if updates:
        return state.model_copy(update=updates)
    return state
