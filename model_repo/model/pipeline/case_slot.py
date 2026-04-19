"""Pronoun case-slot tracker — syntactic role for the upcoming pronoun.

Sets `case_slot` ∈ {0=NONE, 1=SUBJ, 2=OBJ} and `case_wait_words` based
on the grammatical context established by preceding tokens. Runs
AFTER update_pos, update_clause_slot, and update_transitivity so those
fields are already current.

Triggers:
  CASE_OBJ (2) — strong object-slot expectation.
    - last_completed_word is POS_PREPOSITION. Prepositions (to, by,
      with, for, from, in, on, at, upon, against, of, unto, through,
      among) take accusative pronouns: "by thee", "with him", "for me",
      "of them", "to her". "of" is included (it takes NP head with
      very common pronouns: "of him", "of thee", "of us").
    - verb_transitivity just became VT_DO_EXPECTED (transitive main
      verb just completed, taking a direct-object NP that may be a
      pronoun: "slay him", "love me", "know thee").
  CASE_SUBJ (1) — subject-slot expectation.
    - sentence-start position: clause_slot became FRESH AND we're at
      the start of a new sentence (detected via sentence_start_pending
      or via punctuation reset). A pronoun here takes nominative:
      "I do", "he knows", "thou art".
    - after "and"/"but" when clause is POST_OBJ (a fresh clause starts):
      "I fight and he flees", "she came but they left".

Reset to CASE_NONE on:
  - sentence-end punctuation (. ? !)
  - speaker-turn boundary (consecutive_newlines >= 2)
  - comma / semicolon / colon (clausal break clears the prior slot)
  - case_wait_words >= 3 (expectation has staled out past reach)
  - the slot has been filled: last_completed_word POS is
    PRONOUN, NOUN, PROPER_NOUN, ADJECTIVE, ARTICLE, POSSESSIVE
    (these consume the slot or start an NP).

No corpus statistics — all rules from standard English/EME syntax.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB
from .pos import (
    POS_ADJECTIVE,
    POS_ARTICLE,
    POS_CONJUNCTION,
    POS_NOUN,
    POS_POSSESSIVE,
    POS_PREPOSITION,
    POS_PROPER_NOUN,
    POS_PRONOUN,
)


CASE_NONE = 0
CASE_SUBJ = 1
CASE_OBJ = 2

_MAX_WAIT = 5
_STALE_AT = 3

# clause_slot values (mirrored from pipeline/clause_slot.py).
_SLOT_FRESH = 0
_SLOT_POST_OBJ = 3

# verb_transitivity values (mirrored from pipeline/transitivity.py).
_VT_DO_EXPECTED = 1

# POS tags that, when seen as last_completed_word, fill or divert
# the pronoun slot.
_SLOT_FILLERS: frozenset[int] = frozenset({
    POS_PRONOUN,
    POS_NOUN,
    POS_PROPER_NOUN,
    POS_ADJECTIVE,
    POS_ARTICLE,
    POS_POSSESSIVE,
})


def update_case_slot(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Sentence-end punctuation resets.
    if ch in ".?!":
        if state.case_slot == CASE_NONE and state.case_wait_words == 0:
            return state
        return state.model_copy(update={
            "case_slot": CASE_NONE,
            "case_wait_words": 0,
        })

    # Clausal break punctuation resets the prior slot (but doesn't
    # yet open a new one; the next word/trigger will).
    if ch in ",;:":
        if state.case_slot == CASE_NONE and state.case_wait_words == 0:
            return state
        return state.model_copy(update={
            "case_slot": CASE_NONE,
            "case_wait_words": 0,
        })

    # Speaker-turn boundary.
    if ch == "\n" and state.consecutive_newlines >= 2:
        if state.case_slot == CASE_NONE and state.case_wait_words == 0:
            return state
        return state.model_copy(update={
            "case_slot": CASE_NONE,
            "case_wait_words": 0,
        })

    # Word completion drives most logic.
    if not state.just_finished_word or not state.last_completed_word:
        return state

    cur_slot = state.case_slot
    cur_wait = state.case_wait_words
    pos = state.last_word_pos

    # 1) Slot-filling POS — completed word consumes the expected slot
    #    (or diverts it into an NP chain).
    if pos in _SLOT_FILLERS:
        if cur_slot == CASE_NONE and cur_wait == 0:
            new_slot = CASE_NONE
            new_wait = 0
        else:
            new_slot = CASE_NONE
            new_wait = 0
        # Even with slot cleared, we may RE-trigger OBJ after a
        # POSSESSIVE completion inside an NP (e.g., "of my ___" —
        # after "my", a noun is expected, not a pronoun; so just
        # clear and don't re-open).
        if new_slot != cur_slot or new_wait != cur_wait:
            return state.model_copy(update={
                "case_slot": new_slot,
                "case_wait_words": new_wait,
            })
        return state

    # 2) Preposition just completed → CASE_OBJ.
    if pos == POS_PREPOSITION:
        return state.model_copy(update={
            "case_slot": CASE_OBJ,
            "case_wait_words": 0,
        })

    # 3) Transitive verb just completed (read verb_transitivity set
    #    upstream in the same tick).
    if state.verb_transitivity == _VT_DO_EXPECTED and state.vt_wait_words == 0:
        if cur_slot != CASE_OBJ or cur_wait != 0:
            return state.model_copy(update={
                "case_slot": CASE_OBJ,
                "case_wait_words": 0,
            })

    # 4) Coordinating conjunction in a POST_OBJ clause → new CASE_SUBJ.
    if (
        pos == POS_CONJUNCTION
        and state.last_completed_word
        and state.last_completed_word.lower() in ("and", "but", "or")
    ):
        # Only if the clause_slot was at POST_OBJ before the conj (a
        # true inter-clause coordination). After update_clause_slot,
        # the conj typically resets clause_slot to FRESH — we trust
        # that as the trigger.
        if state.clause_slot == _SLOT_FRESH:
            return state.model_copy(update={
                "case_slot": CASE_SUBJ,
                "case_wait_words": 0,
            })

    # 5) Otherwise: age the counter if a slot is open.
    if cur_slot != CASE_NONE:
        new_wait = cur_wait + 1
        if new_wait >= _STALE_AT:
            return state.model_copy(update={
                "case_slot": CASE_NONE,
                "case_wait_words": 0,
            })
        if new_wait > _MAX_WAIT:
            new_wait = _MAX_WAIT
        if new_wait != cur_wait:
            return state.model_copy(update={
                "case_wait_words": new_wait,
            })

    return state
