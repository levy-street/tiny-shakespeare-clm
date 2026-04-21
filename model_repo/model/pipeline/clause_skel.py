"""Pipeline stage — clause skeleton FSM.

Tracks the proposition-building state of the current clause:

  0 EMPTY        — fresh clause, nothing committed yet
  1 SUBJ_OPEN    — NP opener (DET/POSS/ADJ) appeared but head not closed
  2 SUBJ_DONE    — subject head (NOUN/PROPER_NOUN/PRONOUN) completed
  3 VERB_DONE    — finite verb (+ optional AUX/MODAL prefix) completed
  4 COMP_DUE     — verb took an OBJ/PRED complement slot (DET/POSS/PREP
                   seen after verb), complement NP/PP owed
  5 CLAUSE_DONE  — predicate filled; clause ready to close

Resets to EMPTY on:
  * PUNCT_END  (. ? !)
  * PUNCT_MID  (, ; :)
  * Coordinator completion (and/or/but/nor/yet/so), which opens a new
    clause. Coordinator itself stays in PREV state momentarily; we
    simply drop to EMPTY so the next word starts the new clause.
  * Speaker-label entry / speaker-turn boundary

clause_skel_age increments on each just_finished_word and resets on
any reset above.

Reads `last_word_pos` from update_pos and `phrase_slot` from
update_phrase_slot, so runs AFTER both of those. Runs after
update_function_word_chain to keep the "tier 2 POS reactions" pass
sequential.

No corpus statistics.
"""

from __future__ import annotations

from ..state import ModelState
from .linguistic import PUNCT_END, PUNCT_MID
from .pos import (
    POS_ADJECTIVE,
    POS_ARTICLE,
    POS_AUX_VERB,
    POS_CONJUNCTION,
    POS_MODAL,
    POS_NEGATION,
    POS_NOUN,
    POS_POSSESSIVE,
    POS_PREPOSITION,
    POS_PRONOUN,
    POS_PROPER_NOUN,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
)


SK_EMPTY = 0
SK_SUBJ_OPEN = 1
SK_SUBJ_DONE = 2
SK_VERB_DONE = 3
SK_COMP_DUE = 4
SK_CLAUSE_DONE = 5


_NP_OPEN_TAGS = frozenset({POS_ARTICLE, POS_POSSESSIVE, POS_ADJECTIVE})
_NP_HEAD_TAGS = frozenset({POS_NOUN, POS_PROPER_NOUN, POS_PRONOUN})
_VERB_TAGS = frozenset({POS_VERB, POS_VERB_ED, POS_VERB_ING})
# AUX / MODAL before the main verb — transparent, don't trigger
# VERB_DONE on their own. They signal a predicate is forming.
_PREVERB_TAGS = frozenset({POS_AUX_VERB, POS_MODAL})
# Coordinator words that RESTART the clause skeleton: and/or/but/nor/
# yet/so. Classified as POS_CONJUNCTION by pos.py. Reset on any
# conjunction completion.
_COORDINATOR_TAG = POS_CONJUNCTION


def _reset_if_needed(state: ModelState) -> ModelState | None:
    """Handle PUNCT / speaker-label / newline resets. Returns a new state
    if a reset fired, else None."""
    if state.speaker_label_state != 0:
        if state.clause_skel != 0 or state.clause_skel_age != 0:
            return state.model_copy(update={
                "clause_skel": SK_EMPTY,
                "clause_skel_age": 0,
            })
        return state

    if state.last_char_class in (PUNCT_END, PUNCT_MID):
        if state.clause_skel != 0 or state.clause_skel_age != 0:
            return state.model_copy(update={
                "clause_skel": SK_EMPTY,
                "clause_skel_age": 0,
            })
        return state

    if state.consecutive_newlines >= 2:
        if state.clause_skel != 0 or state.clause_skel_age != 0:
            return state.model_copy(update={
                "clause_skel": SK_EMPTY,
                "clause_skel_age": 0,
            })
        return state

    return None


def update_clause_skel(state: ModelState, token_id: int) -> ModelState:
    reset = _reset_if_needed(state)
    if reset is not None:
        return reset

    # Only advance on word completion.
    if not state.just_finished_word:
        return state

    tag = state.last_word_pos
    cur = state.clause_skel
    age = state.clause_skel_age + 1

    # Coordinator completion resets the skeleton (next word begins a
    # new clause). Keep age bumped so predict knows we just reset.
    if tag == _COORDINATOR_TAG:
        new_skel = SK_EMPTY
        new_age = 0
    elif tag in _PREVERB_TAGS:
        # AUX/MODAL — if we had a subject, move to "verb about to form";
        # predict treats this like "expect VERB next". Internally we
        # keep SUBJ_DONE, since the actual verb hasn't landed yet.
        new_skel = cur
        new_age = age
    elif tag in _VERB_TAGS:
        # Finite main verb. Can appear at SUBJ_DONE (normal) or EMPTY
        # (imperative, e.g., "Speak!"). In either case, predicate has
        # started — advance to VERB_DONE.
        if cur in (SK_EMPTY, SK_SUBJ_OPEN, SK_SUBJ_DONE):
            new_skel = SK_VERB_DONE
            new_age = age
        elif cur == SK_VERB_DONE:
            # Chained verb — usually odd; keep VERB_DONE.
            new_skel = SK_VERB_DONE
            new_age = age
        elif cur in (SK_COMP_DUE, SK_CLAUSE_DONE):
            # A second verb after predicate is unusual; treat as a new
            # clause-internal verb (keep CLAUSE_DONE).
            new_skel = cur
            new_age = age
        else:
            new_skel = SK_VERB_DONE
            new_age = age
    elif tag in _NP_OPEN_TAGS:
        # DET/POSS/ADJ — NP is opening.
        if cur == SK_EMPTY:
            new_skel = SK_SUBJ_OPEN
            new_age = age
        elif cur == SK_SUBJ_OPEN:
            new_skel = SK_SUBJ_OPEN  # stay, still building subject NP
            new_age = age
        elif cur == SK_SUBJ_DONE:
            # After subject, an adjective/determiner opens a complement
            # NP. Move toward predicate's object slot.
            new_skel = SK_COMP_DUE
            new_age = age
        elif cur == SK_VERB_DONE:
            new_skel = SK_COMP_DUE
            new_age = age
        elif cur == SK_COMP_DUE:
            new_skel = SK_COMP_DUE  # still building complement NP
            new_age = age
        else:  # CLAUSE_DONE
            new_skel = cur
            new_age = age
    elif tag in _NP_HEAD_TAGS:
        # Noun/proper noun/pronoun — a head lands.
        if cur in (SK_EMPTY, SK_SUBJ_OPEN):
            new_skel = SK_SUBJ_DONE
            new_age = age
        elif cur == SK_SUBJ_DONE:
            # Apposition or second NP — keep SUBJ_DONE.
            new_skel = SK_SUBJ_DONE
            new_age = age
        elif cur == SK_VERB_DONE:
            # Object NP head landed directly (no det needed, e.g., for
            # pronouns or bare plurals) — predicate is filled.
            new_skel = SK_CLAUSE_DONE
            new_age = age
        elif cur == SK_COMP_DUE:
            # Complement NP head lands — clause done.
            new_skel = SK_CLAUSE_DONE
            new_age = age
        else:  # CLAUSE_DONE
            new_skel = cur
            new_age = age
    elif tag == POS_PREPOSITION:
        # Preposition opens a PP. Treat as COMP_DUE if we were past
        # the verb; as NP-opening otherwise.
        if cur == SK_EMPTY:
            new_skel = SK_SUBJ_OPEN  # "Of all the things X..." — PP can
            # head a subject, but usually opens a modifier; treat
            # softly as SUBJ_OPEN so we still expect a noun.
            new_age = age
        elif cur in (SK_SUBJ_OPEN, SK_SUBJ_DONE):
            new_skel = cur  # PP inside NP modifier — stay.
            new_age = age
        elif cur in (SK_VERB_DONE, SK_CLAUSE_DONE):
            new_skel = SK_COMP_DUE
            new_age = age
        elif cur == SK_COMP_DUE:
            new_skel = SK_COMP_DUE
            new_age = age
        else:
            new_skel = cur
            new_age = age
    elif tag == POS_NEGATION:
        # Negation is transparent; don't advance.
        new_skel = cur
        new_age = age
    else:
        # Unknown / other — stay.
        new_skel = cur
        new_age = age

    # Cap age.
    if new_age > 15:
        new_age = 15

    if new_skel == cur and new_age == state.clause_skel_age:
        return state
    return state.model_copy(update={
        "clause_skel": new_skel,
        "clause_skel_age": new_age,
    })
