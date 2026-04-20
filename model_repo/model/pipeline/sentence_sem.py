"""Sentence-scoped semantic field lock.

Orthogonal to `update_noun_class` (per-step single-class bias) and to
`scene_topic` (cross-turn). This tracks within-sentence semantic
field stability.

State transitions (see state/schema.py comment):

  * `sentence_sem_strength == 0`: no candidate field yet. A completed
    content noun of class C establishes a candidate: sem_field=C,
    strength=1.
  * `strength == 1`: a candidate is held but not yet locked. A second
    completed noun of the SAME class locks the field
    (strength=2). A noun of a different class replaces the candidate
    (field=new_class, strength=1).
  * `strength >= 2`: LOCKED. An in-field noun increments strength
    (cap 3). An out-of-field noun is ignored — a single drift
    doesn't break the locked frame, consistent with Shakespeare's
    "heart, hand, tongue — and sword" kind of structure where a
    stray class-switch doesn't topple the dominant register.

Resets on PUNCT_END and speaker-turn boundary.

Must run AFTER `update_noun_class` so the classifier has a fresh view
of the current completed word. Only reads state.last_completed_word
directly (re-classifies), so dependency on update_noun_class is
ordering only — not field-sharing (we want to consider the word even
if last_noun_class didn't change because the same class was held).
"""

from __future__ import annotations

from ..state import ModelState
from ..state.noun_classes import NC_NONE, classify
from .linguistic import PUNCT_END


def update_sentence_sem(state: ModelState, token_id: int) -> ModelState:
    ch = state.last_char

    # Sentence-end reset.
    if state.last_char_class == PUNCT_END:
        if state.sentence_sem_field == 0 and state.sentence_sem_strength == 0:
            return state
        return state.model_copy(update={
            "sentence_sem_field": 0,
            "sentence_sem_strength": 0,
        })

    # Speaker-turn boundary reset.
    if state.consecutive_newlines >= 2 and ch == "\n":
        if state.sentence_sem_field == 0 and state.sentence_sem_strength == 0:
            return state
        return state.model_copy(update={
            "sentence_sem_field": 0,
            "sentence_sem_strength": 0,
        })

    # Only update on word completion outside speaker labels.
    if not state.just_finished_word or not state.last_completed_word:
        return state
    if state.speaker_label_state != 0:
        return state

    cls = classify(state.last_completed_word)
    if cls == NC_NONE:
        return state

    cur_field = state.sentence_sem_field
    cur_strength = state.sentence_sem_strength

    if cur_strength == 0:
        # No candidate: establish.
        return state.model_copy(update={
            "sentence_sem_field": cls,
            "sentence_sem_strength": 1,
        })

    if cur_strength == 1:
        # Unlocked candidate.
        if cls == cur_field:
            # Second hit same class — LOCK.
            return state.model_copy(update={
                "sentence_sem_strength": 2,
            })
        # Different class: replace candidate with most recent noun.
        return state.model_copy(update={
            "sentence_sem_field": cls,
            "sentence_sem_strength": 1,
        })

    # cur_strength >= 2: LOCKED.
    if cls == cur_field:
        new_strength = min(cur_strength + 1, 3)
        if new_strength == cur_strength:
            return state
        return state.model_copy(update={
            "sentence_sem_strength": new_strength,
        })
    # Out-of-field mention in a locked frame: ignore (allow one drift).
    return state
