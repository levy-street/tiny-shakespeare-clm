"""Tier 2/3 — coarse semantic noun-class tracking.

At word completion, classify `state.last_completed_word` against the
12-class noun dictionary in `state.noun_classes`. Update:

  - last_noun_class: most recently matched class id; persists across
    intervening non-noun words so the bias can span a short "noun →
    adj → noun" phrase.
  - noun_class_age: completed-words since last_noun_class was set;
    memory cleared at age >= 8.

At speaker-turn boundary, reset class memory.
"""

from __future__ import annotations

from ..state import ModelState
from ..state.noun_classes import classify, NC_NONE
from ..vocab import VOCAB


def update_noun_class(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-turn boundary: reset.
    if ch == "\n" and state.consecutive_newlines >= 2:
        if state.last_noun_class == 0 and state.noun_class_age == 0:
            return state
        return state.model_copy(update={
            "last_noun_class": 0,
            "noun_class_age": 0,
        })

    if not state.just_finished_word or not state.last_completed_word:
        return state

    cls = classify(state.last_completed_word)
    if cls != NC_NONE:
        return state.model_copy(update={
            "last_noun_class": cls,
            "noun_class_age": 0,
        })

    if state.last_noun_class == 0:
        return state
    new_age = state.noun_class_age + 1
    if new_age >= 8:
        return state.model_copy(update={
            "last_noun_class": 0,
            "noun_class_age": 0,
        })
    return state.model_copy(update={"noun_class_age": new_age})
