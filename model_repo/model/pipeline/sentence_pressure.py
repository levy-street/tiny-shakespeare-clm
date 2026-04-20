"""Tier 2 — sentence_pressure update.

Computes a signed scalar in roughly [-2.0, 2.0]:

  * NEGATIVE (keep going) — sentence is structurally incomplete.
    Pulled down by:
      - missing subject
      - missing finite verb (stronger)
      - np_open (article/possessive/preposition with no head yet)
      - subord_depth > 0 (inside a subordinate clause)
      - last completed word is a conjunction (and/but/or/nor/yet/so)
      - last completed word is a preposition
      - last completed word is an article / possessive
      - very few words yet (words_in_sentence < 3)

  * POSITIVE (ready to end) — backbone is full and sentence has run long.
    Pulled up by:
      - has subject AND has verb AND words_in_sentence >= 5
      - words_in_sentence >= 10 with full backbone

Runs once per token, late in the pipeline (after update_np_head,
update_clause_slot, update_subord, update_sentence_backbone, update_pos).
A single scalar read by a predict layer.
"""

from __future__ import annotations

from ..state import ModelState
from .pos import (
    POS_ARTICLE,
    POS_AUX_VERB,
    POS_CONJUNCTION,
    POS_MODAL,
    POS_POSSESSIVE,
    POS_PREPOSITION,
    POS_WH,
    POS_INTERJECTION,
    POS_NEGATION,
)


def update_sentence_pressure(state: ModelState, token_id: int) -> ModelState:
    # Inside a speaker label the sentence machinery is paused.
    if state.speaker_label_state != 0:
        if state.sentence_pressure != 0.0:
            return state.model_copy(update={"sentence_pressure": 0.0})
        return state

    p = 0.0

    words = state.words_in_sentence
    has_subj = state.sentence_has_subject
    has_verb = state.sentence_has_verb

    # --- Backbone deficits ---
    if not has_verb:
        # Strong "keep going" — a sentence without a finite verb is open.
        if words >= 2:
            p -= 0.6
        elif words >= 1:
            p -= 0.3
    if not has_subj:
        if words >= 2:
            p -= 0.3
        elif words >= 1:
            p -= 0.15

    # --- NP head waiting ---
    if state.np_open:
        # Expecting a noun head; closing here is wrong.
        p -= 0.7
        # Longer waits are even worse.
        if state.np_wait_words >= 2:
            p -= 0.2

    # --- Subordinate clause depth ---
    sd = state.subord_depth
    if sd >= 1:
        p -= 0.35
    if sd >= 2:
        p -= 0.25

    # --- Last-word POS (function-word demand) ---
    last_pos = state.last_word_pos
    # We only trust last_word_pos if we're NOT currently mid-word
    # (letter_run_len == 0) — otherwise last_completed_word is stale
    # relative to what we've emitted in the current buffer.
    if state.letter_run_len == 0 and state.last_completed_word:
        if last_pos == POS_CONJUNCTION:
            # "and", "but", "or", "nor" — a clause must follow.
            p -= 1.0
        elif last_pos == POS_PREPOSITION:
            # Covers np_open overlap; add a little more.
            p -= 0.3
        elif last_pos == POS_ARTICLE or last_pos == POS_POSSESSIVE:
            # Covers np_open overlap; add a little more.
            p -= 0.25
        elif last_pos == POS_AUX_VERB or last_pos == POS_MODAL:
            # "is", "are", "shall", "will" — expecting a verb/complement.
            p -= 0.45
        elif last_pos == POS_WH:
            p -= 0.35
        elif last_pos == POS_NEGATION:
            p -= 0.35
        elif last_pos == POS_INTERJECTION:
            # "O", "alas" — strong expectation of continuation
            # unless it's the whole turn.
            if words >= 1:
                p -= 0.15

    # --- Very short sentence, no terminator yet ---
    if words < 2 and state.letter_run_len == 0:
        # Barely started — don't close.
        p -= 0.4

    # --- Positives (ready to end) ---
    if has_subj and has_verb and not state.np_open and sd == 0:
        if words >= 10:
            p += 0.5
        elif words >= 6:
            p += 0.3
        elif words >= 4:
            p += 0.15

    # Cap range so downstream layers have a predictable magnitude.
    if p > 2.0:
        p = 2.0
    elif p < -2.5:
        p = -2.5

    if abs(p - state.sentence_pressure) > 1e-4:
        return state.model_copy(update={"sentence_pressure": p})
    return state
