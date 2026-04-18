"""Formulaic-phrase progress tracker.

Maintains `state.formula_node` — the current position in a precomputed
trie of common Shakespearean multi-word formulas. Updated at word
completion based on the word that just closed.

On word completion:
  - If the completed word advances the current formula path, move to
    the deeper node.
  - Else, if the completed word starts a fresh formula (is a child of
    root), jump to that fresh path — this lets overlapping formulas
    resynchronize without losing a step.
  - Else, reset to root (0).

Also resets to root on sentence-end punctuation (., ?, !) — formulas
rarely span sentence boundaries — and on speaker-label transitions
(a new speaker starts with a fresh formula context).

Runs after the POS stage (which updates last_completed_word and the
content-word rolling tuple), before clause_slot and friends.
"""

from __future__ import annotations

from ..predict.formula_trie import advance_node
from ..state import ModelState


def update_formula(state: ModelState, token_id: int) -> ModelState:
    # Reset on sentence-end or speaker-label transitions.
    if state.last_char in (".", "?", "!"):
        if state.formula_node != 0:
            return state.model_copy(update={"formula_node": 0})
        return state
    # Reset on blank-line (end of speaker turn / scene change).
    if state.consecutive_newlines >= 2:
        if state.formula_node != 0:
            return state.model_copy(update={"formula_node": 0})
        return state

    # Only advance on word completion.
    if not state.just_finished_word:
        return state

    word = state.last_completed_word
    if not word:
        if state.formula_node != 0:
            return state.model_copy(update={"formula_node": 0})
        return state

    new_node = advance_node(state.formula_node, word)
    if new_node != state.formula_node:
        return state.model_copy(update={"formula_node": new_node})
    return state
