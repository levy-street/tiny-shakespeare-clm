"""Turn-scoped proper-noun rolodex.

Mirrors the add-logic of `proper_noun_memory` but scoped to the
CURRENT turn: the tuple resets at every turn boundary
(consecutive_newlines >= 2). Runs AFTER update_proper_noun_memory so
that when `proper_nouns_seen` acquires a new head entry we can mirror
it into `turn_rolodex`.

Why a turn-scoped sub-rolodex?
-------------------------------
The global rolodex (10 entries rolling) freely carries proper nouns
across scene / play boundaries. In samples this produces injections
like "Tamora" (Titus) and "Bertram" (All's Well) appearing in a HAMLET
turn simply because they were at the top of the global rolodex from
earlier speakers.

Real Shakespeare turns name characters that are IN THE CURRENT SCENE —
the rolodex should scope to the current turn/scene to preserve this
consistency. A turn-local tuple is a coarse but reliable proxy for
"current scene" because speaker-turn boundaries (\\n\\n) are the only
reliable scene-hint in raw character text.

No corpus statistics — the behavior is mechanical.
"""

from __future__ import annotations

from ..state import ModelState

_MAX_TURN_ROLODEX = 5


def update_turn_rolodex(state: ModelState, token_id: int) -> ModelState:
    # Turn boundary (blank line between turns) — reset.
    if state.consecutive_newlines >= 2:
        if state.turn_rolodex:
            return state.model_copy(update={"turn_rolodex": ()})
        return state

    # Mirror: whenever `proper_nouns_seen` head just changed, prepend
    # that same entry into turn_rolodex (deduped). Detect the change
    # by comparing the heads.
    pns = state.proper_nouns_seen
    if not pns:
        return state
    head = pns[0]
    tr = state.turn_rolodex
    if tr and tr[0] == head:
        return state  # no change

    # Prepend head; dedup; cap size.
    deduped: list[str] = [head]
    for existing in tr:
        if existing != head and len(deduped) < _MAX_TURN_ROLODEX:
            deduped.append(existing)
    new_tr = tuple(deduped)
    if new_tr == tr:
        return state
    return state.model_copy(update={"turn_rolodex": new_tr})
