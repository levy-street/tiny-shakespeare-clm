"""Tier 2 — linguistic state updates.

Read Tier 1 fields (tokens_seen, last_token_id) and the incoming token id.
Update fields that a linguistics textbook would recognize: clause depth,
word position, sentence type, morphology markers, verse meter, speaker
label FSM, and so on.

This is the stage to use when your rule has a name an NLP researcher would
give it.

Starts as a no-op. Split into multiple sub-stages (clause.py, word.py,
verse.py, morphology.py, etc.) as the logic grows; call them in sequence
from `update_linguistic`.
"""

from __future__ import annotations

from ..state import ModelState


def update_linguistic(state: ModelState, token_id: int) -> ModelState:
    return state
