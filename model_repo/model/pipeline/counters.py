"""Tier 1 — base counters.

First stage in the pipeline. Unconditional bookkeeping. Later stages read
these fields to decide what to do.
"""

from __future__ import annotations

from ..state import ModelState


def update_basic_counters(state: ModelState, token_id: int) -> ModelState:
    return state.model_copy(
        update={
            "tokens_seen": state.tokens_seen + 1,
            "last_token_id": token_id,
        }
    )
