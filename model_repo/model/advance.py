"""The `advance` half of the two-function model API.

`advance(state, token_id)` threads the token through the stages in
`pipeline.PIPELINE` in order, returning the new state. Each stage is a
pure function and the state is immutable, so this function itself is
pure: same inputs always produce the same output.
"""

from __future__ import annotations

from .pipeline import PIPELINE
from .state import ModelState
from .vocab import VOCAB_SIZE


def advance(state: ModelState, token_id: int) -> ModelState:
    if not 0 <= token_id < VOCAB_SIZE:
        raise ValueError(f"token_id {token_id} out of range [0, {VOCAB_SIZE})")
    for stage in PIPELINE:
        state = stage(state, token_id)
    return state
