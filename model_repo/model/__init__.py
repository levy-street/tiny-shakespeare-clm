"""Public API — the two-function character-level language model.

    from model import ModelState, advance, predict, VOCAB

    state = ModelState()
    logprobs = predict(state)
    state = advance(state, token_id)
"""

from .advance import advance
from .predict import predict
from .state import ModelState
from .vocab import VOCAB, VOCAB_INDEX, VOCAB_SIZE

__all__ = [
    "ModelState",
    "advance",
    "predict",
    "VOCAB",
    "VOCAB_INDEX",
    "VOCAB_SIZE",
]
