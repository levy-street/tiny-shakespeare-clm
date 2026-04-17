"""Predict package.

`predict(state)` is built by composing:
  - a base unigram distribution (once at import),
  - a context-class bias layer (see `context.py`),
  - a letter-level bigram bias layer (see `bigram.py`),
  - a start-of-word bias layer (see `startword.py`).

Each layer adds log-biases to the current logit vector, after which the
combined vector is re-normalized and returned.
"""

from __future__ import annotations

from .compose import predict

__all__ = ["predict"]
