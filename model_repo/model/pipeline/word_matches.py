"""Tier 2 — graded trie-match count.

For the current `word_buffer`, counts how many COMPLETE known words
have that buffer as a prefix. Stored in `state.trie_match_count`.
The prior value is preserved in `prev_trie_match_count` so downstream
predict layers can detect the exact transition when the most recently
added letter took the count to zero.

Unlike `on_word_trie` (which is binary: on the trie at all? yes/no),
this is a graded signal:
  - count == 0: no known word remains possible — forced gibberish.
  - count == 1: exactly one completion left — very sharp prediction.
  - count between 2 and 5: narrow set — lean toward the few completions.
  - count >= 6: broad — near neutral.

Relies on `predict.word_trie.PREFIX_COMPLETE_COUNT` (precomputed at
import). Runs after `update_linguistic` (which sets `word_buffer`).
"""

from __future__ import annotations

from ..state import ModelState
from ..predict.word_trie import PREFIX_COMPLETE_COUNT


def update_word_matches(state: ModelState, token_id: int) -> ModelState:
    buf = state.word_buffer
    if not buf:
        new_count = 0
    else:
        new_count = PREFIX_COMPLETE_COUNT.get(buf, 0)
    if new_count == state.trie_match_count and state.prev_trie_match_count == state.trie_match_count:
        return state
    return state.model_copy(update={
        "prev_trie_match_count": state.trie_match_count,
        "trie_match_count": new_count,
    })
