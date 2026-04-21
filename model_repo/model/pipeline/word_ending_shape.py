"""Tier 2 — word-ending shape score.

Maintains `word_ending_shape_score`, a 3-valued integer that scores
whether terminating the current word_buffer right now would yield a
recognizable English word-shape:

  2 — buffer is itself a complete known word (in the word-trie's
      complete-word set). Treat as definitively endable.
  1 — buffer's tail (last 3 chars) appears as the last 3 chars of
      some known word. This is a strict "the ending exists in real
      English" test — not a pattern-match, but an exact tail lookup
      over the word-trie's own vocabulary.
  0 — neither. Terminating now would yield a word-shaped nonsense
      fragment whose last three letters never end any real word.

The score is recomputed on every letter character and reset to 0 on
any non-letter, non-apostrophe character.

The tail-set comes from the word-trie's hand-authored word list —
the same list used by `predict/word_trie.py`. This is not a corpus
statistic (we never touch the corpus); it's a derived index of the
model's own hand-curated vocabulary.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB

# Cache import — word_trie is optional; compute trie complete-word set
# and tail-trigram set at import time.
from ..predict.word_trie import _WORDS as _TRIE_WORDS  # type: ignore[attr-defined]

_COMPLETE_WORDS: frozenset[str] = frozenset(_TRIE_WORDS)

# Tail trigrams: the set of (last-3-char) endings over every known
# word of length >= 3. Tight enough to be a discriminator for
# gibberish tails ("drt" in "drymudrt" is never a real-word ending).
_TAIL_TRIGRAMS: frozenset[str] = frozenset(
    w[-3:].lower() for w in _TRIE_WORDS if len(w) >= 3
)
# Tail bigrams: fallback for length-2 buffers.
_TAIL_BIGRAMS: frozenset[str] = frozenset(
    w[-2:].lower() for w in _TRIE_WORDS if len(w) >= 2
)

_LETTERS: frozenset[str] = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def update_word_ending_shape(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # In speaker-label territory, we don't score.
    if state.speaker_label_state != 0:
        if state.word_ending_shape_score != 0:
            return state.model_copy(update={"word_ending_shape_score": 0})
        return state

    # On non-letter (and non-apostrophe) char, reset.
    if ch not in _LETTERS and ch != "'":
        if state.word_ending_shape_score != 0:
            return state.model_copy(update={"word_ending_shape_score": 0})
        return state

    wb = state.word_buffer
    if len(wb) < 2:
        if state.word_ending_shape_score != 0:
            return state.model_copy(update={"word_ending_shape_score": 0})
        return state

    wb_low = wb.lower()

    # Score 2: buffer is a complete known word.
    if wb_low in _COMPLETE_WORDS:
        new_score = 2
    elif len(wb_low) >= 3 and wb_low[-3:] in _TAIL_TRIGRAMS:
        new_score = 1
    elif len(wb_low) == 2 and wb_low in _TAIL_BIGRAMS:
        new_score = 1
    else:
        new_score = 0

    if new_score == state.word_ending_shape_score:
        return state
    return state.model_copy(update={"word_ending_shape_score": new_score})
