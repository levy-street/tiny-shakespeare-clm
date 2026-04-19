"""Negation-scope tracker — clause-level polarity memory.

Tracks whether the current sentence is negated and how recently the
triggering negation word fired. Emits three state fields:

  - negation_count         — number of negation-class words completed
                             in this sentence (0..5)
  - words_since_negation   — words since the most recent negation
                             (0..8), or 0 when no negation
  - last_negation_word     — lowercased most recent negation word
                             ("" when no active negation)

Triggers — a word is in the negation class if, lowercased, it is one
of the explicit set:

    not, no, nay, never, none, nothing, naught, nought, nor, neither

OR it equals "cannot" OR it ends with the contracted "n't" (don't,
hasn't, wasn't, can't, ain't, isn't, ...). Contraction detection is
purely lexical; no corpus statistics.

Reset policy:
  - Sentence-end punctuation (. ? !) clears all three.
  - Speaker-turn boundary (consecutive_newlines >= 2 AND last_char
    == "\n") clears all three.

Runs AFTER update_pos (which sets last_completed_word and just_
finished_word) so the "word just completed" signal is reliable.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


_NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "nay", "never", "none", "nothing", "naught",
    "nought", "nor", "neither",
})

_MAX_COUNT: int = 5
_MAX_WAIT: int = 8


def _is_negation(word: str) -> bool:
    if not word:
        return False
    w = word.lower()
    if w in _NEGATION_WORDS:
        return True
    if w == "cannot":
        return True
    # Contracted n't enclitic: don't, hasn't, can't, ain't, isn't,
    # wasn't, wouldn't, couldn't, shouldn't, haven't, shan't, etc.
    # Only accept when the word has at least one letter before n't.
    if len(w) >= 4 and w.endswith("n't"):
        return True
    return False


def update_negation(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Sentence-end punctuation: reset.
    if ch in ".?!":
        if (
            state.negation_count == 0
            and state.words_since_negation == 0
            and state.last_negation_word == ""
        ):
            return state
        return state.model_copy(update={
            "negation_count": 0,
            "words_since_negation": 0,
            "last_negation_word": "",
        })

    # Speaker-turn boundary: second+ consecutive newline.
    # update_basic_counters runs first and sets consecutive_newlines,
    # so reading it here is safe.
    if ch == "\n" and state.consecutive_newlines >= 2:
        if (
            state.negation_count == 0
            and state.words_since_negation == 0
            and state.last_negation_word == ""
        ):
            return state
        return state.model_copy(update={
            "negation_count": 0,
            "words_since_negation": 0,
            "last_negation_word": "",
        })

    # Word completion event — check for negation, else age the counter.
    if not state.just_finished_word or not state.last_completed_word:
        return state

    word = state.last_completed_word
    if _is_negation(word):
        new_count = min(state.negation_count + 1, _MAX_COUNT)
        return state.model_copy(update={
            "negation_count": new_count,
            "words_since_negation": 0,
            "last_negation_word": word.lower(),
        })

    # Non-negation word: age the counter if a negation is live.
    if state.negation_count > 0:
        new_wait = min(state.words_since_negation + 1, _MAX_WAIT)
        if new_wait != state.words_since_negation:
            return state.model_copy(update={
                "words_since_negation": new_wait,
            })
    return state
