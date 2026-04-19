"""Tier 2 — per-line coherence counters.

Maintains two counters of how many words completed ON-TRIE vs OFF-TRIE
in the current verse/prose line.

  line_ontrie_words  — count of completed words that are members of
                       COMPLETE_WORDS (recognized English vocabulary).
  line_offtrie_words — count of completed words NOT in COMPLETE_WORDS
                       (letter-n-gram hallucinations / drift).

Reset rule: both counters zero whenever the cursor is just past a
newline character (state.chars_since_newline == 0 after
update_basic_counters). So fresh-line state is always (0, 0).

Why this exists:
  Per-word drift_streak tracks consecutive off-trie runs, but doesn't
  answer "is THIS LINE salvageable?" — a structural question whose
  answer times the decision to newline-out early. The predict
  consumer (predict/line_coherence.py) uses this to push newline at
  word-end when the line has accumulated multiple garbage words with
  little real vocabulary.

Must run AFTER update_basic_counters (just_finished_word,
last_completed_word, chars_since_newline are fresh) and ideally
AFTER update_drift (for consistency with drift_streak).

No corpus statistics — the trie membership classification comes from
the hand-curated COMPLETE_WORDS vocabulary already used elsewhere.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB
from ..predict.word_trie import COMPLETE_WORDS


def update_line_coherence(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-label territory: skip accounting (a speaker name isn't
    # a "line word" in the prose sense, and coherence should reset on
    # speaker turn anyway via the \n-reset rule).
    if state.speaker_label_state != 0:
        if state.line_ontrie_words != 0 or state.line_offtrie_words != 0:
            return state.model_copy(update={
                "line_ontrie_words": 0,
                "line_offtrie_words": 0,
            })
        return state

    # Newline just emitted: reset counters. `chars_since_newline == 0`
    # means the cursor sits right after a \n character.
    if state.chars_since_newline == 0:
        if state.line_ontrie_words != 0 or state.line_offtrie_words != 0:
            return state.model_copy(update={
                "line_ontrie_words": 0,
                "line_offtrie_words": 0,
            })
        return state

    # Only update on word completion.
    if not state.just_finished_word or not state.last_completed_word:
        return state

    lcw = state.last_completed_word
    # Normalize for lookup — strip surrounding apostrophes ('tis, o'er).
    lookup = lcw
    if lookup and lookup.startswith("'"):
        lookup = lookup[1:]
    if lookup and lookup.endswith("'"):
        lookup = lookup[:-1]

    on_trie = lookup in COMPLETE_WORDS or lookup.lower() in COMPLETE_WORDS

    if on_trie:
        new_on = state.line_ontrie_words + 1
        if new_on > 20:
            new_on = 20
        if new_on != state.line_ontrie_words:
            return state.model_copy(update={"line_ontrie_words": new_on})
        return state
    else:
        new_off = state.line_offtrie_words + 1
        if new_off > 20:
            new_off = 20
        if new_off != state.line_offtrie_words:
            return state.model_copy(update={"line_offtrie_words": new_off})
        return state
