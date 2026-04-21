"""Tier 2 — consecutive off-trie word streak within the current line.

Mirrors `line_coherence` accounting but keeps a CONSECUTIVE streak
rather than a total. An intervening on-trie word resets the streak.

Why two signals, not one?
-------------------------
`line_offtrie_words` sums all off-trie words on the line. If the line
reads "real garbage real garbage garbage", that's 3 off-trie / 2 on-
trie — line_coherence classifies it "failing" and pushes newline at
word boundaries.

But mid-word pressure deserves a sharper signal. A line with
"garbage garbage" at the tail (streak=2) is actively deteriorating —
even if an on-trie word happened earlier on the same line — and
mid-word termination pressure should be applied to the word being
built RIGHT NOW. `line_offtrie_streak` fills exactly that role: it
captures "we are CURRENTLY in a bad run", not "this line has had
some problems".

Update rule
-----------
  * speaker_label_state != 0: reset to 0.
  * newline just emitted (chars_since_newline == 0): reset to 0.
  * just_finished_word + on-trie last_completed_word: reset to 0.
  * just_finished_word + off-trie last_completed_word: increment (cap 8).

Runs AFTER update_line_coherence so that the on/off-trie
classification (done there) is consistent. Mirrors its normalization
logic (strip surrounding apostrophes on the last word for trie
lookup).

No corpus statistics — trie membership comes from the hand-curated
COMPLETE_WORDS.
"""

from __future__ import annotations

from ..state import ModelState
from ..predict.word_trie import COMPLETE_WORDS


_CAP = 8


def update_line_offtrie_streak(state: ModelState, token_id: int) -> ModelState:
    # Speaker-label territory: reset.
    if state.speaker_label_state != 0:
        if state.line_offtrie_streak != 0:
            return state.model_copy(update={"line_offtrie_streak": 0})
        return state

    # Newline just emitted: reset.
    if state.chars_since_newline == 0:
        if state.line_offtrie_streak != 0:
            return state.model_copy(update={"line_offtrie_streak": 0})
        return state

    # Only update on word completion.
    if not state.just_finished_word or not state.last_completed_word:
        return state

    lcw = state.last_completed_word
    lookup = lcw
    if lookup and lookup.startswith("'"):
        lookup = lookup[1:]
    if lookup and lookup.endswith("'"):
        lookup = lookup[:-1]
    on_trie = lookup in COMPLETE_WORDS or lookup.lower() in COMPLETE_WORDS

    if on_trie:
        if state.line_offtrie_streak != 0:
            return state.model_copy(update={"line_offtrie_streak": 0})
        return state

    new_val = state.line_offtrie_streak + 1
    if new_val > _CAP:
        new_val = _CAP
    if new_val == state.line_offtrie_streak:
        return state
    return state.model_copy(update={"line_offtrie_streak": new_val})
