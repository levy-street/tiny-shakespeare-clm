"""Mid-departure extension counter.

Maintains `state.mid_departure_extension`: the number of letters
written since the current word departed the trie, GATED on the
departure having happened at position 3 or 4 specifically.

Existing state already captures the primitives:
  * on_word_trie        — is the current buffer a valid prefix?
  * letters_off_trie    — how many letters since we first left the trie
  * offtrie_depart_pos  — the letter_run_len at which we first left

This stage just gates the computation into the specific regime that
the existing predict layers don't cover:

  * offtrie_depart_bias explicitly returns None for depart_pos in {3,4}
    (see predict/offtrie_depart.py comment at "elif depart_pos <= 4:
    return None"), intending that trie_recovery would handle it.
  * trie_recovery_bias was tuned with term_boost = 0.0, so nothing
    actually pushes the word to close in this regime.

Words in this regime are "had a plausible 3-4 letter real-word
prefix, stepped off, and are now trailing invented morphology" — e.g.
"etustartea" (depart ~3), "Fulfilm", "iegeohce" (depart 2-3).

Must run AFTER update_linguistic (which sets on_word_trie,
letters_off_trie, offtrie_depart_pos, word_buffer).

Reset rule: 0 whenever any gate fails (on-trie, depart_pos 1-2 or
>= 5, no word, speaker-label territory).
"""

from __future__ import annotations

from ..state import ModelState


def update_mid_departure(state: ModelState, token_id: int) -> ModelState:
    # Speaker-label territory: not a normal word.
    if state.speaker_label_state != 0:
        if state.mid_departure_extension != 0:
            return state.model_copy(update={"mid_departure_extension": 0})
        return state

    # No word in progress.
    if not state.word_buffer:
        if state.mid_departure_extension != 0:
            return state.model_copy(update={"mid_departure_extension": 0})
        return state

    # On trie — not departed.
    if state.on_word_trie:
        if state.mid_departure_extension != 0:
            return state.model_copy(update={"mid_departure_extension": 0})
        return state

    # Off trie but departure point is outside the "mid" regime. We
    # include dp in {2, 3, 4}: pos 2 is a very early departure (like
    # "et-" that never extended on-trie), which offtrie_depart_bias
    # handles only when letters_off_trie >= 2; our layer is additive
    # and gives word-end-letter preference that offtrie_depart lacks.
    dp = state.offtrie_depart_pos
    if dp < 2 or dp > 4:
        if state.mid_departure_extension != 0:
            return state.model_copy(update={"mid_departure_extension": 0})
        return state

    # Mid-departure, off trie. Value = letters_off_trie (letters past
    # the departure point, not counting the departure-causing letter
    # itself — same convention as offtrie_depart_bias).
    new_val = min(state.letters_off_trie, 12)
    if new_val == state.mid_departure_extension:
        return state
    return state.model_copy(update={"mid_departure_extension": new_val})
