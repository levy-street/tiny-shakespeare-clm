"""Speaker-label off-trie run tracker.

Counts consecutive letters added to `speaker_buffer` (during FSM states
1 and 2) that have taken the buffer off the known-speaker-prefix trie.
Resets to 0 whenever the buffer is currently on-trie or when we're
outside speaker-label territory.

Must run AFTER update_linguistic (which sets speaker_buffer and
speaker_label_state). The value is consumed by the predict layer to
modulate ":" closure probability as the run grows.
"""

from __future__ import annotations

from ..predict.speaker_trie import is_speaker_prefix
from ..state import ModelState


def update_speaker_offtrie(state: ModelState, token_id: int) -> ModelState:
    sp = state.speaker_label_state
    if sp not in (1, 2):
        # Outside a speaker label — clear.
        if state.speaker_label_offtrie_run != 0:
            return state.model_copy(update={"speaker_label_offtrie_run": 0})
        return state

    buf = state.speaker_buffer
    # Empty buffer (just entered state 1): on-trie.
    if not buf:
        if state.speaker_label_offtrie_run != 0:
            return state.model_copy(update={"speaker_label_offtrie_run": 0})
        return state

    # Check trie membership. The speaker trie is built from UPPERCASE
    # canonical names, and speaker_buffer is uppercased as it's built.
    if is_speaker_prefix(buf):
        if state.speaker_label_offtrie_run != 0:
            return state.model_copy(update={"speaker_label_offtrie_run": 0})
        return state

    # Off-trie: grow the run (capped at 16 to avoid absurd values).
    new_run = min(state.speaker_label_offtrie_run + 1, 16)
    return state.model_copy(update={"speaker_label_offtrie_run": new_run})
