"""Speaker-trie strictness flags.

Consumed by `predict/speaker_label_strict.py`. The existing
`speaker_trie_bias` penalizes only same-case letters outside the
trie-legal next-set at the current `speaker_buffer`. This leaves
space / colon / apostrophe / newline / opposite-case letters
unconstrained at prefixes like "MOUNT" (where only "J" is legal, for
MOUNTJOY). The result: samples produce malformed speaker labels like
"MOUNT tssayl:" that slip through as FSM-valid but are semantically
garbage.

This stage surfaces three precise flags the strict-bias predict layer
can act on:

  speaker_trie_on_trie     — buffer is a prefix of at least one known
                             speaker name (empty prefix counts as True).
  speaker_trie_space_valid — " " is a legal next character (compound
                             labels like "KING HENRY" or "First Citizen").
  speaker_trie_colon_valid — ":" is a legal next character (we're at a
                             node that completes a canonical name).

All three are False outside a speaker label (speaker_label_state not
in {1, 2}) and when the buffer has drifted off-trie.

Runs AFTER update_linguistic (which sets speaker_label_state and
speaker_buffer). No corpus statistics — the trie is hand-authored.
"""

from __future__ import annotations

from ..predict.speaker_trie import SPEAKER_TRIE_NEXTS
from ..state import ModelState


def update_speaker_strict(state: ModelState, token_id: int) -> ModelState:
    sp = state.speaker_label_state
    if sp not in (1, 2):
        # Outside a label — ensure all flags are False.
        if (
            state.speaker_trie_on_trie
            or state.speaker_trie_space_valid
            or state.speaker_trie_colon_valid
        ):
            return state.model_copy(update={
                "speaker_trie_on_trie": False,
                "speaker_trie_space_valid": False,
                "speaker_trie_colon_valid": False,
            })
        return state

    buf = state.speaker_buffer
    nexts = SPEAKER_TRIE_NEXTS.get(buf)
    on_trie = nexts is not None
    if on_trie:
        space_ok = " " in nexts
        colon_ok = ":" in nexts
    else:
        space_ok = False
        colon_ok = False

    if (
        on_trie == state.speaker_trie_on_trie
        and space_ok == state.speaker_trie_space_valid
        and colon_ok == state.speaker_trie_colon_valid
    ):
        return state

    return state.model_copy(update={
        "speaker_trie_on_trie": on_trie,
        "speaker_trie_space_valid": space_ok,
        "speaker_trie_colon_valid": colon_ok,
    })
