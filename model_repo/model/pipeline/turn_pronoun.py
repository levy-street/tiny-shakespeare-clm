"""Tier 2/3 — turn pronoun profile (soliloquy / direct-address / mixed).

Counts first-person-singular and second-person pronoun completions
within the current speaker turn and classifies the turn mode:

    0 = insufficient evidence (< 3 total 1st/2nd person pronouns)
    1 = I-dominant   (i >= 3 AND i >= 2*you)  — soliloquy-ish
    2 = you-dominant (you >= 3 AND you >= 2*i) — direct-address
    3 = mixed        (both >= 2 and ratio within 2x) — dialogue

Reset on speaker-turn boundary (consecutive_newlines >= 2).

Runs after update_linguistic / update_pos (so `just_finished_word` and
`last_completed_word` are current) but before predict reads the mode.

No corpus statistics — the pronoun sets are prior-knowledge closed-class
inventories.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


_I_PRONOUNS: frozenset[str] = frozenset({
    "i", "my", "me", "mine", "myself",
})

_YOU_PRONOUNS: frozenset[str] = frozenset({
    "thou", "thee", "thy", "thine", "thyself",
    "you", "ye", "your", "yours", "yourself",
})


def _classify(i: int, you: int) -> int:
    total = i + you
    if total < 3:
        return 0
    if i >= 3 and i >= 2 * you:
        return 1
    if you >= 3 and you >= 2 * i:
        return 2
    if i >= 2 and you >= 2:
        return 3
    return 0


def update_turn_pronoun(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-turn boundary reset.
    if ch == "\n" and state.consecutive_newlines >= 2:
        if (
            state.turn_i_pronouns == 0
            and state.turn_you_pronouns == 0
            and state.turn_pronoun_mode == 0
        ):
            return state
        return state.model_copy(update={
            "turn_i_pronouns": 0,
            "turn_you_pronouns": 0,
            "turn_pronoun_mode": 0,
        })

    # Skip inside speaker-label.
    if state.speaker_label_state != 0:
        return state

    # Only tick on word completion.
    if not state.just_finished_word:
        return state
    word = state.last_completed_word
    if not word:
        return state

    cur_i = state.turn_i_pronouns
    cur_you = state.turn_you_pronouns

    new_i = cur_i
    new_you = cur_you
    if word in _I_PRONOUNS:
        new_i = cur_i + 1
    elif word in _YOU_PRONOUNS:
        new_you = cur_you + 1

    if new_i == cur_i and new_you == cur_you:
        return state

    new_mode = _classify(new_i, new_you)

    if (
        new_i == state.turn_i_pronouns
        and new_you == state.turn_you_pronouns
        and new_mode == state.turn_pronoun_mode
    ):
        return state
    return state.model_copy(update={
        "turn_i_pronouns": new_i,
        "turn_you_pronouns": new_you,
        "turn_pronoun_mode": new_mode,
    })
