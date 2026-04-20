"""Speaker-buffer vowel counter.

Maintains `speaker_buffer_vowels` — count of vowels in the current
`speaker_buffer` while the speaker-label FSM is active (state 1 or 2).
Resets to 0 on any transition out of those states.

Must run AFTER `update_linguistic` (which sets speaker_buffer and
speaker_label_state).

The count feeds a predict layer that applies a phonotactic gate on
speaker labels: real Shakespeare speaker names always contain at
least one vowel, so a no-vowel buffer of length 2+ signals a phantom
label. No corpus statistics — vowel presence is a structural property
of English orthography.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


# Vowels in uppercase (speaker_buffer is uppercased).
_VOWELS_UP: frozenset[str] = frozenset("AEIOU")
# 'Y' is treated as a vowel at end of names like "TYBALT", "HENRY",
# "MARCY" — we include it below for the vowel gate (but conservatively
# since Y-initial names "YORK" would flag if Y weren't counted).
_VOWELS_UP_WITH_Y: frozenset[str] = frozenset("AEIOUY")


def update_speaker_vowels(state: ModelState, token_id: int) -> ModelState:
    sp = state.speaker_label_state
    if sp not in (1, 2):
        # Not in speaker-label territory — reset.
        if state.speaker_buffer_vowels != 0:
            return state.model_copy(update={"speaker_buffer_vowels": 0})
        return state

    # Recompute from the buffer — simple and authoritative. The buffer
    # is short (capped at 24) so this is cheap.
    buf = state.speaker_buffer
    if not buf:
        if state.speaker_buffer_vowels != 0:
            return state.model_copy(update={"speaker_buffer_vowels": 0})
        return state

    count = 0
    for ch in buf:
        if ch in _VOWELS_UP_WITH_Y:
            count += 1
    if count != state.speaker_buffer_vowels:
        return state.model_copy(update={"speaker_buffer_vowels": count})
    return state
