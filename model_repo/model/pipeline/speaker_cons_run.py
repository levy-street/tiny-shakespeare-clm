"""Speaker-buffer trailing consonant-run counter.

Maintains `speaker_buffer_cons_run` — the length of the trailing
consecutive-consonant letter run inside `speaker_buffer` while the
speaker-label FSM is active (state 1 or 2). Resets to 0 on any
transition out of those states.

Must run AFTER `update_linguistic` (which sets speaker_buffer and
speaker_label_state). Complementary to `speaker_vowels` — vowels
counts how many vowels the whole buffer has; cons_run counts how
many consonants sit at the trailing end.

Phonotactic motivation: real Shakespeare speaker labels essentially
never contain 3+ adjacent consonants in the middle of a name. When
a sampled buffer accumulates a 3+ consonant tail (e.g. "MNN", "LRK",
"BRCKTH"), we're in gibberish-label territory even if earlier vowels
satisfy the whole-buffer vowel gate. No corpus statistics — this is
a structural property of English name phonotactics.
"""

from __future__ import annotations

from ..state import ModelState


_VOWELS_UP: frozenset[str] = frozenset("AEIOUY")


def update_speaker_cons_run(state: ModelState, token_id: int) -> ModelState:
    sp = state.speaker_label_state
    if sp not in (1, 2):
        if state.speaker_buffer_cons_run != 0:
            return state.model_copy(update={"speaker_buffer_cons_run": 0})
        return state

    buf = state.speaker_buffer
    if not buf:
        if state.speaker_buffer_cons_run != 0:
            return state.model_copy(update={"speaker_buffer_cons_run": 0})
        return state

    run = 0
    for ch in reversed(buf):
        # Speaker buffer only contains letters and spaces (per the FSM).
        # A space breaks the consonant run (e.g. "KING HENRY" — after
        # the space, a fresh name word begins so the cons_run restarts).
        if ch == " ":
            break
        if ch in _VOWELS_UP:
            break
        # Non-vowel letter (including the uppercase consonants).
        run += 1

    if run != state.speaker_buffer_cons_run:
        return state.model_copy(update={"speaker_buffer_cons_run": run})
    return state
