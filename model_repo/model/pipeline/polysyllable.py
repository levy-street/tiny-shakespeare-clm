"""Tier 3 — polysyllable density tracking.

Maintains `polysyllable_density`: a rolling [0, 1] EMA over the recent
words' polysyllabic-ness. Updated at the transition WHEN a word closes
(letter → non-letter) outside speaker-label territory. Speaker-turn
boundary (consecutive_newlines >= 2) resets density to 0.5 (neutral).

Polysyllable estimate (no corpus statistics):
  * word length >= 7 characters, OR
  * vowel count >= 3 (counting a y that's not the first letter).

Words shorter than 2 letters are skipped (fragments / stray single
letters like "I", "O" are neither mono nor poly in a meaningful sense
for this rolling texture).

Must run AFTER `update_linguistic` (needs `just_finished_word`,
`last_completed_word`, `consecutive_newlines`) and AFTER `update_pos`
so POS of the completed word is available (not currently read but kept
as ordering convention).

No corpus statistics.
"""

from __future__ import annotations

from ..state import ModelState


_POLY_UP_MIX: float = 0.22    # EMA weight toward 1.0 on a polysyllabic close
_POLY_DOWN_MIX: float = 0.22  # EMA weight toward 0.0 on a monosyllabic close


_VOWELS: frozenset[str] = frozenset("aeiouAEIOU")


def _is_polysyllable(word: str) -> bool:
    if len(word) >= 7:
        return True
    # Vowel count (treating y as a vowel when it is NOT the first letter).
    n = 0
    for i, c in enumerate(word):
        if c in _VOWELS:
            n += 1
        elif (c == "y" or c == "Y") and i > 0:
            n += 1
    return n >= 3


def update_polysyllable(state: ModelState, token_id: int) -> ModelState:
    # Speaker-turn change — reset to neutral.
    if state.consecutive_newlines >= 2 and state.last_char == "\n":
        if state.polysyllable_density == 0.5:
            return state
        return state.model_copy(update={"polysyllable_density": 0.5})

    # Only act at the moment a word closes.
    if not state.just_finished_word:
        return state
    if state.speaker_label_state != 0:
        return state

    word = state.last_completed_word
    if len(word) < 2:
        return state

    is_poly = _is_polysyllable(word)
    cur = state.polysyllable_density
    if is_poly:
        new_d = cur + (1.0 - cur) * _POLY_UP_MIX
    else:
        new_d = cur * (1.0 - _POLY_DOWN_MIX)

    # Clamp to [0, 1].
    if new_d < 0.0:
        new_d = 0.0
    elif new_d > 1.0:
        new_d = 1.0

    if new_d == cur:
        return state
    return state.model_copy(update={"polysyllable_density": new_d})
