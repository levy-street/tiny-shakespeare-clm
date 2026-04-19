"""Tier 3 — enjambment density tracking.

Maintains `enjambment_density` and `prev_line_enjambed`. Fires only at
newlines that close a non-empty, non-speaker-label, verse-plausible
line. A line is:

  * ENJAMBED if the char immediately before \n was a letter (the line
    ran over into the next line syntactically).
  * END-STOPPED if the char before \n was a terminal/clausal
    punctuation (. , ; : ? ! — ').

The rolling density mean-reverts with each closed line. Speaker-turn
boundaries (consecutive_newlines >= 2) reset density.

Must run AFTER `update_linguistic` (needs prev_line_length,
consecutive_newlines, prev_char, last_char) and after `update_rhyme`
is fine too — ordering is independent.

No corpus statistics.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


_ENJAMBMENT_UP: float = 0.35      # pull toward 1.0 on enjambed line
_ENJAMBMENT_DOWN: float = 0.30    # pull toward 0.0 on end-stopped line


def _is_letter(c: str) -> bool:
    return len(c) == 1 and (("a" <= c <= "z") or ("A" <= c <= "Z"))


def update_enjambment(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-turn change — reset to neutral 0.5.
    if ch == "\n" and state.consecutive_newlines >= 2:
        if state.enjambment_density == 0.5 and not state.prev_line_enjambed:
            return state
        return state.model_copy(
            update={
                "enjambment_density": 0.5,
                "prev_line_enjambed": False,
            }
        )

    # A newline that just closed a non-blank, non-label line.
    # state.last_char here is the char BEFORE this \n (since
    # linguistic update already shifted last_char → prev_char at the
    # advance earlier, but counters updates after). Actually under
    # current pipeline ordering, update_basic_counters runs first,
    # making state.last_char == the CURRENT token char. But we are
    # the enjambment stage consuming this token. We need to detect
    # "this is a newline closing a line" using state.last_char == \n
    # and state.prev_char == <prior-line-char>.
    #
    # The rhyme update uses the same pattern: state.last_char == \n
    # and state.prev_char != \n and state.prev_char != ":"
    # (guarded by state.prev_line_length > 0).
    if (
        state.last_char == "\n"
        and state.prev_char != "\n"
        and state.prev_char != ":"
        and state.prev_line_length > 0
    ):
        # Only update on verse-plausible lines (length 12-70) so prose
        # paragraphs don't dilute the signal.
        if not (12 <= state.prev_line_length <= 70):
            return state

        was_enjambed = _is_letter(state.prev_char)
        cur = state.enjambment_density
        if was_enjambed:
            new_density = cur + (1.0 - cur) * _ENJAMBMENT_UP
        else:
            new_density = cur * (1.0 - _ENJAMBMENT_DOWN)

        # Clamp to [0, 1].
        if new_density < 0.0:
            new_density = 0.0
        elif new_density > 1.0:
            new_density = 1.0

        return state.model_copy(
            update={
                "enjambment_density": new_density,
                "prev_line_enjambed": was_enjambed,
            }
        )

    return state
