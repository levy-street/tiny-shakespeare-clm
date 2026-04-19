"""Tier 3 — caesura (mid-line pause) tracking.

Runs after `update_prosody` so `syllables_in_line` is current. Maintains:

  - caesura_syllable: the syllable position within the current line at
    which the last mid-line break fired (,  ;  : outside speaker-label,
    em-dash). -1 means no caesura yet in this line. Reset on newline.
  - has_caesura_this_line: True iff a caesura has fired in the current
    line already.

Caesura definition used here: any of `,  ;  :  -` appearing mid-line
(i.e., not the line-closing char, not inside a speaker label). Shakespeare's
iambic pentameter characteristically balances around a mid-line pause at
syllable 4, 5, or 6; prose similarly employs mid-clause punctuation for
breath. We don't restrict to verse mode in the state (let the consumer
decide), but we do restrict to speaker_label_state == 0 so the colon
after a SPEAKER_NAME doesn't count.

No corpus statistics — the caesura convention is a well-known feature
of English iambic verse.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB

_MID_PUNCT: frozenset[str] = frozenset({",", ";", ":", "-"})


def update_caesura(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Newline: reset (the line is over).
    if ch == "\n":
        if state.caesura_syllable == -1 and not state.has_caesura_this_line:
            return state
        return state.model_copy(
            update={
                "caesura_syllable": -1,
                "has_caesura_this_line": False,
            }
        )

    # Mid-line punctuation inside a body (non-speaker-label) position.
    # Colon inside a speaker label is the turn delimiter, not a caesura.
    if ch in _MID_PUNCT and state.speaker_label_state == 0:
        # Don't count a punctuation char that appears at the very
        # start of a line (e.g., a speaker name's leading dash or a
        # stray colon). syllables_in_line >= 2 ensures at least some
        # line content before marking a caesura.
        if state.syllables_in_line < 2:
            return state
        # Don't re-record if we already had the identical state (idempotent).
        if (
            state.has_caesura_this_line
            and state.caesura_syllable == state.syllables_in_line
        ):
            return state
        return state.model_copy(
            update={
                "caesura_syllable": state.syllables_in_line,
                "has_caesura_this_line": True,
            }
        )

    return state
