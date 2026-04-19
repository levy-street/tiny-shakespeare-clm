"""Tier 2: parenthetical-dash scope tracker.

Shakespeare's "--" is a mid-sentence parenthetical break. It opens a
short aside — typically 1–5 words, often followed by a newline and a
capitalised new-clause opener — and is closed by another "--", by
sentence-ending punctuation, or by a speaker-turn boundary.

State fields:
  * `in_dash_aside` (bool) — True while we're inside an unclosed --.
  * `chars_since_dash_open` (int) — characters emitted since the
    opening '--' completed (aged per char).
  * `words_since_dash_open` (int) — completed words since open.

Transitions:
  * A '-' followed by another '-' flips `in_dash_aside`: if currently
    inside, this is the closer; otherwise this is the opener.
    Detected by reading `state.last_char == '-'` AND emitted char ==
    '-'. When we open, also zero the counters.
  * On '.', '?', '!', the aside is force-closed (sentence end).
  * On speaker-turn boundary (consecutive_newlines >= 2) same —
    close and zero counters.
  * On any other char while `in_dash_aside`, age `chars_since_dash_open`.
  * On `just_finished_word` while `in_dash_aside`, bump
    `words_since_dash_open`. Cap at a small max (8) so it doesn't
    grow unbounded.

No corpus statistics — rules are syntactic bookkeeping.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB

_WORD_CAP = 8
_CHAR_CAP = 64


def update_dash_aside(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-turn boundary — force-close.
    if ch == "\n" and state.consecutive_newlines >= 2:
        if state.in_dash_aside or state.chars_since_dash_open or state.words_since_dash_open:
            return state.model_copy(update={
                "in_dash_aside": False,
                "chars_since_dash_open": 0,
                "words_since_dash_open": 0,
            })
        return state

    # Sentence-ending punctuation — force-close.
    if ch in (".", "?", "!"):
        if state.in_dash_aside or state.chars_since_dash_open or state.words_since_dash_open:
            return state.model_copy(update={
                "in_dash_aside": False,
                "chars_since_dash_open": 0,
                "words_since_dash_open": 0,
            })
        return state

    # Second '-' of a '--' sequence — toggle.
    if ch == "-" and state.last_char == "-":
        if state.in_dash_aside:
            # Closing dash.
            return state.model_copy(update={
                "in_dash_aside": False,
                "chars_since_dash_open": 0,
                "words_since_dash_open": 0,
            })
        # Opening dash.
        return state.model_copy(update={
            "in_dash_aside": True,
            "chars_since_dash_open": 0,
            "words_since_dash_open": 0,
        })

    # Inside an aside: age counters.
    if state.in_dash_aside:
        updates: dict = {}
        new_chars = min(state.chars_since_dash_open + 1, _CHAR_CAP)
        if new_chars != state.chars_since_dash_open:
            updates["chars_since_dash_open"] = new_chars
        if state.just_finished_word:
            new_words = min(state.words_since_dash_open + 1, _WORD_CAP)
            if new_words != state.words_since_dash_open:
                updates["words_since_dash_open"] = new_words
        if updates:
            return state.model_copy(update=updates)

    return state
