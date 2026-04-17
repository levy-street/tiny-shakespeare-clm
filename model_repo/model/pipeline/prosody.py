"""Tier 3 — prosody / syllable tracking.

Runs after `update_linguistic`. Reads the incoming character and updates:
  - in_vowel_group: True iff the new char is part of a vowel cluster.
  - syllables_in_word: incremented at every consonant->vowel transition
    inside a word; reset at word boundaries.
  - syllables_in_line: incremented at every consonant->vowel transition
    inside the line; reset at newlines.
  - prev_line_syllables: captured when a newline closes a non-blank line.

Rationale: Shakespeare's verse is dominantly iambic pentameter (10
syllables per line, 11 with feminine endings). Tracking the syllable
count within a line lets the predict layer strongly favor a newline at
positions 9–11 when we're inside a verse passage (verse_score > 0), and
suppress the newline before we've reached a plausible pentameter cadence.

Syllables are measured as consonant->vowel transitions where a vowel is
a/e/i/o/u (lower or upper). This is a coarse proxy but captures the
gross cadence cue without any statistics over the corpus.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB

_VOWELS: frozenset[str] = frozenset("aeiouAEIOU")


def _is_vowel(ch: str) -> bool:
    return ch in _VOWELS


def _is_letter(ch: str) -> bool:
    return len(ch) == 1 and (
        ("a" <= ch <= "z") or ("A" <= ch <= "Z")
    )


def update_prosody(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Determine the incoming char's vowel/letter status.
    is_vowel = _is_vowel(ch)
    is_letter = _is_letter(ch)

    # A new syllable is counted at the consonant->vowel transition
    # (i.e., the incoming char is a vowel AND we were not already in a
    # vowel group). This counts each vowel cluster exactly once.
    starts_new_syllable = is_vowel and not state.in_vowel_group

    # Updates.
    if ch == "\n":
        # Line just ended. Capture syllable count before resetting.
        if state.syllables_in_line > 0:
            prev_line_syllables = state.syllables_in_line
        else:
            prev_line_syllables = state.prev_line_syllables
        syllables_in_line = 0
        syllables_in_word = 0
        in_vowel_group = False
    elif not is_letter:
        # Word boundary (space, punct, apostrophe, etc.). Reset the
        # word syllable count; keep the line count.
        prev_line_syllables = state.prev_line_syllables
        syllables_in_line = state.syllables_in_line
        syllables_in_word = 0
        in_vowel_group = False
    else:
        # Letter inside a word.
        prev_line_syllables = state.prev_line_syllables
        if starts_new_syllable:
            syllables_in_line = state.syllables_in_line + 1
            syllables_in_word = state.syllables_in_word + 1
        else:
            syllables_in_line = state.syllables_in_line
            syllables_in_word = state.syllables_in_word
        in_vowel_group = is_vowel

    return state.model_copy(
        update={
            "syllables_in_line": syllables_in_line,
            "syllables_in_word": syllables_in_word,
            "in_vowel_group": in_vowel_group,
            "prev_line_syllables": prev_line_syllables,
        }
    )
