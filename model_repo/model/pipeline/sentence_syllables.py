"""Sentence-scoped syllable counter.

Maintains `syllables_in_sentence` (running count), and at each PUNCT_END
shifts it into `prev_sentence_syllables` / `prev_prev_sentence_syllables`
for downstream rhythmic-parallelism biasing. Resets on speaker-turn
boundary.

Runs BEFORE `update_prosody` in the pipeline so the pre-tick value of
`state.in_vowel_group` (set by last tick's prosody) is what we read
here. That lets us detect a new-syllable event identically to prosody
without needing to stash a prev-tick shadow copy.

Syllables are measured as consonant->vowel transitions where a vowel is
a/e/i/o/u (lower or upper). Matches pipeline/prosody.py's definition.
"""

from __future__ import annotations

from ..state import ModelState
from .linguistic import PUNCT_END


_VOWELS: frozenset[str] = frozenset("aeiouAEIOU")


def _is_letter(ch: str) -> bool:
    return len(ch) == 1 and (("a" <= ch <= "z") or ("A" <= ch <= "Z"))


def update_sentence_syllables(state: ModelState, token_id: int) -> ModelState:
    ch = state.last_char

    # Speaker-turn boundary: wipe per-sentence + memory so each turn's
    # rhythm is local. Matches rhyme.py / sentence_backbone.py behavior.
    if state.consecutive_newlines >= 2 and ch == "\n":
        if (
            state.syllables_in_sentence == 0
            and state.prev_sentence_syllables == 0
            and state.prev_prev_sentence_syllables == 0
        ):
            return state
        return state.model_copy(update={
            "syllables_in_sentence": 0,
            "prev_sentence_syllables": 0,
            "prev_prev_sentence_syllables": 0,
        })

    # Sentence-end: shift memory, reset current.
    if state.last_char_class == PUNCT_END:
        cur = state.syllables_in_sentence
        if cur == 0 and state.prev_sentence_syllables == 0:
            # Nothing to do.
            return state
        return state.model_copy(update={
            "syllables_in_sentence": 0,
            "prev_sentence_syllables": cur,
            "prev_prev_sentence_syllables": state.prev_sentence_syllables,
        })

    # Count a new syllable on consonant->vowel transition INSIDE a
    # word. Equivalent to prosody.py's `starts_new_syllable` logic but
    # gated at word-interior (letter_run_len >= 1 means we're in or
    # just entered a word).
    if not _is_letter(ch):
        return state
    is_vowel = ch in _VOWELS
    if not is_vowel:
        return state
    # state.in_vowel_group here reflects the prev tick's value (prosody
    # runs AFTER us). If prev tick was not in a vowel group and this
    # tick is a vowel letter inside a word, a new syllable begins.
    if state.in_vowel_group:
        return state
    if state.letter_run_len < 1:
        # Defensive: should be >=1 once we're committed to the letter.
        return state
    # Don't count inside speaker labels — labels aren't sentence-body.
    if state.speaker_label_state != 0:
        return state

    return state.model_copy(update={
        "syllables_in_sentence": state.syllables_in_sentence + 1,
    })
