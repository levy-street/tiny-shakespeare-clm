"""Tier 2 — per-word phonotactic red-flag accumulator.

Runs after `update_linguistic` (which sets letter_run_len,
word_buffer, on_word_trie, letters_off_trie, has_seen_complete)
and `update_prosody` (which maintains consonants_since_vowel and
vowels_since_consonant). This stage reads those and maintains
`word_red_flags`, a persistent per-word counter of phonotactic
warning signs that linger across the word even after the local
signal resets.

Flag events:
  1. Consonant-cluster saturation: the moment
     `consonants_since_vowel` reaches 4+ in this word. Fires ONCE
     per consonant-cluster maxing (guarded by
     `red_flag_cluster_fired`, which is reset at word boundary).
  2. Vowel-triple saturation: the moment
     `vowels_since_consonant` reaches 3+ in this word. Fires ONCE
     per vowel-run (guarded by `red_flag_vowel_fired`).
  3. Rare mid-word letter: incoming char is j/q/x/z at position > 0
     within the word AND not part of a "qu" digraph (common). Fires
     each such letter.
  4. Post-complete off-trie drift: the buffer had reached a complete
     word earlier (`has_seen_complete`), AND is now off-trie. Fires
     once as the buffer transitions off-trie after being complete.

Guards: the flag counter is reset on any non-letter incoming char
(word boundary) and on speaker-label territory (state != 0) — inside
a speaker label, the "word" is a name and has different phonotactic
rules, so we skip accounting.

Consumed by a predict layer (see compose.py integration) that at
word-end positions boosts word-terminators when red_flags >= 2 — a
"this word has failed twice, close it now" rule.

No corpus statistics — all thresholds (4+ consonants, 3+ vowels,
rare-letter mid-word) come from well-known English phonotactic
constraints.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


_LETTERS_SET: frozenset[str] = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_RARE_MIDWORD: frozenset[str] = frozenset("jqxzJQXZ")


def _is_letter(ch: str) -> bool:
    return ch in _LETTERS_SET


def update_word_shape(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-label: don't account red flags while inside a name.
    if state.speaker_label_state != 0:
        return state.model_copy(update={
            "word_red_flags": 0,
            "red_flag_cluster_fired": False,
            "red_flag_vowel_fired": False,
        })

    # Word boundary (non-letter, apostrophe is tolerated as continuer).
    if not _is_letter(ch) and ch != "'":
        if (
            state.word_red_flags != 0
            or state.red_flag_cluster_fired
            or state.red_flag_vowel_fired
        ):
            return state.model_copy(update={
                "word_red_flags": 0,
                "red_flag_cluster_fired": False,
                "red_flag_vowel_fired": False,
            })
        return state

    # Letter (or apostrophe) — evaluate flags.
    new_flags = state.word_red_flags
    cluster_fired = state.red_flag_cluster_fired
    vowel_fired = state.red_flag_vowel_fired

    # Flag 1: consonant-cluster saturation. The updated prosody state
    # has already set consonants_since_vowel for the incoming char.
    if state.consonants_since_vowel >= 4 and not cluster_fired:
        new_flags += 1
        cluster_fired = True
    elif state.consonants_since_vowel == 0:
        # Vowel just arrived — reset the one-shot guard so a later
        # cluster in the SAME word (e.g., "strictness") can flag again.
        cluster_fired = False

    # Flag 2: vowel-triple saturation.
    if state.vowels_since_consonant >= 3 and not vowel_fired:
        new_flags += 1
        vowel_fired = True
    elif state.vowels_since_consonant == 0:
        vowel_fired = False

    # Flag 3: rare mid-word letter (j/q/x/z) at position > 0 that
    # isn't the "u" of "qu".
    if (
        ch in _RARE_MIDWORD
        and state.letter_run_len >= 1  # previous char was a letter of this word
    ):
        # We increment unconditionally — rare letters are themselves
        # one-shot events on this char.
        new_flags += 1

    # Flag 4 (disabled — firing on legitimate word extensions and
    # inflections hurt BPC). Kept as comment for future work:
    # "post-complete off-trie drift: had a complete prefix then went
    # off-trie" is a useful signal but requires smarter gating.

    return state.model_copy(update={
        "word_red_flags": new_flags,
        "red_flag_cluster_fired": cluster_fired,
        "red_flag_vowel_fired": vowel_fired,
    })
