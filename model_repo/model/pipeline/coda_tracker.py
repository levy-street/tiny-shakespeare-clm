"""Tier 2 — post-vowel consonant cluster tracker.

Maintains `post_vowel_cluster`: the lowercase string of consonants
emitted since the most recent vowel within the current word.

Purpose: a syllable-coda FSM signal. In English, once a vowel has
appeared in the current word, subsequent consonants form either the
coda of that syllable or the onset of a new syllable. The literal
cluster string (e.g., "nd", "st", "ngth", "mpt") lets downstream
layers check:

  * Is the cluster a legal English word-final coda? → endable.
  * Is it a legal *prefix* of some coda? → still extending legally.
  * Is it NOT any legal coda-prefix? → phonotactic dead-end; close now.

Rules:
  - Reset to "" on word boundary (non-letter, non-apostrophe char).
  - Reset to "" on any vowel (a/e/i/o/u) — the vowel starts a new
    nucleus, so whatever consonants follow will be a fresh cluster.
  - Reset to "" on 'y' when it acts as a vowel (has a prior vowel in
    the current word; English treats word-internal y as vowel).
  - Skip (unchanged) on apostrophe — contractions like "can't" or
    "hath'd" don't interrupt the coda phonotactics.
  - Skip in speaker-label territory (proper names have loose
    phonotactics). Reset-and-skip.
  - Append the lowercased consonant ONLY when at least one vowel has
    appeared in the current word. Before the first vowel, the
    accumulating consonants are the onset of the first syllable and
    shouldn't be scored as a coda.
  - Cap the stored length at 8 (deeper than any real English coda).

Runs after `update_flow` (which maintains consonants_since_vowel /
vowels_in_word) so the updated counters are already in state.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


_LETTERS: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
_STRICT_VOWELS: frozenset[str] = frozenset("aeiouAEIOU")


def _is_letter(ch: str) -> bool:
    return ch in _LETTERS


def update_coda_tracker(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-label: reset.
    if state.speaker_label_state != 0:
        if state.post_vowel_cluster:
            return state.model_copy(update={"post_vowel_cluster": ""})
        return state

    # Apostrophe: invisible to coda tracking.
    if ch == "'":
        return state

    # Non-letter: word boundary, reset.
    if not _is_letter(ch):
        if state.post_vowel_cluster:
            return state.model_copy(update={"post_vowel_cluster": ""})
        return state

    ch_low = ch.lower()

    # Vowel (strict a/e/i/o/u): reset cluster — a new nucleus has started.
    if ch_low in ("a", "e", "i", "o", "u"):
        if state.post_vowel_cluster:
            return state.model_copy(update={"post_vowel_cluster": ""})
        return state

    # y: treat as a vowel when the current word already has a strict
    # vowel earlier in it ("rhythm" → y is vowel; "yes" → y is consonant).
    if ch_low == "y":
        if state.vowels_in_word >= 1:
            # Acts as vowel — reset cluster.
            if state.post_vowel_cluster:
                return state.model_copy(update={"post_vowel_cluster": ""})
            return state
        # y as consonant at word-start: falls through to the consonant
        # branch below — but it's the first letter of the word, so no
        # vowel has occurred yet. The append-guard below will keep the
        # cluster empty anyway.

    # Consonant. Append ONLY if we've already seen a vowel in this word
    # (otherwise we're in the onset of the first syllable, which is
    # NOT a coda).
    if state.vowels_in_word < 1:
        # Still in pre-first-vowel onset. Keep cluster empty.
        if state.post_vowel_cluster:
            return state.model_copy(update={"post_vowel_cluster": ""})
        return state

    # Append this consonant to the cluster.
    new_cluster = state.post_vowel_cluster + ch_low
    if len(new_cluster) > 8:
        new_cluster = new_cluster[-8:]
    if new_cluster == state.post_vowel_cluster:
        return state
    return state.model_copy(update={"post_vowel_cluster": new_cluster})
