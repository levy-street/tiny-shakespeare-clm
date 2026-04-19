"""Consonant-vowel alternation bias for polysyllabic word interiors.

English polysyllabic words exhibit strong consonant-vowel alternation:
  - After a vowel, the next letter is more likely a consonant (CVC
    structure)
  - After a consonant, the next letter is more likely a vowel (onset+
    nucleus) — except in common consonant clusters (st, tr, nd, ng,
    ...).

This bias fires only INSIDE polysyllabic words — syllables_in_word >= 2
AND letter_run_len >= 4 — where the CVC rhythm is established and
the word's length alone isn't a reliable guide. It is GENTLE: the
letter-ngram layers (bigram, trigram) already encode much of this
pattern for common sequences; this layer adds a small prior for the
long-tail of rare n-grams that the n-gram tables under-serve.

Additionally, when consonants_since_vowel >= 3 (we have a 3-consonant
cluster in progress), we're likely in an implausible sequence for
word-interior position (real English clusters rarely exceed 3 letters,
and those tend to be word-initial like "str-"). Push toward vowel
emission.

Similarly, when vowels_since_consonant >= 2 (2 vowels without a
consonant), we've likely formed a diphthong — push toward consonant
to close the vowel cluster.

All weights from prior knowledge of English phonotactics — no corpus
statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_VOWEL_LETTERS: dict[str, float] = {
    "a": 1.0, "e": 1.0, "i": 0.85, "o": 0.85, "u": 0.60, "y": 0.40,
}
_COMMON_CONSONANTS: dict[str, float] = {
    "t": 1.0, "n": 0.95, "s": 0.90, "r": 0.85, "l": 0.75, "d": 0.70,
    "m": 0.55, "c": 0.55, "h": 0.50, "b": 0.40, "p": 0.40, "g": 0.35,
    "f": 0.35, "w": 0.30, "k": 0.25, "v": 0.25,
}


def _build_vowel_push_vec(scale: float) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch, w in _VOWEL_LETTERS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += scale * w
    # Penalize consonants slightly (mostly the cluster-extending ones).
    for ch in "bcdfghjklmnpqrstvwxz":
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] -= scale * 0.15
    return vec


def _build_cons_push_vec(scale: float) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch, w in _COMMON_CONSONANTS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += scale * w
    for ch in "aeiou":
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] -= scale * 0.15
    return vec


# Pre-build vectors at different intensity levels.
_VOWEL_PUSH_WEAK = _build_vowel_push_vec(0.08)
_VOWEL_PUSH_MED = _build_vowel_push_vec(0.14)
_VOWEL_PUSH_STRONG = _build_vowel_push_vec(0.24)
_CONS_PUSH_WEAK = _build_cons_push_vec(0.05)
_CONS_PUSH_MED = _build_cons_push_vec(0.10)


def cv_alternation_bias(
    syllables_in_word: int,
    letter_run_len: int,
    consonants_since_vowel: int,
    vowels_since_consonant: int,
    on_word_trie: bool,
    letters_off_trie: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Gentle C-V alternation bias for polysyllabic word interiors.
    Fires only when we're past the typical short-word zone
    (syllables_in_word >= 2 AND letter_run_len >= 4).
    Fires stronger when off-trie, where the ngram tables are noisier
    and a phonotactic prior helps more. Returns None when not
    applicable."""
    if speaker_label_state != 0:
        return None
    if letter_run_len < 4:
        return None
    if syllables_in_word < 2:
        return None

    # Consonant-cluster push: drive toward vowel emission when we've
    # accumulated 3+ consecutive consonants inside a polysyllabic word.
    if consonants_since_vowel >= 3:
        if consonants_since_vowel >= 4:
            return _VOWEL_PUSH_STRONG
        # Scale escalates with off-trie depth.
        if not on_word_trie and letters_off_trie >= 1:
            return _VOWEL_PUSH_MED
        return _VOWEL_PUSH_WEAK

    # Vowel-cluster push: drive toward consonant emission when we've
    # accumulated 2+ consecutive vowels inside a polysyllabic word
    # (real diphthongs are usually length 2; 3+ is implausible).
    if vowels_since_consonant >= 2:
        if vowels_since_consonant >= 3:
            return _CONS_PUSH_MED
        if not on_word_trie and letters_off_trie >= 1:
            return _CONS_PUSH_WEAK
        return None  # 2 vowels is common (diphthong) — no push

    return None
