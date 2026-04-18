"""Phonotactic red-flag close-out bias.

Reads `state.word_red_flags`, a persistent per-word count of
phonotactic warning events (4+ consonant cluster, 3+ vowel run,
rare mid-word letter, post-complete off-trie drift) maintained by
`pipeline/word_shape.py`. Unlike the local signals that reset
(consonants_since_vowel, vowels_since_consonant), this count
*persists across the word* — a word that had a cluster then a
vowel still shows the flag.

When the word has accumulated 2+ red flags AND the buffer has at
least 3 letters, the word is very likely gibberish. We boost space
and clausal punctuation to let it close.

No corpus statistics — the red-flag thresholds come from well-known
English phonotactic constraints, not corpus frequencies.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def red_flags_close_bias(
    word_red_flags: int,
    letter_run_len: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if word_red_flags < 3:
        return None
    if letter_run_len < 5:
        return None

    # Escalate with more flags.
    if word_red_flags >= 4:
        sc = 1.0
    else:
        sc = 0.5

    vec = [0.0] * VOCAB_SIZE
    if " " in VOCAB_INDEX:
        vec[VOCAB_INDEX[" "]] += sc
    if "," in VOCAB_INDEX:
        vec[VOCAB_INDEX[","]] += sc * 0.5
    if "." in VOCAB_INDEX:
        vec[VOCAB_INDEX["."]] += sc * 0.4
    if ";" in VOCAB_INDEX:
        vec[VOCAB_INDEX[";"]] += sc * 0.3
    if "\n" in VOCAB_INDEX:
        vec[VOCAB_INDEX["\n"]] += sc * 0.3
    return vec
