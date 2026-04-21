"""Illegal same-vowel-doubling penalty.

English (and Shakespeare's Early Modern English) allow very few
doubled vowels inside a word:

  ee  — common (see, bee, tree, keep, deep, sleep, feet, ...)
  oo  — common (too, soon, moon, book, look, foot, blood, ...)
  ii  — essentially never (only foreign loanwords like "skiing",
        "radii"; vanishingly rare in Shakespeare)
  uu  — essentially never in Shakespeare; only modern loans
        ("vacuum", "continuum")
  aa  — essentially never in Shakespeare; only loanwords
        ("aardvark", "bazaar")
  yy  — never in English

Samples drift produces "faate", "raeu", "liemafyso", "p'ebrohn'eydenda"
-style substrings where a second same-vowel is emitted right after a
first. The existing letter_repeat_penalty provides ~0.10 at count 2
(mild) and the CV alternation layer only fires when 3+ vowels accumulate
without a consonant. This layer specifically targets the ADJACENT
doubled-vowel case — where the preceding letter is a vowel X and the
model is about to emit X again — with a sharper, position-aware penalty.

Legal doublings "ee" and "oo" get ZERO penalty (they're mainstream
English doublings). "ii", "uu", "aa", "yy" get strong penalties.

Gates:
  * word_buffer last char is a vowel letter in {a, i, u, y, e, o}
  * speaker_label_state == 0 (speaker labels have their own FSM)
  * letter_run_len >= 1

Fires regardless of on-trie / off-trie: same-vowel doubling is
pathological in both regimes, and the word-trie's positive vote for
legitimate doublings ("ee", "oo") isn't blocked by this layer because
those specific pairs get 0 penalty.

No corpus statistics — all rules from English orthography.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Penalty per same-vowel double. 0 = no penalty (legal). Negative
# values suppress emission.
_DOUBLE_PENALTY: dict[str, float] = {
    "e": 0.0,    # "ee" — legal (see, keep, bee, tree, ...)
    "o": 0.0,    # "oo" — legal (too, soon, book, moon, ...)
    "a": -3.0,   # "aa" — essentially never in Shakespeare
    "i": -3.5,   # "ii" — essentially never
    "u": -3.5,   # "uu" — essentially never
    "y": -4.0,   # "yy" — never in English
}


def illegal_vowel_double_bias(
    word_buffer: str,
    letter_run_len: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if letter_run_len < 1:
        return None
    if not word_buffer:
        return None
    last = word_buffer[-1].lower()
    pen = _DOUBLE_PENALTY.get(last)
    if pen is None:
        return None
    if pen == 0.0:
        return None  # legal doublet — no bias at all

    vec = [0.0] * VOCAB_SIZE
    # Penalize both cases of the same-vowel letter. Uppercase mid-word
    # is already heavily penalized by word_cap_integrity, but include
    # here for robustness.
    if last in VOCAB_INDEX:
        vec[VOCAB_INDEX[last]] += pen
    up = last.upper()
    if up in VOCAB_INDEX:
        vec[VOCAB_INDEX[up]] += pen
    return vec
