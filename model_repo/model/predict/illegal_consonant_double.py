"""Illegal same-consonant-doubling penalty.

Complement to `illegal_vowel_double`. English allows many doubled
consonants (ll, nn, ss, tt, pp, ff, mm, rr, dd, gg, bb, cc, zz, kk
in a few words), but a specific subset is essentially never valid in
Shakespearean English:

  hh  — never (only modern compound words like "bathhouse" with
         hidden boundary; Shakespeare never doubles h)
  jj  — never
  qq  — never
  vv  — essentially never ("savvy" is modern slang; no Shakespeare)
  ww  — essentially never
  xx  — essentially never

Legal (from common Shakespeare vocab): ll, nn, ss, tt, pp, ff, mm,
rr, dd, gg, bb, cc, zz, kk (rarely), yy? no (yy is in vowel layer)

Samples produce substrings like "hhwwi", "vvent", "qqur" — these
should be blocked. The existing phonotactic layer covers within-word
illegal trigrams but doesn't target the adjacent-same-consonant-
double specifically at letter_run_len 1+.

Gates:
  * word_buffer last char is in the illegal-double set
  * speaker_label_state == 0
  * letter_run_len >= 1

No corpus statistics — rules from English orthography + Shakespeare
lexical knowledge.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Per-letter penalty for emitting the same consonant again.
# Only letters that are near-universally illegal to double get a
# non-zero penalty here; legal doublings (ll, nn, ss, tt, pp, ...)
# receive no bias.
_DOUBLE_PENALTY: dict[str, float] = {
    "h": -3.0,   # "hh" — never in Shakespeare
    "j": -3.5,   # "jj" — never
    "q": -3.5,   # "qq" — never (q is always followed by u)
    "v": -2.8,   # "vv" — essentially never
    "w": -3.0,   # "ww" — essentially never
    "x": -3.5,   # "xx" — essentially never
}


def illegal_consonant_double_bias(
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

    vec = [0.0] * VOCAB_SIZE
    if last in VOCAB_INDEX:
        vec[VOCAB_INDEX[last]] += pen
    up = last.upper()
    if up in VOCAB_INDEX:
        vec[VOCAB_INDEX[up]] += pen
    return vec
