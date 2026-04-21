"""Penalize "x" emission right after a non-vowel.

In English, "x" almost always appears between a vowel and a vowel,
or between a vowel and a word-terminator:

  axe, wax, box, fox, six, fix, tax, vex, hex, nix
  axis, axle, exist, exit, expert, extent, oxen
  text, next, Mixtress (rare), sixth

The "x"-after-consonant case is essentially confined to one cluster
(compound morphology of "lynx", "jinx", "sphinx" — post-nasal) and
these are rare in Shakespeare. "rx", "tx", "px", "mx", "wx", "kx",
etc. are near-zero occurrences.

At letter_run_len >= 1, when word_buffer's last char is a non-vowel
consonant letter (not in {a,e,i,o,u,y,n} — we allow "n" for nx
patterns), suppress "x" emission.

Gates:
  * word_buffer non-empty, last char a consonant != n
  * speaker_label_state == 0

No corpus statistics — phonotactic rule from English.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_VOWEL_OR_NASAL = frozenset("aeiouyAEIOUYnN")  # "n" allowed (lynx, jinx)
_CONSONANT_RANGE = set("bcdfghjklmpqrstvwxz" + "BCDFGHJKLMPQRSTVWXZ")

_PENALTY = -3.5


def cons_x_rule_bias(
    word_buffer: str,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if not word_buffer:
        return None
    last = word_buffer[-1]
    # Must be a consonant and NOT 'n' (which allows -nx- cluster).
    if last in _VOWEL_OR_NASAL:
        return None
    if last not in _CONSONANT_RANGE:
        return None
    vec = [0.0] * VOCAB_SIZE
    if "x" in VOCAB_INDEX:
        vec[VOCAB_INDEX["x"]] += _PENALTY
    if "X" in VOCAB_INDEX:
        vec[VOCAB_INDEX["X"]] += _PENALTY
    return vec
