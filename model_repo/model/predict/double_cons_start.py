"""Double-consonant word-start penalty.

English words almost never start with a doubled consonant. "frr-",
"tt-", "ss-" (outside onomatopoeia), "pp-", "bb-", "dd-", "kk-",
"rr-", "ll-" (rare, "llama"), "mm-", "nn-", "vv-", "ww-", "xx-",
"zz-" (except "zzz" sound-words) — all implausible.

Fires only at word-start position 1 (the second letter), penalizing
emission of the same consonant just emitted. Lowercase and uppercase
match.

No corpus stats — pure prior knowledge of English orthography.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

_CONSONANTS = frozenset("bcdfghjklmnpqrstvwxz")

# "ll" at word start is rare but occurs in Welsh loanwords (Llewellyn)
# and "llama". Give a smaller penalty.
_SOFT_DOUBLES = frozenset("l")
# Not all doubles are implausible at start: "oo" (ooze), "aa" (aardvark),
# "ee" (very rare). Vowels doubled are less penalized. We only penalize
# consonant doubles.


def double_consonant_penalty(first_letter: str) -> list[float] | None:
    if not first_letter:
        return None
    ch = first_letter.lower()
    if ch not in _CONSONANTS:
        return None
    penalty = -1.0 if ch in _SOFT_DOUBLES else -3.0
    vec = [0.0] * VOCAB_SIZE
    if ch in VOCAB_INDEX:
        vec[VOCAB_INDEX[ch]] = penalty
    up = first_letter.upper()
    if up in VOCAB_INDEX:
        vec[VOCAB_INDEX[up]] = penalty * 0.5
    return vec
