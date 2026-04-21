"""Predict layer — coordinator-parallelism first-letter bias.

Reads `state.coord_echo_pos` and `state.coord_echo_pending` set by
pipeline/coord.py. When pending AND at the word-start position
(letter_run_len == 0, last char is space / apostrophe slot), nudge
the first letter of the upcoming word toward letters that typically
open words of the echoed POS class.

Magnitudes are modest (~+0.12 to +0.25) — the echo is a preference,
not a hard constraint, and the existing letter-level priors
(startword / next_word / trie) still dominate. We only fire on POS
classes whose first-letter distribution is narrow enough to be
informative:

  PROPER_NOUN  — capitalize: strong push on A-Z, penalty on lowercase.
  PRONOUN      — {i, t(hou/hey), h(e/im/er), s(he), w(e), y(ou)}
  POSSESSIVE   — {m(y/ine), t(hy/hine), h(is/er), o(ur), y(our)}
  PREPOSITION  — {o(f/n), i(n/nto), t(o), w(ith), f(or/rom), b(y), a(t/s), u(pon)}
  NEGATION     — {n(ot/o/ay/ever/or)}
  NUMBER       — {o(ne), t(wo/hree), f(our/ive), s(ix/even), e(ight), n(ine/ought)}
  WH           — {w(ho/hat/hen/here/hy/hich), h(ow)}
  INTERJECTION — {o, a(h/las), h(a/ark), l(o), f(ie)}
  AUX_VERB     — {i(s), a(re/m), w(as/ere/ert), b(e/een), h(ath/ad/as), d(o/oth/id)}
  MODAL        — {s(hall/hould), w(ill/ould), c(ould/an), m(ay/ight/ust)}
  ARTICLE      — {t(he), a(n)}  (narrow)

Broad POS classes (NOUN, VERB, VERB_ING, VERB_ED, ADJECTIVE, ADVERB,
CONJUNCTION, UNKNOWN) yield no informative first-letter bias and are
skipped.

For each class the letter list is a prior-knowledge catalog of the
typical starter letters. No corpus statistics — these are derived
from the small closed-class inventories inherent in the language.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# POS constants (must match pipeline/pos.py — keep in sync).
POS_UNKNOWN = 0
POS_ARTICLE = 1
POS_PRONOUN = 2
POS_POSSESSIVE = 3
POS_PREPOSITION = 4
POS_CONJUNCTION = 5
POS_AUX_VERB = 6
POS_MODAL = 7
POS_INTERJECTION = 8
POS_NEGATION = 9
POS_ADVERB = 10
POS_VERB_ING = 11
POS_VERB_ED = 12
POS_NOUN = 13
POS_ADJECTIVE = 14
POS_PROPER_NOUN = 15
POS_VERB = 16
POS_NUMBER = 17
POS_WH = 18


# Per-POS typical first-letter catalogs (lowercase). Each entry is the
# minimal set of letters that covers the closed-class inventory of
# that POS.
_POS_LETTERS: dict[int, str] = {
    POS_ARTICLE: "ta",
    POS_PRONOUN: "ithswy",
    POS_POSSESSIVE: "mthoy",
    POS_PREPOSITION: "oitwfbau",
    POS_NEGATION: "n",
    POS_NUMBER: "otfsen",
    POS_WH: "wh",
    POS_INTERJECTION: "oahlf",
    POS_AUX_VERB: "iawbhd",
    POS_MODAL: "swcm",
}

# Magnitudes per POS class (the push applied to each matching letter
# in lowercase; the UPPER equivalent gets half-strength since lowercase
# is the common case for these closed-class words).
_POS_PUSH: dict[int, float] = {
    POS_ARTICLE: 0.18,
    POS_PRONOUN: 0.15,
    POS_POSSESSIVE: 0.18,
    POS_PREPOSITION: 0.14,
    POS_NEGATION: 0.30,  # only 'n' — sharp signal
    POS_NUMBER: 0.15,
    POS_WH: 0.20,
    POS_INTERJECTION: 0.15,
    POS_AUX_VERB: 0.12,
    POS_MODAL: 0.15,
}


def coord_parallel_bias(
    coord_echo_pending: bool,
    coord_echo_pos: int,
    coord_echo_first_letter: str,
    coord_echo_was_capital: bool,
    letter_run_len: int,
    speaker_label_state: int,
    word_buffer: str,
) -> list[float] | None:
    if not coord_echo_pending:
        return None
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None
    if word_buffer:
        # Not exactly at a clean word-start (mid-word or post-apos).
        return None

    vec = [0.0] * VOCAB_SIZE
    any_push = False

    # Case echo — if pre-coord word was a mid-sentence cap (proper-
    # noun-like), the next word is overwhelmingly likely to also be
    # capitalized. ("Romeo and Juliet", "Cassio and Iago".)
    if coord_echo_was_capital:
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += 0.75
        for ch in "abcdefghijklmnopqrstuvwxyz":
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] -= 0.30
        any_push = True

    # Alliterative first-letter echo — Shakespeare's coord pairs are
    # frequently alliterative: "fair and foul", "kith and kin",
    # "tooth and nail", "safe and sound", "short and sweet".
    if coord_echo_first_letter:
        fl = coord_echo_first_letter
        idx_lo = VOCAB_INDEX.get(fl)
        idx_hi = VOCAB_INDEX.get(fl.upper())
        if coord_echo_was_capital:
            if idx_hi is not None:
                vec[idx_hi] += 0.45
        else:
            if idx_lo is not None:
                vec[idx_lo] += 0.35
            if idx_hi is not None:
                vec[idx_hi] += 0.12
        any_push = True

    # POS-based echoes are deliberately NOT applied here — we tried
    # them and they yielded zero signal on BPC because (a) many
    # content-word POSes end up UNKNOWN, and (b) for the narrow
    # closed-class cases, the existing startword/next_word machinery
    # already pushes those letters. The first-letter + case echo
    # above captures the bulk of the coord-parallelism regularity.

    if not any_push:
        return None
    return vec
