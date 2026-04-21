"""English "q must be followed by u" orthographic rule.

In English, the letter 'q' is almost always followed by 'u' — so tightly
that it's a near-universal orthographic convention:

  queen, quick, quiet, quote, quest, queer, quench, quiver, quill,
  equal, equity, equip, aqua, acquire, require, inquire, antique

Exceptions are loanwords (qat, qi, qoph) essentially absent from
Shakespeare. The only other Shakespearean pattern is word-final 'q'
(which doesn't occur) or 'q' at the edge of a name (Qnet, etc. —
also absent).

Samples drift occasionally produces implausible q-followups like
"uqur", "qhrs", "qct" — the existing onset_cluster penalizes "q + non-
u" at letter_run_len==1 but NOT at mid-word positions.

This layer: whenever `word_buffer` ends in 'q' (any position in the
word), boost 'u' very strongly and penalize all other letters.

Gates:
  * word_buffer last char == 'q' or 'Q'
  * speaker_label_state == 0

No corpus statistics — universal English orthographic rule.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_U_BOOST = 2.0        # Push "u" sharply
_OTHER_LETTER_PEN = -2.5  # Penalize other letters
# Word-terminators at this position are rare but not impossible (word
# might be ending on "q" — essentially never in English). Penalize
# mildly so the preferred move is to emit "u".
_TERM_PEN = -1.5
_TERMINATORS = (" ", ",", ".", ";", ":", "!", "?", "\n")


def qu_rule_bias(
    word_buffer: str,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if not word_buffer:
        return None
    last = word_buffer[-1]
    if last not in ("q", "Q"):
        return None

    vec = [0.0] * VOCAB_SIZE

    # Boost 'u' and 'U' (uppercase pretty unlikely mid-word but keep).
    if "u" in VOCAB_INDEX:
        vec[VOCAB_INDEX["u"]] += _U_BOOST
    if "U" in VOCAB_INDEX:
        vec[VOCAB_INDEX["U"]] += _U_BOOST * 0.5

    # Penalize all OTHER ASCII letters.
    for ch in "abcdefghijklmnoprstvwxyz":  # exclude 'u', 'q'
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += _OTHER_LETTER_PEN
    for ch in "ABCDEFGHIJKLMNOPRSTVWXYZ":  # exclude 'U', 'Q'
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += _OTHER_LETTER_PEN * 0.7
    # Don't penalize 'q' / 'Q' again (already a separate rule).

    # Word-terminators: mild penalty (word essentially never ends on q).
    for t in _TERMINATORS:
        idx = VOCAB_INDEX.get(t)
        if idx is not None:
            vec[idx] += _TERM_PEN

    return vec
