"""Mid-word capitalization integrity bias.

Enforces a simple orthographic rule that English (and Shakespeare's
early-modern spelling) follows almost without exception:

  * Once a word has started lowercase, NO uppercase letter appears
    anywhere else in the word.
  * Once a word has emitted its capital first letter (proper noun
    or sentence-opener), subsequent letters are almost always
    lowercase. The only common exceptions are hyphenated compounds
    and all-caps names — but those are followed by a hyphen or
    punctuation before the second uppercase, so the in-the-same-
    letter-run position is still lowercase.

The fix: a sharp negative bias on all uppercase codepoints whenever
`letter_run_len >= 1` and we're not inside a speaker label (which has
its own FSM — speaker labels are often all-caps).

Does not fire at the FIRST letter of a word (letter_run_len == 0)
because that's where the capitalization is legitimately chosen.

No corpus statistics — this is an orthographic invariant.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def word_cap_integrity_bias(
    letter_run_len: int,
    current_word_started_cap: bool,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if letter_run_len < 1:
        return None

    # The rule: after the first letter of a word, uppercase is almost
    # never legal. Slightly stronger penalty for a lowercase-started
    # word (impossible) than a capitalized word (rare compound).
    if current_word_started_cap:
        # "Hamlet" → next letter must be lowercase. Upper would be a
        # compound like "McBeth", which in Shakespeare is vanishingly
        # rare. Still penalize firmly.
        penalty = -6.0
    else:
        # "speak" → next letter absolutely cannot be upper. Nearly
        # impossible orthographic violation.
        penalty = -8.5

    vec = [0.0] * VOCAB_SIZE
    for ch in _UPPER:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] = penalty
    return vec
