"""Committed-word letter bias.

When `state.committed_word` is set (by pipeline/word_commit.py), we've
decided — based on deep formula context — what the next word is going
to be. This layer biases each subsequent letter toward the target's
next letter, holding onto the commitment across positions instead of
relying on independent n-gram signals at every step.

Strength is substantial (+4.0 target letter) but not overwhelming,
so that if the underlying prior strongly disagrees (indicating a
formula-trie false positive), the prediction can still escape — the
pipeline stage will notice the mismatch and clear the commit on the
next step, restoring the normal letter flow.

At the final letter of the committed word (pos == len-1), we also
softly discourage word-continuation letters — once "thee" is emitted,
a space or punct should follow rather than a run-on "theee".

No corpus statistics — magnitudes are hand-tuned from prior knowledge
that Shakespearean formulas like "I pray thee", "my good lord",
"alas poor Yorick" commit to their targets robustly.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Magnitudes.
_TARGET_BOOST = 4.0        # boost on committed_word[pos]
_COMPETITOR_PEN = 0.35     # mild penalty on all other a-z at the same slot
_UPPER_MIRROR = 0.8        # also boost uppercase mirror at pos 0 (word-start)
_TERMINATOR_PEN_MIDWORD = 0.8  # penalize word-end chars while we have letters left
_TERMINATOR_BOOST_AT_END = 0.6  # gentle nudge toward space/punct once word done


def word_commit_bias(
    committed_word: str,
    committed_word_pos: int,
    letter_run_len: int,
) -> list[float] | None:
    if not committed_word:
        return None
    if committed_word_pos < 0 or committed_word_pos > len(committed_word):
        return None
    # Only apply when letter_run_len matches committed_word_pos — otherwise
    # state is stale or something unusual is happening; back off.
    if letter_run_len != committed_word_pos:
        return None

    vec = [0.0] * VOCAB_SIZE

    if committed_word_pos < len(committed_word):
        # Still emitting the word.
        target = committed_word[committed_word_pos]
        # Target letter, lowercase.
        idx = VOCAB_INDEX.get(target)
        if idx is not None:
            vec[idx] += _TARGET_BOOST
        # Competitor penalty on every other lowercase a-z.
        for ch in "abcdefghijklmnopqrstuvwxyz":
            if ch == target:
                continue
            i = VOCAB_INDEX.get(ch)
            if i is not None:
                vec[i] -= _COMPETITOR_PEN
        # At the first letter (pos 0) also boost the uppercase mirror
        # — sentence-start / verse-line-start conditions may demand it.
        if committed_word_pos == 0:
            up = target.upper()
            ui = VOCAB_INDEX.get(up)
            if ui is not None:
                vec[ui] += _UPPER_MIRROR
        # Penalize word-terminators mid-word — we still have letters to go.
        for term in " \n\t,.;:!?":
            ti = VOCAB_INDEX.get(term)
            if ti is not None:
                vec[ti] -= _TERMINATOR_PEN_MIDWORD
    # Else: committed_word_pos == len(committed_word). We may be at end
    # of a committed full word OR at the end of a committed common-prefix
    # (where alternative children branch after the prefix). Leave the
    # choice between continuation-letter and terminator to the
    # underlying layers, which read word-trie / letter-ngram context.

    return vec
