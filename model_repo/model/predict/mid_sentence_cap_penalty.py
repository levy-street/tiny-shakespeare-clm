"""Predict layer — mid-sentence capitalization penalty.

Reads `state.mid_sentence_word_start` (set by the pipeline stage).
When True, we're about to emit the first letter of a word that is
mid-sentence (not sentence-start, not verse-line-start, not
post-speaker-label). Shakespeare's convention: such words are
lowercase unless they are a proper noun or the standalone pronoun
"I" or vocative "O".

Sample pathology this targets:
    "Know the Is th" / "Our Is nobs Is aesv" / "Ay his divert for me"
where function-class words ("Is", "Our", "Interest") are
incorrectly capitalized mid-sentence.

Penalty schedule:
  * Uppercase letters EXCEPT "I" / "O": -0.75 (strong)
  * Uppercase "I": +0.15 (valid standalone pronoun)
  * Uppercase "O": -0.10 (mild — O is valid mid-sentence but rare
                        and usually followed by "! " / ", ", forming
                        a standalone interjection)
  * Lowercase letters: +0.05 gentle boost
  * Non-letter chars: unaffected

The penalty is GENTLE on proper-noun-likely letters (J, K, M, P)?
No — we can't distinguish proper nouns here without more context.
The word_trie layer already biases toward legitimate proper nouns
(Hamlet, Claudius, etc.) and will override this penalty for those.
Function-word uppercase gets properly suppressed.

No corpus statistics — rule from English / Shakespeare convention.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"


def _build_vec() -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch in _UPPER:
        idx = VOCAB_INDEX.get(ch)
        if idx is None:
            continue
        if ch == "I":
            vec[idx] += 0.15
        elif ch == "O":
            vec[idx] += -0.10
        else:
            vec[idx] += -0.75
    for ch in _LOWER:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += 0.05
    return vec


_BIAS_VEC = _build_vec()


def mid_sentence_cap_penalty_bias(
    mid_sentence_word_start: bool,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if not mid_sentence_word_start:
        return None
    return _BIAS_VEC
