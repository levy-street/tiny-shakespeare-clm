"""Block second apostrophe inside a single word.

Shakespearean words essentially never contain two apostrophes:

  o'er'reach — does not occur
  t'mercy's  — does not occur

The canonical in-word apostrophe cases are:
  't<letter>     ('tis, 'twas, 'twere, 'twill, 'twixt)
  <word>'d / 's / 't / 're / 've / 'll / 'n
  o'er, ne'er, e'er, where'er, how'er

All at most ONE apostrophe per word.

Samples produce drift like "p'ebrohn'eydenda", "aen'reg", "bone'o'er"
— the second apostrophe is a clear tell for word-level gibberish.

This layer reads `state.had_apostrophe_this_word`: if True, strongly
penalize emitting another apostrophe in the same word.

Gates:
  * had_apostrophe_this_word == True (previous apostrophe seen)
  * speaker_label_state == 0 (labels handled separately)

No corpus statistics — morphological rule.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_PENALTY = -1.5


def second_apostrophe_block_bias(
    had_apostrophe_this_word: bool,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if not had_apostrophe_this_word:
        return None
    vec = [0.0] * VOCAB_SIZE
    if "'" in VOCAB_INDEX:
        vec[VOCAB_INDEX["'"]] = _PENALTY
    return vec
