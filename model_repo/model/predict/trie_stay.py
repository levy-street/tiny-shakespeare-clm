"""Trie-stay terminator suppression.

Counterpart to FORCE_END_BIAS (which pushes terminators when off-trie).
This layer mildly SUPPRESSES word-terminators (space, comma, period,
semicolon, colon, !, ?, newline, apostrophe) when:

  * The current buffer is ON the word-trie,
  * The trie has narrowed to a small set of completions
    (trie_match_count <= 4),
  * We have a real prefix (letter_run_len in [2, 6]),
  * The unique-completion target (when trie_match_count == 1) still
    has at least 1 letter remaining, OR trie_match_count > 1
    indicating continuation is the dominant move.

The motivation: samples have artifacts like "Citiz?" / "estate. Is"
where the model commits to a premature word-terminator while sitting
on a sharp on-trie continuation prefix (citizen/citizens, estatement
or simply continuing). The existing trie_completion_hint boosts the
next LETTER, but doesn't suppress the punctuation alternative — so a
moderately-attractive ',' or '?' can win against the boosted letter.
This layer closes that loop with a small, gated terminator penalty.

Gates avoid hurting BPC on legitimate single-syllable closures (e.g.,
"the." / "I,") by requiring letter_run_len >= 2 and excluding
in-speaker-label territory.

No corpus statistics — purely structural ("don't terminate while a
short, sharp completion is right there").
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE
from .word_trie import PREFIX_COMPLETE_COUNT, PREFIX_UNIQUE_COMPLETION


_TERMINATORS = (" ", ",", ".", ";", ":", "?", "!", "\n")


def trie_stay_bias(
    word_buffer: str,
    letter_run_len: int,
    on_word_trie: bool,
    trie_match_count: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if not on_word_trie:
        return None
    if not word_buffer:
        return None
    if letter_run_len < 2 or letter_run_len > 6:
        return None
    if trie_match_count < 1 or trie_match_count > 4:
        return None

    # Determine remaining-tail context.
    # If unique completion exists, require >=1 letter remaining.
    target = PREFIX_UNIQUE_COMPLETION.get(word_buffer)
    if target is not None:
        tail_remaining = len(target) - len(word_buffer)
        if tail_remaining < 1:
            return None
        # Suppression strength: stronger when more letters to go and
        # match_count is sharper.
        if tail_remaining >= 2:
            penalty = -0.7
        else:
            penalty = -0.4
    else:
        # Non-unique but few candidates — gentler suppression.
        if trie_match_count <= 2:
            penalty = -0.5
        else:
            penalty = -0.25

    vec = [0.0] * VOCAB_SIZE
    for t in _TERMINATORS:
        if t in VOCAB_INDEX:
            vec[VOCAB_INDEX[t]] = penalty
    # Apostrophe also a (mild) word-end signal in many contractions.
    if "'" in VOCAB_INDEX:
        vec[VOCAB_INDEX["'"]] = penalty * 0.4
    return vec
