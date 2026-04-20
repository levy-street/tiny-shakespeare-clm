"""Trie-unique completion hint bias.

Structural companion to `word_commit_bias` for the general word lexicon.
`word_commit_bias` fires only when the *formula trie* commits to a
specific next-word identity — a very narrow regime. This layer fires
whenever the general word-trie has narrowed to exactly one candidate
completion mid-buffer, which happens much more often (any prefix with
trie_match_count == 1).

Key difference from word_commit_bias:
  * Much gentler (+0.8 target boost vs +4.0), no penalty on other
    letters, no penalty on terminators. The corpus vocabulary is far
    larger than our word-trie; a trie-unique completion tells us "IF
    the target is in our lexicon, it's THIS word" — which is a partial
    signal, not certainty. Over-committing via penalties costs BPC when
    the actual word is one of the many not in our list.
  * Only fires when the remaining tail is short (1-3 letters). With a
    short tail, the chance that the corpus word is a different (longer)
    word whose prefix happens to match is low: any longer extension
    wouldn't have yielded a count of 1 here (else the prefix would have
    had multiple candidates including the longer one).

Gate:
  * speaker_label_state == 0 (proper-noun zones have their own logic)
  * letter_run_len >= 4 (allow real English ambiguity to resolve first)
  * letter_run_len + tail_remaining <= target length — invariant
  * on_word_trie == True, trie_match_count == 1
  * tail_remaining in {1, 2, 3}

No corpus statistics — purely derived from the hand-written word list.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE
from .word_trie import PREFIX_UNIQUE_COMPLETION


def trie_completion_hint_bias(
    word_buffer: str,
    letter_run_len: int,
    speaker_label_state: int,
    on_word_trie: bool,
    trie_match_count: int,
) -> list[float] | None:
    """Return a gentle bias pushing the next letter toward the unique
    trie completion's next letter, or None if not applicable.
    """
    if speaker_label_state != 0:
        return None
    if not on_word_trie:
        return None
    if trie_match_count != 1:
        return None
    if letter_run_len < 4:
        return None
    if not word_buffer:
        return None
    target = PREFIX_UNIQUE_COMPLETION.get(word_buffer)
    if target is None:
        return None
    buf_len = len(word_buffer)
    tail_remaining = len(target) - buf_len
    if tail_remaining < 1 or tail_remaining > 3:
        return None
    if not target.startswith(word_buffer):
        return None

    # Next letter in the target sequence.
    next_letter = target[buf_len]
    vec = [0.0] * VOCAB_SIZE

    # Scale by confidence: tail_remaining == 1 is the strongest
    # (we're at the penultimate letter — the next char is near-certain
    # to complete the word). tail_remaining == 2 is moderate; 3 is weak.
    if tail_remaining == 1:
        boost = 1.6
    elif tail_remaining == 2:
        boost = 1.0
    else:
        boost = 0.5

    idx = VOCAB_INDEX.get(next_letter)
    if idx is not None:
        vec[idx] += boost

    return vec
