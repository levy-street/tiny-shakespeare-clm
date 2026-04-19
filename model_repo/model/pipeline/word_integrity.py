"""Word-integrity monitor — targets gibberish word-runs.

Samples regularly contain sequences like "etustarse", "daetfaanwetfimnly",
"rotxouddfser" — letter-runs that never appear in real Shakespeare but
arise when our predict stack drifts off the word-trie and letter-n-gram
backoff produces phonotactic-ish noise.

This stage scores the current `word_buffer` on shape plausibility:

  - buffer_has_vowel: any of a/e/i/o/u/y seen in buffer (early vowel is
    a near-universal English word property; an all-consonant 4+ buffer
    is almost certainly garbage).

  - buffer_last_vowel_pos: 1-indexed position of most recent vowel in
    buffer; 0 if none. A long drought since the last vowel indicates a
    broken pronunciation (real English rarely has 3+ consonants after
    a vowel mid-word: "strength", "twelfth" are edge cases).

  - buffer_consonant_run: length of the trailing consonant run.
    buffer_consonant_run >= 4 is a strong gibberish signal.

  - word_integrity: aggregated score in [0.0, 1.0]. Starts at 1.0, is
    pulled down by structural red flags, pulled up by trie match.

Runs AFTER update_linguistic (which sets word_buffer and letter_run_len)
and AFTER update_word_matches (which sets trie_match_count). Resets all
fields when word_buffer has just emptied.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


_VOWELS = frozenset("aeiouyAEIOUY")
_LETTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def update_word_integrity(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]
    wb = state.word_buffer

    # Empty buffer (word just ended, or between words) → reset.
    if not wb:
        updates: dict[str, object] = {}
        if state.word_integrity != 1.0:
            updates["word_integrity"] = 1.0
        if state.buffer_has_vowel:
            updates["buffer_has_vowel"] = False
        if state.buffer_last_vowel_pos != 0:
            updates["buffer_last_vowel_pos"] = 0
        if state.buffer_consonant_run != 0:
            updates["buffer_consonant_run"] = 0
        if updates:
            return state.model_copy(update=updates)
        return state

    # Character-level step.
    is_letter = ch in _LETTERS
    is_vowel = ch in _VOWELS
    is_apos = ch == "'"

    # Compute new buffer-derived fields.
    if is_letter:
        if is_vowel:
            buffer_has_vowel = True
            buffer_last_vowel_pos = len(wb)
            buffer_consonant_run = 0
        else:
            buffer_has_vowel = state.buffer_has_vowel
            buffer_last_vowel_pos = state.buffer_last_vowel_pos
            buffer_consonant_run = state.buffer_consonant_run + 1
    elif is_apos:
        # Apostrophe inside buffer doesn't break a consonant run or add vowel.
        buffer_has_vowel = state.buffer_has_vowel
        buffer_last_vowel_pos = state.buffer_last_vowel_pos
        buffer_consonant_run = state.buffer_consonant_run
    else:
        # Some non-letter, non-apostrophe char (shouldn't happen when
        # word_buffer is non-empty, but be defensive).
        buffer_has_vowel = state.buffer_has_vowel
        buffer_last_vowel_pos = state.buffer_last_vowel_pos
        buffer_consonant_run = state.buffer_consonant_run

    # Compute integrity score.
    blen = len(wb)
    score = 1.0

    # Strong trie signal: if we're still a prefix of a known word, the
    # buffer is by definition well-shaped. Only boost up; don't let a
    # later flag drag a trie-hit below 0.8.
    on_trie = state.on_word_trie
    if on_trie:
        score = 1.0
    else:
        # No vowel within the first 4 letters → severe red flag.
        if blen >= 4 and not buffer_has_vowel:
            score -= 0.7
        elif blen >= 3 and not buffer_has_vowel:
            score -= 0.4

        # Long trailing consonant run → red flag, escalates.
        if buffer_consonant_run >= 5:
            score -= 0.8
        elif buffer_consonant_run == 4:
            score -= 0.5
        elif buffer_consonant_run == 3 and blen >= 5:
            score -= 0.2

        # Vowel drought: 4+ letters since the last vowel (in a buffer
        # that has had at least one vowel).
        if buffer_has_vowel:
            since_vowel = blen - buffer_last_vowel_pos
            if since_vowel >= 5:
                score -= 0.6
            elif since_vowel == 4:
                score -= 0.3

        # Buffer long with no trie match → graded global penalty.
        # Real Shakespeare has legitimate long off-trie words (rare
        # vocab, coined forms) but 8+ letters with no trie match and
        # no high-value mid-word bias is almost always drifting. Lean
        # toward termination pressure — the BPC cost on legit long
        # words is small (they terminate a beat early), but the sample
        # benefit of killing gibberish tails is large.
        if blen >= 10:
            score -= 0.45
        elif blen >= 8:
            score -= 0.3
        elif blen >= 6:
            score -= 0.12

    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0

    updates = {}
    if abs(score - state.word_integrity) > 1e-6:
        updates["word_integrity"] = score
    if buffer_has_vowel != state.buffer_has_vowel:
        updates["buffer_has_vowel"] = buffer_has_vowel
    if buffer_last_vowel_pos != state.buffer_last_vowel_pos:
        updates["buffer_last_vowel_pos"] = buffer_last_vowel_pos
    if buffer_consonant_run != state.buffer_consonant_run:
        updates["buffer_consonant_run"] = buffer_consonant_run
    if updates:
        return state.model_copy(update=updates)
    return state
