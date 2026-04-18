"""Adjacent-word repeat blocker.

Penalizes producing two identical words back-to-back ("of of", "the the",
"and and", "I I"). The clause-wide repetition layer exempts short
function words so they can appear multiple times in a clause, but it
still lets immediate adjacent repetition leak through. In Shakespeare,
even function words almost never doubles up side-by-side.

Fires in three positions:
  1. At the first letter of a fresh word (word_buffer=="") — penalize
     the first letter of last_completed_word.
  2. Mid-word when the buffer is a strict prefix of last_completed_word
     — penalize the next matching letter.
  3. At end-of-buffer when buffer == last_completed_word — penalize the
     terminators (" ", ",", ".", etc.) that would complete the repeat.

A narrow whitelist of words that DO legitimately repeat in Shakespeare
(emphatic/refrain) is exempted entirely:
  "no no", "ha ha", "O O", "oh oh", "ay ay", "fie fie", "hear hear",
  "come come", "alas alas", "war war", "well well".

No corpus statistics — this is a grammar rule.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# Words Shakespeare does legitimately double-up for rhetorical effect.
_ALLOW_DOUBLE: frozenset[str] = frozenset({
    "no", "ha", "o", "oh", "ay", "aye", "fie", "alas", "alack",
    "hear", "come", "away", "well", "nay", "war", "peace",
    "hark", "soft", "away", "on", "die", "kill",
    "more", "so",
})


def adjacent_repeat_bias(
    word_buffer: str,
    last_completed_word: str,
    letter_run_len: int,
    speaker_label_state: int,
    consecutive_newlines: int,
) -> list[float] | None:
    """Penalize building an exact adjacent repeat of last_completed_word."""
    if speaker_label_state != 0:
        return None
    if not last_completed_word:
        return None
    if consecutive_newlines >= 2:
        # Speaker turn — new speaker may echo, don't penalize.
        return None
    if last_completed_word in _ALLOW_DOUBLE:
        return None
    # Skip trivially short words where prediction space is tiny
    # (e.g., "a") — too many false positives on word-initial "a".
    if len(last_completed_word) < 2:
        return None

    # Case 3: buffer == last_completed_word. Emitting a terminator now
    # completes the repeat. Penalize terminators strongly.
    if word_buffer == last_completed_word:
        vec = [0.0] * VOCAB_SIZE
        pen = -2.5
        for ch in (" ", ",", ".", ";", ":", "!", "?", "\n"):
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += pen
        return vec

    # Case 2: buffer is a strict prefix of last_completed_word,
    # non-empty, and shorter. Penalize the next matching letter.
    if (
        word_buffer
        and last_completed_word.startswith(word_buffer)
        and len(word_buffer) < len(last_completed_word)
    ):
        next_ch = last_completed_word[len(word_buffer)]
        vec = [0.0] * VOCAB_SIZE
        # Gentler penalty mid-word — there might be other real words
        # sharing this prefix.
        pen = -0.9
        if next_ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[next_ch]] += pen
        return vec

    # Case 1: empty buffer, word about to start. letter_run_len == 0
    # and we're right after a non-letter. Penalize the first letter.
    if letter_run_len == 0 and not word_buffer:
        first = last_completed_word[0]
        vec = [0.0] * VOCAB_SIZE
        # Mild — lots of real same-first-letter transitions occur
        # ("of old", "the thing", "and after").
        pen = -0.45
        if first in VOCAB_INDEX:
            vec[VOCAB_INDEX[first]] += pen
        up = first.upper()
        if up != first and up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] += pen * 0.5
        return vec

    return None
