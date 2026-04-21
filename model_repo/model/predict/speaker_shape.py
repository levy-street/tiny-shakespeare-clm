"""Predict layer — enforce speaker-label morphology.

Speaker labels in Shakespeare follow a tight pattern:
  * 1-3 space-separated WORDS
  * Each word has >= 2 letters (never "K a" or "L o")
  * Each word starts with an UPPERCASE letter
  * No digits, no punctuation until the terminating ':'
  * Characters after the first word's letter are always letters or
    a single space separating to a new word

Failure examples observed in length-400 samples:
  * "UZST:"             — first letter plausible but U-Z-S-T has no
                          vowel — name-letter runs should allow
                          vowels after 2+ consonants
  * "R a:"              — single-letter "words" separated by spaces
  * "E wcaaei:"         — second word starts lowercase and is garbage
  * "T e Home ruin o:"  — multiple spaces with single-letter words
  * "CORNER one ia:"    — second and third words should be uppercase
  * "PIST3lonesome;"    — digit inside label broke the FSM; handled
                          by the ORTHOGRAPHIC forbid mask already
  * "OCTAVIUS lone ttdsiaw:" — after valid OCTAVIUS, continuation is
                               garbage. The FSM stays in state 2.

This layer derives two quantities from `speaker_buffer` + FSM state:
  * `label_word_count` — number of words so far (1 + number of
    internal spaces in the buffer, while buffer is non-empty).
  * `current_word_len` — length of the current (possibly in-progress)
    word within the label.

And applies these biases in speaker_label_state == 2:

  A. After a letter with current_word_len == 1, suppress the
     next char being ' ' or ':' (words of length 1 are unreal).
     Push UP on continuation letters.

  B. After a space (current_word_len == 0), suppress ':' (a label
     can't end immediately after a space), and push UP on
     UPPERCASE letters only.

  C. After 3+ words have started (label_word_count >= 3) and the
     current word has at least 2 letters, gently push ':' (typical
     labels are 1-3 words max).

  D. After any single word >= 10 letters and we're off-trie already,
     push ':' firmly.

No corpus statistics — name-morphology rules are universal English
convention.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"


def speaker_shape_bias(
    speaker_buffer: str,
    speaker_label_state: int,
    speaker_label_saw_lower: bool,
    speaker_label_offtrie_run: int,
) -> list[float] | None:
    if speaker_label_state != 2:
        return None
    if not speaker_buffer:
        return None

    # Count words + current word length from speaker_buffer.
    # speaker_buffer stores an uppercased form with internal spaces.
    trimmed = speaker_buffer.rstrip(" ")
    leading_chars = speaker_buffer.split(" ")
    # If buffer ends with space, current_word_len == 0 and we're
    # between words.
    ends_with_space = speaker_buffer.endswith(" ")
    # Word count = number of whitespace-separated chunks so far,
    # counting in-progress current word if any.
    parts = speaker_buffer.split(" ")
    # Drop empty strings (from consecutive spaces — shouldn't happen
    # but defensive).
    parts = [p for p in parts if p]
    word_count = len(parts) if not ends_with_space else len(parts) + 1
    if word_count == 0:
        word_count = 1

    if ends_with_space:
        current_word_len = 0
    else:
        current_word_len = len(parts[-1]) if parts else 0

    vec = [0.0] * VOCAB_SIZE

    # Rule B: just after a space (between words). Next char must be
    # an uppercase letter; never ':' or another space.
    if ends_with_space:
        # Forbid ":" and space.
        idx = VOCAB_INDEX.get(":")
        if idx is not None:
            vec[idx] -= 6.0
        idx = VOCAB_INDEX.get(" ")
        if idx is not None:
            vec[idx] -= 6.0
        # Push up uppercase letters (gentle — existing biases already
        # prefer upper in this state).
        if not speaker_label_saw_lower:
            for ch in _UPPER:
                idx = VOCAB_INDEX.get(ch)
                if idx is not None:
                    vec[idx] += 0.25
            # Suppress lowercase.
            for ch in _LOWER:
                idx = VOCAB_INDEX.get(ch)
                if idx is not None:
                    vec[idx] -= 1.5
        return vec

    # Rule A: current word length == 1 (just started a new word).
    # Suppress ' ' and ':' so we can't close an effectively empty
    # word.
    if current_word_len == 1:
        idx = VOCAB_INDEX.get(" ")
        if idx is not None:
            vec[idx] -= 3.5
        idx = VOCAB_INDEX.get(":")
        if idx is not None:
            vec[idx] -= 3.5
        # Push continuation letters up modestly. In all-caps mode,
        # upper; in mixed-case mode, allow lower (second char of
        # "First" or "Second" is lowercase).
        if not speaker_label_saw_lower:
            for ch in _UPPER:
                idx = VOCAB_INDEX.get(ch)
                if idx is not None:
                    vec[idx] += 0.18
        # else: mixed-case — leave continuation letters neutral
        # beyond what other layers do.
        return vec

    # Rule C: 3+ words already started and current word has letters.
    # Gently push ':' to close (labels > 3 words are very rare in
    # Shakespeare: "FIRST CITIZEN" (2), "LADY MACBETH" (2),
    # "FIRST LORD" (2), "KING HENRY IV" (3)).
    if word_count >= 3 and current_word_len >= 2:
        idx = VOCAB_INDEX.get(":")
        if idx is not None:
            vec[idx] += 1.0
        # Also suppress another space (too many words).
        idx = VOCAB_INDEX.get(" ")
        if idx is not None:
            vec[idx] -= 0.8
        # Continue; fall through in case other rules stack.

    # Rule D: single word >= 12 letters and off-trie — almost
    # certainly ran away from a real name. Push ':' firmly.
    if (
        word_count == 1
        and current_word_len >= 12
        and speaker_label_offtrie_run >= 5
    ):
        idx = VOCAB_INDEX.get(":")
        if idx is not None:
            vec[idx] += 1.5

    if not any(v != 0.0 for v in vec):
        return None
    return vec
