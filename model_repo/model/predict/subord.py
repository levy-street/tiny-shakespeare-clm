"""Subordinate-clause-depth predict consumers.

Reads `state.subord_depth` and `state.subord_words_since_open` to
produce two biases:

  1. **Deep-nesting sentence-end suppression** — inside a subord
     clause (depth >= 1), a period/question/exclamation is much
     less likely than at main-clause level (the main clause still
     needs to be closed). Suppress end-punctuation boosts.
  2. **Subord-close pressure** — when subord_words_since_open has
     grown large (>= 5), a comma / conjunction is due to close the
     dependent clause and return to the main clause. Boost comma.
  3. **Inside-subord verb-form shift** — inside a subordinate clause
     the verb is often a participle or -eth form rather than a
     tensed main verb. Gentle lean toward -ing / -eth mid-word
     endings when inside a subord.

This consumes a genuinely new axis that no existing predict layer
uses.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def subord_word_end_bias(
    subord_depth: int,
    subord_words_since_open: int,
    letter_run_len: int,
    word_buffer: str,
    on_word_trie: bool,
    chars_since_sentence_end: int,
    speaker_label_state: int,
) -> list[float] | None:
    """At word-end positions, modify punctuation preferences based on
    subord state.

    Returns a bias vector or None.
    """
    if speaker_label_state != 0:
        return None
    if subord_depth <= 0:
        return None
    # Fire only at plausible word-end positions.
    if letter_run_len < 2:
        return None
    if not word_buffer:
        return None

    vec = [0.0] * VOCAB_SIZE
    any_bias = False

    # Suppress sentence-end punctuation inside a subordinate clause.
    # The deeper the nesting, the stronger the suppression.
    sentence_end_suppression = 0.70 * subord_depth  # depth=1:-0.70 ; depth=3:-2.10
    for ch in ".?!":
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] -= sentence_end_suppression
            any_bias = True

    # Boost comma when the subord has run long enough to close.
    if subord_words_since_open >= 3:
        comma_boost = 0.20 + 0.15 * min(subord_words_since_open - 2, 5)
        cm = VOCAB_INDEX.get(",")
        if cm is not None:
            vec[cm] += comma_boost
            any_bias = True
        # Also lightly boost semicolon.
        sc = VOCAB_INDEX.get(";")
        if sc is not None:
            vec[sc] += comma_boost * 0.5
            any_bias = True

    if not any_bias:
        return None
    return vec


def subord_midword_bias(
    subord_depth: int,
    subord_words_since_open: int,
    letter_run_len: int,
    word_buffer: str,
    clause_slot: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Mid-word letter bias inside a subordinate clause.

    At HAS_SUBJ inside a subord, subordinate-clause verbs often take
    -eth / -ing / -ed endings (rather than plain tensed forms):
      "the king who speaketh..."
      "the man that loveth..."
      "when night cometh..."
    So bias toward "e" (for -eth/-ed/-es) and "i" (for -ing).

    Also at buffer ends with "s", boost "h" for -th/-ath/-eth.
    """
    if speaker_label_state != 0:
        return None
    if subord_depth <= 0:
        return None
    if letter_run_len < 2:
        return None
    if clause_slot != 1:  # only at HAS_SUBJ inside subord (verb position)
        return None
    if not word_buffer or not word_buffer.isalpha():
        return None

    vec = [0.0] * VOCAB_SIZE
    any_bias = False

    # If ends in "et" → favor "h" (complete -eth)
    if len(word_buffer) >= 2 and word_buffer[-2:].lower() == "et":
        idx_h = VOCAB_INDEX.get("h")
        if idx_h is not None:
            vec[idx_h] += 0.20
            any_bias = True

    # If ends in vowel + "s" (not immediately after a short prefix),
    # favor "t" (complete -est).
    if (
        letter_run_len >= 3
        and len(word_buffer) >= 2
        and word_buffer[-1].lower() == "s"
        and word_buffer[-2].lower() in "aeiouy"
    ):
        idx_t = VOCAB_INDEX.get("t")
        if idx_t is not None:
            vec[idx_t] += 0.12
            any_bias = True

    if not any_bias:
        return None
    return vec
