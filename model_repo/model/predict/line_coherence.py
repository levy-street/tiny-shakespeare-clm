"""Predict layer — per-line coherence bias.

Reads `state.line_ontrie_words` and `state.line_offtrie_words` (set
by pipeline/line_coherence.py) and applies a word-end bias that
times the newline decision based on the line's accumulated quality.

Three regimes at word-end positions:

  "failing" line (off >= 2, on <= 1):
      Line has accumulated garbage with little real vocabulary. Boost
      newline to cut losses; mild boost to sentence-end terminators
      (. ? !) as alternative closures. Also gently discourage comma
      / semicolon (these extend the line, we want to kill it).

  "healthy" line (on >= 3, off == 0):
      Line is well-formed and contentful. Mild anti-newline nudge at
      on-trie word-ends so a good line can breathe to natural length
      instead of closing prematurely.

  middle ground: no bias.

Fires only at legitimate word-close opportunities:
  - word_buffer is a complete on-trie word, OR
  - we're off-trie with letter_run_len >= 3 (plausible close point)
  - speaker_label_state == 0 (not inside speaker name)
  - letter_run_len >= 2 (not at word-start)

Bias magnitudes are modest so this can't single-handedly force a
terrible newline on a slightly-unlucky line. The signal stacks
with drift_recovery_midword and red_flags_close for strong
compound pressure only when multiple signals agree.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def line_coherence_wordend_bias(
    line_ontrie_words: int,
    line_offtrie_words: int,
    letter_run_len: int,
    on_word_trie: bool,
    speaker_label_state: int,
    word_buffer: str,
    complete_words: frozenset[str] | set[str],
    chars_since_sentence_end: int,
    consecutive_newlines: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if letter_run_len < 2:
        return None
    # Don't double-push newline if we just emitted one.
    if consecutive_newlines >= 1:
        return None

    # Classify the close-opportunity:
    is_complete_word = (
        on_word_trie and word_buffer in complete_words
    )
    is_plausible_offtrie_close = (
        not on_word_trie and letter_run_len >= 4
    )
    if not (is_complete_word or is_plausible_offtrie_close):
        return None

    # "Failing line" — push newline to abandon it.
    if line_offtrie_words >= 2 and line_ontrie_words <= 1:
        # Ramp with how badly the line has failed.
        excess = line_offtrie_words - line_ontrie_words
        if excess >= 4:
            scale = 1.0
        elif excess >= 3:
            scale = 0.75
        elif excess >= 2:
            scale = 0.55
        else:
            scale = 0.35
        # Prose-like: heavier newline push when line has also gone
        # sentence-long without a period (csse large → we're overdue
        # for ANY break).
        if chars_since_sentence_end >= 40:
            scale *= 1.2

        vec = [0.0] * VOCAB_SIZE
        if "\n" in VOCAB_INDEX:
            vec[VOCAB_INDEX["\n"]] += 2.6 * scale
        if "." in VOCAB_INDEX:
            vec[VOCAB_INDEX["."]] += 1.4 * scale
        if "?" in VOCAB_INDEX:
            vec[VOCAB_INDEX["?"]] += 0.7 * scale
        if "!" in VOCAB_INDEX:
            vec[VOCAB_INDEX["!"]] += 0.7 * scale
        # Discourage extenders — these keep us in the failing line.
        if "," in VOCAB_INDEX:
            vec[VOCAB_INDEX[","]] -= 0.9 * scale
        if ";" in VOCAB_INDEX:
            vec[VOCAB_INDEX[";"]] -= 0.5 * scale
        if ":" in VOCAB_INDEX:
            vec[VOCAB_INDEX[":"]] -= 0.4 * scale
        # Space slightly penalized too — we don't want another word
        # appended to a failing line; we want out.
        if " " in VOCAB_INDEX:
            vec[VOCAB_INDEX[" "]] -= 0.5 * scale
        return vec

    # "Healthy line" — mild anti-newline, slight extension nudge.
    # Only at on-trie complete-word positions so we don't cling to a
    # bad line just because the current word happened to close well.
    if line_ontrie_words >= 3 and line_offtrie_words == 0 and is_complete_word:
        # Scale with how many on-trie words we have. Cap at 5.
        if line_ontrie_words >= 5:
            scale = 1.0
        else:
            scale = (line_ontrie_words - 2) / 3.0
        # If the line is already quite long (csse high), don't resist
        # newline — natural close is approaching.
        if chars_since_sentence_end >= 60:
            return None

        vec = [0.0] * VOCAB_SIZE
        if "\n" in VOCAB_INDEX:
            vec[VOCAB_INDEX["\n"]] -= 0.35 * scale
        if " " in VOCAB_INDEX:
            vec[VOCAB_INDEX[" "]] += 0.15 * scale
        if "," in VOCAB_INDEX:
            vec[VOCAB_INDEX[","]] += 0.10 * scale
        return vec

    return None
