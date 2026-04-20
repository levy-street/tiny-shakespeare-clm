"""Line-word-count cadence bias — consumer of recent_line_word_counts.

Reads `state.recent_line_word_counts` and `state.line_word_count`
(both maintained by pipeline/line_word_cadence.py) and tilts the
newline-character probability at plausible word-end positions based
on whether the current line has reached the cadence established by
the preceding 2-3 lines in the same turn.

Why this helps Shakespearean verse: a speaker's blank-verse lines
cluster tightly around a shared word-count within a single speech
(~6-9 words per pentameter line). Once we've seen two such lines,
the next body-line is very likely to close in the same range.

Activation conditions:
  - not inside a speaker label
  - at a complete-word position (letter_run_len >= 2, word at
    complete-word boundary or buffer empty)
  - at least 2 non-zero entries in recent_line_word_counts
  - running mean of entries in a plausible verse range [4, 14]
  - consecutive_newlines == 0 (we're actually on an in-progress
    line, not right after a newline)
  - chars_since_newline >= 10 (past the line's first word)

Bias shape (mild magnitudes — this is texture, not an enforcer):
  delta = line_word_count - target
    delta >= +2  : +0.45 on \n (we've blown past the established
                               cadence; close the line)
    delta in [0, 1] : +0.30 on \n (we're at the cadence; natural
                               break)
    delta == -1 : +0.10 on \n (one word shy of cadence; ok to
                               break)
    delta <= -2 : -0.18 on \n (too early; keep going)

Gated by speaker_label_state and letter_run_len to fire only at
true word-end, so we don't over-apply inside mid-word territory.

No corpus statistics — magnitudes from prior knowledge of blank-
verse rhythm (~7-word pentameter lines in Shakespeare).
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def line_word_cadence_bias(
    recent_line_word_counts: tuple[int, ...],
    line_word_count: int,
    speaker_label_state: int,
    letter_run_len: int,
    word_buffer_len: int,
    consecutive_newlines: int,
    chars_since_newline: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if consecutive_newlines != 0:
        return None
    if chars_since_newline < 10:
        return None
    # Word-end shape gate.
    if letter_run_len < 2:
        return None

    non_zero = [c for c in recent_line_word_counts if c > 0]
    if len(non_zero) < 2:
        return None
    total = sum(non_zero)
    target = total / len(non_zero)
    if target < 4.0 or target > 14.0:
        return None

    delta = line_word_count - target

    nl = VOCAB_INDEX.get("\n")
    if nl is None:
        return None

    vec = [0.0] * VOCAB_SIZE
    if delta >= 2.0:
        vec[nl] += 0.45
    elif delta >= 0.0:
        vec[nl] += 0.30
    elif delta >= -1.0:
        vec[nl] += 0.10
    elif delta <= -2.0:
        vec[nl] -= 0.18
    else:
        return None

    return vec
