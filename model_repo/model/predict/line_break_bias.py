"""Line-break propriety bias.

Reads `state.line_break_propriety` (maintained by
pipeline/line_break.py) and biases the newline character based on
whether the current position is a grammatical place to close a verse
line:

  propriety 0 (DEEP mid-phrase) → strong \n penalty
  propriety 1 (WEAK)            → mild \n penalty
  propriety 2 (PHRASE_END)      → no change
  propriety 3 (CLAUSE_END)      → mild \n boost

Gated by verse_score (only applies in verse-plausible context) and
chars_since_newline (no-op before line-length threshold).

This layer is complementary to the chars_since_newline-driven newline
biases in compose.py — those say "a break is overdue"; this says
"a break here would be ungrammatical". The combination produces
more Shakespearean verse-line structure in samples.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def line_break_newline_bias(
    line_break_propriety: int,
    verse_score: float,
    chars_since_newline: int,
    speaker_label_state: int,
    in_prose_line: bool,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if in_prose_line:
        # Prose lines have much freer break points than verse; don't
        # suppress mid-phrase \n in prose because the model relies on
        # other signals (character count, paragraph-like breaks).
        return None
    # Only apply in verse-ish contexts.
    if verse_score < 0.6:
        return None
    # No-op before minimum line length — the existing line-length
    # biases handle short-line behaviour.
    if chars_since_newline < 20:
        return None

    if line_break_propriety == 0:
        # Deep mid-phrase — strong suppression.
        pen = 1.6
        bonus = 0.0
    elif line_break_propriety == 1:
        # Weak — mild suppression.
        pen = 0.6
        bonus = 0.0
    elif line_break_propriety == 2:
        # Phrase-end — neutral.
        return None
    elif line_break_propriety == 3:
        # Clause-end — mild boost (strongest break target).
        pen = 0.0
        bonus = 0.4
    else:
        return None

    vec = [0.0] * VOCAB_SIZE
    if "\n" in VOCAB_INDEX:
        vec[VOCAB_INDEX["\n"]] += (bonus - pen)
    return vec
