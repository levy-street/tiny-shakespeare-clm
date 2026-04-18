"""Pentameter / iambic-meter bias at end-of-word in verse passages.

Shakespeare's verse is dominantly iambic pentameter — 10 syllables per
line, 11 for feminine endings, with 9 and 12 as occasional variants.
The existing newline-at-long-line bias uses character count (csn >= 20,
25, 30, ...) as a proxy. Syllable count is the *actual* metric
Shakespeare's ear was timing to.

`pipeline/prosody.py` maintains `syllables_in_line` (count of C->V
transitions since the last newline) and `prev_line_syllables` (last
non-empty line's count). This layer consumes both.

Fires at word-end positions in verse passages:
  - verse_score > 0 AND verse_line_run >= 1 (we're in a verse run)
  - speaker_label_state == 0
  - letter_run_len >= 2, on_word_trie, word_buffer a complete word
    (i.e., a real word-end where newline is legal)

Targets:
  - If prev_line_syllables in {9, 10, 11}: target = prev_line_syllables
    (match the just-established meter)
  - Else: target = 10 (pentameter default)

Bumps to newline at word-end:
  - syllables_in_line == target:     +1.8  (prime line-end)
  - syllables_in_line == target+1:   +1.0  (feminine ending plausible)
  - syllables_in_line == target-1:   +0.3  (slight — feminine inverse)
  - syllables_in_line >= target+2:   +2.5  (line overrunning — close it)
  - syllables_in_line <= target-2:   -0.6  (too short — don't close yet)

All bumps are gentle-to-moderate. They're additive on top of the
existing char-count newline biases — when both agree (e.g., csn>=30
AND syllables==10), the composite is strong but natural; when they
disagree (csn==30 but syllables==7), the syllable layer resists the
premature newline.

No corpus statistics — pentameter's 10-syllable target is a
well-known feature of Shakespeare's verse.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def pentameter_wordend_bias(
    syllables_in_line: int,
    prev_line_syllables: int,
    verse_score: float,
    verse_line_run: int,
    chars_since_newline: int,
) -> list[float] | None:
    """Return a bias vector for \n at word-end in verse passages.

    Caller must have already checked that we're at a legal word-end
    (letter_run_len >= 2, on_word_trie, word_buffer a complete word,
    speaker_label_state == 0).
    """
    # Only fire deep in verse mode. Require a strong verse signal
    # (score and an established prev_line_syllables in pentameter
    # range). Without a calibrated target, any syllable bump risks
    # shifting prose-paragraph newlines.
    if verse_score < 0.6:
        return None
    if verse_line_run < 2:
        return None
    # Only fire when prev line was a credible pentameter line —
    # that's our target. If prev wasn't pentameter, we have no
    # anchor.
    if not (9 <= prev_line_syllables <= 11):
        return None
    if syllables_in_line < 5:
        return None
    if chars_since_newline < 18:
        return None

    target = prev_line_syllables
    diff = syllables_in_line - target

    vec = [0.0] * VOCAB_SIZE
    nl_idx = VOCAB_INDEX.get("\n")
    if nl_idx is None:
        return None

    if diff <= -3:
        # Line way too short; \n resistance.
        vec[nl_idx] -= 0.4
    elif diff == -2:
        # Two syllables short; gentle \n resistance.
        vec[nl_idx] -= 0.15
    elif diff >= 3:
        # Overrunning by 3+ syllables; \n nudge.
        vec[nl_idx] += 0.6
    elif diff == 2:
        # Over by 2 syllables; gentle \n nudge.
        vec[nl_idx] += 0.15
    else:
        return None

    return vec
