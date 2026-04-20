"""Line-terminal-word first-letter echo bias.

Reads `state.recent_line_end_words` (populated by pipeline/
line_end_memory.py). At a word-start that looks like the FINAL word of
an upcoming verse line, bias the first letter toward the first letters
of recent line-ending words. Supports Shakespeare's epistrophe —
rhetorical recurrence of a closing word across consecutive lines.

Activation conditions (all must hold):
  - letter_run_len == 0 (word-start)
  - last_char_class ∈ {SPACE, NEWLINE} (actually at a word-start)
  - speaker_label_state == 0 (not inside a speaker label)
  - in a verse context: meter_confidence >= 0.25
  - approaching line-end: syllables_until_line_end <= 3 AND
    line_length >= 24 (we're ~3-4 syllables from the pentameter close)

Strength is modest — a +0.35 first-letter boost per remembered
line-ender, aggregated across the tuple. Caps the aggregate bump per
letter to keep the competing-letter prior intact when several recent
line-enders share a starter.

No corpus statistics — line-terminal word echo is a rhetorical figure
(epistrophe) Shakespeare applies by authorial choice.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_PER_MATCH_BUMP = 0.35
_MAX_BUMP_PER_LETTER = 0.9  # cap if several recent enders share a starter


def line_end_echo_bias(
    recent_line_end_words: tuple,
    letter_run_len: int,
    last_char_class: int,
    speaker_label_state: int,
    meter_confidence: float,
    syllables_until_line_end: int,
    line_length: int,
) -> list[float] | None:
    if not recent_line_end_words:
        return None
    if letter_run_len != 0:
        return None
    # last_char_class: SPACE (5) or NEWLINE (6) per linguistic codes.
    if last_char_class not in (5, 6):
        return None
    if speaker_label_state != 0:
        return None
    if meter_confidence < 0.25:
        return None
    if syllables_until_line_end > 3:
        return None
    if line_length < 24:
        return None

    vec = [0.0] * VOCAB_SIZE
    totals: dict[str, float] = {}
    for w in recent_line_end_words:
        if not w:
            continue
        first = w[0]
        totals[first] = totals.get(first, 0.0) + _PER_MATCH_BUMP

    if not totals:
        return None
    for ch, val in totals.items():
        if val > _MAX_BUMP_PER_LETTER:
            val = _MAX_BUMP_PER_LETTER
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += val
        # Mirror to uppercase (rare at mid-line word-start, but e.g.
        # after ". " mid-line punctuation).
        up = ch.upper()
        if up != ch:
            uidx = VOCAB_INDEX.get(up)
            if uidx is not None:
                vec[uidx] += val * 0.4

    return vec
