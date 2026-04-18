"""Rhyme-position bias — at late verse-line positions, nudge the next
letter toward one that would rhyme with the previous line's tail.

This is a letter-level proxy for phonological rhyme: boost the last
letter of `prev_line_tail` at word positions near the natural end of
a verse line. Most English near-rhymes share their last 1-2 letters
(-ay/-ay, -ore/-ore, -ight/-ight); boosting the final letter at the
right moment makes rhymed closes more likely without forcing them.

Gating:
  - verse_line_run >= 1 (we're in a run of verse-plausible lines)
  - speaker_label_state == 0 (not inside a speaker label)
  - chars_since_newline in the "approaching line-end" window
  - mid-word (word_buffer present)

Weight ramps up as we approach line-end, then tapers off past ~46
chars (overrun — we don't care about rhyme anymore, just closing).
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def rhyme_midword_bias(
    prev_line_tail: str,
    prev_prev_line_tail: str,
    verse_line_run: int,
    chars_since_newline: int,
    word_buffer: str,
) -> list[float] | None:
    if verse_line_run < 1:
        return None
    if not prev_line_tail:
        return None
    if chars_since_newline < 24:
        return None
    if not word_buffer:
        return None

    # Weight schedule by line position. Kept gentle because a single
    # letter boost can't encode real rhyme; at best we nudge the tail
    # toward a compatible phonological cluster. Too strong and we
    # penalize legitimate line closures that don't happen to rhyme.
    # CURRENT TUNING: even gentle letter-level rhyme boost hurts BPC
    # (~0.0001-0.0005 per seed). We leave the infrastructure in and
    # disable the bias for now — the state fields (prev_line_tail,
    # verse_line_run) are useful for future layers. Set `enabled`
    # to True and pick a scale to re-activate.
    enabled = False
    if not enabled:
        return None
    cs = chars_since_newline
    if cs < 32:
        return None
    elif cs < 40:
        scale = 0.03
    elif cs < 46:
        scale = 0.05
    else:
        return None

    vec = [0.0] * VOCAB_SIZE
    any_hit = False

    # Primary: last letter of prev line.
    target = prev_line_tail[-1]
    if target in VOCAB_INDEX:
        vec[VOCAB_INDEX[target]] += scale
        any_hit = True

    # (ABAB alternate-target branch removed — adding prev_prev as an
    # alternate rhyme target blurs the signal and hurts BPC.)

    if not any_hit:
        return None
    return vec
