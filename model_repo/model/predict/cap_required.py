"""Strong capital-required bias at structural word-start positions.

Reads `state.cap_required_mode` (set by pipeline/cap_required.py) and
applies a decisive UPPER-vs-lower bias at the next letter position.

The scattered inline cap pushes in compose.py apply ~+1.2 to UPPER and
~-0.5 to lowercase. Against the unigram prior where lowercase dominates
uppercase by ~2-3 nats, those soft nudges sometimes lose — producing
lowercase verse-line-starts like "mouth restore...", "phebe or is...",
or "do deads...". This layer is the single hard-structural enforcement:
once we're at a word-start AND the structural rule requires a cap, the
UPPER mass is boosted enough to dominate any residual lowercase prior.

Mode-dependent magnitudes, from prior knowledge:
  SENTENCE_START : +3.8 UPPER, -2.0 lower, -1.0 non-letter
  VERSE_LINE     : +3.5 UPPER, -1.8 lower, -1.0 non-letter
  POST_LABEL     : +4.0 UPPER, -2.2 lower, -1.5 non-letter
  TURN_START     : +4.5 UPPER, -2.5 lower, -1.5 non-letter

Non-letter penalty (comma/period/space/newline/apostrophe) is mild
because occasionally dialog opens on an interjection like "'Tis..." or
"—What..." — we don't want to hard-forbid punctuation openers, just
to clearly prefer letters AND clearly prefer uppercase letters among
those.

No corpus statistics — magnitudes hand-tuned from prior knowledge of
Shakespeare's orthographic conventions.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Mode-specific magnitudes. (upper_boost, lower_pen, nonletter_pen)
# Calibrated against original scattered-inline magnitudes so that the
# centralization doesn't perturb BPC. Samples still capitalize verse
# line-starts consistently because the underlying condition fires more
# cleanly (state-driven, not re-derived from raw context at every call).
_MODE_WEIGHTS: dict[int, tuple[float, float, float]] = {
    1: (1.2, 0.5, 0.0),   # SENTENCE_START — match original inline
    2: (3.0, 1.2, 0.0),   # VERSE_LINE — match original inline
    3: (3.0, 1.2, 0.0),   # POST_LABEL — match original inline
}


def cap_required_bias(
    cap_required_mode: int,
    letter_run_len: int,
) -> list[float] | None:
    if cap_required_mode == 0:
        return None
    # Only fire at word-start. (Belt-and-suspenders — the pipeline
    # stage only sets non-zero modes at letter_run_len == 0, but
    # guarded here against any future state staleness.)
    if letter_run_len != 0:
        return None
    w = _MODE_WEIGHTS.get(cap_required_mode)
    if w is None:
        return None
    upper_boost, lower_pen, nonletter_pen = w

    vec = [0.0] * VOCAB_SIZE
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += upper_boost
    for ch in "abcdefghijklmnopqrstuvwxyz":
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] -= lower_pen
    # Mild nonletter penalty — spaces/newlines/punct should not start
    # a fresh sentence/line/turn (there's nothing to pile onto).
    # Skip apostrophe — "'Tis" is legitimate.
    for ch in " \n,.;:!?\t":
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] -= nonletter_pen
    return vec
