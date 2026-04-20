"""Tier 2/3 — iambic meter tracking.

Runs after `update_prosody` (so `syllables_in_line` and
`prev_line_syllables` are current) and updates three fields:

  - meter_confidence: rolling [0, 1] confidence that we're inside a
    pentameter verse passage. Strongly bumped when a line closes at
    9–11 syllables, gently decayed on each word so long prose drifts
    the score toward 0.
  - expected_stress: 0 (weak / offbeat) or 1 (strong / ictus) — the
    predicted metrical weight of the NEXT syllable onset, assuming
    iambic. Iambic feet are xX (weak–strong), so within a pentameter
    line syllables 2, 4, 6, 8, 10 carry the ictus. The next syllable
    index is `syllables_in_line + 1`; if that index is even → STRONG,
    else WEAK.
  - syllables_until_line_end: projected syllables remaining before a
    pentameter close. `max(0, 10 - syllables_in_line)`. Zero means a
    line-end is immediately plausible; smaller values should compress
    word-length expectations (short word or direct line-end).

Prose passages drift meter_confidence to 0 so the downstream predict
layer silently disables itself. No corpus statistics — iambic
pentameter is a well-known feature of Shakespeare's dramatic verse.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


# --- meter_confidence update constants ----------------------------
# On line-end, map prev_line_syllables to a delta applied to the
# rolling meter_confidence. These numbers are by prior knowledge of
# iambic pentameter conventions, not corpus-fit.
_LINE_END_BUMP: dict[int, float] = {
    9: 0.18,
    10: 0.25,  # canonical pentameter
    11: 0.15,  # feminine ending
    8: 0.05,
    12: 0.02,
    7: -0.05,
    13: -0.05,
    6: -0.08,
    14: -0.10,
}
# Hard decay for severe under/overshoot (prose) or blank line.
_LINE_END_DECAY_EXTREME = 0.55

# Per-word multiplicative decay toward zero (applied inside `advance`
# at word-completion events). Long verse-less stretches eventually
# cool the confidence to 0.
_PER_WORD_DECAY = 0.985


def _line_end_delta(prev_line_syllables: int) -> tuple[str, float]:
    """Return (mode, value). mode == 'bump' → additive, value in
    [-0.10, +0.25]. mode == 'decay' → multiplicative, value in
    (0, 1]."""
    if prev_line_syllables == 0:
        return ("bump", 0.0)  # blank line — neutral
    if prev_line_syllables in _LINE_END_BUMP:
        return ("bump", _LINE_END_BUMP[prev_line_syllables])
    if prev_line_syllables <= 5 or prev_line_syllables >= 15:
        return ("decay", _LINE_END_DECAY_EXTREME)
    # 15-ish fallthrough: gentle decay.
    return ("decay", 0.80)


def _compute_expected_stress(syllables_in_line: int) -> int:
    """Next-syllable stress under iambic convention.
    Iambic pentameter: weak on 1, strong on 2, weak on 3, ...
    Next syllable index = syllables_in_line + 1.
    STRONG (ictus) iff that index is even."""
    next_idx = syllables_in_line + 1
    return 1 if (next_idx % 2 == 0) else 0


def _compute_syll_remaining(syllables_in_line: int) -> int:
    rem = 10 - syllables_in_line
    if rem < 0:
        return 0
    if rem > 10:
        return 10
    return rem


def update_meter(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Newline ended a line: update meter_confidence from line length.
    # (prosody already captured prev_line_syllables before the reset.)
    if ch == "\n":
        mode, val = _line_end_delta(state.prev_line_syllables)
        if mode == "bump":
            new_conf = state.meter_confidence + val
        else:
            new_conf = state.meter_confidence * val
        if new_conf < 0.0:
            new_conf = 0.0
        elif new_conf > 1.0:
            new_conf = 1.0
        # After newline, syllables_in_line resets to 0 → next syllable
        # index 1 → WEAK (standard iambic line opener).
        new_stress = 0
        new_rem = 10
        updates: dict = {}
        if abs(new_conf - state.meter_confidence) > 1e-6:
            updates["meter_confidence"] = new_conf
        if new_stress != state.expected_stress:
            updates["expected_stress"] = new_stress
        if new_rem != state.syllables_until_line_end:
            updates["syllables_until_line_end"] = new_rem
        if updates:
            return state.model_copy(update=updates)
        return state

    # Per-word tiny decay (once per word completion). `just_finished_word`
    # is set by the linguistic stage which runs earlier in the pipeline.
    # We use it here to apply a gentle decay each time a word closes.
    conf = state.meter_confidence
    if state.just_finished_word and conf > 0.0:
        conf = conf * _PER_WORD_DECAY
        if conf < 1e-4:
            conf = 0.0

    # Recompute expected_stress / syllables_until_line_end from current
    # syllables_in_line (prosody ran earlier this token, so this value
    # is post-update).
    new_stress = _compute_expected_stress(state.syllables_in_line)
    new_rem = _compute_syll_remaining(state.syllables_in_line)

    updates: dict = {}
    if abs(conf - state.meter_confidence) > 1e-6:
        updates["meter_confidence"] = conf
    if new_stress != state.expected_stress:
        updates["expected_stress"] = new_stress
    if new_rem != state.syllables_until_line_end:
        updates["syllables_until_line_end"] = new_rem
    if updates:
        return state.model_copy(update=updates)
    return state
