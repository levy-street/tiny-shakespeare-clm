"""Predict layer — kinetic register (motion ↔ stasis) word-start bias.

Reads `state.motion_register` in [-1.0, +1.0]. When the recent
content-word diction has drifted strongly MOTION (>= +0.25) or
STASIS (<= -0.25), biases next-word first letters toward letters
typical of kinetic-mode-matching vocabulary.

Gates:
  * speaker_label_state == 0
  * letter_run_len == 0 (word-start position)
  * last_char_class in (1, 7) — post-space / mid-punct
  * |motion_register| >= 0.25

Letter magnitudes hand-graded from prior knowledge of Early Modern
English kinetic vocabulary:

  MOTION starters — letters whose common word-starters are dominated
  by motion verbs/adverbs:
    'c' — come, chase, charge, climb (strong motion; 'calm' counter)
    'r' — run, ride, rush, rise (strong motion; some neutral)
    'f' — fly, flee, fall, forth, fetch, follow (strong motion)
    'h' — haste, hither, hence (motion; 'here' stasis counter; net
          slight motion lean, capped modest)
    'm' — march, mount (motion)
    'g' — go, gone (clear motion)
    's' — sail, speed, strike, soar, send (motion)

  STASIS starters — letters whose common word-starters land on
  stasis vocabulary:
    's' — stand, stay, sit, sleep (stasis; but also above on motion
          — net ambiguous, excluded from stasis primary)
    'd' — dwell (clear but 'd' is small)
    'l' — lie, lingering, long, lay (some motion from 'leap'; net
          slight stasis)
    'a' — abide, abode (stasis)
    'r' — rest, remain, rest (stasis; but 'r' is motion-tilted net —
          excluded)
    'b' — bide, between (stasis)
    'y' — yonder (stasis location)
    'w' — wait, within (stasis; 'weep' excluded emotional)

No corpus statistics — hand-authored from prior knowledge.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Letters tilting toward MOTION openers. Magnitudes reflect net
# strength of motion vs stasis-carrying words starting with that
# letter in Shakespeare's common vocabulary.
_MOTION_STARTERS: dict[str, float] = {
    "c": 0.40,   # come, chase, charge, climb
    "r": 0.35,   # run, ride, rush, rise (net motion)
    "f": 0.45,   # fly, flee, fall, forth, fetch, follow, forward
    "m": 0.30,   # march, mount
    "g": 0.35,   # go, gone
}

# Letters tilting toward STASIS openers.
_STASIS_STARTERS: dict[str, float] = {
    "d": 0.25,   # dwell
    "a": 0.30,   # abide, abode, await, always
    "b": 0.25,   # bide, between
    "y": 0.35,   # yonder, yet
    "l": 0.25,   # lie, linger, long, lay
}


def motion_register_bias(
    motion_register: float,
    letter_run_len: int,
    last_char_class: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None
    # 1 = SPACE, 7 = PUNCT_MID
    if last_char_class not in (1, 7):
        return None
    m = motion_register
    if abs(m) < 0.25:
        return None

    mag = min(abs(m), 1.0)
    # Scale ramps from ~0.06 at |m|=0.25 to ~0.22 at |m|=1.0. Gentle.
    # Deliberately conservative: the motion/stasis first-letter signal
    # overlaps with many other content-word starters, so the bias
    # mostly expresses itself when |m| is already high and the scene
    # is clearly committed to one kinetic mode.
    scale = 0.06 + 0.16 * ((mag - 0.25) / 0.75)

    vec = [0.0] * VOCAB_SIZE
    if m > 0:
        primary = _MOTION_STARTERS
        rival = _STASIS_STARTERS
    else:
        primary = _STASIS_STARTERS
        rival = _MOTION_STARTERS

    for ch, w in primary.items():
        for c in (ch, ch.upper()):
            idx = VOCAB_INDEX.get(c)
            if idx is not None:
                vec[idx] += scale * w

    for ch, w in rival.items():
        for c in (ch, ch.upper()):
            idx = VOCAB_INDEX.get(c)
            if idx is not None:
                vec[idx] -= scale * w * 0.5

    return vec
