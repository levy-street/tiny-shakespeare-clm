"""Sensory-charge first-letter bias.

Reads `sensory_charge` (flow-tier, [-3, +3] corporeal ↔ abstract axis)
and, at word-start positions, biases first-letter choice toward
continuation of the dominant register:

  + high (>= +1.0): corporeal / tragic-lyric mode — boost the first
    letters of sensory vocabulary: b (blood/bones/blade/breath),
    e (eye/ears), h (heart/hand/head/heaven/hell), f (fire/flame/fear),
    s (sword/sun/sea/storm/soul), t (tears/tongue/tomb/thunder),
    n (night), g (grave/gold), d (dark/death), w (wound/wind/weep),
    m (moon/murder), r (rose/rain), l (lamb/lightning).

  − low (<= −1.0): abstract / discursive mode — boost the first
    letters of reasoning vocabulary: c (cause/case/counsel/charge),
    m (matter/mind/means), r (reason/right), q (question), t (truth),
    j (justice), h (honour), v (virtue), d (duty/doubt), p (purpose/
    promise/policy), s (sense/state/service), f (fault/faith),
    o (office/order/opinion), w (word/wit/way).

Neutral charge (|.| < 1.0) → no bias (return None).

Scale ramps with |charge| up to a modest cap so this never dominates
word_trie or next_word — it's a mood tilt, not a hard constraint.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# First letters to boost in each register mode. Weights are relative.
_SENSORY_START: dict[str, float] = {
    "b": 1.0,  # blood, bones, blade, breath, breast, bosom
    "e": 0.8,  # eye, ears, earth
    "h": 1.0,  # heart, hand, head, heaven, hell, horse
    "f": 0.9,  # fire, flame, fear, flesh, feet, face
    "s": 1.0,  # sword, sun, sea, storm, soul, shadow, stone
    "t": 0.9,  # tears, tongue, tomb, thunder, torch
    "n": 0.7,  # night
    "g": 0.7,  # grave, gold, ground
    "d": 0.8,  # dark, death, dust, dawn, dew
    "w": 0.8,  # wound, wind, weep, water, wave
    "m": 0.6,  # moon, murder, mist
    "r": 0.6,  # rose, rain, river, rock
    "l": 0.6,  # lamb, lightning, lion, leaves
    "p": 0.5,  # plague, poison, pulse
    "c": 0.4,  # crown, cup, cloud, crow
    "a": 0.3,  # arms, arrow, ashes
}

_ABSTRACT_START: dict[str, float] = {
    "c": 1.0,  # cause, case, counsel, charge, circumstance, conscience
    "m": 0.9,  # matter, mind, means, motive, mercy, manner
    "r": 1.0,  # reason, right, regard, report, rule, respect
    "q": 0.9,  # question
    "t": 0.7,  # truth, time, term
    "j": 0.8,  # justice
    "h": 0.5,  # honour, hope
    "v": 0.7,  # virtue
    "d": 0.7,  # duty, doubt, decree, despair
    "p": 0.9,  # purpose, promise, policy, pity, practice, proof, prayer
    "s": 0.7,  # sense, state, service, suit, shame, sin
    "f": 0.7,  # fault, faith, fashion
    "o": 0.8,  # office, order, opinion, occasion, order
    "w": 0.6,  # word, wit, way, wrong
    "i": 0.5,  # intent, intention, intent
    "g": 0.5,  # grace, government, guilt
    "n": 0.5,  # news, nature
}


def _build_vec(starts: dict[str, float], scale: float) -> list[float]:
    """Build a bias vec from a letter → weight dict, scaled uniformly."""
    vec = [0.0] * VOCAB_SIZE
    # Tiny negative on unlisted letters so the listed ones stand out.
    negative = -0.15 * scale
    for ch in "abcdefghijklmnopqrstuvwxyz":
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = negative
    for ch, w in starts.items():
        if ch not in VOCAB_INDEX:
            continue
        bias = scale * w
        vec[VOCAB_INDEX[ch]] = bias
        up = ch.upper()
        if up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] = bias * 0.6
    return vec


def sensory_charge_start_bias(
    sensory_charge: float,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a bias vector for the first letter of a new word, given
    the sensory_charge flow field. Caller is expected to only invoke
    when we are at a word-start cursor position (last char = space or
    single newline) — compose.py's START_BIAS gate.
    """
    if speaker_label_state != 0:
        return None
    if sensory_charge >= 1.0:
        intensity = min((sensory_charge - 1.0) / 2.0, 1.0)
        scale = 0.10 + 0.12 * intensity
        return _build_vec(_SENSORY_START, scale)
    if sensory_charge <= -1.0:
        intensity = min((-sensory_charge - 1.0) / 2.0, 1.0)
        scale = 0.10 + 0.12 * intensity
        return _build_vec(_ABSTRACT_START, scale)
    return None
