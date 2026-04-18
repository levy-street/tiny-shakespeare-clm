"""Ornament-density word-start bias.

Reads `state.ornament_density` — a Tier 3 flow field tracking how
ornate (adjective-stacked) the recent text has been. Fires at
word-start outside speaker-label territory.

Behavior:
  - When ornament_density is high (>= 0.4) AND np_open is True:
    the NP has already accumulated ornaments; push harder toward
    the head noun (noun starter letters) rather than more adjectives.
  - When ornament_density is low (< 0.25) AND np_open is True:
    room for more ornament; give a small bump to adjective starters.
  - When ornament_density is high AND np_open is False (no open NP):
    the recent texture is ornate — favor adjective-starter letters
    at word-start to maintain the ornate feel.
  - When ornament_density is low: no effect (don't interfere with
    default behavior).

The magnitudes are gentle — this is a texture prior, not a hard rule.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Adjective-starter letters (common pre-head modifiers in Shakespeare).
_ADJ_LETTERS: dict[str, float] = {
    "g": 0.4,  # good/great/gentle/green/golden/gracious
    "s": 0.3,  # sweet/sacred/sad/silent/sure/soft
    "f": 0.3,  # fair/false/foul/full/fond/faint
    "d": 0.3,  # dear/dark/deep/dead/divine/dumb
    "t": 0.2,  # true/tender
    "l": 0.2,  # little/long/last/low/loose
    "h": 0.2,  # high/holy/happy/harsh/humble
    "n": 0.2,  # noble/new/near/naked
    "p": 0.2,  # poor/pale/proud/precious
    "b": 0.3,  # bright/brave/brief/bold/black/base/blessed
    "m": 0.2,  # mad/mere/meek/mighty
    "o": 0.15, # old/own
    "y": 0.1,  # young
    "e": 0.2,  # every/evil/eternal/empty
    "w": 0.2,  # wise/weak/weary/wild/worthy
    "r": 0.2,  # rich/rude/rare/royal
    "c": 0.2,  # cold/common/cruel/cursed/clear
    "v": 0.2,  # vile/valiant/vain
}

# Noun-starter letters (typical concrete nouns), duplicated here from
# np_head but re-weighted for ornament context.
_NOUN_LETTERS: dict[str, float] = {
    "h": 0.7, "l": 0.7, "m": 0.7, "s": 0.9,
    "f": 0.6, "w": 0.5, "d": 0.5, "k": 0.5,
    "b": 0.6, "c": 0.6, "p": 0.5, "e": 0.5,
    "g": 0.4, "n": 0.4, "t": 0.4, "r": 0.4,
}


def ornament_start_bias(
    ornament_density: float,
    np_open: bool,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a word-start bias vector based on ornament texture."""
    if speaker_label_state != 0:
        return None
    od = ornament_density
    if od < 0.10:
        return None  # no signal

    vec = [0.0] * VOCAB_SIZE

    if od >= 0.35 and np_open:
        # Ornament-saturated NP — push toward head noun, dampen adj.
        overshoot = min(od - 0.35, 0.65)  # 0..0.65
        noun_scale = 0.35 + 0.8 * overshoot
        adj_damp = 0.18 + 0.35 * overshoot
        for ch, w in _NOUN_LETTERS.items():
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += noun_scale * w
        for ch, w in _ADJ_LETTERS.items():
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] -= adj_damp * w
        return vec

    if od < 0.25 and np_open:
        # Room for ornament: mildly encourage adjective-starters.
        adj_scale = 0.12 * (0.25 - od) / 0.15  # 0..0.12
        for ch, w in _ADJ_LETTERS.items():
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += adj_scale * w
        return vec

    if od >= 0.45 and not np_open:
        # Maintain ornate texture across NP boundaries — nudge
        # adjective-ish openers at word-start.
        adj_scale = 0.15 * min((od - 0.45) / 0.35, 1.0)
        for ch, w in _ADJ_LETTERS.items():
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += adj_scale * w
        return vec

    return None
