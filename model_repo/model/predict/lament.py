"""Predict layer — lament register word-start bias.

Reads `state.lament_register` (rolling [0, 1] float set by
pipeline/lament.py). When the register is high enough, boost the
first-letter mass toward the grief lexicon:

  a — alas, alack
  w — woe, weep, weary, weeping
  s — sorrow, sigh, sad
  g — grief, groan, grave
  h — heavy, heart, heavens
  t — tears
  m — mourn, miserable
  p — pity, piteous, poor, pain
  d — death, dread, doleful, dying, dead
  l — lament, lost, loss
  o — O (apostrophe of grief) — boosted only at sentence-start
      (chars_since_sentence_end small)

Threshold: lament_register >= 0.35 to fire.
Scaling: linear with register, capped at full weight at register = 1.0.

No corpus statistics — all weights from prior knowledge of
Shakespearean lament diction.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Grief-lexicon starters for general word-start boost.
_GRIEF_STARTERS: dict[str, float] = {
    "a": 0.35,   # alas, alack, anguish
    "w": 0.45,   # woe, weep, weeping, weary
    "s": 0.40,   # sorrow, sigh, sad
    "g": 0.35,   # grief, groan, grave
    "h": 0.35,   # heavy, heart, heavens
    "t": 0.30,   # tears
    "m": 0.30,   # mourn, miserable, misery
    "p": 0.30,   # pity, piteous, poor, pain
    "d": 0.35,   # death, dread, doleful, dying, dead
    "l": 0.25,   # lament, lost, loss
}

_THRESHOLD = 0.20


def lament_start_bias(
    lament_register: float,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if lament_register < _THRESHOLD:
        return None

    # Linear scale above threshold.
    raw = (lament_register - _THRESHOLD) / (1.0 - _THRESHOLD)
    if raw > 1.0:
        raw = 1.0
    scale = raw

    vec = [0.0] * VOCAB_SIZE
    for ch, w in _GRIEF_STARTERS.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
    return vec


def lament_sentence_start_bias(
    lament_register: float,
    speaker_label_state: int,
) -> list[float] | None:
    """At sentence-start, additionally boost 'O' — the apostrophe
    of grief. Called at sentence-first-letter positions."""
    if speaker_label_state != 0:
        return None
    if lament_register < _THRESHOLD:
        return None

    raw = (lament_register - _THRESHOLD) / (1.0 - _THRESHOLD)
    if raw > 1.0:
        raw = 1.0
    scale = raw

    vec = [0.0] * VOCAB_SIZE
    if "O" in VOCAB_INDEX:
        vec[VOCAB_INDEX["O"]] += 0.60 * scale
    if "A" in VOCAB_INDEX:
        vec[VOCAB_INDEX["A"]] += 0.30 * scale  # Alas, Alack, Ah
    if "W" in VOCAB_INDEX:
        vec[VOCAB_INDEX["W"]] += 0.20 * scale  # Woe
    return vec
