"""Predict layer — gravitas register word-start bias.

Reads `state.gravitas_register` (rolling [0, 1] float). When the
register crosses the threshold, boost gravitas-lexicon first
letters at word-start:

  h — honour, heaven, holy
  v — virtue, virtuous
  s — soul, sin, sacred, shame, spirit
  d — duty, doom, divine, death, deed
  t — truth, time, thy
  c — conscience, crown
  j — justice, just
  m — mortal, mercy
  e — earth, eternal
  f — fate, faith
  r — reason, right
  g — god, grace, glory
  p — power, pity, peace
  n — nature

Plus "O" at sentence-start for the gravitas apostrophe ("O heavens!",
"O my soul!").

Threshold: gravitas_register >= 0.25.
Scaling: linear above threshold, capped at 1.0.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_GRAVITAS_STARTERS: dict[str, float] = {
    # Stronger only on the most distinctively-gravitas letters.
    "v": 0.22,   # virtue — dense gravitas cluster
    "j": 0.20,   # justice, just
    "h": 0.14,   # honour, heaven, holy
    "e": 0.12,   # earth, eternal
    "c": 0.10,   # conscience, crown
    "g": 0.10,   # god, grace, glory
    "f": 0.10,   # fate, faith
    "s": 0.08,   # soul, sin, sacred, spirit
    "d": 0.08,   # duty, doom, divine, deed
    "p": 0.08,   # power, pity, peace
    "m": 0.06,   # mortal, mercy
    "r": 0.06,   # reason, right
    "n": 0.04,   # nature
    # t, a, i, o, etc. are used for so many non-gravitas words we
    # don't boost them — would over-boost function words.
}

_THRESHOLD = 0.25


def gravitas_start_bias(
    gravitas_register: float,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if gravitas_register < _THRESHOLD:
        return None

    raw = (gravitas_register - _THRESHOLD) / (1.0 - _THRESHOLD)
    if raw > 1.0:
        raw = 1.0
    scale = raw

    vec = [0.0] * VOCAB_SIZE
    for ch, w in _GRAVITAS_STARTERS.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
        up = ch.upper()
        up_idx = VOCAB_INDEX.get(up)
        if up_idx is not None:
            vec[up_idx] += w * scale * 0.55
    return vec


def gravitas_sentence_start_bias(
    gravitas_register: float,
    speaker_label_state: int,
) -> list[float] | None:
    """At sentence-start, lift 'O' (gravitas apostrophe: 'O heavens!',
    'O my soul!', 'O my conscience!')."""
    if speaker_label_state != 0:
        return None
    if gravitas_register < _THRESHOLD:
        return None
    raw = (gravitas_register - _THRESHOLD) / (1.0 - _THRESHOLD)
    if raw > 1.0:
        raw = 1.0
    scale = raw
    vec = [0.0] * VOCAB_SIZE
    if "O" in VOCAB_INDEX:
        vec[VOCAB_INDEX["O"]] += 0.40 * scale
    return vec
