"""Predict layer — tenderness register word-start bias.

Reads `state.tenderness_register` (rolling [0, 1] float). When the
register crosses the threshold, boost tenderness-lexicon first
letters at word-start:

  l — love, lover, light
  s — sweet, soft, silver
  f — fair, fond, flower, fine
  d — dear, divine, delight
  g — gentle, grace, good
  k — kind, kiss
  m — mild, mine, my, mistress
  t — tender, true
  b — beauteous, beloved, bright, blossom
  r — rose, rare, radiant

Plus "O" at sentence-start for the tender apostrophe ("O my love").

Threshold: tenderness_register >= 0.35.
Scaling: linear above threshold, capped at 1.0.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_TENDER_STARTERS: dict[str, float] = {
    "l": 0.45,   # love, lovely, light, lady
    "s": 0.40,   # sweet, soft, silver
    "f": 0.40,   # fair, fond, flower, fine
    "d": 0.35,   # dear, divine, delight
    "g": 0.30,   # gentle, grace, good
    "k": 0.25,   # kind, kiss
    "m": 0.30,   # mine, my, mild, mistress
    "t": 0.25,   # tender, true
    "b": 0.30,   # beauteous, beloved, bright, blossom
    "r": 0.22,   # rose, rare, radiant
}

_THRESHOLD = 0.20


def tenderness_start_bias(
    tenderness_register: float,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if tenderness_register < _THRESHOLD:
        return None

    raw = (tenderness_register - _THRESHOLD) / (1.0 - _THRESHOLD)
    if raw > 1.0:
        raw = 1.0
    scale = raw

    vec = [0.0] * VOCAB_SIZE
    for ch, w in _TENDER_STARTERS.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
    return vec


def tenderness_sentence_start_bias(
    tenderness_register: float,
    speaker_label_state: int,
) -> list[float] | None:
    """At sentence-start, additionally lift 'O' (tender apostrophe:
    'O my love!', 'O sweet!', 'O beauteous!')."""
    if speaker_label_state != 0:
        return None
    if tenderness_register < _THRESHOLD:
        return None
    raw = (tenderness_register - _THRESHOLD) / (1.0 - _THRESHOLD)
    if raw > 1.0:
        raw = 1.0
    scale = raw

    vec = [0.0] * VOCAB_SIZE
    if "O" in VOCAB_INDEX:
        vec[VOCAB_INDEX["O"]] += 0.45 * scale
    if "M" in VOCAB_INDEX:
        vec[VOCAB_INDEX["M"]] += 0.20 * scale  # "My sweet..."
    if "S" in VOCAB_INDEX:
        vec[VOCAB_INDEX["S"]] += 0.15 * scale  # "Sweet..."
    if "F" in VOCAB_INDEX:
        vec[VOCAB_INDEX["F"]] += 0.12 * scale  # "Fair..."
    return vec
