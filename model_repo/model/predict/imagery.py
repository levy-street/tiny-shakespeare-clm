"""Imagery-density bias layer.

Reads `state.imagery_density` (a rolling [0, 1] float tracking how
much sensory/concrete language has appeared in the recent window)
and, at word-start outside speaker labels, nudges next-word first-
letter choice toward letters that commonly begin concrete, sensory
words.

Shakespeare's imagistic passages are self-reinforcing: once a scene
turns toward sensory language (body parts, weapons, weather, color,
gesture) the next word is far more likely to continue in that mode.
This layer captures that texture.

Distinct from:
  - tonal_weight: dark/light valence — "blood" (dark image) and
    "sword" (dark image) and "moon" (neutral image) all bump imagery
    the same way, while tonal_weight distinguishes them.
  - archaic_density: formal/archaic register — "hath", "prithee",
    "thou" bump archaic but not imagery.
  - topic clusters: topic clusters are categorical (DARK/LIGHT/ROYAL);
    imagery is continuous and cross-cutting.

All weights come from prior knowledge of concrete English vocabulary.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# Letters that most commonly begin sensory / concrete words.
# Weighted by rough coverage of the imagery lexicon.
_IMAGERY_STARTERS: dict[str, float] = {
    "s": 1.0,  # sword, star, sun, sea, storm, snow, shadow, skin, stone, smile, serpent, song
    "b": 1.0,  # blood, body, bone, blade, brow, breath, bosom, bird, blow, bright, blue, bough
    "h": 0.95, # hand, head, hair, heart, house, horse, hell, hollow
    "f": 0.9,  # face, fire, flower, feet, finger, flame, frost, flies, field, forest, flood
    "e": 0.8,  # eye, ear, earth, edge
    "r": 0.85, # rose, river, rock, robe, ring, rain, red, rough
    "c": 0.9,  # cheek, chain, cloud, cup, crown, cloak, coin, cut, cold, cave
    "m": 0.85, # moon, music, mouth, mirror, mist, mud, meadow, mouth
    "w": 0.85, # wave, wind, wound, water, wall, window, worm, white, warm, weapon
    "t": 0.9,  # tear, throat, tongue, thorn, tree, throne, tower, touch, tempest
    "l": 0.85, # lip, leaf, light, lion, letter, line, limb, lock
    "g": 0.75, # gold, grass, gate, garden, gem, goblet, ground, green, gray
    "p": 0.75, # pearl, path, pulse, palm, paint, petal, pale, purple
    "d": 0.7,  # dagger, dust, dove, door, dagger, dew, dawn, dark, dry
    "a": 0.55, # arm, arrow, ash, air
    "n": 0.55, # night, nail, nerve, noise
    "k": 0.65, # kiss, key, knee, knife, kingdom
    "o": 0.35, # ocean
    "v": 0.45, # voice, veil, vault, vein
    "i": 0.30, # ice, iron, island
    "y": 0.25,
    "u": 0.15,
}


def _build_vec() -> list[float]:
    """Centered starter-letter vector (sum ≈ 0 across alphabet) so the
    scale multiplier controls pure mass-shift, not level."""
    vec = [0.0] * VOCAB_SIZE
    pos_letters: list[tuple[str, float]] = []
    for ch, w in _IMAGERY_STARTERS.items():
        pos_letters.append((ch, w))
        lo = ch
        up = ch.upper()
        if lo in VOCAB_INDEX:
            vec[VOCAB_INDEX[lo]] = w
        if up in VOCAB_INDEX and up != lo:
            vec[VOCAB_INDEX[up]] = w * 0.6
    # Center across lowercase alphabet.
    total_pos = sum(w for _, w in pos_letters)
    n_neg = 0
    for ch in "abcdefghijklmnopqrstuvwxyz":
        if ch in VOCAB_INDEX and vec[VOCAB_INDEX[ch]] == 0.0:
            n_neg += 1
    if n_neg > 0:
        neg_per = -total_pos / 26.0 * 0.2
        for ch in "abcdefghijklmnopqrstuvwxyz":
            if ch in VOCAB_INDEX and vec[VOCAB_INDEX[ch]] == 0.0:
                vec[VOCAB_INDEX[ch]] = neg_per
    return vec


_IMAGERY_VEC: list[float] = _build_vec()


# Threshold and gain: only apply when the rolling density has risen
# clearly above baseline. Scale is linear in (density - threshold).
_THRESHOLD = 0.25
_GAIN = 1.0
_MAX_SCALE = 0.70


def imagery_start_bias(density: float) -> list[float] | None:
    """Return a word-start letter-bias vector scaled by imagery_density."""
    if density <= _THRESHOLD:
        return None
    scale = min((density - _THRESHOLD) * _GAIN, _MAX_SCALE)
    if scale <= 0.0:
        return None
    return [scale * v for v in _IMAGERY_VEC]
