"""Predict layer — sonority-level biases.

Consumes `state.sonority_level` (a rolling [-1, +1] phonetic-texture
float) and nudges letter-choice toward or away from in-register
phonemes. Positive sonority → melodic passages → favor vowels and
liquids; negative sonority → percussive passages → favor stops.

Fires at mid-word positions only (where letter-bigram/trigram are
also firing). The magnitude is intentionally tiny because it sits on
top of heavy n-gram priors — we just tilt the balance slightly toward
the ambient phonetic texture.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Per-letter weights scaled by sonority_level:
#   positive  =  boosted when sonority_level > 0 (melodic)
#   negative  =  boosted when sonority_level < 0 (percussive)
# Magnitudes are small (<= 1.0) because sonority_level is bounded
# and overall scale in compose is capped further.
_SONORITY_LETTER_WEIGHTS: dict[str, float] = {
    # Vowels → strongly positive (melodic)
    "a": 0.40, "e": 0.40, "i": 0.40, "o": 0.40, "u": 0.40,
    # Liquids / nasals → mildly positive
    "l": 0.30, "m": 0.30, "n": 0.30, "r": 0.30,
    # Approximants → weakly positive
    "w": 0.18, "y": 0.18,
    # Voiceless fricatives → neutralish
    "h": 0.10, "s": 0.05, "f": 0.08,
    # Voiced consonants → mildly negative (percussive pull)
    "v": -0.12, "c": -0.15,
    # Hard voiceless stops → negative
    "k": -0.35, "t": -0.25, "p": -0.30,
    # Voiced stops → negative
    "g": -0.25, "b": -0.22, "d": -0.20,
    # Rare harsh consonants → strongly negative
    "j": -0.50, "q": -0.45, "x": -0.50, "z": -0.45,
}

# Precompute a reusable per-letter weight vector for quick scaling.
_LETTER_WEIGHT_VEC: list[float] = [0.0] * VOCAB_SIZE
for _ch, _w in _SONORITY_LETTER_WEIGHTS.items():
    _idx = VOCAB_INDEX.get(_ch)
    if _idx is not None:
        _LETTER_WEIGHT_VEC[_idx] = _w


# Global scale — small because this sits on top of n-gram priors.
_GLOBAL_SCALE = 0.08


def sonority_midword_bias(
    sonority_level: float,
    speaker_label_state: int,
) -> list[float] | None:
    """Tilt letter choice toward the ambient sonority.

    Fires when a letter is the expected next character — we return
    a vector and compose.py adds it after bigram/trigram biases. The
    caller is responsible for gating by position (letter_run_len >= 1,
    speaker label, etc.).
    """
    if speaker_label_state != 0:
        return None
    # Below threshold — no meaningful texture yet.
    if abs(sonority_level) < 0.08:
        return None
    scale = sonority_level * _GLOBAL_SCALE
    return [w * scale for w in _LETTER_WEIGHT_VEC]
