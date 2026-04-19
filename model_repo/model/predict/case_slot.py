"""Predict layer — pronoun case-slot word-start bias.

Reads `state.case_slot` (set by pipeline/case_slot.py) and biases
pronoun-starter first letters at word-start based on the expected
grammatical case:

  CASE_SUBJ (1) — subject slot, nominative case expected:
    I (capital), h (he), s (she), t (thou/they), w (we), y (ye)

  CASE_OBJ (2) — object slot (of verb or preposition), accusative:
    m (me), t (thee/them), h (him/her), u (us), y (you)

The bias is intentionally modest because these letters overlap with
many other word classes (t starts "the/to/thy/that/this"; h starts
"his/her/him/how"; m starts "my/mine/methinks"). The signal is in
the *relative* boost between the case-appropriate pronouns and their
case-inappropriate counterparts.

Scale decays with case_wait_words to avoid over-extending the bias
past the immediate next word:
  wait 0: full strength
  wait 1: 0.65x
  wait 2: 0.35x
  wait 3+: silenced (update_case_slot resets by then)

No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

CASE_NONE = 0
CASE_SUBJ = 1
CASE_OBJ = 2


# Subject-slot first-letter weights. Pronoun-specific targeting.
#   I (capital) — always the 1sg nominative. Unique as-is.
#   h — he (nominative); also his/her/him but those are orthogonal.
#   s — she
#   t — thou (nom.), they (nom.); also thee/them/thy/thine: mixed.
#   w — we (nom.); also who, which.
#   y — ye (nom.); also your, you.
_SUBJ_WEIGHTS: dict[str, float] = {
    "I": 1.50,   # always-capital; 1sg nom
    "h": 0.25,   # he
    "s": 0.16,   # she
    "w": 0.20,   # we
    "t": 0.12,   # thou, they
    "y": 0.07,   # ye
}

# Object-slot first-letter weights. VERY conservative because after
# prepositions the dominant continuation is "the"/"my"/common noun,
# not a pronoun.
_OBJ_WEIGHTS: dict[str, float] = {
    "h": 0.08,   # him, her, his
    "u": 0.12,   # us, upon, unto (all legitimate OBJ continuations)
    "m": 0.05,   # me, my
}


def _build_vec(weights: dict[str, float], scale: float) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch, w in weights.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
    return vec


def case_slot_start_bias(
    case_slot: int,
    case_wait_words: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if case_slot == CASE_NONE:
        return None

    if case_wait_words >= 3:
        return None
    if case_wait_words == 0:
        scale = 1.0
    elif case_wait_words == 1:
        scale = 0.65
    else:  # 2
        scale = 0.35

    if case_slot == CASE_SUBJ:
        return _build_vec(_SUBJ_WEIGHTS, scale)
    if case_slot == CASE_OBJ:
        return _build_vec(_OBJ_WEIGHTS, scale)
    return None
