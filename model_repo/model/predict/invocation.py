"""Predict layer — invocation-mode biases.

Consumes `state.invocation_mode` (a rolling [0, 1] float tracking whether
the current speaker is in rhetorical / declamatory / apostrophe-address
mode) and returns three kinds of bias:

1. `invocation_sentence_start_bias(m)` — at sentence starts, nudge the
   first letter toward canonical invocation openers:
      O / Oh   → "O"
      Alas/Ah  → "A"
      Hark/Hail/Hear/How → "H"
      Lo        → "L"
      What/Why/Whence → "W"
   Scale grows linearly with mode magnitude.

2. `invocation_sentence_end_bias(m)` — at potential sentence-end
   positions (post word-end inside the clause), boost "!" vs "." to
   reflect that invocation passages chain exclamations.

3. `invocation_word_start_bias(m)` — mid-sentence word-start in
   invocation mode: favor vocative-lead modifiers:
      m=my, t=thy/thou, g=good/gentle/great, s=sweet/sacred,
      n=noble, d=dear, f=fair/fell, h=holy.

All biases return None (no-op) when `m <= 0.0` or in speaker-label mode.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Per-letter weights for sentence-start invocation opener boost.
# O is the crown jewel (O + Oh).
# A covers Alas/Ah/Ay.
# H covers Hark/Hail/Hear/How (and capital-H start of Heaven).
# L covers Lo and Lord.
# W covers What/Why/Whence/Wherefore.
_START_CAPITAL_WEIGHTS: dict[str, float] = {
    "O": 1.45,
    "A": 0.65,
    "H": 0.60,
    "L": 0.35,
    "W": 0.45,
    "B": 0.25,  # Behold
}
# Small negative bump for letters that open invocation-unlikely
# starts (e.g., boring declaratives).
_START_CAPITAL_NEG: dict[str, float] = {
    # Intentionally empty — the prior baseline already accounts for
    # "The/This/That" prevalence; pushing against it here hurt BPC.
}


def _build_start_vec(scale: float) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch, w in _START_CAPITAL_WEIGHTS.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] = w * scale
    for ch, w in _START_CAPITAL_NEG.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] = w * scale
    return vec


def invocation_sentence_start_bias(
    invocation_mode: float,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if invocation_mode <= 0.05:
        return None
    scale = min(1.0, invocation_mode) * 0.40
    return _build_start_vec(scale)


def invocation_sentence_end_bias(
    invocation_mode: float,
    speaker_label_state: int,
    word_is_complete: bool,
    chars_since_sentence_end: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if invocation_mode <= 0.1:
        return None
    # Only fire when we're at a plausible sentence-end position: we just
    # finished a word (or are at a word-end boundary) and the sentence
    # isn't freshly started.
    if not word_is_complete:
        return None
    if chars_since_sentence_end < 15:
        return None
    vec = [0.0] * VOCAB_SIZE
    # Boost "!" proportionally to mode; weak penalty on "." to tip
    # the balance without suppressing absolutely.
    m = min(1.0, invocation_mode)
    bang = VOCAB_INDEX.get("!")
    dot = VOCAB_INDEX.get(".")
    if bang is not None:
        vec[bang] = 0.55 * m
    if dot is not None:
        vec[dot] = -0.20 * m
    return vec


# Per-letter weights for mid-sentence word-start invocation boost.
# Favor the modifier/vocative-lead letters that cluster in invocation
# passages: "my noble lord", "thy sweet grace", "good my lord",
# "dear friend", "most fair".
_WORD_START_WEIGHTS: dict[str, float] = {
    "m": 0.28,  # my, most, mighty
    "t": 0.18,  # thy, thou, thee (though lowered to avoid "the" boost)
    "g": 0.26,  # good, gentle, great, grace
    "s": 0.20,  # sweet, sacred
    "n": 0.28,  # noble
    "d": 0.22,  # dear
    "f": 0.18,  # fair, fell, faithful
    "h": 0.14,  # holy, heaven-prefixed (mid-sentence lowercase)
}
# Gentle penalty on function-word starters that don't fit invocation
# texture (e.g., plain conjunctions).
_WORD_START_NEG: dict[str, float] = {
    # Intentionally empty.
}


def _build_word_start_vec(scale: float) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch, w in _WORD_START_WEIGHTS.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] = w * scale
    for ch, w in _WORD_START_NEG.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] = w * scale
    return vec


def invocation_word_start_bias(
    invocation_mode: float,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if invocation_mode <= 0.15:
        return None
    # Disabled: word-start letter bias within invocation hurts BPC.
    # The invocation-opener sentence-start and the !/. shift carry the
    # signal; pushing m/t/g/s/n/d at every word-start inside an
    # invocation sentence is too blunt.
    return None
    scale = min(1.0, invocation_mode) * 0.10
    return _build_word_start_vec(scale)
