"""Doubt-register predict layer.

Reads `state.doubt_register` (rolling [-1, +1] float) and produces:

  1. Word-start first-letter bias: boost doubt-starter letters in
     doubt mode; assertion-starter letters in assertion mode. Fires
     only when |register| > 0.25.

  2. Sentence-end punctuation bias: boost "?" in doubt mode; boost "!"
     and mildly "." in assertion mode. Fires at the sentence-end
     decision point (chars_since_sentence_end >= 15 and word-end).

All weights hand-specified from prior knowledge; no corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Weights applied at word-start in DOUBT mode (register > 0).
_DOUBT_START: dict[str, float] = {
    "p": 0.22,  # perhaps, perchance, peradventure
    "m": 0.18,  # may, might, methinks, maybe
    "h": 0.14,  # haply
    "b": 0.12,  # belike
    "s": 0.12,  # seem, seems, some (suggestive)
    "i": 0.10,  # if
    "w": 0.10,  # whether, wonder
    "o": 0.08,  # or
}
_DOUBT_START_CAPS: dict[str, float] = {
    "P": 0.12, "M": 0.10, "H": 0.08, "B": 0.06, "S": 0.06,
    "I": 0.08, "W": 0.08,
}

# Weights applied at word-start in ASSERTION mode (register < 0).
_ASSERT_START: dict[str, float] = {
    "v": 0.22,  # verily
    "s": 0.16,  # surely, so, shall
    "i": 0.14,  # indeed, I
    "k": 0.14,  # know, knew, known
    "t": 0.14,  # truly, the, that, this
    "a": 0.10,  # assuredly, ay
    "d": 0.10,  # doubtless
    "c": 0.10,  # certain, certes
    "f": 0.08,  # forsooth
}
_ASSERT_START_CAPS: dict[str, float] = {
    "V": 0.14, "S": 0.08, "I": 0.10, "K": 0.06, "T": 0.08,
    "A": 0.06, "C": 0.06,
}


def doubt_word_start_bias(
    doubt_register: float,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    mag = abs(doubt_register)
    if mag < 0.25:
        return None

    scale = min(1.0, (mag - 0.25) / 0.55) * 0.70  # 0..0.70

    vec = [0.0] * VOCAB_SIZE
    if doubt_register > 0:
        src = _DOUBT_START
        caps = _DOUBT_START_CAPS
    else:
        src = _ASSERT_START
        caps = _ASSERT_START_CAPS
    for ch, w in src.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
    for ch, w in caps.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
    return vec


def doubt_sentence_end_bias(
    doubt_register: float,
    speaker_label_state: int,
    word_is_complete: bool,
    chars_since_sentence_end: int,
) -> list[float] | None:
    """At a plausible sentence-end decision point, boost "?" in doubt
    mode or "!" in assertion mode. Kept mild — overlaps with
    invocation / sentence_type biases already.
    """
    if speaker_label_state != 0:
        return None
    if not word_is_complete:
        return None
    if chars_since_sentence_end < 15:
        return None

    mag = abs(doubt_register)
    if mag < 0.20:
        return None
    scale = min(1.0, (mag - 0.20) / 0.60) * 0.40  # 0..0.40

    vec = [0.0] * VOCAB_SIZE
    q = VOCAB_INDEX.get("?")
    bang = VOCAB_INDEX.get("!")
    dot = VOCAB_INDEX.get(".")
    if doubt_register > 0:
        if q is not None:
            vec[q] += 0.60 * scale
        if bang is not None:
            vec[bang] -= 0.10 * scale
    else:
        if bang is not None:
            vec[bang] += 0.40 * scale
        if dot is not None:
            vec[dot] += 0.15 * scale
        if q is not None:
            vec[q] -= 0.15 * scale
    return vec
