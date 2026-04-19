"""Discourse-rhythm first-letter bias.

Reads `state.recent_sentence_types` (see pipeline/sentence.py) — a
rolling tuple of the last up-to-4 completed sentence types. This
captures DISCOURSE-LEVEL patterns that 1-back memory
(prev_sentence_type) can't see:

  * Question-chain mode: two or three questions in a row. A third
    question is highly likely; the next sentence tends to ALSO open
    with a WH/aux starter ("Why... What... How...?"). Often this
    chain ends with an answer (I/we/thou/the...).
  * Exclamation-chain mode: two exclamations in a row. The third
    sentence continues the exclamative register ("O my lord! My
    God! Alas, that ever..."). Continuation often starts with
    interjection letters (o, a, h).
  * Declarative flow: three declaratives in a row — the speaker is in
    a narrative/philosophical groove. Next sentence likely starts with
    a continuer/connective (and, but, yet, so, therefore...).
  * Mixed short-then-long: short questions followed by a long
    declarative answer — the answer tends to open with "I", "thou",
    "the", or a pronoun.

Fires only at sentence-start positions (sentence_start_pending or
words_in_sentence == 0) outside speaker-label territory, and only
when recent_sentence_types has >= 2 entries (enough to detect a
pattern).

Bias scale is gentle — this is a long-range discourse signal stacking
on top of existing 1-back conditioning.

All weights from prior knowledge of Shakespearean dialogue rhythm —
no corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Sentence type constants (mirror pipeline/sentence.py).
SENT_UNKNOWN = 0
SENT_DECL = 1
SENT_INTERROG = 2
SENT_EXCLAM = 3
SENT_IMPER = 4


_GLOBAL_SCALE = 0.12


def _vec_from_weights(weights: dict[str, float], global_scale: float) -> list[float]:
    """Build a bias vector from {letter: weight}. Listed letters get
    boosted; unlisted get a mild negative to maintain distribution
    shape. Also tilts uppercase counterparts (sentence-initial)."""
    vec = [0.0] * VOCAB_SIZE
    total = sum(weights.values()) or 1.0
    mean = 1.0 / 26.0
    for ch in "abcdefghijklmnopqrstuvwxyz":
        w = weights.get(ch, 0.0)
        frac = w / total
        bias = global_scale * (frac - mean) * 3.0
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += bias
        up = ch.upper()
        if up in VOCAB_INDEX:
            # Sentence-starts are capitalized; pass bias through to
            # the capital letter too (same sign).
            vec[VOCAB_INDEX[up]] += bias
    return vec


# Patterns keyed on (last_type, second_last_type), values = letter
# weights for the UPCOMING sentence's first letter.

# Question-chain continuation: after 2+ questions, the next sentence
# often EITHER continues interrogatively (wh/aux starters) OR shifts
# to an answer opening with pronoun/article.
_Q_CHAIN_WEIGHTS: dict[str, float] = {
    "w": 5,  # What/Why/When/Where/Whence/Who/Which/Would
    "h": 4,  # How/Hath/Hast/Have/He/Here
    "i": 4,  # Is/I
    "a": 4,  # Art/Are/Am
    "d": 3,  # Dost/Did/Do/Does
    "s": 3,  # Shall/Should/Say
    "c": 2,  # Canst/Can
    "t": 4,  # Thou/The/Tell me/That/This/They
    "y": 2,  # You/Ye
    "n": 2,  # Nay/No
    "o": 2,  # O/Oh
    "m": 2,  # My/Must
    "f": 1,
    "b": 1,
}

# Exclamation-chain continuation: after 2+ exclamations, emotional
# register locked. Next sentence often opens with interjection or
# vocative.
_EXCLAM_CHAIN_WEIGHTS: dict[str, float] = {
    "o": 7,   # O/Oh
    "a": 5,   # Alas/Ah
    "h": 3,   # Hark/Ha/Help
    "m": 4,   # My
    "f": 3,   # Fie/For shame
    "w": 3,   # Would to God
    "t": 3,   # Thou/The
    "g": 2,   # Good
    "s": 2,   # Sweet
    "n": 2,   # Never/Nay
    "d": 2,   # Dear
    "i": 2,   # I
    "y": 2,   # Ye/You
    "b": 1,
    "c": 1,
    "l": 1,
}

# Declarative flow (3 decls in a row): continuer / connective opening.
_DECL_FLOW_WEIGHTS: dict[str, float] = {
    "a": 5,   # And
    "b": 4,   # But/Be
    "t": 5,   # The/Thus/Then/That/This
    "s": 3,   # So
    "y": 3,   # Yet
    "f": 3,   # For
    "n": 3,   # Now/Nor
    "o": 2,   # Or
    "w": 3,   # When/While
    "i": 3,   # If/I
    "h": 2,   # Here/He
    "m": 2,   # My
    "l": 1,
    "u": 2,   # Upon/Until
    "c": 1,
}

# After Q then DECL: probably answered question; next is another
# declarative continuation (same weights as decl flow but tilted).
_Q_THEN_DECL_WEIGHTS: dict[str, float] = {
    "a": 4,   # And (answer chain)
    "y": 4,   # Yes/Yea/Yet
    "n": 3,   # No/Nay
    "i": 4,   # I/It/Indeed
    "t": 5,   # The/Thou/That/This/Thus
    "s": 3,   # So/Sir
    "h": 3,   # He/Here
    "m": 2,
    "o": 2,
    "w": 2,
    "b": 2,
    "f": 2,
}

# After Exclam then DECL: speaker is coming down from emphasis, often
# explaining. Similar to decl flow.
_EXCLAM_THEN_DECL_WEIGHTS: dict[str, float] = {
    "t": 5,
    "a": 4,
    "b": 3,
    "i": 3,
    "s": 3,
    "y": 2,
    "f": 2,
    "n": 2,
    "o": 2,
    "h": 2,
    "m": 2,
    "w": 2,
}


_Q_CHAIN_VEC = _vec_from_weights(_Q_CHAIN_WEIGHTS, _GLOBAL_SCALE)
_EXCLAM_CHAIN_VEC = _vec_from_weights(_EXCLAM_CHAIN_WEIGHTS, _GLOBAL_SCALE)
_DECL_FLOW_VEC = _vec_from_weights(_DECL_FLOW_WEIGHTS, _GLOBAL_SCALE)
_Q_THEN_DECL_VEC = _vec_from_weights(_Q_THEN_DECL_WEIGHTS, _GLOBAL_SCALE * 0.7)
_EXCLAM_THEN_DECL_VEC = _vec_from_weights(
    _EXCLAM_THEN_DECL_WEIGHTS, _GLOBAL_SCALE * 0.7
)


def discourse_rhythm_start_bias(
    recent_sentence_types: tuple[int, ...],
    words_in_sentence: int,
    letter_run_len: int,
    word_buffer: str,
    speaker_label_state: int,
) -> list[float] | None:
    """Bias the first letter of the upcoming sentence based on the
    pattern of the last 2-3 completed sentence types. Fires only at
    positions where the model is about to emit the first letter of
    the sentence's first word (words_in_sentence == 0, word_buffer
    empty, letter_run_len == 0). Returns None when no pattern matches."""
    if speaker_label_state != 0:
        return None
    if words_in_sentence != 0:
        return None
    if letter_run_len != 0:
        return None
    if word_buffer:
        return None
    if len(recent_sentence_types) < 2:
        return None

    last = recent_sentence_types[-1]
    second = recent_sentence_types[-2]

    # Question chain: 2+ questions in a row.
    if last == SENT_INTERROG and second == SENT_INTERROG:
        return _Q_CHAIN_VEC

    # Exclam chain: 2+ exclamations in a row.
    if last == SENT_EXCLAM and second == SENT_EXCLAM:
        return _EXCLAM_CHAIN_VEC

    # Declarative flow: 3 declaratives in a row.
    if (
        last == SENT_DECL
        and second == SENT_DECL
        and len(recent_sentence_types) >= 3
        and recent_sentence_types[-3] == SENT_DECL
    ):
        return _DECL_FLOW_VEC

    # Q-then-DECL: previous was question, current-last was declarative
    # (common answer pattern).
    if last == SENT_DECL and second == SENT_INTERROG:
        return _Q_THEN_DECL_VEC

    # Exclam-then-DECL: exclamatory then calm declarative.
    if last == SENT_DECL and second == SENT_EXCLAM:
        return _EXCLAM_THEN_DECL_VEC

    return None
