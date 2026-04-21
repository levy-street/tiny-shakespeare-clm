"""Predict layer — turn pronoun profile bias.

Reads `state.turn_pronoun_mode` set by pipeline/turn_pronoun.py.

Two biases:

(1) SENTENCE-START PRONOUN BIAS. At the start of a fresh sentence
    within a soliloquy-mode turn (mode 1), push capital "I". Within
    a direct-address turn (mode 2), push capital "T" (Thou) and "Y"
    (You/Ye). Within mixed mode (3), split the push.

    Gates: sentence-start (letter_run_len == 0, last char is a space
    that follows sentence-end punct or a newline) and no letters yet
    in the word buffer.

(2) CONTENT-WORD LEXICAL BIAS. Inside a soliloquy-mode turn, push
    first letters typical of introspective / philosophical vocabulary:
    t (think/truth), d (death/doubt/dream), l (life/love/light),
    m (mind/memory/mercy/man), s (soul/sorrow/self), h (heart/hope/
    honour), r (reason/rest), b (breath/being). Inside a direct-
    address turn, push letters typical of direct-address vocabulary:
    s (sir/speak/see), g (go/good), h (hear/hence/hie/hark), l (look/
    leave), c (come/cease/cry), m (mark/meet), t (tell/take/turn).

    Magnitudes are deliberately small (~0.08 — 0.18): the speaker's
    existing lexical layers still dominate. This layer is a texture
    reinforcement.

No corpus statistics — letter sets and thresholds are prior-knowledge
characterizations of Shakespearean soliloquy vs direct-address register.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Mode constants (mirror pipeline/turn_pronoun.py).
_MODE_NONE = 0
_MODE_I = 1
_MODE_YOU = 2
_MODE_MIXED = 3


# Content-word starter letter sets.
_SOLILOQUY_LETTERS: str = "tdlmshrb"
_DIRECT_ADDRESS_LETTERS: str = "sghlcmt"


def _letter_push_vec(letters: str, push_lower: float, push_upper: float) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch in letters:
        idx_lo = VOCAB_INDEX.get(ch)
        if idx_lo is not None:
            vec[idx_lo] += push_lower
        idx_hi = VOCAB_INDEX.get(ch.upper())
        if idx_hi is not None:
            vec[idx_hi] += push_upper
    return vec


# Pre-built content-word vectors.
_SOL_CONTENT_VEC = _letter_push_vec(_SOLILOQUY_LETTERS, 0.06, 0.03)
_SOL_CONTENT_VEC_STRONG = _letter_push_vec(_SOLILOQUY_LETTERS, 0.10, 0.05)
_ADDR_CONTENT_VEC = _letter_push_vec(_DIRECT_ADDRESS_LETTERS, 0.05, 0.02)
_ADDR_CONTENT_VEC_STRONG = _letter_push_vec(_DIRECT_ADDRESS_LETTERS, 0.08, 0.04)


def turn_pronoun_sentence_start_bias(
    turn_pronoun_mode: int,
    speaker_label_state: int,
    letter_run_len: int,
    word_buffer: str,
    at_sentence_start: bool,
) -> list[float] | None:
    """Push 'I' / 'Thou' / 'You' at a fresh sentence start based on mode.

    Caller passes `at_sentence_start` — True when last_cls is SPACE and
    chars_since_sentence_end == 1 (freshly capitalized after period), or
    when we're at the first word of a turn.
    """
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None
    if word_buffer:
        return None
    if not at_sentence_start:
        return None
    if turn_pronoun_mode == _MODE_NONE:
        return None

    vec = [0.0] * VOCAB_SIZE
    if turn_pronoun_mode == _MODE_I:
        idx = VOCAB_INDEX.get("I")
        if idx is not None:
            vec[idx] += 0.28
    elif turn_pronoun_mode == _MODE_YOU:
        idx = VOCAB_INDEX.get("T")  # Thou / Thee / Thy
        if idx is not None:
            vec[idx] += 0.18
        idx = VOCAB_INDEX.get("Y")  # You / Ye / Your
        if idx is not None:
            vec[idx] += 0.12
    elif turn_pronoun_mode == _MODE_MIXED:
        idx = VOCAB_INDEX.get("I")
        if idx is not None:
            vec[idx] += 0.10
        idx = VOCAB_INDEX.get("T")
        if idx is not None:
            vec[idx] += 0.08
        idx = VOCAB_INDEX.get("Y")
        if idx is not None:
            vec[idx] += 0.04

    return vec


def turn_pronoun_content_bias(
    turn_pronoun_mode: int,
    turn_i_pronouns: int,
    turn_you_pronouns: int,
    speaker_label_state: int,
    letter_run_len: int,
    word_buffer: str,
    last_char_class: int,
) -> list[float] | None:
    """Push content-word first letters based on soliloquy vs direct-address
    register. Fires only at mid-turn word-starts (last char = SPACE)."""
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None
    if word_buffer:
        return None
    # last_char_class == 1 (SPACE). We don't bias sentence-starts here
    # (those get turn_pronoun_sentence_start_bias which is separate).
    if last_char_class != 1:
        return None
    if turn_pronoun_mode == _MODE_NONE:
        return None

    # Mid-turn word-start push: in I-mode, boost capital "I" as a
    # likely mid-sentence word start ("I think", "I am", "I love")
    # — Shakespeare's soliloquies tile "I"-clauses densely. In
    # you-mode, boost capital "T" (Thou) similarly at mid-sentence.
    # Magnitudes modest since most word-starts are NOT pronouns.
    if turn_pronoun_mode == _MODE_I and turn_i_pronouns >= 3:
        vec = [0.0] * VOCAB_SIZE
        idx = VOCAB_INDEX.get("I")
        if idx is not None:
            vec[idx] += 1.75 if turn_i_pronouns < 6 else 2.20
        return vec
    if turn_pronoun_mode == _MODE_YOU and turn_you_pronouns >= 3:
        vec = [0.0] * VOCAB_SIZE
        idx = VOCAB_INDEX.get("T")
        if idx is not None:
            vec[idx] += 1.20 if turn_you_pronouns < 6 else 1.55
        idx = VOCAB_INDEX.get("Y")
        if idx is not None:
            vec[idx] += 0.75 if turn_you_pronouns < 6 else 1.05
        return vec
    return None
