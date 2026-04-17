"""Context-class bias layer.

Maps a coarse context key (derived from state.last_char_class, newline
state, speaker-label FSM, etc.) to per-class log-biases that nudge the
distribution toward the kinds of characters that tend to follow.
"""

from __future__ import annotations

from ..pipeline.linguistic import (
    APOSTROPHE,
    CLASS_OF_TOKEN,
    DASH,
    LOWER_CONS,
    LOWER_VOWEL,
    NEWLINE,
    OTHER,
    PUNCT_END,
    PUNCT_MID,
    SPACE,
    UPPER,
)
from ..state import ModelState
from ..vocab import VOCAB_SIZE

# Context identifiers.
CTX_START = 0
CTX_AFTER_DOUBLE_NL = 1
CTX_AFTER_SINGLE_NL = 2
CTX_AFTER_SPACE_SENT_START = 3
CTX_AFTER_SPACE = 4
CTX_AFTER_PUNCT_END = 5
CTX_AFTER_PUNCT_MID = 6
CTX_AFTER_APOS = 7
CTX_AFTER_DASH = 8
CTX_IN_SPEAKER_LABEL = 9
CTX_AFTER_COLON_LABEL = 10
CTX_IN_WORD_SHORT = 11
CTX_IN_WORD_MID = 12
CTX_IN_WORD_LONG = 13
CTX_AFTER_UPPER_START = 14
CTX_IN_MIXED_LABEL_WORD = 15  # inside a mixed-case speaker label word
CTX_SPEAKER_LABEL_AMBIGUOUS = 16  # first letter after label start (could be any)
CTX_OTHER = 17

N_CTX = 18

_BIAS: list[list[float]] = [[0.0] * 10 for _ in range(N_CTX)]


def _set(ctx: int, values: dict[int, float]) -> None:
    for k, v in values.items():
        _BIAS[ctx][k] = v


_set(
    CTX_START,
    {
        UPPER: 6.0,
        LOWER_CONS: 1.0,
        LOWER_VOWEL: 0.5,
        NEWLINE: -4.0,
        SPACE: -4.0,
        PUNCT_END: -6.0,
        PUNCT_MID: -6.0,
        APOSTROPHE: -4.0,
        DASH: -4.0,
        OTHER: -6.0,
    },
)
_set(
    CTX_AFTER_DOUBLE_NL,
    {
        UPPER: 6.5,
        LOWER_CONS: -3.5,
        LOWER_VOWEL: -3.5,
        NEWLINE: -1.5,
        SPACE: -3.5,
        PUNCT_END: -6.0,
        PUNCT_MID: -6.0,
        APOSTROPHE: -4.0,
        DASH: -4.0,
        OTHER: -6.0,
    },
)
_set(
    CTX_AFTER_SINGLE_NL,
    {
        UPPER: 1.6,
        LOWER_CONS: 0.6,
        LOWER_VOWEL: 0.5,
        NEWLINE: 0.4,
        SPACE: -3.0,
        PUNCT_END: -5.0,
        PUNCT_MID: -5.0,
        APOSTROPHE: -2.5,
        DASH: -2.0,
        OTHER: -5.0,
    },
)
_set(
    CTX_AFTER_SPACE_SENT_START,
    {
        UPPER: 3.5,
        LOWER_CONS: 0.6,
        LOWER_VOWEL: 0.3,
        NEWLINE: -5.0,
        SPACE: -6.0,
        PUNCT_END: -6.0,
        PUNCT_MID: -6.0,
        APOSTROPHE: 0.2,
        DASH: -4.0,
        OTHER: -6.0,
    },
)
_set(
    CTX_AFTER_SPACE,
    {
        UPPER: -0.5,
        LOWER_CONS: 1.5,
        LOWER_VOWEL: 1.3,
        NEWLINE: -5.0,
        SPACE: -6.0,
        PUNCT_END: -5.0,
        PUNCT_MID: -5.0,
        APOSTROPHE: -0.5,
        DASH: -3.0,
        OTHER: -5.0,
    },
)
_set(
    CTX_AFTER_PUNCT_END,
    {
        SPACE: 3.0,
        NEWLINE: 2.0,
        UPPER: -3.0,
        LOWER_CONS: -4.0,
        LOWER_VOWEL: -4.0,
        PUNCT_END: -4.0,
        PUNCT_MID: -4.0,
        APOSTROPHE: -2.0,
        DASH: -3.0,
        OTHER: -4.0,
    },
)
_set(
    CTX_AFTER_PUNCT_MID,
    {
        SPACE: 3.5,
        NEWLINE: 1.5,
        UPPER: -4.0,
        LOWER_CONS: -4.5,
        LOWER_VOWEL: -4.5,
        PUNCT_END: -5.0,
        PUNCT_MID: -5.0,
        APOSTROPHE: -3.0,
        DASH: -3.0,
        OTHER: -5.0,
    },
)
_set(
    CTX_AFTER_APOS,
    {
        LOWER_CONS: 3.0,
        LOWER_VOWEL: 1.5,
        UPPER: -4.0,
        SPACE: -2.5,
        NEWLINE: -4.0,
        PUNCT_END: -4.0,
        PUNCT_MID: -4.0,
        APOSTROPHE: -4.0,
        DASH: -4.0,
        OTHER: -5.0,
    },
)
_set(
    CTX_AFTER_DASH,
    {
        SPACE: 2.0,
        LOWER_CONS: 1.0,
        LOWER_VOWEL: 0.6,
        UPPER: -1.0,
        NEWLINE: -2.0,
        PUNCT_END: -3.0,
        PUNCT_MID: -3.0,
        APOSTROPHE: -3.0,
        DASH: -2.0,
        OTHER: -4.0,
    },
)
_set(
    CTX_IN_SPEAKER_LABEL,
    {
        UPPER: 4.0,
        SPACE: 1.5,
        PUNCT_MID: 2.5,
        LOWER_CONS: -2.0,
        LOWER_VOWEL: -2.0,
        NEWLINE: -4.0,
        PUNCT_END: -5.0,
        APOSTROPHE: -4.0,
        DASH: -4.0,
        OTHER: -5.0,
    },
)
_set(
    CTX_AFTER_COLON_LABEL,
    {
        NEWLINE: 5.0,
        SPACE: -1.0,
        UPPER: -4.0,
        LOWER_CONS: -4.0,
        LOWER_VOWEL: -4.0,
        PUNCT_END: -5.0,
        PUNCT_MID: -5.0,
        APOSTROPHE: -4.0,
        DASH: -4.0,
        OTHER: -5.0,
    },
)
_set(
    CTX_IN_WORD_SHORT,
    {
        LOWER_CONS: 2.0,
        LOWER_VOWEL: 2.0,
        UPPER: -3.0,
        SPACE: 0.3,
        NEWLINE: -2.0,
        PUNCT_END: -2.5,
        PUNCT_MID: -2.5,
        APOSTROPHE: 0.2,
        DASH: -3.0,
        OTHER: -5.0,
    },
)
_set(
    CTX_IN_WORD_MID,
    {
        LOWER_CONS: 1.5,
        LOWER_VOWEL: 1.3,
        UPPER: -3.5,
        SPACE: 1.4,
        NEWLINE: -1.0,
        PUNCT_END: -1.5,
        PUNCT_MID: -1.5,
        APOSTROPHE: 0.2,
        DASH: -3.0,
        OTHER: -5.0,
    },
)
_set(
    CTX_IN_WORD_LONG,
    {
        LOWER_CONS: 0.5,
        LOWER_VOWEL: 0.2,
        UPPER: -4.0,
        SPACE: 2.6,
        NEWLINE: 0.2,
        PUNCT_END: -0.5,
        PUNCT_MID: 0.0,
        APOSTROPHE: 0.0,
        DASH: -2.0,
        OTHER: -5.0,
    },
)
_set(
    CTX_AFTER_UPPER_START,
    {
        LOWER_CONS: 2.2,
        LOWER_VOWEL: 2.2,
        UPPER: -2.0,
        SPACE: -0.5,
        NEWLINE: -3.0,
        PUNCT_END: -3.0,
        PUNCT_MID: -3.0,
        APOSTROPHE: -0.5,
        DASH: -3.0,
        OTHER: -5.0,
    },
)
_set(
    # Inside a Mixed-Case speaker label word (e.g., "First Citizen"):
    # after the initial capital, the rest of the word is lowercase;
    # between words is a single space; the label ends with ":".
    CTX_IN_MIXED_LABEL_WORD,
    {
        LOWER_CONS: 2.0,
        LOWER_VOWEL: 2.0,
        UPPER: -1.5,
        SPACE: 1.0,
        PUNCT_MID: 2.0,  # ":" terminator
        NEWLINE: -4.0,
        PUNCT_END: -5.0,
        APOSTROPHE: -4.0,
        DASH: -4.0,
        OTHER: -5.0,
    },
)
_set(
    # Ambiguous position right after the first capital of a speaker
    # label: could be another capital (KING) or a lowercase (First).
    # Balanced bias.
    CTX_SPEAKER_LABEL_AMBIGUOUS,
    {
        UPPER: 1.5,
        LOWER_CONS: 1.5,
        LOWER_VOWEL: 1.5,
        SPACE: 0.3,
        PUNCT_MID: 0.8,
        NEWLINE: -4.0,
        PUNCT_END: -5.0,
        APOSTROPHE: -4.0,
        DASH: -4.0,
        OTHER: -5.0,
    },
)


def context_key(state: ModelState) -> int:
    if state.tokens_seen == 0:
        return CTX_START
    cls = state.last_char_class
    if cls == NEWLINE:
        if state.consecutive_newlines >= 2:
            return CTX_AFTER_DOUBLE_NL
        return CTX_AFTER_SINGLE_NL
    if cls == SPACE:
        if state.prev_char_class == PUNCT_END:
            return CTX_AFTER_SPACE_SENT_START
        return CTX_AFTER_SPACE
    if cls == PUNCT_END:
        return CTX_AFTER_PUNCT_END
    if cls == PUNCT_MID:
        if state.speaker_label_state == 3:
            return CTX_AFTER_COLON_LABEL
        return CTX_AFTER_PUNCT_MID
    if cls == APOSTROPHE:
        return CTX_AFTER_APOS
    if cls == DASH:
        return CTX_AFTER_DASH
    if cls == UPPER:
        if state.speaker_label_state == 2:
            # Inside speaker label; if we've already seen a lowercase,
            # this is a Mixed-Case label (next letter probably lowercase).
            if state.speaker_label_saw_lower:
                return CTX_IN_MIXED_LABEL_WORD
            return CTX_IN_SPEAKER_LABEL
        if state.upper_run_len == 1:
            return CTX_AFTER_UPPER_START
        return CTX_IN_SPEAKER_LABEL
    # Lowercase letter(s) in a speaker label → mixed-case context.
    if state.speaker_label_state == 2 and state.speaker_label_saw_lower:
        return CTX_IN_MIXED_LABEL_WORD
    pos = state.letter_run_len
    if pos <= 3:
        return CTX_IN_WORD_SHORT
    if pos <= 6:
        return CTX_IN_WORD_MID
    return CTX_IN_WORD_LONG


def context_bias_vector(ctx: int) -> list[float]:
    biases = _BIAS[ctx]
    return [biases[CLASS_OF_TOKEN[i]] for i in range(VOCAB_SIZE)]


# Precompute per-ctx VOCAB_SIZE vectors.
CTX_BIAS_VECTORS: list[list[float]] = [
    context_bias_vector(c) for c in range(N_CTX)
]
