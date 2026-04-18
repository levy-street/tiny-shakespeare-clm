"""Addressee-memory bias — when vocative_expectation fires AND the
current turn has already used a specific vocative noun, bias strongly
toward repeating that same noun.

Two positions of action:

  - Word-start: inside vocative expectation, boost the first letter of
    last_vocative (and penalize other vocative-noun first letters so
    the bias crowds out generic vocative choices).

  - Mid-word: when the current word_buffer is a prefix of last_vocative,
    boost the next-matching letter to extend toward that noun.

Scales with turn_vocative_count — the more times this speaker has used
this vocative, the more confident we are that the next one will match.

This sits on top of the generic vocative first-letter bias. It does
not fire when no vocative has been recorded (turn_vocative_count == 0)
— in that case the generic prior is all we have.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def addressee_start_bias(
    last_vocative: str,
    turn_vocative_count: int,
    vocative_expectation: bool,
    speaker_label_state: int,
) -> list[float] | None:
    """Boost first letter of last_vocative at vocative-expected word-start."""
    if speaker_label_state != 0:
        return None
    if not vocative_expectation:
        return None
    if not last_vocative or turn_vocative_count < 1:
        return None

    # Confidence scales with repeat count.
    # 1 use: moderate hint; 2+ uses: strong hint.
    if turn_vocative_count == 1:
        boost = 1.4
    elif turn_vocative_count == 2:
        boost = 2.0
    else:
        boost = 2.6

    vec = [0.0] * VOCAB_SIZE
    first = last_vocative[0]
    if first in VOCAB_INDEX:
        vec[VOCAB_INDEX[first]] += boost
    up = first.upper()
    if up != first and up in VOCAB_INDEX:
        vec[VOCAB_INDEX[up]] += boost * 0.5

    # Gently suppress OTHER common vocative-first-letters so the memory
    # dominates the generic vocative prior.
    _OTHER_VOC_STARTS = "lsmfpbckqdh"
    suppress = min(boost * 0.25, 0.6)
    for c in _OTHER_VOC_STARTS:
        if c == first:
            continue
        if c in VOCAB_INDEX:
            vec[VOCAB_INDEX[c]] -= suppress
        cu = c.upper()
        if cu in VOCAB_INDEX:
            vec[VOCAB_INDEX[cu]] -= suppress * 0.5
    return vec


def addressee_midword_bias(
    last_vocative: str,
    turn_vocative_count: int,
    word_buffer: str,
    vocative_expectation: bool,
    speaker_label_state: int,
) -> list[float] | None:
    """When buffer is a prefix of last_vocative and we're mid-word in
    a vocative-expected position, boost the next continuing letter.
    """
    if speaker_label_state != 0:
        return None
    if not vocative_expectation:
        return None
    if not last_vocative or turn_vocative_count < 1:
        return None
    if not word_buffer or len(word_buffer) >= len(last_vocative):
        return None
    if not last_vocative.startswith(word_buffer):
        return None

    next_ch = last_vocative[len(word_buffer)]
    if turn_vocative_count == 1:
        boost = 0.9
    elif turn_vocative_count == 2:
        boost = 1.3
    else:
        boost = 1.7

    vec = [0.0] * VOCAB_SIZE
    if next_ch in VOCAB_INDEX:
        vec[VOCAB_INDEX[next_ch]] += boost
    return vec
