"""Phonotactic gate on speaker labels: every real speaker name has a vowel.

Reads `state.speaker_buffer_vowels` (set by pipeline/speaker_vowels.py)
alongside the FSM state and buffer length. When the buffer has
accumulated 2+ letters with zero vowels, the label is almost certainly
phantom — real Shakespeare speaker labels ALWAYS contain at least one
vowel (A, E, I, O, U, or Y). Examples of no-vowel gibberish caught:
"TCK:", "BNR:", "CCM:", "HMM:" …

This layer applies:
  * strong ":" penalty (prevent the phantom label from closing)
  * strong per-consonant penalty (prevent further gibberish letters)
  * vowel boost (if the label IS a legitimate unknown name prefix
    that just hasn't landed on a vowel yet, let it proceed)
  * mild \\n boost (escape hatch if the FSM is truly stuck)

Gating:
  * speaker_label_state == 2 (inside a label name)
  * speaker_buffer_vowels == 0
  * len(speaker_buffer) >= 2 (single-letter start is fine: many
    names begin with a consonant)

Escalates with length: a 2-letter no-vowel buffer is still
plausible (name is in progress); a 4-letter no-vowel buffer is a
near-certain phantom.

No corpus statistics — relies on the structural fact that every
English name contains a vowel.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Uppercase vowels — speaker names are uppercase.
_VOWELS_UP: tuple[str, ...] = ("A", "E", "I", "O", "U")
# Uppercase consonants — everything non-vowel in A-Z.
_CONSONANTS_UP: tuple[str, ...] = tuple(
    ch for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if ch not in set(_VOWELS_UP + ("Y",))
)


def speaker_vowel_gate_bias(
    speaker_label_state: int,
    speaker_buffer: str,
    speaker_buffer_vowels: int,
    speaker_label_saw_lower: bool,
) -> list[float] | None:
    if speaker_label_state != 2:
        return None
    if speaker_buffer_vowels != 0:
        return None
    n = len(speaker_buffer)
    if n < 2:
        return None

    # Escalation schedule by buffer length.
    # n=2: mild (name might still become "BR..." → "BRUTUS")
    # n=3: strong (3 consonants with no vowel is already implausible)
    # n=4+: extreme (no English name goes 4 consonants before a vowel)
    if n == 2:
        colon_pen = -1.5
        cons_pen = -0.35
        vowel_boost = 0.80
        nl_boost = 0.0
    elif n == 3:
        colon_pen = -3.0
        cons_pen = -0.80
        vowel_boost = 1.60
        nl_boost = 0.40
    else:  # n >= 4
        colon_pen = -5.0
        cons_pen = -1.50
        vowel_boost = 2.40
        nl_boost = 1.20

    vec = [0.0] * VOCAB_SIZE

    # Colon penalty — prevent phantom closure.
    idx = VOCAB_INDEX.get(":")
    if idx is not None:
        vec[idx] += colon_pen

    # Consonant penalty — both uppercase and (if mixed-case)
    # lowercase case variants. The FSM routes case based on prior
    # saw_lower, so target the active case primarily.
    for ch in _CONSONANTS_UP:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += cons_pen
        # Also apply to lowercase consonants if the label has gone
        # mixed-case (e.g. "Tck" → shouldn't happen but guard).
        if speaker_label_saw_lower:
            lc = ch.lower()
            lidx = VOCAB_INDEX.get(lc)
            if lidx is not None:
                vec[lidx] += cons_pen

    # Vowel boost — let the buffer continue toward a real name.
    for ch in _VOWELS_UP:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += vowel_boost
        if speaker_label_saw_lower:
            lc = ch.lower()
            lidx = VOCAB_INDEX.get(lc)
            if lidx is not None:
                vec[lidx] += vowel_boost

    # Y is vowel-ish at end-of-name; give a smaller boost than the
    # core five.
    y_idx = VOCAB_INDEX.get("Y")
    if y_idx is not None:
        vec[y_idx] += vowel_boost * 0.5
    if speaker_label_saw_lower:
        yl_idx = VOCAB_INDEX.get("y")
        if yl_idx is not None:
            vec[yl_idx] += vowel_boost * 0.5

    # Newline escape hatch at deep no-vowel drift.
    if nl_boost > 0.0:
        nl_idx = VOCAB_INDEX.get("\n")
        if nl_idx is not None:
            vec[nl_idx] += nl_boost

    return vec
