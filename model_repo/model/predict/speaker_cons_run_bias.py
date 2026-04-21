"""Phonotactic gate on trailing consonant runs inside speaker labels.

Reads `state.speaker_buffer_cons_run` (set by pipeline/speaker_cons_run.py)
alongside the FSM state. When the buffer has accumulated 3+ adjacent
consonants at its trailing end, the label is drifting into gibberish:
real English names rarely have 3 consonants in a row mid-name, and
essentially never 4+.

Complementary to `speaker_vowel_gate`:
  * speaker_vowel_gate fires only when the ENTIRE buffer has zero
    vowels. It catches early gibberish like "TCK:", "BNR:", but lets
    "MNNEILRKHI" pass once an "E" appears.
  * speaker_cons_run_bias fires when the LAST N letters are all
    consonants, catching the "MNN…LRK…" pattern where vowels are
    scattered but consonant clusters accumulate.

The bias:
  * penalizes further consonant letters
  * boosts vowels (so the label can escape into a plausible name)
  * boosts newline (escape hatch at extreme drift)
  * slight colon/space penalty (prevent phantom closure; we want the
    model to fix the label rather than close it)

Gating:
  * speaker_label_state == 2 (inside a label name)
  * speaker_buffer_cons_run >= 3

Escalates with run length: run==3 is mild (some real names have 3-
consonant endings like "-NTZ"); run==4 is severe; run==5+ extreme.

No corpus statistics — structural English phonotactic rule.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_VOWELS_UP: tuple[str, ...] = ("A", "E", "I", "O", "U")
_CONSONANTS_UP: tuple[str, ...] = tuple(
    ch for ch in "BCDFGHJKLMNPQRSTVWXZ"
)


def speaker_cons_run_bias(
    speaker_label_state: int,
    speaker_buffer_cons_run: int,
    speaker_label_saw_lower: bool,
) -> list[float] | None:
    if speaker_label_state != 2:
        return None
    if speaker_buffer_cons_run < 3:
        return None

    # Escalation schedule by run length.
    # run==3: mild (many real endings have a 3-consonant coda like
    #         -NTZ, -RCH, but only at word-terminal; still, the
    #         signal is gently against further consonants)
    # run==4: severe (essentially no real English name has a 4-
    #         consonant mid-name cluster)
    # run==5+: extreme (purely gibberish)
    n = speaker_buffer_cons_run
    if n == 3:
        cons_pen = -0.60
        vowel_boost = 1.20
        nl_boost = 0.25
        colon_pen = -0.50
        space_pen = -0.20
    elif n == 4:
        cons_pen = -1.50
        vowel_boost = 2.20
        nl_boost = 0.90
        colon_pen = -1.50
        space_pen = -0.60
    else:  # n >= 5
        cons_pen = -2.50
        vowel_boost = 3.00
        nl_boost = 1.80
        colon_pen = -2.50
        space_pen = -1.00

    vec = [0.0] * VOCAB_SIZE

    # Consonant penalty — uppercase (primary) and lowercase (if FSM
    # drifted to mixed-case).
    for ch in _CONSONANTS_UP:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += cons_pen
        if speaker_label_saw_lower:
            lc = ch.lower()
            lidx = VOCAB_INDEX.get(lc)
            if lidx is not None:
                vec[lidx] += cons_pen

    # Vowel boost — give the label a way out.
    for ch in _VOWELS_UP:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += vowel_boost
        if speaker_label_saw_lower:
            lc = ch.lower()
            lidx = VOCAB_INDEX.get(lc)
            if lidx is not None:
                vec[lidx] += vowel_boost

    # Y is vowel-ish; smaller boost.
    y_idx = VOCAB_INDEX.get("Y")
    if y_idx is not None:
        vec[y_idx] += vowel_boost * 0.4
    if speaker_label_saw_lower:
        yl_idx = VOCAB_INDEX.get("y")
        if yl_idx is not None:
            vec[yl_idx] += vowel_boost * 0.4

    # Newline escape at deep drift.
    if nl_boost > 0.0:
        nl_idx = VOCAB_INDEX.get("\n")
        if nl_idx is not None:
            vec[nl_idx] += nl_boost

    # Space: mild penalty (don't let a gibberish mid-cluster segment
    # on a space — that would produce two bad-fragment names).
    if space_pen < 0.0:
        sp_idx = VOCAB_INDEX.get(" ")
        if sp_idx is not None:
            vec[sp_idx] += space_pen

    # Colon: penalty — block closing a gibberish label.
    if colon_pen < 0.0:
        c_idx = VOCAB_INDEX.get(":")
        if c_idx is not None:
            vec[c_idx] += colon_pen

    return vec
