"""Predict consumer — enjambment-density biasing at word-end.

Reads `state.enjambment_density` (see pipeline/enjambment.py). Fires only
at a word-end position (letter_run_len >= 2, on-trie, word in
COMPLETE_WORDS) and only when the current line is in the "could plausibly
close" zone (verse_line_run in [22, 75], verse_score >= 0.3).

The density rolls over verse-plausible lines:
  * High density ⇒ recent lines have run over (no terminal punct before
    \n). Lean into it: boost direct letter→\n and slightly suppress
    terminal punct before line-close.
  * Low density ⇒ recent lines have been end-stopped. Lean into it:
    boost "." "," ";" "!" "?" at word-end and slightly suppress direct
    \n (which would break the end-stopped habit).

Amplitude is intentionally small — this is a mood modulator, not a
primary signal. Other layers (pentameter_wordend_bias,
clause_rhythm_comma_bias, etc.) continue to decide the bulk of word-end
distribution.

No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_AMP: float = 0.25  # max log-unit swing from density extremes


def enjambment_wordend_bias(
    enjambment_density: float,
    verse_line_run: int,
    verse_score: float,
    speaker_label_state: int,
    chars_since_newline: int,
) -> list[float] | None:
    """Return a log-bias vector or None when the layer shouldn't fire."""
    if speaker_label_state != 0:
        return None
    if verse_score < 0.30:
        return None
    # Only bias inside the "line could plausibly close here" zone.
    # Below 22 chars it's too early; above 75 chars is non-verse prose.
    if not (22 <= chars_since_newline <= 75):
        return None
    # Must have been running verse (line-streak) for the density to mean
    # anything. Before establishment, density is just 0.0 default.
    if verse_line_run < 2:
        return None

    # Center around 0.5 so neutral density produces no bias.
    signed = (enjambment_density - 0.5) * 2.0  # in [-1, +1]
    # Stronger bias when verse streak is longer (up to 5-line cap).
    streak_gain = min(verse_line_run, 5) / 5.0  # 0.2 .. 1.0
    amp = _AMP * streak_gain

    # Positive signed → enjambed rhythm → boost \n direct, suppress
    # terminal punct. Negative signed → end-stopped → boost punct,
    # slightly suppress \n.
    nl_bias = signed * amp
    punct_bias = -signed * amp * 0.60

    vec = [0.0] * VOCAB_SIZE
    if "\n" in VOCAB_INDEX:
        vec[VOCAB_INDEX["\n"]] += nl_bias
    for ch in (".", ",", ";", ":", "!", "?"):
        if ch in VOCAB_INDEX:
            # Weight specific marks — "." and "," are the dominant
            # line-closers; ";" and ":" are intermediate; "!" and "?"
            # are reserved for exclamatory/interrog usage and should
            # move less with this axis.
            if ch == "." or ch == ",":
                vec[VOCAB_INDEX[ch]] += punct_bias
            elif ch == ";" or ch == ":":
                vec[VOCAB_INDEX[ch]] += punct_bias * 0.60
            else:  # "!" "?"
                vec[VOCAB_INDEX[ch]] += punct_bias * 0.30
    return vec
