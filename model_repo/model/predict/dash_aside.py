"""Predict layer — parenthetical-dash aside biases.

Consumes `state.in_dash_aside`, `state.chars_since_dash_open`,
and `state.words_since_dash_open` (set by pipeline/dash_aside.py).

Two regimes:

1. JUST-OPENED (chars_since_dash_open == 0, in_dash_aside == True):
   We've just emitted the second '-' of a "--" run. The next char
   is empirically overwhelmingly a newline, a space, or a capital
   opener letter of a discourse particle: For/And/But/Yet/So/Which/
   If/O/I/This/These/The.

2. INSIDE-ASIDE (in_dash_aside == True, post-opening): bias toward
   short-aside termination. As `words_since_dash_open` grows past ~4,
   bias toward closing-dash "-" after a space (sets up "-- "
   closure), or toward sentence punctuation.

No corpus statistics — rules from reading the corpus's use of "--"
as a mid-sentence parenthetical convention.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def dash_aside_open_bias(
    in_dash_aside: bool,
    chars_since_dash_open: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Fires at the moment just AFTER '--' was completed.
    chars_since_dash_open == 0 on the step right after opening.
    """
    if speaker_label_state != 0:
        return None
    if not in_dash_aside:
        return None
    if chars_since_dash_open != 0:
        return None

    vec = [0.0] * VOCAB_SIZE
    # Strong newline boost — half the -- asides in the corpus go
    # "--\n" before continuing.
    if "\n" in VOCAB_INDEX:
        vec[VOCAB_INDEX["\n"]] += 1.8
    # Space — the other common continuation is "-- " then a word.
    if " " in VOCAB_INDEX:
        vec[VOCAB_INDEX[" "]] += 0.9
    # After the space/newline, a capital opener of a discourse particle
    # is common. Boost capital letters of the common turn-opener set;
    # these biases only fire directly after the dash, so they're a
    # modest nudge not a dominant effect.
    for ch, w in (
        ("F", 0.25),  # For
        ("A", 0.20),  # And, Ay
        ("B", 0.22),  # But
        ("Y", 0.18),  # Yet
        ("S", 0.20),  # So
        ("W", 0.18),  # Which, Why, What
        ("I", 0.25),  # I, If
        ("O", 0.25),  # O
        ("T", 0.20),  # This, The, To
        ("N", 0.15),  # Now
        ("H", 0.15),  # He, Here
    ):
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += w
    return vec


def dash_aside_close_bias(
    in_dash_aside: bool,
    words_since_dash_open: int,
    letter_run_len: int,
    last_char: str,
    speaker_label_state: int,
) -> list[float] | None:
    """Encourages aside closure as the aside grows long.
    Fires at word-boundary positions (letter_run_len == 0) inside
    an aside with words_since_dash_open >= 3.
    """
    if speaker_label_state != 0:
        return None
    if not in_dash_aside:
        return None
    if letter_run_len != 0:
        return None
    if words_since_dash_open < 3:
        return None
    # Only meaningful right after a space, so that "-- " closing is
    # a natural continuation into "--".
    if last_char != " ":
        return None

    # Scale grows with words spent inside aside.
    if words_since_dash_open == 3:
        scale = 0.30
    elif words_since_dash_open == 4:
        scale = 0.55
    elif words_since_dash_open == 5:
        scale = 0.85
    else:
        scale = 1.20

    vec = [0.0] * VOCAB_SIZE
    # Boost "-" to set up the closing "--".
    if "-" in VOCAB_INDEX:
        vec[VOCAB_INDEX["-"]] += scale
    return vec
