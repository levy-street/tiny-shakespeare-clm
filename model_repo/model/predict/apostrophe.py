"""Predict layer — apostrophe-mode biases.

Reads `state.apostrophe_mode` (set by pipeline/apostrophe.py):

  0 — off / normal discourse. No bias.
  1 — primed. Speaker just opened with "O" / "Oh" / "Ye" / "Alas" at
      sentence start; we expect an abstract-noun target to follow. Mild
      boost at word-start to first-letters of apostrophe-target nouns.
  2 — active. Abstract target named. Exclamation terminators are the
      rhetorical-figure's natural punctuation; tilt "!" slightly over
      "." at sentence-end-punct positions. Resist premature newline
      during the apostrophe expansion (Shakespearean apostrophes are
      typically long expansive exclamations).
  3 — locked. Stronger versions of mode 2. Apostrophe reinforced by a
      second invocation or an imperative directed at the target.

All weights are small and hand-chosen from prior knowledge of how
Shakespearean apostrophe passages scan. No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Lowercase word-start letter preferences for abstract apostrophe
# targets (shared across families):
#   heaven, heart, honour, hope, hell     -> h
#   love                                  -> l
#   night, nature                         -> n
#   death, doom, destiny                  -> d
#   fortune, fate, faith, fire            -> f
#   grief, gods, grave                    -> g
#   time, truth                           -> t
#   earth                                 -> e
#   sun, stars, soul, sorrow              -> s
#   moon, mind, muse, mercy               -> m
#   pity, peace, pride                    -> p
#   beauty                                -> b
_TARGET_LC_STARTERS: dict[str, float] = {
    "h": 0.55, "l": 0.40, "n": 0.45, "d": 0.45, "f": 0.45,
    "g": 0.35, "t": 0.35, "e": 0.30, "s": 0.35, "m": 0.30,
    "p": 0.25, "b": 0.25, "o": 0.20,
}

# Uppercase starters (rare — mid-sentence, but possible right after "O")
_TARGET_UC_STARTERS: dict[str, float] = {
    "H": 0.30, "L": 0.25, "N": 0.25, "D": 0.25, "F": 0.25,
    "G": 0.20, "T": 0.20, "E": 0.18, "S": 0.20, "M": 0.18,
}


def apostrophe_start_bias(
    apostrophe_mode: int,
    letter_run_len: int,
    last_char_class: int,  # SPACE=2, NEWLINE=3 at word-start
    speaker_label_state: int,
) -> list[float] | None:
    """Word-start (letter_run_len == 0) tilt toward apostrophe-target
    starters when apostrophe mode is primed (1) or active (2).
    """
    if speaker_label_state != 0:
        return None
    if apostrophe_mode < 1:
        return None
    if letter_run_len != 0:
        return None
    # Scale: strongest at priming (mode 1) so the target lands; milder
    # at mode 2/3 where target is already named. Kept conservative: the
    # base distribution has already been shaped by many upstream layers
    # and the goal here is a gentle tilt, not a dominating force.
    if apostrophe_mode == 1:
        scale = 0.55
    elif apostrophe_mode == 2:
        scale = 0.30
    else:  # 3
        scale = 0.20

    vec = [0.0] * VOCAB_SIZE
    for ch, w in _TARGET_LC_STARTERS.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
    for ch, w in _TARGET_UC_STARTERS.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale * 0.5
    return vec


def apostrophe_terminator_bias(
    apostrophe_mode: int,
    last_char_class: int,  # unused but kept for signature compatibility
    speaker_label_state: int,
    chars_since_newline: int,
) -> list[float] | None:
    """Sentence-end terminator tilt toward "!" over "." when
    apostrophe mode is active (2+). Fires at positions where either
    terminator would be grammatical; the bias magnitudes are small
    so they only matter when the base distribution is close to a tie.
    """
    if speaker_label_state != 0:
        return None
    if apostrophe_mode < 2:
        return None
    # Only active when we're deep enough in a line that a terminator
    # decision is plausible (avoid the very first tokens of a line).
    if chars_since_newline < 6:
        return None

    if apostrophe_mode == 2:
        excl = 0.22
        period = -0.08
    else:  # 3
        excl = 0.40
        period = -0.15

    vec = [0.0] * VOCAB_SIZE
    idx = VOCAB_INDEX.get("!")
    if idx is not None:
        vec[idx] += excl
    idx = VOCAB_INDEX.get(".")
    if idx is not None:
        vec[idx] += period
    # Also nudge "," since expansive apostrophe passages often have
    # long clauses separated by commas ("Come, night; come, Romeo;
    # come, thou day in night!").
    idx = VOCAB_INDEX.get(",")
    if idx is not None:
        vec[idx] += excl * 0.25
    return vec
