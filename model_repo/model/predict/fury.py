"""Predict consumer for `state.fury_register`.

When the rage/wrath texture is elevated, bias the letter stream
toward characteristic invective vocabulary:

  Word-start (letter_run_len == 0, outside speaker label):
    Boost first letters of fury-class starters:
      d (damn, devil, dog, die)
      h (hell, hate, hag)
      w (wrath, wretch, witch, woe)
      v (vile, villain, venom, viper, vengeance)
      c (curse, cur, coward)
      r (rage, rogue, rascal, rot)
      f (foul, fiend, false, fool)
      p (plague, putrid, poison)
    Mildly discourage peace/love first letters (l, s): fury doesn't
    reach for soft lexicon.

  Sentence-end punctuation (word just completed, last_is_vowel or
  letter_run_len >= 2 with a word boundary coming):
    When sentence_start_pending is False and punctuation is a plausible
    next token, boost "!" over "." and ";". Captures the fact that
    fury speech tends to punch endings with exclamations.

Magnitudes scale with (fury_register - threshold) so the bias is
gentle at threshold and grows with intensity.

No corpus statistics — Shakespearean invective lexicon from prior
knowledge of the plays' rage registers.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Threshold below which no bias fires.
_START_THRESHOLD = 0.30
_END_THRESHOLD = 0.35

# Word-start fury-starter weights (applied to lowercase and a smaller
# share to the capitalized form).
_FURY_STARTERS: dict[str, float] = {
    "d": 0.50,  # damn, devil, die, death, dog, dogged
    "h": 0.40,  # hell, hate, hag, horror
    "w": 0.50,  # wrath, wretch, witch, woe
    "v": 0.60,  # vile, villain, venom, viper, vengeance
    "c": 0.35,  # curse, cur, coward
    "r": 0.45,  # rage, rogue, rascal, rot
    "f": 0.35,  # foul, fiend, false, fool
    "p": 0.40,  # plague, putrid, poison
    "k": 0.25,  # knave, kill
    "b": 0.22,  # base, beast, bastard, bloody, burn
    "m": 0.18,  # murder, monster, mad, malice
    "s": 0.18,  # slave, scorn, strike, spite
    "t": 0.15,  # traitor, torment
}

# Mildly penalize tender/soft first letters (overlap with love register).
_COUNTER_STARTERS: dict[str, float] = {
    "l": -0.12,  # love, lovely, lullaby
    # "s" is ambiguous (slave-as-insult vs sweet-as-tender). Skip.
    "g": -0.08,  # gentle, grace
}


def fury_start_bias(
    fury_register: float,
    letter_run_len: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None
    if fury_register <= _START_THRESHOLD:
        return None

    # Scale linearly above threshold, capped at fury=1.0 → scale 1.0.
    scale = fury_register - _START_THRESHOLD
    # scale is in [~0, 0.70]; normalize mildly so max is ~0.5.
    if scale > 0.5:
        scale = 0.5

    vec = [0.0] * VOCAB_SIZE
    for ch, w in _FURY_STARTERS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += w * scale
        up = ch.upper()
        if up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] += w * scale * 0.55
    for ch, w in _COUNTER_STARTERS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += w * scale
        up = ch.upper()
        if up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] += w * scale * 0.55
    return vec


def fury_end_bias(
    fury_register: float,
    letter_run_len: int,
    word_buffer: str,
    speaker_label_state: int,
    words_in_sentence: int,
) -> list[float] | None:
    """At word-end positions (letter_run_len >= 2 with a complete-ish
    buffer), if fury is high and the sentence is already a few words
    long, boost "!" over "." and ";".
    """
    if speaker_label_state != 0:
        return None
    if fury_register <= _END_THRESHOLD:
        return None
    if letter_run_len < 2:
        return None
    if not word_buffer:
        return None
    # Only if the sentence has enough content to plausibly be ending.
    if words_in_sentence < 2:
        return None

    scale = fury_register - _END_THRESHOLD
    if scale > 0.4:
        scale = 0.4

    vec = [0.0] * VOCAB_SIZE
    if "!" in VOCAB_INDEX:
        vec[VOCAB_INDEX["!"]] += 0.60 * scale
    if "." in VOCAB_INDEX:
        vec[VOCAB_INDEX["."]] -= 0.20 * scale
    if ";" in VOCAB_INDEX:
        vec[VOCAB_INDEX[";"]] -= 0.08 * scale
    return vec
