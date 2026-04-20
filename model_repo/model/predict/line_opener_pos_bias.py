"""Line-opener POS anaphora bias.

Reads `state.recent_line_opener_pos` — the POS tags of the first word
of each of the last up-to-4 completed lines (most-recent LAST). When
the recent openers show a same-POS run of 2+, boost first-letters
that are typical for that POS class at the NEXT line-start.

This complements `predict/anaphora.py` which boosts shared starter
LETTERS when the letters themselves agree. The POS bias fires when
the words were different but in the same grammatical class — a
looser but more common Shakespearean anaphora pattern:

  "I know not what I seek. I cannot tell the hour. I would that..."
    → 3 PRONOUN openers; pattern-detected even though 'know', 'cannot',
      'would' never match letter-wise.

  "Hard is the way. Sharp the tongue. Cold the hand."
    → 3 ADJECTIVE openers starting H, S, C — no shared letter but a
      shared POS.

Gating:
  * speaker_label_state == 0
  * consecutive_newlines == 1 (single newline — new line within a turn,
    NOT a turn-start where the first word is shaped by a speaker label)
  * recent_line_opener_pos has >= 2 entries all sharing the same POS
  * last_char == "\\n"
  * letter_run_len == 0, word_buffer == "" (word hasn't started yet)

No corpus statistics — first-letter mappings come from prior knowledge
of which POS classes typically open English words.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# POS tag constants — must match pipeline/pos.py.
_POS_UNKNOWN = 0
_POS_ARTICLE = 1
_POS_PRONOUN = 2
_POS_POSSESSIVE = 3
_POS_PREPOSITION = 4
_POS_CONJUNCTION = 5
_POS_AUX_VERB = 6
_POS_MODAL = 7
_POS_INTERJECTION = 8
_POS_NEGATION = 9
_POS_ADVERB = 10
_POS_VERB_ING = 11
_POS_VERB_ED = 12
_POS_NOUN = 13
_POS_ADJECTIVE = 14
_POS_PROPER_NOUN = 15
_POS_VERB = 16
_POS_NUMBER = 17
_POS_WH = 18


# Per-POS first-letter weights (lowercase + capital variants when
# appropriate). Lines after the first line of a turn are capitalized
# by convention but verse sometimes continues lowercase enjambed; we
# bias both case variants, with a slight tilt toward capitals since
# line-start in verse is conventionally capitalized.
_POS_LETTERS: dict[int, dict[str, float]] = {
    _POS_PRONOUN: {
        "I": 1.0, "T": 0.7, "H": 0.6, "Y": 0.5, "W": 0.5, "M": 0.5,
        "S": 0.5, "O": 0.3,  # She, Our
        "i": 0.4, "t": 0.35, "h": 0.3, "y": 0.25, "w": 0.25, "m": 0.25,
        "s": 0.25,
    },
    _POS_INTERJECTION: {
        "O": 1.0, "A": 0.7, "F": 0.6, "H": 0.5, "L": 0.4,
        "o": 0.45, "a": 0.35, "f": 0.3, "h": 0.25, "l": 0.2,
    },
    _POS_ADVERB: {
        "N": 0.7, "T": 0.7, "H": 0.6, "S": 0.6, "Y": 0.5,
        "n": 0.35, "t": 0.35, "h": 0.3, "s": 0.3, "y": 0.25,
    },
    _POS_CONJUNCTION: {
        "A": 0.9, "B": 0.8, "O": 0.6, "Y": 0.5, "N": 0.5, "S": 0.5,
        "a": 0.45, "b": 0.4, "o": 0.3, "y": 0.25, "n": 0.25, "s": 0.25,
    },
    _POS_PREPOSITION: {
        "O": 0.7, "I": 0.6, "T": 0.6, "W": 0.6, "F": 0.6, "U": 0.5,
        "B": 0.5, "A": 0.5,
        "o": 0.35, "i": 0.3, "t": 0.3, "w": 0.3, "f": 0.3, "u": 0.25,
        "b": 0.25, "a": 0.25,
    },
    _POS_MODAL: {
        "S": 0.8, "W": 0.8, "M": 0.7, "C": 0.6,
        "s": 0.4, "w": 0.4, "m": 0.35, "c": 0.3,
    },
    _POS_AUX_VERB: {
        "I": 0.8, "A": 0.7, "H": 0.7, "B": 0.6, "D": 0.5, "W": 0.5,
        "i": 0.4, "a": 0.35, "h": 0.35, "b": 0.3, "d": 0.25, "w": 0.25,
    },
    _POS_NEGATION: {
        "N": 1.0, "n": 0.5,
    },
    _POS_ARTICLE: {
        "T": 0.9, "A": 0.7,
        "t": 0.45, "a": 0.35,
    },
    _POS_POSSESSIVE: {
        "M": 0.7, "T": 0.7, "H": 0.6, "O": 0.5, "Y": 0.5,
        "m": 0.35, "t": 0.35, "h": 0.3, "o": 0.25, "y": 0.25,
    },
    _POS_WH: {
        "W": 1.0, "H": 0.6,
        "w": 0.5, "h": 0.3,
    },
    _POS_ADJECTIVE: {
        # Many possible starters; spread and modest.
        "F": 0.4, "S": 0.4, "G": 0.4, "H": 0.4, "B": 0.4, "W": 0.3,
        "P": 0.3, "D": 0.3, "C": 0.3, "L": 0.3, "M": 0.3, "T": 0.3,
        "f": 0.2, "s": 0.2, "g": 0.2, "h": 0.2, "b": 0.2, "w": 0.15,
        "p": 0.15, "d": 0.15, "c": 0.15, "l": 0.15, "m": 0.15, "t": 0.15,
    },
    _POS_NOUN: {
        "L": 0.4, "D": 0.4, "H": 0.4, "T": 0.4, "M": 0.3, "F": 0.3,
        "S": 0.3, "W": 0.3, "K": 0.3, "P": 0.3, "N": 0.3,
        "l": 0.2, "d": 0.2, "h": 0.2, "t": 0.2, "m": 0.15, "f": 0.15,
        "s": 0.15, "w": 0.15, "k": 0.15, "p": 0.15, "n": 0.15,
    },
    _POS_VERB: {
        "S": 0.4, "G": 0.4, "C": 0.4, "L": 0.4, "M": 0.4, "T": 0.3,
        "F": 0.3, "H": 0.3, "P": 0.3, "R": 0.3, "B": 0.3,
        "s": 0.2, "g": 0.2, "c": 0.2, "l": 0.2, "m": 0.2, "t": 0.15,
        "f": 0.15, "h": 0.15, "p": 0.15, "r": 0.15, "b": 0.15,
    },
    # PROPER_NOUN, NUMBER, UNKNOWN, VERB_ING, VERB_ED — we don't build
    # a pattern bias for these (too diffuse or too specific).
}


def line_opener_pos_bias(
    recent_line_opener_pos: tuple[int, ...],
    speaker_label_state: int,
    consecutive_newlines: int,
    last_char: str,
    letter_run_len: int,
    word_buffer: str,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if consecutive_newlines != 1:
        return None
    if last_char != "\n":
        return None
    if letter_run_len != 0 or word_buffer:
        return None
    if len(recent_line_opener_pos) < 2:
        return None

    # Detect same-POS run in the last 2 or 3 entries.
    last = recent_line_opener_pos[-1]
    # Minimum: last two agree.
    if recent_line_opener_pos[-2] != last:
        return None
    # Depth-3 run boosts scale.
    if len(recent_line_opener_pos) >= 3 and recent_line_opener_pos[-3] == last:
        scale = 0.70
    else:
        scale = 0.40

    letters = _POS_LETTERS.get(last)
    if letters is None:
        return None

    vec = [0.0] * VOCAB_SIZE
    for ch, w in letters.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += scale * w
    return vec
