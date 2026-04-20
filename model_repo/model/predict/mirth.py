"""Predict layer — mirth register word-start bias.

Reads `state.mirth_register` (rolling [0, 1] float). When the register
crosses the threshold, boost mirth-lexicon first letters at word-start:

  m — mirth, merry, music, merry, mad, mock
  j — jest, jolly, joy, jape, jig
  l — laugh, laughter
  f — fool, feast, frolic, festive, fun, fine
  s — sport, song, sing, smile, sack
  c — cheer, carol, cup, caper, clown
  g — gay, glad, glee
  p — play, pleasant, pipe
  r — revel, revels
  d — dance, drink
  w — wit, wine, wedding

Plus "O" / "Ha" starters at sentence-start for comic apostrophe
("Ha, ha!" or "O merry morn!").

Threshold: mirth_register >= 0.20.
Scaling: linear above threshold, capped at 1.0.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_MIRTH_STARTERS: dict[str, float] = {
    "m": 0.20,   # merry, mirth, music, mock, masque
    "j": 0.28,   # jest, jolly, joy, jape, jig
    "l": 0.14,   # laugh, laughter
    "f": 0.20,   # fool, feast, frolic, festive
    "s": 0.16,   # sport, song, sing, smile, sack
    "c": 0.14,   # cheer, carol, cup, caper, clown
    "g": 0.14,   # gay, glad, glee
    "p": 0.12,   # play, pleasant, pipe
    "r": 0.10,   # revel, revels
    "d": 0.10,   # dance, drink
    "w": 0.12,   # wit, wine, wedding
}

_THRESHOLD = 0.28


def mirth_start_bias(
    mirth_register: float,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if mirth_register < _THRESHOLD:
        return None

    raw = (mirth_register - _THRESHOLD) / (1.0 - _THRESHOLD)
    if raw > 1.0:
        raw = 1.0
    scale = raw

    vec = [0.0] * VOCAB_SIZE
    for ch, w in _MIRTH_STARTERS.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
    return vec


def mirth_sentence_start_bias(
    mirth_register: float,
    speaker_label_state: int,
) -> list[float] | None:
    """At sentence-start, lift capitalized comic openers: 'O', 'Ha',
    'What', 'Come'. Shakespeare's comic scenes love 'O merry', 'Ha,
    ha!', 'Come, a song!'."""
    if speaker_label_state != 0:
        return None
    if mirth_register < _THRESHOLD:
        return None
    raw = (mirth_register - _THRESHOLD) / (1.0 - _THRESHOLD)
    if raw > 1.0:
        raw = 1.0
    scale = raw

    vec = [0.0] * VOCAB_SIZE
    # "O merry morn!"
    if "O" in VOCAB_INDEX:
        vec[VOCAB_INDEX["O"]] += 0.16 * scale
    # "Ha!" "Ho!"
    if "H" in VOCAB_INDEX:
        vec[VOCAB_INDEX["H"]] += 0.10 * scale
    # "Come, a song!"
    if "C" in VOCAB_INDEX:
        vec[VOCAB_INDEX["C"]] += 0.08 * scale
    # "Why, ..."
    if "W" in VOCAB_INDEX:
        vec[VOCAB_INDEX["W"]] += 0.06 * scale
    # "Marry, ..."
    if "M" in VOCAB_INDEX:
        vec[VOCAB_INDEX["M"]] += 0.08 * scale
    return vec
