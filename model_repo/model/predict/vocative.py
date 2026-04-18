"""Vocative-noun first-letter bias.

Consumed when `state.vocative_expectation` is True at word-start. The
expected next word is a vocative noun — a person-denoting noun used as
address. These cluster on a small set of first letters in Shakespeare:

  l — lord, lady, liege, love, lieutenant
  s — sir, son, sister, signior, sovereign
  m — madam, master, mistress, mother, maid
  f — friend, father, fellow, fool
  p — prince, princess
  b — brother, boy, baron
  c — cousin, captain, child, count
  k — king, knight, kinsman
  q — queen
  d — daughter, duke, dame
  h — husband

Values are modest; this layer sits on top of startbigram/next_word/
word_trie and is only active when the narrow LEAD+ADJ pattern fires.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

_VOCATIVE_LEAN: dict[str, float] = {
    "l": 1.8,
    "s": 1.3,
    "m": 1.2,
    "f": 1.0,
    "p": 0.75,
    "b": 0.75,
    "c": 0.6,
    "k": 0.55,
    "q": 0.5,
    "d": 0.5,
    "h": 0.45,
}


def _build() -> list[float]:
    v = [0.0] * VOCAB_SIZE
    for ch, lean in _VOCATIVE_LEAN.items():
        if ch in VOCAB_INDEX:
            v[VOCAB_INDEX[ch]] = lean
        up = ch.upper()
        if up in VOCAB_INDEX:
            # Capital version — slightly less (most vocatives mid-sentence are lowercase).
            v[VOCAB_INDEX[up]] = lean * 0.5
    return v


VOCATIVE_START_BIAS: list[float] = _build()
