"""Tier-2 — contraction closure tracker.

Maintains `contraction_tail_ok` (bool). Starts True; flips to False
when we emit an apostrophe inside a non-empty letter buffer (mid-word
apostrophe); flips back to True once a valid contraction tail has been
emitted after that apostrophe. Resets to True at word-end (empty
buffer).

Valid tails (the short documented elision repertoire):

  position 1:  s, d, t, m            →  's / 'd / 't / 'm
  position 2:  ll, re, ve, er, en,
               em, st                →  'll / 're / 've / 'er / 'en / 'em / 'st
  position 3:  tw(a|e|i) / th(a|e|o) →  'twas-extension / 'thou-extension
                                        (embedded, e.g. 'where'er')

NOTE on timing: update_word_cap_apos runs FIRST and bumps lsa by one
with each token. The convention after word_cap_apos is:
  lsa=1 right after the apostrophe (0 letters of tail emitted yet),
  lsa=2 after ONE tail letter, lsa=3 after two, lsa=4 after three.

Word-initial apostrophes ('tis, 'twas, 'gainst, 'bout, 'pon, 'tween)
are left un-blocked — they have longer, more varied tails outside the
documented elision repertoire, and `apostrophe_elision_bias` already
handles their position-3+ shape.

Runs after `update_word_cap_apos`. No corpus statistics.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB
from .linguistic import LOWER_CONS, LOWER_VOWEL, UPPER


_POS1_CLOSERS: frozenset[str] = frozenset({"s", "d", "t", "m"})
_POS2_CLOSERS: frozenset[tuple[str, str]] = frozenset({
    ("l", "l"),  # 'll
    ("r", "e"),  # 're
    ("v", "e"),  # 've
    ("e", "r"),  # 'er
    ("e", "n"),  # 'en
    ("e", "m"),  # 'em
    ("s", "t"),  # 'st  (know'st, lov'st, thou'st-style)
})
_TW_POS3_CLOSERS: frozenset[str] = frozenset({"a", "e", "i"})
_TH_POS3_CLOSERS: frozenset[str] = frozenset({"o", "e", "a", "i"})


def update_contraction_tail(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]
    cls = state.last_char_class
    wb = state.word_buffer
    is_letter = cls in (UPPER, LOWER_VOWEL, LOWER_CONS)
    is_apos = ch == "'"

    cur = state.contraction_tail_ok

    # Word ended → reset to True.
    if not wb:
        if cur is not True:
            return state.model_copy(update={"contraction_tail_ok": True})
        return state

    # Apostrophe just emitted inside a word with a preceding letter.
    if is_apos:
        if len(wb) >= 2 and wb[-2].isalpha():
            if cur is True:
                return state.model_copy(update={"contraction_tail_ok": False})
            return state
        # Word-initial apostrophe — do not open a block.
        return state

    if is_letter:
        lsa = state.letters_since_apostrophe
        if lsa == 0 or not state.had_apostrophe_this_word:
            if cur is not True:
                return state.model_copy(update={"contraction_tail_ok": True})
            return state

        if cur is True:
            return state

        lower_last = wb[-1].lower()

        # One letter after apos (pos-1 single-letter closers).
        if lsa == 2:
            if lower_last in _POS1_CLOSERS:
                return state.model_copy(update={"contraction_tail_ok": True})
            return state

        # Two letters after apos (pos-2 pair closers).
        if lsa == 3 and len(wb) >= 2:
            lower_prev = wb[-2].lower()
            if (lower_prev, lower_last) in _POS2_CLOSERS:
                return state.model_copy(update={"contraction_tail_ok": True})
            return state

        # Three letters after apos (embedded 'tw- / 'th- extensions).
        if lsa == 4 and len(wb) >= 3:
            a = wb[-3].lower()
            b = wb[-2].lower()
            c = lower_last
            if a == "t" and b == "w" and c in _TW_POS3_CLOSERS:
                return state.model_copy(update={"contraction_tail_ok": True})
            if a == "t" and b == "h" and c in _TH_POS3_CLOSERS:
                return state.model_copy(update={"contraction_tail_ok": True})
            return state

        return state

    return state
