"""POS-based next-word first-letter bias layer.

Given the POS tag of the last completed word, bias the first letter of
the next word toward letters that commonly begin words that plausibly
follow words of that POS. A coarse word-level bigram prior, applied at
word starts (after a space or single newline).

This complements `next_word.py` (which has exact per-word bigrams for a
handful of extremely frequent words) by generalizing across whole POS
classes for the long tail of words.

All weights come from prior knowledge of English / Shakespearean text —
no corpus statistics.
"""

from __future__ import annotations

import math

from ..pipeline.pos import (
    N_POS,
    POS_ADJECTIVE,
    POS_ADVERB,
    POS_ARTICLE,
    POS_AUX_VERB,
    POS_CONJUNCTION,
    POS_INTERJECTION,
    POS_MODAL,
    POS_NEGATION,
    POS_NOUN,
    POS_NUMBER,
    POS_POSSESSIVE,
    POS_PREPOSITION,
    POS_PRONOUN,
    POS_UNKNOWN,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
    POS_WH,
)
from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# Per-POS: relative weight of each first-letter of the next word.
# Letters NOT in a dict get a small negative bump.
#
# Heuristic targets:
#  - after ARTICLE: nouns/adjectives (diverse consonant-initial)
#  - after POSSESSIVE: nouns (body/heart/eyes/hand/lord/love/good/sweet)
#  - after PREPOSITION: articles/possessives/nouns (t=the/this/that,
#    m=my, h=his/her, a=a/an)
#  - after PRONOUN: verbs/aux (a=am, d=do/did, h=have/hath, s=shall, w=will/would)
#  - after AUX_VERB: participles/adjectives/adverbs (broad, a bit of everything)
#  - after MODAL: bare verbs (b=be, n=not, h=have, g=go, s=see/say)
#  - after CONJUNCTION: sentence-initial variety (t=the/this/that,
#    i=I, w=we/when, h=he/his, s=so, y=you)
#  - after NEGATION: verbs/adjectives (b=be, t=the, m=more, s=so)
#  - after VERB_ING: prepositions/articles (t=the/to, o=of/on, w=with, i=in)
#  - after VERB_ED: same as VERB_ING (post-verb context)
#  - after NOUN: verbs/connectives (i=is, a=are/and, w=was/will/who, t=that/the)
#  - after ADJECTIVE: nouns (same profile as after ARTICLE)
#  - after ADVERB: verbs/connectives (varied)
#  - after WH: verbs/auxiliaries (i=is, a=art, d=do, h=hath, s=shall)
#  - after NUMBER: nouns (o=of, y=years, d=days, etc.)
#  - after INTERJECTION: pronouns/direct address (m=my, t=thou/thee, w=what)
#
# Values are small integer weights; they're log-normalized into a bias.
_POS_NEXT: dict[int, dict[str, int]] = {
    POS_ARTICLE: {
        "s": 5, "m": 5, "l": 4, "w": 4, "k": 4, "f": 4, "h": 4, "c": 4,
        "n": 4, "b": 4, "d": 4, "p": 4, "t": 4, "r": 3, "g": 3, "e": 3,
        "o": 3, "i": 2, "a": 2, "y": 2, "v": 2, "u": 1, "q": 1, "j": 1,
    },
    POS_POSSESSIVE: {
        "l": 5, "h": 5, "f": 4, "d": 4, "s": 4, "b": 4, "g": 3, "m": 3,
        "p": 3, "t": 3, "c": 3, "w": 3, "n": 3, "e": 3, "o": 3, "a": 3,
        "y": 2, "r": 3, "v": 2, "k": 2, "i": 2, "u": 2,
    },
    POS_PREPOSITION: {
        "t": 7, "m": 5, "h": 5, "a": 4, "y": 4, "o": 3, "s": 3, "w": 3,
        "b": 3, "e": 3, "i": 3, "l": 3, "n": 3, "d": 3, "f": 3, "g": 3,
        "c": 3, "p": 3, "r": 3, "u": 2, "v": 2, "k": 2, "j": 1, "q": 1,
    },
    POS_PRONOUN: {
        "a": 6, "w": 5, "h": 5, "s": 5, "d": 5, "c": 3, "m": 3, "l": 3,
        "t": 3, "k": 3, "b": 3, "g": 3, "n": 3, "p": 3, "f": 3, "r": 3,
        "e": 3, "o": 3, "y": 2, "i": 2, "u": 2, "v": 2,
    },
    POS_AUX_VERB: {
        "t": 5, "n": 5, "a": 4, "s": 4, "m": 4, "h": 4, "w": 4, "b": 3,
        "i": 3, "o": 3, "c": 3, "d": 3, "f": 3, "g": 3, "l": 3, "p": 3,
        "r": 3, "e": 3, "y": 3, "u": 2, "v": 2, "k": 2,
    },
    POS_MODAL: {
        "b": 7, "n": 5, "h": 4, "s": 4, "g": 3, "c": 3, "d": 3, "f": 3,
        "l": 3, "m": 3, "p": 3, "r": 3, "t": 3, "w": 3, "y": 2, "a": 2,
        "e": 2, "i": 2, "o": 2, "u": 2, "v": 2, "k": 2,
    },
    POS_CONJUNCTION: {
        "t": 5, "i": 5, "w": 4, "h": 4, "y": 4, "s": 4, "a": 3, "m": 3,
        "o": 3, "b": 3, "c": 3, "d": 3, "e": 3, "f": 3, "g": 3, "l": 3,
        "n": 3, "p": 3, "r": 3, "u": 2, "v": 2, "k": 2,
    },
    POS_NEGATION: {
        "t": 4, "a": 4, "b": 4, "m": 4, "s": 4, "h": 3, "w": 3, "i": 3,
        "o": 3, "y": 3, "f": 3, "g": 3, "l": 3, "n": 3, "c": 3, "r": 3,
        "p": 3, "d": 3, "e": 3, "u": 2, "v": 2, "k": 2,
    },
    POS_VERB_ING: {
        "t": 6, "o": 5, "w": 4, "i": 4, "u": 3, "a": 3, "b": 3, "f": 3,
        "m": 3, "h": 3, "s": 3, "y": 2, "d": 2, "g": 2, "l": 2, "p": 2,
        "r": 2, "n": 2, "c": 2, "e": 2,
    },
    POS_VERB_ED: {
        "t": 6, "o": 5, "w": 4, "i": 4, "b": 3, "a": 3, "m": 3, "f": 3,
        "h": 3, "s": 3, "u": 3, "y": 2, "d": 2, "g": 2, "l": 2, "p": 2,
        "r": 2, "n": 2, "c": 2, "e": 2,
    },
    POS_VERB: {
        "t": 5, "o": 4, "w": 4, "i": 4, "a": 3, "b": 3, "h": 3, "m": 3,
        "s": 3, "f": 3, "u": 2, "y": 2, "d": 2, "g": 2, "l": 2, "p": 2,
        "r": 2, "n": 2, "c": 2, "e": 2,
    },
    POS_NOUN: {
        "i": 5, "a": 5, "w": 5, "t": 4, "h": 4, "o": 4, "s": 4, "m": 3,
        "b": 3, "c": 3, "d": 3, "f": 3, "g": 3, "l": 3, "n": 3, "p": 3,
        "r": 3, "y": 3, "e": 3, "u": 2, "v": 2, "k": 2,
    },
    POS_ADJECTIVE: {
        "s": 4, "m": 4, "l": 4, "w": 3, "k": 3, "f": 3, "h": 3, "c": 3,
        "n": 3, "b": 3, "d": 3, "p": 3, "t": 3, "r": 3, "g": 3, "e": 3,
        "o": 3, "i": 2, "a": 2, "y": 2, "v": 2, "u": 1, "q": 1,
    },
    POS_ADVERB: {
        "t": 5, "w": 4, "i": 4, "a": 4, "h": 4, "s": 4, "m": 3, "b": 3,
        "c": 3, "d": 3, "f": 3, "g": 3, "l": 3, "n": 3, "p": 3, "r": 3,
        "o": 3, "e": 3, "y": 3, "u": 2, "v": 2, "k": 2,
    },
    POS_WH: {
        "i": 5, "a": 4, "d": 4, "h": 4, "s": 4, "w": 4, "c": 3, "m": 3,
        "t": 3, "b": 3, "f": 3, "l": 3, "n": 3, "p": 3, "r": 3, "o": 3,
        "e": 2, "y": 2, "g": 2, "u": 2,
    },
    POS_NUMBER: {
        "o": 5, "y": 4, "d": 4, "h": 4, "t": 3, "m": 3, "s": 3, "w": 3,
        "a": 3, "i": 3, "e": 3, "b": 2, "c": 2, "f": 2, "g": 2, "l": 2,
        "n": 2, "p": 2, "r": 2, "u": 2,
    },
    POS_INTERJECTION: {
        "m": 5, "t": 5, "w": 4, "h": 4, "g": 3, "s": 3, "y": 3, "i": 3,
        "a": 3, "o": 3, "e": 3, "b": 2, "c": 2, "d": 2, "f": 2, "l": 2,
        "n": 2, "p": 2, "r": 2, "u": 2, "v": 2, "k": 2,
    },
}


_GLOBAL_SCALE = 0.1


def _build_vectors() -> list[list[float] | None]:
    out: list[list[float] | None] = [None] * N_POS
    for tag, nexts in _POS_NEXT.items():
        vec = [0.0] * VOCAB_SIZE
        total = sum(nexts.values())
        # Baseline penalty on letters not listed (letters only).
        listed = set(nexts.keys())
        for ch in "abcdefghijklmnopqrstuvwxyz":
            if ch not in listed and ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] = -0.3 * _GLOBAL_SCALE
        for ch, w in nexts.items():
            if ch not in VOCAB_INDEX:
                continue
            frac = w / total
            bias = _GLOBAL_SCALE * math.log((frac + 0.02) / 0.05)
            vec[VOCAB_INDEX[ch]] = bias
            up = ch.upper()
            if up in VOCAB_INDEX:
                vec[VOCAB_INDEX[up]] = bias * 0.5
        out[tag] = vec
    return out


_POS_NEXT_BIAS: list[list[float] | None] = _build_vectors()


def pos_next_bias(last_word_pos: int) -> list[float] | None:
    """Return a bias vector over first-letter of next word given POS of
    the previous word, or None if no table entry exists (UNKNOWN)."""
    if 0 <= last_word_pos < len(_POS_NEXT_BIAS):
        return _POS_NEXT_BIAS[last_word_pos]
    return None
