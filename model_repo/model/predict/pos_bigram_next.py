"""POS-bigram next-word first-letter bias layer.

Existing `pos_next.py` uses only `last_word_pos` (1-back). This layer
uses `(prev_word_pos, last_word_pos)` — a 2-back content-level bigram
— to pick up patterns that unigram POS context cannot see:

  * (DET, NOUN): subject NP just completed → main VERB expected.
    The NOUN alone doesn't tell us whether it's subject-pos or
    object-pos; the preceding DET does.
  * (POSS, NOUN): same as DET,NOUN — subject-flavored NP done.
  * (ADJ, NOUN): NP done → VERB expected.
  * (PREP, NOUN): prep-phrase done → VERB or CONJ next.
  * (PRONOUN, AUX): "I am", "thou art", "he is" → predicate expected.
  * (PRONOUN, MODAL): "I shall", "thou wilt" → bare verb next.
  * (AUX, VERB_ING): progressive → prep/object next.
  * (AUX, VERB_ED): passive/perfect → "by" or prep next.
  * (MODAL, VERB): modal+bare-verb → object/prep next.
  * (NOUN, VERB): post-subject verb → object NP opener (DET/POSS) next.
  * (NOUN, AUX): "king is", "lord hath" → predicate ADJ / VERB_ED next.
  * (VERB, NOUN): V+O → prep or conj next.

Only a bigram table with these 12 patterns is encoded; other bigram
combinations fall through (this layer returns None) and `pos_next`
handles the 1-back signal. Weights are deliberately modest so the
composite with `pos_next` remains smooth.

All weights from English / Shakespeare prior knowledge — no corpus
statistics.
"""

from __future__ import annotations

import math

from ..pipeline.pos import (
    POS_ADJECTIVE,
    POS_ADVERB,
    POS_ARTICLE,
    POS_AUX_VERB,
    POS_CONJUNCTION,
    POS_MODAL,
    POS_NEGATION,
    POS_NOUN,
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

# Encode (prev, last) → {first_letter: weight}. Weights are small
# integers interpreted as relative preference; they're log-normalized
# into a bias with a modest global scale.
#
# Letters NOT listed in a bigram's dict get a small negative bump
# (they're not excluded, just slightly dispreferred).
_BIGRAM_NEXT: dict[tuple[int, int], dict[str, int]] = {
    # ── Subject NP completed → VERB expected ──────────────────
    # boost aux/modal/verb starters: i=is, a=are/art, w=was/were/will/
    # would, h=has/hath/have/had/had, s=shall/should, d=do/did, b=be/
    # been, m=may/must, c=can/could. Plus conjunctions t=that, o=or.
    (POS_ARTICLE, POS_NOUN): {
        "i": 5, "a": 5, "w": 5, "h": 5, "s": 4, "d": 4, "b": 3,
        "m": 3, "c": 3, "t": 3, "o": 2, "y": 2, "n": 2, "l": 2,
    },
    (POS_POSSESSIVE, POS_NOUN): {
        "i": 5, "a": 5, "w": 5, "h": 5, "s": 4, "d": 4, "b": 3,
        "m": 3, "c": 3, "t": 3, "o": 2, "y": 2, "n": 2, "l": 2,
    },
    (POS_ADJECTIVE, POS_NOUN): {
        "i": 5, "a": 5, "w": 5, "h": 5, "s": 4, "d": 4, "b": 3,
        "m": 3, "c": 3, "t": 3, "o": 2, "y": 2, "n": 2, "l": 2,
    },
    # ── PP completed → VERB or CONJ expected ─────────────────
    (POS_PREPOSITION, POS_NOUN): {
        "i": 4, "a": 5, "w": 5, "h": 4, "s": 4, "d": 3, "b": 3,
        "m": 3, "c": 3, "t": 3, "o": 2, "n": 2, "y": 2, "f": 2,
    },
    # ── Pronoun + aux → predicate next (ADJ, VERB_ED, NOUN) ─
    # Common predicates: good/great/gone/glad/gentle/given/grown,
    # dead/done/dear, mad/made/mortal, noble/nothing, sad/safe/set,
    # born/bound/blessed, fair/full/fled/false, well/wise/weary/wed.
    (POS_PRONOUN, POS_AUX_VERB): {
        "g": 4, "d": 4, "m": 4, "s": 4, "b": 4, "f": 4, "w": 4,
        "n": 3, "c": 3, "l": 3, "p": 3, "r": 3, "h": 3, "t": 3,
        "a": 3, "i": 2, "o": 2, "e": 2, "y": 2, "u": 2,
    },
    # ── Pronoun + modal → bare VERB next (b=be, g=go/get, c=come/
    #    call, d=die/do, s=speak/see/say/stay, k=know/keep, l=love/
    #    live/lie, m=make/meet, t=take/tell, h=hold/hear/help) ──
    (POS_PRONOUN, POS_MODAL): {
        "b": 6, "n": 5, "g": 4, "c": 4, "d": 4, "s": 4, "k": 4,
        "l": 4, "m": 4, "t": 4, "h": 4, "p": 3, "r": 3, "f": 3,
        "w": 3, "a": 2, "e": 2, "i": 2, "o": 2, "u": 2, "y": 2,
    },
    # ── AUX + -ING → obj/prep/adv next (t=to/the, o=of/on, i=in/it,
    #    w=with/when, a=at/and, f=for/from, u=upon) ──────────
    (POS_AUX_VERB, POS_VERB_ING): {
        "t": 5, "o": 4, "i": 4, "w": 4, "a": 4, "f": 3, "u": 3,
        "b": 3, "h": 3, "m": 3, "s": 2, "y": 2, "d": 2, "g": 2,
        "l": 2, "p": 2, "r": 2, "n": 2, "c": 2, "e": 2,
    },
    # ── AUX + -ED → passive/perfect → prep, often "by" ──────
    (POS_AUX_VERB, POS_VERB_ED): {
        "b": 6, "t": 4, "o": 4, "i": 4, "w": 4, "a": 3, "f": 3,
        "u": 3, "h": 3, "m": 3, "s": 2, "y": 2, "d": 2, "g": 2,
        "l": 2, "p": 2, "r": 2, "n": 2, "c": 2, "e": 2,
    },
    # ── MODAL + VERB → obj/prep next ────────────────────────
    (POS_MODAL, POS_VERB): {
        "t": 5, "o": 4, "i": 4, "w": 4, "a": 4, "f": 3, "u": 3,
        "b": 3, "h": 3, "m": 3, "s": 2, "y": 2, "d": 2, "g": 2,
        "l": 2, "p": 2, "r": 2, "n": 2, "c": 2, "e": 2,
    },
    # ── NOUN + VERB → object NP opener: determiner/possessive/
    #    pronoun starters (t=the/this/that, a=a/an/all, h=his/her,
    #    m=my/me, y=your, o=our/one, s=such/some, n=no, e=every) ──
    (POS_NOUN, POS_VERB): {
        "t": 6, "a": 5, "h": 5, "m": 4, "y": 4, "o": 4, "s": 4,
        "n": 3, "e": 3, "w": 3, "i": 3, "b": 2, "c": 2, "d": 2,
        "f": 2, "g": 2, "l": 2, "p": 2, "r": 2, "u": 2,
    },
    # ── PRONOUN + VERB → same as NOUN+VERB ─────────────────
    (POS_PRONOUN, POS_VERB): {
        "t": 6, "a": 5, "h": 5, "m": 4, "y": 4, "o": 4, "s": 4,
        "n": 3, "e": 3, "w": 3, "i": 3, "b": 2, "c": 2, "d": 2,
        "f": 2, "g": 2, "l": 2, "p": 2, "r": 2, "u": 2,
    },
    # ── NOUN + AUX → predicate (ADJ / VERB_ED / NOUN) ──────
    (POS_NOUN, POS_AUX_VERB): {
        "g": 4, "d": 4, "m": 4, "s": 4, "b": 4, "f": 4, "w": 4,
        "n": 3, "c": 3, "l": 3, "p": 3, "r": 3, "h": 3, "t": 3,
        "a": 3, "i": 2, "o": 2, "e": 2, "y": 2, "u": 2,
    },
    # ── VERB + NOUN → post-V object done → prep/conj ────────
    (POS_VERB, POS_NOUN): {
        "a": 5, "w": 5, "t": 4, "o": 4, "i": 3, "b": 3, "f": 3,
        "u": 3, "h": 3, "s": 3, "m": 3, "n": 2, "d": 2, "c": 2,
        "l": 2, "p": 2, "r": 2, "y": 2, "e": 2, "g": 2,
    },
    # ── NOUN + NOUN → apposition or NP complete → VERB/PREP
    (POS_NOUN, POS_NOUN): {
        "i": 5, "a": 5, "w": 4, "h": 4, "s": 4, "d": 3, "b": 3,
        "o": 3, "m": 3, "c": 3, "t": 3, "f": 2, "n": 2, "l": 2,
        "y": 2, "p": 2, "r": 2, "g": 2, "e": 2, "u": 2,
    },
}


_GLOBAL_SCALE = 0.08


def _build_vectors() -> dict[tuple[int, int], list[float]]:
    out: dict[tuple[int, int], list[float]] = {}
    for key, weights in _BIGRAM_NEXT.items():
        vec = [0.0] * VOCAB_SIZE
        total = sum(weights.values())
        listed = set(weights.keys())
        for ch in "abcdefghijklmnopqrstuvwxyz":
            if ch not in listed and ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] = -0.25 * _GLOBAL_SCALE
        for ch, w in weights.items():
            if ch not in VOCAB_INDEX:
                continue
            frac = w / total
            bias = _GLOBAL_SCALE * math.log((frac + 0.02) / 0.05)
            vec[VOCAB_INDEX[ch]] = bias
            up = ch.upper()
            if up in VOCAB_INDEX:
                vec[VOCAB_INDEX[up]] = bias * 0.5
        out[key] = vec
    return out


_POS_BIGRAM_NEXT_BIAS: dict[tuple[int, int], list[float]] = _build_vectors()


def pos_bigram_next_bias(
    prev_word_pos: int,
    last_word_pos: int,
) -> list[float] | None:
    """Return a first-letter bias given the 2-back POS bigram, or None
    if the bigram has no encoded entry."""
    return _POS_BIGRAM_NEXT_BIAS.get((prev_word_pos, last_word_pos))
