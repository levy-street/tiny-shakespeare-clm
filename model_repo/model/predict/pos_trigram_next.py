"""POS-trigram next-word first-letter bias layer.

Structural step up from pos_bigram_next. Uses the triple
`(prev_prev_word_pos, prev_word_pos, last_word_pos)` — a 3-back
positional POS window — to pick out patterns where the 3rd-back
context meaningfully narrows the next-word class.

Shakespeare's syntax has many skeletons that a bigram can't distinguish
but a trigram can:

  * (DET, ADJ, NOUN): full subject NP complete — "the fair maid",
    "a noble lord". Expect main VERB / AUX next. Much sharper than
    (ADJ, NOUN) alone, which could also be the tail of a bare modifier.
  * (POSS, ADJ, NOUN): "my dear friend", "his good queen" — same.
  * (PRON, AUX, VERB_ED): "I am slain", "thou art dead" — passive/
    perfect complete; expect sentence-end or "by" / prep.
  * (PRON, AUX, ADJECTIVE): "I am mad", "thou art fair" — predicate
    adjective; expect conjunction / sentence-end.
  * (PRON, MODAL, VERB): "I shall go", "thou wilt speak" — bare verb
    complete; expect object NP opener (det/poss) or prep.
  * (NOUN, PREP, NOUN): "son of man", "lord of hosts" — PP done,
    larger NP complete, expect VERB or punct.
  * (PREP, DET, NOUN): "in the field", "on the sea" — PP done; expect
    VERB / CONJ / punct.
  * (VERB, DET, NOUN): "see the king", "kill my liege" — post-verb
    object NP complete; expect CONJ / punct / prep.
  * (VERB, POSS, NOUN): "seek thy peace", "know my mind" — same.
  * (CONJ, PRON, VERB): "and I love", "but thou hast" — new clause
    with verb filled; expect object NP opener.
  * (PRON, VERB, PRON): "I love thee", "he knows me" — clause done;
    expect CONJ / punct.
  * (NOUN, AUX, ADJECTIVE): "king is fair" — pred ADJ; expect CONJ
    / punct.
  * (NOUN, AUX, VERB_ED): "king is slain" — passive pred; expect
    "by" / prep / punct.
  * (PRON, VERB, PREP): "I come to", "thou goest for" — expect bare
    VERB or NOUN starters.

Weights and first-letter priors come from English / Shakespeare prior
knowledge. No corpus statistics.

This layer is ADDITIVE with pos_next (1-back) and pos_bigram_next
(2-back). The three form a graded backoff, each firing when its
context window has an encoded pattern.
"""

from __future__ import annotations

import math

from ..pipeline.pos import (
    POS_ADJECTIVE,
    POS_ARTICLE,
    POS_AUX_VERB,
    POS_CONJUNCTION,
    POS_MODAL,
    POS_NEGATION,
    POS_NOUN,
    POS_POSSESSIVE,
    POS_PREPOSITION,
    POS_PRONOUN,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
)
from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# (prev_prev, prev, last) → {first_letter: integer weight}.
# Letters NOT listed get a small negative bump.
_TRIGRAM_NEXT: dict[tuple[int, int, int], dict[str, int]] = {
    # ── Full subject NP (DET/POSS + ADJ + NOUN) → VERB/AUX next ──
    # Boost verb/aux/modal starters: i=is, a=are/art, w=was/were/will/
    # would, h=has/hath/have/had, s=shall/should, d=do/did/doth, b=be,
    # m=may/must, c=can/could. Light conj/adv: t=that/then, o=or, y=yet.
    (POS_ARTICLE, POS_ADJECTIVE, POS_NOUN): {
        "i": 6, "a": 6, "w": 5, "h": 6, "s": 5, "d": 5, "b": 3,
        "m": 4, "c": 3, "t": 3, "o": 2, "y": 2, "n": 2, "l": 2,
    },
    (POS_POSSESSIVE, POS_ADJECTIVE, POS_NOUN): {
        "i": 6, "a": 6, "w": 5, "h": 6, "s": 5, "d": 5, "b": 3,
        "m": 4, "c": 3, "t": 3, "o": 2, "y": 2, "n": 2, "l": 2,
    },
    # ── Pronoun + aux + past-part → sentence close or "by"/prep ──
    # "I am slain.", "thou art dead." — boost punct (via space then
    # period handled by newline-layer; here first-letter of NEXT word
    # in case more content continues) "b"=by, "f"=for, "w"=with/
    # when, "a"=and, "o"=of.
    (POS_PRONOUN, POS_AUX_VERB, POS_VERB_ED): {
        "b": 6, "f": 5, "w": 5, "a": 5, "o": 4, "i": 4, "t": 4,
        "u": 3, "n": 2, "y": 2, "s": 2,
    },
    # ── Pronoun + aux + adjective → conj/punct next.  "I am mad,
    # thou art fair" — a=and, b=but, y=yet, t=then, o=or, n=nor.
    (POS_PRONOUN, POS_AUX_VERB, POS_ADJECTIVE): {
        "a": 6, "b": 5, "y": 5, "t": 4, "o": 4, "n": 3,
        "i": 2, "w": 3, "f": 2, "s": 2, "h": 2,
    },
    # ── Pronoun + modal + verb → object NP opener.  "I shall go
    # home": t=the/to/this, a=a/an/all, h=him/her/his, m=my/me,
    # y=your, o=our, s=some/such, n=no.
    (POS_PRONOUN, POS_MODAL, POS_VERB): {
        "t": 6, "a": 5, "h": 5, "m": 5, "y": 4, "o": 4, "s": 4,
        "n": 3, "f": 3, "w": 3, "i": 3, "b": 2, "d": 2, "u": 2,
    },
    # ── Noun + prep + noun → PP done → VERB/CONJ.  "son of man
    # [verb]", "lord of hosts [verb]".
    (POS_NOUN, POS_PREPOSITION, POS_NOUN): {
        "i": 5, "a": 5, "w": 5, "h": 5, "s": 4, "d": 4, "b": 3,
        "m": 3, "c": 3, "t": 3, "o": 2, "y": 2, "n": 2,
    },
    # ── Prep + det + noun → PP done → VERB/CONJ.  "in the field"
    (POS_PREPOSITION, POS_ARTICLE, POS_NOUN): {
        "i": 5, "a": 5, "w": 5, "h": 5, "s": 4, "d": 4, "b": 3,
        "m": 3, "c": 3, "t": 3, "o": 2, "y": 2, "n": 2,
    },
    (POS_PREPOSITION, POS_POSSESSIVE, POS_NOUN): {
        "i": 5, "a": 5, "w": 5, "h": 5, "s": 4, "d": 4, "b": 3,
        "m": 3, "c": 3, "t": 3, "o": 2, "y": 2, "n": 2,
    },
    # ── Verb + det + noun → object NP complete → CONJ/PREP/punct.
    # "see the king" → a=and/at, t=to/that/then, o=of/or/on, w=with/
    # when, f=for/from, b=but/by, i=in/is, u=upon, y=yet.
    (POS_VERB, POS_ARTICLE, POS_NOUN): {
        "a": 6, "t": 5, "o": 5, "w": 5, "f": 4, "b": 4, "i": 3,
        "u": 3, "y": 3, "n": 3, "s": 2, "h": 2, "m": 2,
    },
    (POS_VERB, POS_POSSESSIVE, POS_NOUN): {
        "a": 6, "t": 5, "o": 5, "w": 5, "f": 4, "b": 4, "i": 3,
        "u": 3, "y": 3, "n": 3, "s": 2, "h": 2, "m": 2,
    },
    # ── Conj + pron + verb → object NP opener.  "and I see [the]"
    (POS_CONJUNCTION, POS_PRONOUN, POS_VERB): {
        "t": 6, "a": 5, "h": 5, "m": 4, "y": 4, "o": 4, "s": 4,
        "n": 3, "e": 3, "w": 3, "i": 3, "b": 2, "f": 2,
    },
    # ── Pron + verb + pron → clause done → CONJ/punct.  "I love
    # thee" → a=and, b=but, y=yet, t=then/that, o=or, n=nor, f=for.
    (POS_PRONOUN, POS_VERB, POS_PRONOUN): {
        "a": 6, "b": 5, "y": 5, "t": 4, "o": 4, "n": 3, "f": 3,
        "w": 3, "i": 2, "s": 2, "h": 2,
    },
    # ── Noun + aux + adj → pred ADJ complete → CONJ/punct.
    (POS_NOUN, POS_AUX_VERB, POS_ADJECTIVE): {
        "a": 6, "b": 5, "y": 5, "t": 4, "o": 4, "n": 3, "f": 3,
        "w": 3, "i": 2, "s": 2, "h": 2,
    },
    # ── Noun + aux + past-part → passive complete → "by"/prep/punct
    (POS_NOUN, POS_AUX_VERB, POS_VERB_ED): {
        "b": 6, "f": 5, "w": 5, "a": 4, "o": 4, "i": 3, "t": 4,
        "u": 3, "n": 2, "y": 2,
    },
    # ── Pron + verb + prep → expect bare VERB / NOUN starter.
    # "I come to [speak]", "thou goest for [the]".  t=the (NOUN),
    # s=speak/see, k=know, l=live/love, m=make/meet, d=die/do,
    # b=be/be/bear, c=come/call, h=hold/hear, a=a/an (NOUN).
    (POS_PRONOUN, POS_VERB, POS_PREPOSITION): {
        "t": 5, "s": 4, "k": 4, "l": 4, "m": 4, "d": 4, "b": 4,
        "c": 4, "h": 4, "a": 4, "g": 3, "p": 3, "r": 3, "f": 3,
        "w": 3, "n": 2, "e": 2, "i": 2, "o": 2, "u": 2, "y": 2,
    },
    # ── Noun + verb + prep → same as above for 3rd-person subject.
    (POS_NOUN, POS_VERB, POS_PREPOSITION): {
        "t": 5, "s": 4, "k": 4, "l": 4, "m": 4, "d": 4, "b": 4,
        "c": 4, "h": 4, "a": 4, "g": 3, "p": 3, "r": 3, "f": 3,
        "w": 3, "n": 2, "e": 2, "i": 2, "o": 2, "u": 2, "y": 2,
    },
    # ── Pron + neg + verb → "I not know" / obj NP opener next.
    (POS_PRONOUN, POS_NEGATION, POS_VERB): {
        "t": 6, "a": 5, "h": 5, "m": 4, "y": 4, "o": 4, "s": 4,
        "n": 3, "e": 3, "w": 3, "i": 3, "b": 2, "f": 2,
    },
    # ── Modal + verb + det → "shall see the" — NP head expected.
    # Boost NOUN / ADJ first letters: k=king/knight, l=lord/love,
    # w=world/word, m=man/maid/morn, d=day/death, n=night/name,
    # s=soul/sun, h=heart/head, e=eye, f=father/fair, g=god/good,
    # p=power, q=queen, r=rose, t=tongue.
    (POS_MODAL, POS_VERB, POS_ARTICLE): {
        "k": 5, "l": 5, "w": 5, "m": 5, "d": 4, "n": 4, "s": 4,
        "h": 4, "e": 4, "f": 4, "g": 4, "p": 3, "q": 2, "r": 3,
        "t": 3, "b": 3, "c": 3, "a": 2, "i": 2, "o": 2, "u": 2,
        "y": 2,
    },
    # ── Aux + verb + det → "hath seen the" — NP head.
    (POS_AUX_VERB, POS_VERB, POS_ARTICLE): {
        "k": 5, "l": 5, "w": 5, "m": 5, "d": 4, "n": 4, "s": 4,
        "h": 4, "e": 4, "f": 4, "g": 4, "p": 3, "q": 2, "r": 3,
        "t": 3, "b": 3, "c": 3, "a": 2, "i": 2, "o": 2, "u": 2,
        "y": 2,
    },
    # ── Aux + verb_ing + prep → "am going to" — bare VERB / NOUN.
    (POS_AUX_VERB, POS_VERB_ING, POS_PREPOSITION): {
        "t": 5, "s": 4, "k": 4, "l": 4, "m": 4, "d": 4, "b": 4,
        "c": 4, "h": 4, "a": 4, "g": 3, "p": 3, "r": 3, "f": 3,
        "w": 3, "n": 2, "e": 2, "i": 2, "o": 2, "u": 2, "y": 2,
    },
    # ── Prep + adj + noun → PP with modifier done → VERB/CONJ.
    (POS_PREPOSITION, POS_ADJECTIVE, POS_NOUN): {
        "i": 5, "a": 5, "w": 5, "h": 5, "s": 4, "d": 4, "b": 3,
        "m": 3, "c": 3, "t": 3, "o": 2, "y": 2, "n": 2,
    },
}


_GLOBAL_SCALE = 0.08


def _build_vectors() -> dict[tuple[int, int, int], list[float]]:
    out: dict[tuple[int, int, int], list[float]] = {}
    for key, weights in _TRIGRAM_NEXT.items():
        vec = [0.0] * VOCAB_SIZE
        total = sum(weights.values())
        listed = set(weights.keys())
        for ch in "abcdefghijklmnopqrstuvwxyz":
            if ch not in listed and ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] = -0.20 * _GLOBAL_SCALE
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


_POS_TRIGRAM_NEXT_BIAS: dict[tuple[int, int, int], list[float]] = _build_vectors()


def pos_trigram_next_bias(
    prev_prev_word_pos: int,
    prev_word_pos: int,
    last_word_pos: int,
) -> list[float] | None:
    """Return a first-letter bias for the 3-back POS trigram, or None
    if the specific trigram has no encoded entry."""
    return _POS_TRIGRAM_NEXT_BIAS.get(
        (prev_prev_word_pos, prev_word_pos, last_word_pos)
    )
