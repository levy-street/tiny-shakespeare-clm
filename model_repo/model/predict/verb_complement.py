"""Predict layer — verb-complement class start bias.

Reads `state.verb_complement_class` (set by pipeline/verb_complement.py).
At word-start (letter_run_len == 0), bias the first letter toward the
expected complement opener:

  VCC_THAT (mental/speech verbs): expect "that/if/whether/how/why/
       what/when/where/who/which" OR an NP-opener (the/a/my/thy/his/
       her) OR a pronoun (I/thou/he/him/me/us).
       Top letters: t (that, the, thou, thy), i (if, I, it), w
       (what, when, where, who, whether), h (how, he, him, his, her),
       m (me, my), a (a, an, as).
  VCC_PP (motion verbs): expect preposition first letters.
       Top letters: t (to, toward, through), f (from, for), i (in,
       into), u (upon, up), w (with), b (by, before, beyond),
       a (at, after, above, around), o (on, of, over), n (near).
  VCC_PPART (have/hath/had): expect past-participle verb.
       Top letters: s (seen, spoken, sworn, said, sent, slain,
       struck), d (done, done), g (gone, given), t (taken, thought,
       told, torn), b (been, broken, borne), c (come, caught,
       chosen), f (fallen, fought, felt, forgotten), l (lost,
       laid, left), m (made, met), h (heard, held, hidden).
  VCC_INF (shall/will/would/must/etc.): expect bare infinitive verb.
       Top letters: b (be, bear, bring, believe), h (have, hear,
       help, hold), s (see, say, speak, stand, seek, swear),
       g (go, give, get, grant), m (make, meet), l (love, look,
       live, lie), t (take, tell), c (come, call, change), d (do,
       die, draw), f (find, feel, follow, fall), k (know, keep),
       p (pray, put, prove).
  VCC_PRED (is/are/was/be/etc.): expect predicate - ADJ or VERB_ING
       or NP.
       Top letters: n (not, never), a (a, all, an, alone, as), t
       (the, that, thou, too), s (so, still), g (good, great,
       gone), h (he, her, his, high), m (my, mine, more, made),
       f (for, fair, full, free), d (dead, dear, done), w (with,
       well, worthy, willing).

Scale fades with vcc_wait_words — the further we are from the verb,
the weaker the expectation.

No corpus statistics — all weights from English / Shakespeare prior
knowledge.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

VCC_NONE = 0
VCC_THAT = 1
VCC_PP = 2
VCC_PPART = 3
VCC_INF = 4
VCC_PRED = 5


# Per-class first-letter weight maps. Weights are RELATIVE rank
# (compressed to 0-1) that get log-multiplied into bias.
_WEIGHTS: dict[int, dict[str, float]] = {
    VCC_THAT: {
        # THAT / NP-opener / pronoun
        "t": 0.50,   # that, the, thou, thy, to
        "i": 0.40,   # if, I, it, in
        "w": 0.35,   # what, when, where, who, whether, which
        "h": 0.35,   # how, he, him, his, her
        "m": 0.25,   # me, my, mine
        "a": 0.22,   # a, an, as, all
        "n": 0.15,   # no, none, not (negated clause)
        "s": 0.15,   # she, so
        "b": 0.10,   # because, but
        "o": 0.10,   # our
    },
    VCC_PP: {
        "t": 0.55,   # to, toward, through
        "f": 0.45,   # from, for
        "i": 0.40,   # in, into
        "u": 0.35,   # upon, under, unto, up
        "w": 0.30,   # with, within, without
        "b": 0.30,   # by, before, beyond, beside, beneath
        "a": 0.30,   # at, after, above, around, against, among
        "o": 0.25,   # on, of, out, off, over
        "n": 0.15,   # near
        "s": 0.10,   # since (temporal)
    },
    VCC_PPART: {
        "s": 0.45,   # seen, spoken, sworn, said, sent, slain, struck, shown, slept, sought
        "d": 0.35,   # done
        "g": 0.30,   # gone, given, grown
        "t": 0.35,   # taken, thought, told, torn, taught
        "b": 0.35,   # been, broken, borne, brought, bought
        "c": 0.30,   # come, caught, chosen, crept
        "f": 0.30,   # fallen, fought, felt, forgotten, found, flown
        "l": 0.25,   # lost, laid, left, led
        "m": 0.25,   # made, met
        "h": 0.25,   # heard, held, hidden, hung
        "k": 0.18,   # kept, known
        "w": 0.22,   # written, worn, wept, wrung, wrought
        "r": 0.22,   # risen, ridden, read, rung
        "e": 0.18,   # eaten
        "p": 0.15,   # paid
    },
    VCC_INF: {
        "b": 0.40,   # be, bear, bring, believe, break, begin
        "h": 0.35,   # have, hear, help, hold, have, hurt
        "s": 0.40,   # see, say, speak, stand, seek, swear, stay, stop
        "g": 0.30,   # go, give, get, grant, grow
        "m": 0.28,   # make, meet, move
        "l": 0.28,   # love, look, live, lie, learn, let
        "t": 0.30,   # take, tell, teach, try, turn, think
        "c": 0.30,   # come, call, change, carry, close, command
        "d": 0.30,   # do, die, draw, dwell, deliver
        "f": 0.28,   # find, feel, follow, fall, fight, fail, free
        "k": 0.22,   # know, keep, kill
        "p": 0.22,   # pray, put, prove, pass, please
        "r": 0.22,   # rise, run, remain, return, receive, reign, rest
        "w": 0.22,   # wait, weep, win, work, wish, walk, wear, wish
        "a": 0.18,   # assay, appear, abide, ask
    },
    VCC_PRED: {
        "n": 0.35,   # not, never, no (negation)
        "a": 0.30,   # a, an, all, alone, as
        "t": 0.30,   # the, that, thou, too
        "s": 0.30,   # so, still, such, sweet, sick, sure
        "g": 0.25,   # good, great, gone, gentle, glad
        "h": 0.25,   # he, her, his, high, half, happy
        "m": 0.25,   # my, mine, more, made, much, merry
        "f": 0.22,   # for, fair, full, free, false, foolish
        "d": 0.22,   # dead, dear, done, done, done
        "w": 0.22,   # with, well, worthy, willing, wild, wise, weary
        "i": 0.20,   # in, ill
        "b": 0.20,   # be, but, bold, brave
        "l": 0.18,   # like, living, lost
        "p": 0.15,   # poor, proud, pale
        "c": 0.15,   # content, certain, clear, cold
    },
}


# Waiting-word decay: scale drops fast after the verb.
def _wait_scale(wait: int) -> float:
    if wait == 0:
        return 1.0
    if wait == 1:
        return 0.65
    if wait == 2:
        return 0.35
    if wait == 3:
        return 0.15
    return 0.0


_BASE_SCALE = 0.30


def verb_complement_start_bias(
    verb_complement_class: int,
    vcc_wait_words: int,
    speaker_label_state: int,
    letter_run_len: int,
    word_buffer: str,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if verb_complement_class == VCC_NONE:
        return None
    if letter_run_len != 0:
        return None
    if word_buffer:
        return None

    ws = _wait_scale(vcc_wait_words)
    if ws <= 0.0:
        return None

    entries = _WEIGHTS.get(verb_complement_class)
    if entries is None:
        return None

    vec = [0.0] * VOCAB_SIZE
    scale = _BASE_SCALE * ws
    for ch, w in entries.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
        # Capital mirror for sentence-starts (modest).
        up = ch.upper()
        if up != ch:
            uidx = VOCAB_INDEX.get(up)
            if uidx is not None:
                vec[uidx] += w * scale * 0.30

    return vec
