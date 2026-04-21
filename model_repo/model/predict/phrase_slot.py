"""Predict layer — bias word-start first letter by phrase-slot FSM.

Reads `state.phrase_slot` (maintained by `pipeline/phrase_slot.py`)
at word-start (letter_run_len == 0, post-space) and biases toward
first letters of the slot-appropriate POS class.

Slot meanings:
  SLOT_POST_DET (1): after article/possessive. Prefer adjective/noun
                     openers; suppress verb/modal/aux/another-det.
  SLOT_POST_ADJ (2): inside NP after an adjective. Prefer adj/noun;
                     suppress verb/modal/det.
  SLOT_POST_NOUN (3): head noun complete. Prefer prep/verb/conj;
                     suppress another det/adj (which would start a
                     new bare NP).

Scales are modest — this is a nudge that interacts with other word-
start biases. The slot_len component escalates pressure the longer an
NP has been "open" (no noun yet).

No corpus statistics. All letter-class mappings come from prior
knowledge of common English word-initial letters per POS class.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# First letters that commonly begin each POS class. Values are relative
# weights (will be scaled at call time). Hand-graded from prior
# knowledge of Early Modern English vocabulary.

# NOUN/PROPER-NOUN/ADJECTIVE word-starts (content-word openers).
# Nouns: man, woman, king, queen, lord, lady, heart, love, life, death,
# time, day, night, world, heaven, mind, soul, eye, hand, word...
# Adjectives: good, fair, great, true, false, bold, sweet, dear, sick,
# glad, gentle, honest, noble, valiant, poor, rich, dead, fresh...
_NOUN_ADJ_STARTS: dict[str, float] = {
    "m": 0.80,  # man, mind, mother, murder, mercy, mild, much
    "h": 0.80,  # heart, hand, head, heaven, honest, honour, hot, high
    "l": 0.75,  # love, life, lord, lady, light, little, loyal, long
    "d": 0.70,  # day, death, dear, deep, dark, doubt, dread, dire
    "f": 0.85,  # fair, fear, father, false, fine, fool, friend, fresh
    "g": 0.75,  # god, good, great, gold, gentle, grief, green, glad
    "w": 0.70,  # woman, word, world, way, war, white, wild, wise, weak
    "t": 0.60,  # time, true, thing, tongue, twain, tender, tame, trusty
    "s": 0.70,  # sun, soul, son, sir, sweet, sure, sad, sharp, sick
    "n": 0.55,  # night, name, noble, noise, new, near, nimble, naked
    "b": 0.70,  # body, blood, book, bone, beast, bold, black, bright
    "p": 0.60,  # power, prince, peace, pain, poor, proud, perfect, pale
    "c": 0.65,  # care, child, crown, court, cold, clean, clear, close
    "k": 0.40,  # king, knight, kin, kind, keen, keen
    "r": 0.55,  # rose, reason, rule, right, rude, red, raw, rash, rank
    "e": 0.50,  # earth, eye, end, enemy, eager, empty, early, easy
    "o": 0.30,  # one, other, open, old
    "a": 0.45,  # arm, arms, angel, anger, apt, able, alone, angry
    "y": 0.35,  # year, youth, young
    "v": 0.30,  # voice, virtue, vile, vain
    "j": -0.20,
    "q": -0.50,
    "x": -0.90,
    "z": -0.60,
    # Capitalized starts (proper nouns / line-start) — mild positive,
    # since at a non-sentence-start position we prefer lowercase.
    "M": 0.05, "H": 0.05, "L": 0.05, "D": 0.05, "F": 0.05, "G": 0.05,
    "W": 0.05, "T": 0.05, "S": 0.05, "N": 0.05, "B": 0.05, "P": 0.05,
    "C": 0.05, "K": 0.05, "R": 0.05, "E": 0.05,
}

# VERB word-starts (main + common irregulars).
# go, come, know, think, say, see, make, take, tell, give, find,
# leave, bring, hold, speak, stand, hear, keep, let, set, put, cut,
# run, sit, eat, get, read, seek, meet, write, buy, send, spend, lose,
# win, lay, die, lie, kill, love, hate, fear, pray, weep, sleep, fly,
# grow, blow, show, throw, break, wake, rise, fall, fight, want, need,
# feel, look, seem, turn, call, work, play, move, live, stay, try, use,
# ask, wish, bear, beat, swear, wear, tear
_VERB_STARTS: dict[str, float] = {
    "s": 0.85,  # see, say, speak, stand, sit, sleep, seek, send, swear, stay, set, show
    "t": 0.80,  # take, tell, think, turn, try, tear, throw, touch
    "l": 0.80,  # let, lay, lie, leave, live, look, love, lose, learn
    "g": 0.80,  # go, give, get, grow, grant
    "c": 0.80,  # come, call, cut, care, climb, catch, change
    "f": 0.75,  # find, fight, fall, feel, fear, fly, follow
    "m": 0.75,  # make, meet, move, mean, mark, mind
    "b": 0.70,  # bring, bear, beat, break, blow, buy, beg
    "h": 0.70,  # have, hear, hold, help, hope, hate, hide, hang
    "r": 0.65,  # run, read, rise, rest, ride, reach, rule
    "p": 0.60,  # put, pray, play, pass, push, pull, praise, prove
    "w": 0.70,  # weep, wake, wear, win, want, wish, walk, watch, write
    "k": 0.60,  # know, kill, keep, kiss
    "d": 0.65,  # die, do, draw, drink, dwell, dare
    "e": 0.40,  # eat
    "o": 0.25,  # open
    "a": 0.50,  # ask, answer, act, arrive
    "n": 0.25,  # not (aux-like); need
    "y": 0.15,  # yield
    "v": 0.30,  # vow, view, vex
    "j": -0.20,
    "q": -0.50,
    "x": -0.90,
    "z": -0.60,
}

# DETERMINER word-starts (first letters of articles/possessives).
# the, a, an, this, that, these, those, my, thy, his, her, our, your,
# their, its, some, any, every, each, one, no, what, which
_DET_STARTS: dict[str, float] = {
    "t": 1.00,  # the, this, that, these, those, thy, their
    "a": 0.85,  # a, an, any, all
    "m": 0.65,  # my, mine
    "h": 0.65,  # his, her
    "o": 0.60,  # our, one
    "y": 0.55,  # your, yon, yonder
    "s": 0.40,  # some, such
    "e": 0.45,  # every, each (but e is rare start overall)
    "n": 0.35,  # no
    "w": 0.50,  # what, which, whose
}

# PREPOSITION / CONJUNCTION word-starts.
# of, to, in, on, with, for, by, at, as, from, into, unto, upon, from,
# before, after, between, beyond, above, within, without, during...
# and, but, or, nor, yet, so, if, though, because, since, when, while...
_PREP_CONJ_STARTS: dict[str, float] = {
    "o": 0.85,  # of, on, or, over
    "t": 0.85,  # to, through, though, than, then
    "i": 0.85,  # in, into, if
    "w": 0.95,  # with, when, while, where, whether, whose
    "f": 0.80,  # for, from
    "b": 0.85,  # by, but, before, because, between, beyond, beneath
    "a": 0.80,  # at, as, and, after, against, above, about, amid, among
    "u": 0.70,  # unto, upon, until, under
    "s": 0.55,  # so, since
    "n": 0.45,  # nor
    "y": 0.40,  # yet
    "e": 0.30,  # ere (archaic)
}


def _build_vec(weights: dict[str, float], scale: float) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch, w in weights.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += scale * w
    return vec


def _sub_vec(primary: dict[str, float], penalize: dict[str, float],
             scale: float, penalty_frac: float = 0.4) -> list[float]:
    """Build bias: + on primary letters, - on penalize letters.
    penalty_frac scales the suppression magnitude relative to primary."""
    vec = [0.0] * VOCAB_SIZE
    for ch, w in primary.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += scale * w
    for ch, w in penalize.items():
        # Penalize by subtracting a smaller version.
        if ch in VOCAB_INDEX:
            # Only penalize letters NOT strongly in primary (to avoid
            # double-sign noise).
            if primary.get(ch, 0.0) < 0.3:
                vec[VOCAB_INDEX[ch]] -= scale * penalty_frac * w
    return vec


def phrase_slot_bias(
    phrase_slot: int,
    phrase_slot_len: int,
    letter_run_len: int,
    last_char_class: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a word-start bias keyed to the current phrase slot."""
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None
    # Only at post-space / post-mid-punct.
    if last_char_class not in (1, 7):
        return None
    if phrase_slot == 0:
        return None

    # Scale with slot_len — longer NP without closure = more pressure.
    base = 0.18
    if phrase_slot_len >= 3:
        scale = base * 1.8
    elif phrase_slot_len == 2:
        scale = base * 1.3
    else:
        scale = base

    if phrase_slot == 1:
        # POST_DET — want ADJ or NOUN next. Penalize DET/PREP/CONJ/VERB.
        return _sub_vec(
            primary=_NOUN_ADJ_STARTS,
            penalize={
                **{k: v * 0.6 for k, v in _DET_STARTS.items()},
                **{k: v * 0.4 for k, v in _VERB_STARTS.items()},
            },
            scale=scale,
            penalty_frac=0.35,
        )
    elif phrase_slot == 2:
        # POST_ADJ — want ADJ or NOUN. Suppress VERB/DET more strongly;
        # at slot_len >= 2, pull harder toward noun (need to close NP).
        if phrase_slot_len >= 2:
            scale *= 1.2
        return _sub_vec(
            primary=_NOUN_ADJ_STARTS,
            penalize={
                **{k: v * 0.5 for k, v in _VERB_STARTS.items()},
                **{k: v * 0.5 for k, v in _DET_STARTS.items()},
            },
            scale=scale,
            penalty_frac=0.30,
        )
    elif phrase_slot == 3:
        # POST_NOUN — want PREP / VERB / CONJ / terminator. Suppress
        # another DET / another ADJ (would start a new bare NP).
        combined: dict[str, float] = {}
        for k, v in _PREP_CONJ_STARTS.items():
            combined[k] = max(combined.get(k, 0.0), v)
        for k, v in _VERB_STARTS.items():
            combined[k] = max(combined.get(k, 0.0), v * 0.8)
        return _sub_vec(
            primary=combined,
            penalize={
                **{k: v for k, v in _DET_STARTS.items()},
            },
            scale=scale * 0.8,
            penalty_frac=0.35,
        )
    return None
