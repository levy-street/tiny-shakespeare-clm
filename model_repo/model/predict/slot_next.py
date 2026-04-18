"""Syntactic-slot-aware next-word first-letter bias.

Reads `state.clause_slot` (the syntactic-position state machine
maintained by pipeline/clause_slot.py) and returns a first-letter
bias vector for the next word. This gives the predict layer a real
syntactic prior that the n-gram / word-bigram layers don't see:

  - FRESH (post-sentence-end / post-break): expect subject-like
    elements — pronouns ("I", "thou", "he", "she", "we", "they"),
    demonstratives ("this", "that"), interjections ("O", "A"lack,
    "F"ie), WH words ("W"hat/"W"ho), articles ("T"he/"A"), or
    determiners ("M"y/"T"hy/"Y"our).
  - HAS_SUBJ: expect verb phrase — aux/modal starting with
    "h"ast/"h"ath, "a"re/"a"m/"a"rt, "w"ill/"w"as/"w"ere,
    "i"s, "d"o/"d"id/"d"oth, "s"hall/"s"hould, "c"an/"c"ould,
    "m"ay/"m"ight/"m"ust, or main verb consonants.
  - HAS_VERB: expect object / complement — articles ("t"he),
    possessives ("m"y/"t"hy/"y"our/"h"is/"h"er), prepositions
    ("t"o/"o"f/"i"n/"o"n/"w"ith/"f"or/"b"y), or content nouns.
  - POST_OBJ: clause is complete — conjunctions ("a"nd/"b"ut/"o"r/
    "n"or/"y"et), follow-on prepositions, or sentence-terminating
    punctuation (handled elsewhere).

Biases are small (0.1-0.5) because they sit on top of several
existing layers. They express priors, not certainties.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

SLOT_FRESH = 0
SLOT_HAS_SUBJ = 1
SLOT_HAS_VERB = 2
SLOT_POST_OBJ = 3


# Per-slot letter leans. Lowercase values apply to lowercase starters
# (mid-sentence word continuations); capital leans are 0.6x the
# lowercase value at sentence-start.
_SLOT_LOWER_LEAN: list[dict[str, float]] = [
    # FRESH — subject / determiner / interjection starters
    {
        "i": 0.30,  # I, it, if
        "t": 0.25,  # the, thou, thy, this, that
        "o": 0.25,  # O, our, of (interjection + article)
        "m": 0.22,  # my, mine, me, methinks
        "a": 0.22,  # a, an, alas, all, art, and (overlap)
        "w": 0.20,  # we, what, who, where, when (WH)
        "h": 0.18,  # he, his, her, how
        "y": 0.15,  # you, ye, your, yet
        "s": 0.10,  # she, so (some subject-starters)
        "n": 0.08,  # no, now, never (negation interjection)
        "b": 0.08,  # but (conjunction / imperative start)
        "f": 0.05,  # fie, for
    },
    # HAS_SUBJ — expect verb / auxiliary
    {
        "h": 0.35,  # have/has/hath/hast/had
        "a": 0.32,  # am/are/art (aux)
        "w": 0.30,  # will/would/was/were
        "i": 0.25,  # is
        "d": 0.28,  # do/doth/did/didst
        "s": 0.22,  # shall/shalt/should
        "c": 0.18,  # can/could/canst
        "m": 0.18,  # may/might/must
        "l": 0.10,  # look, let (imperatives)
        "g": 0.08,  # go, gave
        "t": 0.08,  # take, tell
        # Penalize new-subject letters:
        "y": -0.15,  # "you" after already-subject is unlikely
        "o": -0.10,
    },
    # HAS_VERB — expect object / complement / preposition
    {
        "t": 0.32,  # the, to, thy, this, that
        "a": 0.25,  # a, an, all, any
        "m": 0.25,  # my, mine, me (objects)
        "h": 0.22,  # his, her, him, how
        "o": 0.22,  # of, on, one, our, o'er
        "i": 0.20,  # in, it, into
        "w": 0.18,  # with, what, which
        "b": 0.15,  # by, both
        "f": 0.15,  # for, from
        "y": 0.12,  # you, your, ye
        "n": 0.10,  # no, not (negation of object)
        "s": 0.10,  # such, some
        "u": 0.08,  # upon, unto, under
    },
    # POST_OBJ — expect conjunction, prep, or clause break
    {
        "a": 0.30,  # and, as
        "b": 0.25,  # but, by, because, before
        "o": 0.25,  # or, of, on, o'er
        "y": 0.18,  # yet
        "n": 0.20,  # nor, not
        "t": 0.15,  # to, that
        "w": 0.15,  # with, when, while
        "f": 0.12,  # for, from
        "s": 0.10,  # so, since
        "i": 0.10,  # in, if
        # Penalize re-subject letters:
        "h": -0.08,
        "m": -0.08,
    },
]


_SCALE = 0.0


def _build() -> list[list[float]]:
    out: list[list[float]] = []
    for slot_dict in _SLOT_LOWER_LEAN:
        vec = [0.0] * VOCAB_SIZE
        for ch, lean in slot_dict.items():
            v = lean * _SCALE
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] = v
            up = ch.upper()
            if up in VOCAB_INDEX and up != ch:
                vec[VOCAB_INDEX[up]] = v * 0.6
        out.append(vec)
    return out


SLOT_BIAS_VECTORS: list[list[float]] = _build()


def slot_start_bias(slot: int) -> list[float] | None:
    if 0 <= slot < len(SLOT_BIAS_VECTORS):
        return SLOT_BIAS_VECTORS[slot]
    return None
