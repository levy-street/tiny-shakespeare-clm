"""Parallel-structure word-start bias (post-conjunction).

When the last completed word was a coordinating conjunction
("and", "or", "nor", "but", "yet"), the word after it is very
often parallel in POS to the word that came BEFORE the conjunction:

  "the king and queen"    — NOUN and NOUN
  "red and green"         — ADJECTIVE and ADJECTIVE
  "sweet and fair"        — ADJECTIVE and ADJECTIVE
  "fought and fled"       — VERB_ED and VERB_ED
  "love and be merry"     — VERB and VERB
  "I and thou"            — PRONOUN and PRONOUN

This reads `prev_word_pos` (the POS two words ago) and biases the
next word's starter letter toward that POS family. The existing
pos_next_bias fires on last_word_pos only — it sees CONJUNCTION,
which gives a broad prior. This layer adds the parallelism signal
the n-gram POS-next can't see.

Fires at word-start (letter_run_len == 0, buffer empty) when
last_completed_word is in the small coordinating-conjunction set.
"""

from __future__ import annotations

from ..pipeline.pos import (
    POS_ADJECTIVE,
    POS_ADVERB,
    POS_CONJUNCTION,
    POS_NOUN,
    POS_PRONOUN,
    POS_PROPER_NOUN,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
)
from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# Coordinating conjunctions that trigger parallel-structure expectation.
_COORD_CONJ: frozenset[str] = frozenset({
    "and", "or", "nor", "but", "yet",
})

# Starter-letter weights keyed by POS family. Hand-weighted from
# typical Shakespearean lexicon (no corpus statistics).
_NOUN_STARTERS: dict[str, float] = {
    "h": 0.9, "l": 0.9, "m": 0.9, "s": 1.0,
    "f": 0.8, "w": 0.7, "d": 0.7, "k": 0.5,
    "b": 0.8, "c": 0.8, "p": 0.7, "e": 0.6,
    "g": 0.5, "n": 0.5, "t": 0.5, "r": 0.5,
}
_ADJ_STARTERS: dict[str, float] = {
    "g": 0.6, "s": 0.5, "f": 0.5, "d": 0.5,
    "t": 0.3, "l": 0.3, "h": 0.3, "n": 0.3,
    "p": 0.3, "b": 0.5, "m": 0.3, "o": 0.2,
    "y": 0.2, "w": 0.3, "r": 0.3, "c": 0.3,
}
_VERB_STARTERS: dict[str, float] = {
    "s": 0.6,  # say, see, speak, stand, strike
    "t": 0.5,  # take, tell, think
    "l": 0.5,  # look, love, live, lie, lose
    "k": 0.4,  # know, kill
    "c": 0.5,  # come, call, cry
    "f": 0.5,  # fall, find, fight, fear, feel
    "g": 0.5,  # go, give, grow
    "r": 0.4,  # run, read, rise
    "d": 0.4,  # do, die
    "b": 0.4,  # be, bear, bring
    "h": 0.5,  # have, hear, hold
    "m": 0.3,  # make, move, meet
    "p": 0.3,  # pray, play
    "w": 0.4,  # wish, weep, win
}
# VERB_ED: past-tense/participle starters are mostly same as VERB
# but favor words that naturally form -ed/-en forms.
_VERB_ED_STARTERS: dict[str, float] = {
    "s": 0.4, "t": 0.4, "l": 0.4, "f": 0.4,
    "c": 0.4, "k": 0.3, "p": 0.3, "g": 0.3,
    "b": 0.4, "d": 0.4, "m": 0.3, "w": 0.3,
    "h": 0.3, "r": 0.3,
}
# VERB_ING: likewise, with a slight skew toward motion/action roots.
_VERB_ING_STARTERS: dict[str, float] = {
    "s": 0.5, "r": 0.4, "w": 0.4, "f": 0.5,
    "c": 0.4, "l": 0.3, "t": 0.3, "b": 0.3,
    "d": 0.3, "m": 0.3, "p": 0.3, "g": 0.3,
}
_PRONOUN_STARTERS: dict[str, float] = {
    "h": 0.8,  # he/him/his/her
    "t": 0.8,  # thou/thee/they/them/their
    "i": 0.4,  # I
    "m": 0.5,  # me/my
    "s": 0.4,  # she
    "w": 0.5,  # we/ye (w shared)
    "y": 0.5,  # you/ye/your
    "o": 0.3,  # one
}
_ADVERB_STARTERS: dict[str, float] = {
    "s": 0.4,  # sweetly, softly, sadly
    "g": 0.4,  # gently, gladly, greatly
    "f": 0.3,  # fairly, freely, faintly
    "q": 0.3,  # quickly, quietly
    "t": 0.3,  # truly
    "h": 0.3,  # hardly, happily
    "n": 0.4,  # now, never, not
    "e": 0.3,  # ever
    "o": 0.3,  # often, only
    "a": 0.3,  # again, always
    "w": 0.4,  # well, where, when
    "b": 0.3,  # but
    "d": 0.3,  # dearly
    "m": 0.4,  # most, much
}

_POS_TO_STARTERS: dict[int, dict[str, float]] = {
    POS_NOUN: _NOUN_STARTERS,
    POS_PROPER_NOUN: _NOUN_STARTERS,
    POS_ADJECTIVE: _ADJ_STARTERS,
    POS_VERB: _VERB_STARTERS,
    POS_VERB_ED: _VERB_ED_STARTERS,
    POS_VERB_ING: _VERB_ING_STARTERS,
    POS_PRONOUN: _PRONOUN_STARTERS,
    POS_ADVERB: _ADVERB_STARTERS,
}


def parallel_start_bias(
    last_completed_word: str,
    last_word_pos: int,
    prev_word_pos: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a word-start bias matching prev_word_pos when
    last_completed_word is a coordinating conjunction.
    """
    if speaker_label_state != 0:
        return None
    if last_word_pos != POS_CONJUNCTION:
        return None
    if last_completed_word not in _COORD_CONJ:
        return None
    starters = _POS_TO_STARTERS.get(prev_word_pos)
    if starters is None:
        return None
    vec = [0.0] * VOCAB_SIZE
    scale = 0.45  # pos_next sets a POS-based prior; this adds parallelism
    for ch, w in starters.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += scale * w
        # Capital form too.
        up = ch.upper()
        if up != ch and up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] += scale * w * 0.4
    return vec
