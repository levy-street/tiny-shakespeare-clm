"""Verb-semantic-class post-verb object bias.

Reads `state.verb_class` (set by pipeline/verb_class.py) and biases the
first letter of the next content word by the *semantic* class of the
most recent verb in the current clause.

This is a new axis distinct from transitivity (which only says "an
object is expected") and word_form (which only constrains the
morphological slot). verb_class says: given THAT a post-verb word is
coming, what sort of object-noun or pronoun is semantically plausible?

All biases come from prior English / Shakespeare knowledge, not from
corpus counts. The biases are small in magnitude — this is a nudge,
not a hard gate — because any class has wide object-variety.

Gated:
  - word-start position only (letter_run_len == 0, word_buffer empty)
  - outside speaker labels
  - vc_wait_words 0..2 inclusive (object slot still live)
  - verb_class != NONE
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Per-class first-letter leans. Each dict maps lowercase letter to a
# logit bump applied at word-start. The caps version gets a smaller
# bump (so proper-noun objects also benefit, modestly).
#
# Rationale for each class:
#   PERCEPT    (see, hear, feel, behold): objects are often person-
#              pronouns ("him", "her", "thee", "you", "us", "me",
#              "them") or abstract-noun perceptual targets.
#   COGNITION  (know, think, believe, doubt): "not", "naught",
#              "nothing", "that", "what", "thy" (thy mind), "him"
#              (clause subject), negation words.
#   SPEECH     (say, tell, speak, cry, swear, promise): addressee
#              pronouns (me/thee/him/her/you/us), "truth", "that",
#              "no"/"nay"/"yes"/"ay" for answers, "O" (vocative).
#   MOTION     (go, come, follow, lead, seek, meet, hunt): location
#              prepositions that follow ("to", "from", "with"), or
#              person-object for transitive motion ("follow him",
#              "seek her", "meet thee").
#   GIVE_TAKE  (give, take, bring, send, offer, keep, hold, grant):
#              indirect+direct object: "me/thee/him/her/us/them"
#              or "the/a/an/this/that" article-nominal.
#   VIOLENCE   (kill, slay, strike, wound, break, hurt): patient is
#              almost always person ("him/her/thee/them"), or body-
#              part / weapon noun.
#   EMOTION    (love, hate, fear, curse, bless, praise, mourn): the
#              emotion-target is usually person-pronoun ("him/her/
#              thee/you/my") or abstract ("life/death/fortune").
#   BE_EXIST   (is, are, be, seem, become, appear, prove, remain):
#              complement is article/possessive/adjective/adverb:
#              "a/an/the/my/thy/his/her/our/your/their/not/no/so/
#              too/very/more/most/as".

_PERCEPT_LEAN: dict[str, float] = {
    # pronouns / demonstratives (him/her/thee/them/it/this/that/you/us)
    "h": 0.55, "t": 0.50, "i": 0.35, "u": 0.30, "y": 0.25,
    # common nouns (eyes, face, man, woman, light, dark, nothing)
    "e": 0.15, "f": 0.15, "m": 0.12, "n": 0.12, "l": 0.10, "d": 0.10,
    # discourse
    "a": 0.12, "s": 0.10,
}

_COGNITION_LEAN: dict[str, float] = {
    # negation + discourse complements
    "n": 0.55,   # not, no, nothing, naught, ne'er
    "t": 0.45,   # that, thou, the, truth, thy
    "w": 0.30,   # what, why, whether, when
    "h": 0.30,   # him, her, how, he
    # other common
    "i": 0.20, "m": 0.18, "s": 0.15, "a": 0.15, "y": 0.12,
    "o": 0.10, "b": 0.10, "e": 0.08,
}

_SPEECH_LEAN: dict[str, float] = {
    # addressee pronouns
    "m": 0.55, "t": 0.55, "y": 0.45, "u": 0.40, "h": 0.35,
    # answer words / vocative
    "n": 0.35,   # no, nay
    "a": 0.30,   # ay, all
    "o": 0.30,   # O, of
    # speech-act content
    "s": 0.20, "w": 0.18, "i": 0.18, "g": 0.12,
}

_MOTION_LEAN: dict[str, float] = {
    # prepositions / directions
    "t": 0.55,   # to, toward
    "f": 0.40,   # from, forth, forward
    "w": 0.35,   # with, where, whither
    "i": 0.30,   # into, in
    "a": 0.25,   # away, after, along
    "u": 0.22,   # up, unto, upon, us
    "h": 0.30,   # him / her / home / hence
    "o": 0.22,   # on, out, off, of
    "b": 0.15,   # by, back
    "d": 0.15,   # down
    "n": 0.10,   # near
}

_GIVE_TAKE_LEAN: dict[str, float] = {
    # pronouns (indirect object primarily)
    "m": 0.55, "t": 0.55, "h": 0.45, "u": 0.30, "y": 0.28,
    # article / determiner
    "a": 0.35, "i": 0.22,
    # common objects (hand, heart, thanks, sword, crown, gold, pardon)
    "s": 0.15, "g": 0.15, "p": 0.12, "c": 0.12, "l": 0.12,
    # discourse
    "o": 0.18, "n": 0.10,
}

_VIOLENCE_LEAN: dict[str, float] = {
    # patient pronouns
    "h": 0.65,   # him, her
    "t": 0.55,   # thee, them, the, that, this
    "m": 0.40,   # me, my, man, master
    "y": 0.30,   # you, your, yourself
    "u": 0.25,   # us, upon
    # weapons / targets
    "s": 0.15, "b": 0.15, "f": 0.12, "d": 0.10, "w": 0.10,
    # articles
    "a": 0.15, "n": 0.10,
}

_EMOTION_LEAN: dict[str, float] = {
    # emotion-target pronouns
    "h": 0.55, "t": 0.50, "m": 0.40, "y": 0.40, "u": 0.25,
    # abstract nouns often in emotion clauses
    "l": 0.20,   # life, love
    "d": 0.20,   # death
    "f": 0.18,   # fate, father, face
    "n": 0.18,   # not, nothing
    "g": 0.15,   # god, grace
    "a": 0.15, "s": 0.12, "w": 0.12, "i": 0.10,
}

_BE_EXIST_LEAN: dict[str, float] = {
    # articles, possessives, negations, adverbs, comparatives
    "a": 0.55,   # a, an, all, as, already
    "t": 0.50,   # the, that, thy, too, this
    "n": 0.45,   # no, not, never, naught
    "m": 0.35,   # my, more, most, me
    "h": 0.25,   # his, her, he
    "s": 0.30,   # so, such, some
    "o": 0.22,   # our, one
    "y": 0.22,   # your, yet
    "i": 0.18,   # in, it, indeed
    "w": 0.18,   # we, well
    "f": 0.12,   # for, full
    "g": 0.10,   # good
    "b": 0.10,   # but
    "l": 0.08,   # like
}


_CLASS_TABLE: dict[int, dict[str, float]] = {
    1: _PERCEPT_LEAN,
    2: _COGNITION_LEAN,
    3: _SPEECH_LEAN,
    4: _MOTION_LEAN,
    5: _GIVE_TAKE_LEAN,
    6: _VIOLENCE_LEAN,
    7: _EMOTION_LEAN,
    8: _BE_EXIST_LEAN,
}


def verb_object_class_start_bias(
    verb_class: int,
    vc_wait_words: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if verb_class == 0:
        return None
    lean = _CLASS_TABLE.get(verb_class)
    if lean is None:
        return None
    # Decay: strongest right after the verb; weaker as other words pile up.
    # Overall magnitude kept small — this is a nudge that composes with
    # stronger structural biases, not a gate.
    if vc_wait_words == 0:
        scale = 0.22
    elif vc_wait_words == 1:
        scale = 0.15
    elif vc_wait_words == 2:
        scale = 0.08
    else:
        return None

    vec = [0.0] * VOCAB_SIZE
    for ch, lean_val in lean.items():
        v = lean_val * scale
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += v
        up = ch.upper()
        if up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] += v * 0.55
    return vec
