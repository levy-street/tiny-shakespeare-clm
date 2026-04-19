"""Content-backbone next-word first-letter bias.

Complements pos_bigram_next and pos_trigram_next by using a
*function-word-filtered* view of the recent POS history. The state's
`recent_pos_backbone` is the rolling tuple of up to 4 recent content
backbone tags (NOUN / VERB / AUX / MODAL / PREP / ADJ / ADV /
VERB_ING / VERB_ED), with ARTICLE / POSSESSIVE / PRONOUN /
CONJUNCTION / NEGATION / INTERJECTION / WH skipped so that the
content skeleton shines through.

Example:
    text:          "I do not love the queen"
    positional:    (PRON, AUX, NEG, VERB, ART, NOUN)
    backbone:                    (NOUN, VERB, AUX)  # most-recent first

The positional trigram is (ART, NOUN) for the last two — but the
backbone trigram is (AUX, VERB, NOUN), a content-level view. The
positional layer and this one see different things:

  * pos_trigram_next sees raw 3-back positional windows — catches
    patterns like (DET, ADJ, NOUN) where function words ARE the
    signal.
  * backbone_next sees content-level 2-back / 3-back — catches
    patterns like (AUX, VERB, NOUN) regardless of what closed-class
    words intervened. After an (AUX, VERB, NOUN) backbone we've
    filled subject, aux, main verb, and object: a conjunction or
    sentence-end is strongly expected, regardless of whether the
    literal sequence was "I have sworn the oath" or "I ought to
    have sworn a truer oath" (both end with the same backbone).

Encoded backbone contexts (most-recent-first notation):

  * (NOUN, VERB, AUX)      — after subject+aux+verb (full predicate):
      expect CONJ / sentence-end / PREP. "I have sworn [and/now/./for]"
  * (AUX, VERB, NOUN)      — same as above, different surface order
      ("has sworn the X").
  * (NOUN, VERB, NOUN)     — subject+verb+object: expect CONJ / PREP /
      punct. "The king slew his foe [and/in/./with]".
  * (NOUN, AUX)            — subject+aux (predicate still forming):
      expect VERB_ED / ADJ predicate starters.
  * (NOUN, AUX, VERB_ED)   — passive/perfect: expect prep ("by") or
      sentence-end.
  * (NOUN, MODAL)          — after "subject + modal": expect bare VERB.
  * (NOUN, MODAL, VERB)    — after full modal clause: expect object
      NP opener (backbone sees DET filtered out so the opener is the
      head NOUN or ADJ of the object NP).
  * (VERB, PREP, NOUN)     — after V + PP: expect CONJ / sentence-end.
  * (NOUN, PREP, NOUN)     — after NP + PP-tail: expect VERB / CONJ.
  * (ADJ, NOUN)            — head NP done: expect VERB/AUX.
  * (NOUN, ADJ, NOUN)      — apposition or compound: expect VERB.
  * (VERB_ING, PREP, NOUN) — "going to the field": expect CONJ/punct.

Weights come from English prior knowledge. No corpus statistics.
"""

from __future__ import annotations

import math

from ..pipeline.pos import (
    POS_ADJECTIVE,
    POS_ADVERB,
    POS_AUX_VERB,
    POS_MODAL,
    POS_NOUN,
    POS_PREPOSITION,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
)
from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Most-recent-first backbone triples → first-letter weights.
_BACKBONE3: dict[tuple[int, int, int], dict[str, int]] = {
    # Full predicate built (NOUN subj + AUX + main-VERB-form): expect
    # conj/punct-before-next/prep.
    (POS_NOUN, POS_VERB, POS_AUX_VERB): {
        "a": 6, "b": 5, "f": 5, "y": 4, "t": 4, "o": 4, "n": 3,
        "w": 3, "i": 3, "s": 3, "h": 3,
    },
    (POS_AUX_VERB, POS_VERB, POS_NOUN): {
        "a": 6, "b": 5, "f": 5, "y": 4, "t": 4, "o": 4, "n": 3,
        "w": 3, "i": 3, "s": 3, "h": 3,
    },
    (POS_NOUN, POS_VERB, POS_NOUN): {
        "a": 6, "b": 5, "f": 5, "y": 4, "t": 4, "o": 4, "n": 3,
        "w": 3, "i": 3, "s": 3, "h": 3,
    },
    # Passive/perfect: "X is / has been Y-ed". Expect "by" or prep.
    (POS_NOUN, POS_AUX_VERB, POS_VERB_ED): {
        "b": 6, "f": 5, "w": 5, "a": 5, "o": 4, "i": 4, "t": 4,
        "u": 3, "n": 2, "y": 2,
    },
    # After modal + main verb + object (filtered): expect conj/punct.
    (POS_NOUN, POS_MODAL, POS_VERB): {
        "a": 6, "b": 5, "y": 5, "t": 4, "o": 4, "n": 3, "f": 3,
        "w": 3, "i": 2, "s": 2, "h": 2,
    },
    # Verb + PP: clause complete — conj/punct.
    (POS_NOUN, POS_PREPOSITION, POS_VERB): {
        "a": 6, "b": 5, "y": 5, "t": 4, "o": 4, "n": 3, "f": 3,
        "w": 3, "i": 2, "s": 2, "h": 2,
    },
    # NP + PP tail: expect VERB/AUX next.
    (POS_NOUN, POS_PREPOSITION, POS_NOUN): {
        "i": 6, "a": 6, "w": 5, "h": 5, "s": 5, "d": 4, "b": 3,
        "m": 4, "c": 3, "t": 3, "o": 2, "y": 2, "n": 2,
    },
    # -ING gerund followed by PP: "going to the field, ..."
    (POS_NOUN, POS_PREPOSITION, POS_VERB_ING): {
        "a": 6, "b": 5, "y": 5, "t": 4, "o": 4, "n": 3, "f": 3,
        "w": 3, "i": 2, "s": 2, "h": 2,
    },
    # Full ADJ + NOUN subject NP + VERB: "good king rode [to/through/and]"
    (POS_VERB, POS_NOUN, POS_ADJECTIVE): {
        "t": 5, "o": 5, "a": 4, "w": 4, "i": 4, "f": 3, "u": 3,
        "b": 3, "h": 3, "m": 3, "s": 3, "d": 2, "n": 2,
    },
    # Two NOUNs in a row in backbone — apposition/compound: expect VERB/AUX.
    (POS_NOUN, POS_NOUN): {
        "i": 5, "a": 5, "w": 5, "h": 5, "s": 4, "d": 4, "b": 3,
        "m": 3, "c": 3, "t": 3, "o": 2, "y": 2, "n": 2,
    },
}

# Backbone pairs (when backbone has only 2 entries — early in a sentence
# or after a reset). Less specific than triples but still useful.
_BACKBONE2: dict[tuple[int, int], dict[str, int]] = {
    # (ADJ, NOUN): head noun done after modifier — expect VERB/AUX.
    (POS_ADJECTIVE, POS_NOUN): {
        "i": 5, "a": 5, "w": 5, "h": 5, "s": 4, "d": 4, "b": 3,
        "m": 3, "c": 3, "t": 3, "o": 2, "y": 2, "n": 2,
    },
    # (NOUN, AUX): subject + aux — predicate starter next.
    (POS_AUX_VERB, POS_NOUN): {
        "g": 4, "d": 4, "m": 4, "s": 4, "b": 4, "f": 4, "w": 4,
        "n": 3, "c": 3, "l": 3, "p": 3, "r": 3, "h": 3, "t": 3,
    },
    # (NOUN, MODAL): subject + modal — bare verb starter next.
    (POS_MODAL, POS_NOUN): {
        "b": 6, "g": 4, "c": 4, "d": 4, "s": 4, "k": 4, "l": 4,
        "m": 4, "t": 4, "h": 4, "p": 3, "r": 3, "f": 3, "w": 3,
    },
    # (VERB, NOUN): V+O — prep/conj/punct.
    (POS_NOUN, POS_VERB): {
        "a": 6, "t": 5, "o": 5, "w": 5, "f": 4, "b": 4, "i": 3,
        "u": 3, "y": 3, "n": 3,
    },
}


_GLOBAL_SCALE_3 = 0.08
_GLOBAL_SCALE_2 = 0.05


def _build_vec(weights: dict[str, int], scale: float) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    total = sum(weights.values())
    listed = set(weights.keys())
    for ch in "abcdefghijklmnopqrstuvwxyz":
        if ch not in listed and ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = -0.20 * scale
    for ch, w in weights.items():
        if ch not in VOCAB_INDEX:
            continue
        frac = w / total
        bias = scale * math.log((frac + 0.02) / 0.05)
        vec[VOCAB_INDEX[ch]] = bias
        up = ch.upper()
        if up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] = bias * 0.5
    return vec


_BB3_VEC = {k: _build_vec(v, _GLOBAL_SCALE_3) for k, v in _BACKBONE3.items()}
_BB2_VEC = {k: _build_vec(v, _GLOBAL_SCALE_2) for k, v in _BACKBONE2.items()}


def backbone_next_bias(
    recent_pos_backbone: tuple[int, ...],
) -> list[float] | None:
    """Return a first-letter bias given the filtered content-backbone
    tuple (most-recent first). Try the 3-entry lookup first; fall back
    to the 2-entry; return None if neither matches.
    """
    if len(recent_pos_backbone) >= 3:
        key3 = (
            recent_pos_backbone[0],
            recent_pos_backbone[1],
            recent_pos_backbone[2],
        )
        v = _BB3_VEC.get(key3)
        if v is not None:
            return v
    if len(recent_pos_backbone) >= 2:
        key2 = (recent_pos_backbone[0], recent_pos_backbone[1])
        return _BB2_VEC.get(key2)
    return None
