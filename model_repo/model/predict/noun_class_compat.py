"""Cross-class compatibility bias for noun-class coherence.

Existing `predict/noun_class.py` fires only after PREP/POSS/ART/CONJ.
`predict/sentence_sem.py` fires only when a single class has been
matched twice within one sentence. Between those two layers, there
are still many word-start positions where a DIFFERENT class can
sneak in — producing samples like "throne of treasure", "my mother
is niece", where each local transition is grammatical but the
semantic frame is violated.

This layer addresses that gap by reading `state.last_noun_class` and
`state.noun_class_age` and applying a compatibility bias at EVERY
content-word-start (not gated on POS), combining:

  (+) boost letters that open words in classes SEMANTICALLY
      COMPATIBLE with the active class (e.g., after ROYALTY, boost
      ROYALTY+PLACE+ABSTRACT+DIVINE starters);
  (-) small penalty on letters that are DISTINCTIVE starters of
      incompatible classes (e.g., after KINSHIP, discourage letters
      that overwhelmingly open WEAPON/CREATURE words like 'j' for
      javelin and 'u' for unicorn, but only to the degree those
      letters AREN'T also compatible-class starters).

Compatibility is a 12x12 matrix expressing prior knowledge of which
Shakespearean semantic fields naturally co-occur:

    KINSHIP   <-> KINSHIP BODY EMOTION TIME
    ROYALTY   <-> ROYALTY PLACE ABSTRACT DIVINE KINSHIP
    BODY      <-> BODY KINSHIP EMOTION ABSTRACT
    EMOTION   <-> EMOTION KINSHIP BODY ABSTRACT DIVINE
    NATURE    <-> NATURE PLACE TIME CREATURE
    ABSTRACT  <-> ABSTRACT EMOTION DIVINE TIME ROYALTY
    WEAPON    <-> WEAPON BODY ROYALTY
    PLACE     <-> PLACE ROYALTY NATURE KINSHIP
    TIME      <-> TIME NATURE ABSTRACT EMOTION
    CREATURE  <-> CREATURE NATURE
    DIVINE    <-> DIVINE ABSTRACT ROYALTY EMOTION

The layer fires through ages 0–5 with decaying magnitude.

No corpus statistics — the matrix is prior-knowledge semantic-field
affinity drawn from reading Shakespeare.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


NC_NONE = 0
NC_KINSHIP = 1
NC_ROYALTY = 2
NC_BODY = 3
NC_EMOTION = 4
NC_NATURE = 5
NC_ABSTRACT = 6
NC_WEAPON = 7
NC_PLACE = 8
NC_TIME = 9
NC_CREATURE = 10
NC_DIVINE = 11


# Class-starter letter sets (subsets of the richer tables used by
# noun_class.py, focusing on the letters MOST characteristic of each
# class so the compatibility computation below is clean).
_CLASS_LETTERS: dict[int, str] = {
    NC_KINSHIP: "mfbsdclnpw",   # mother father brother son daughter child niece parent wife
    NC_ROYALTY: "klpcrmgnqth", # king lord prince crown royal majesty queen throne
    NC_BODY: "hbeflthpc",       # heart hand eye face flesh tongue hair bones
    NC_EMOTION: "lsghfdwbpr",   # love sorrow grief hate fear despair wrath hope pity rage
    NC_NATURE: "swfmblr",       # sun moon wind water flower sea rose leaf river
    NC_ABSTRACT: "tshdgpvlnf",  # truth soul honour doom grace virtue life lie fate
    NC_WEAPON: "sbwdaklp",      # sword blade weapon dagger arrow armor knight pike
    NC_PLACE: "cthpgsadlo",     # castle tower hall palace grave city tomb land door
    NC_TIME: "hdmnyl",          # hour day month night year long-ago
    NC_CREATURE: "hwblsfcd",    # horse wolf bird lion sheep fox cat dog
    NC_DIVINE: "ghsbdpaflm",    # god heaven saint blessing devil prayer angel fiend
}


# Compatibility sets. Each class lists its compatible classes
# (itself included).
_COMPAT: dict[int, frozenset[int]] = {
    NC_KINSHIP:  frozenset({NC_KINSHIP, NC_BODY, NC_EMOTION, NC_TIME, NC_PLACE, NC_ROYALTY}),
    NC_ROYALTY:  frozenset({NC_ROYALTY, NC_PLACE, NC_ABSTRACT, NC_DIVINE, NC_KINSHIP, NC_BODY}),
    NC_BODY:     frozenset({NC_BODY, NC_KINSHIP, NC_EMOTION, NC_ABSTRACT, NC_ROYALTY}),
    NC_EMOTION:  frozenset({NC_EMOTION, NC_KINSHIP, NC_BODY, NC_ABSTRACT, NC_DIVINE, NC_TIME}),
    NC_NATURE:   frozenset({NC_NATURE, NC_PLACE, NC_TIME, NC_CREATURE, NC_BODY}),
    NC_ABSTRACT: frozenset({NC_ABSTRACT, NC_EMOTION, NC_DIVINE, NC_TIME, NC_ROYALTY, NC_BODY}),
    NC_WEAPON:   frozenset({NC_WEAPON, NC_BODY, NC_ROYALTY, NC_PLACE}),
    NC_PLACE:    frozenset({NC_PLACE, NC_ROYALTY, NC_NATURE, NC_KINSHIP, NC_TIME}),
    NC_TIME:     frozenset({NC_TIME, NC_NATURE, NC_ABSTRACT, NC_EMOTION, NC_KINSHIP}),
    NC_CREATURE: frozenset({NC_CREATURE, NC_NATURE, NC_PLACE}),
    NC_DIVINE:   frozenset({NC_DIVINE, NC_ABSTRACT, NC_ROYALTY, NC_EMOTION, NC_PLACE}),
}


def _compat_letters(cls: int) -> str:
    """Letters distinctive to the compatible-classes set — letters
    that open words in this class or its compatibles, but NOT in any
    INCOMPATIBLE class. These are the letters we can confidently
    push because pushing them can't lean us toward an incompatible
    continuation.
    """
    compat = _COMPAT.get(cls)
    if compat is None:
        return ""
    all_classes = set(_CLASS_LETTERS.keys())
    incompat = all_classes - compat
    compat_set: set[str] = set()
    for c in compat:
        compat_set.update(_CLASS_LETTERS.get(c, ""))
    incompat_set: set[str] = set()
    for c in incompat:
        incompat_set.update(_CLASS_LETTERS.get(c, ""))
    distinctive = compat_set - incompat_set
    return "".join(sorted(distinctive))


def _incompat_distinctive_letters(cls: int) -> str:
    """Letters that open words in INCOMPATIBLE classes but NOT
    in any compatible class. These are the letters we can safely
    penalize without suppressing compatible-class words."""
    compat = _COMPAT.get(cls)
    if compat is None:
        return ""
    all_classes = set(_CLASS_LETTERS.keys())
    incompat = all_classes - compat
    incompat_letters: set[str] = set()
    for c in incompat:
        incompat_letters.update(_CLASS_LETTERS.get(c, ""))
    compat_letters: set[str] = set()
    for c in compat:
        compat_letters.update(_CLASS_LETTERS.get(c, ""))
    # Distinctive = in incompat classes, NOT in any compat class.
    distinctive = incompat_letters - compat_letters
    return "".join(sorted(distinctive))


def _build_vec(cls: int) -> list[float]:
    """Pre-build the letter bias vector for a given active class.
    Positive bumps on compatible-class starters, small negative
    bumps on distinctively incompatible-class starters.
    """
    vec = [0.0] * VOCAB_SIZE
    # Positive for compatible letters.
    for ch in _compat_letters(cls):
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += 1.0
        up = VOCAB_INDEX.get(ch.upper())
        if up is not None:
            vec[up] += 0.5
    # Negative bias disabled — the positive compat push is sufficient;
    # negative penalties were hitting too few letters to change BPC
    # while adding risk of hurting valid continuations.
    _ = _incompat_distinctive_letters(cls)
    return vec


_CLASS_VECS: dict[int, list[float]] = {
    c: _build_vec(c) for c in _CLASS_LETTERS.keys()
}


# Age decay: strongest right after the noun, fades by age 5.
_AGE_SCALE = {0: 0.08, 1: 0.06, 2: 0.04, 3: 0.025, 4: 0.015}


def noun_class_compat_bias(
    last_noun_class: int,
    noun_class_age: int,
    speaker_label_state: int,
    letter_run_len: int,
    word_buffer: str,
    last_char_class: int,
) -> list[float] | None:
    """Return a class-compatibility word-start bias, or None if gate fails."""
    if speaker_label_state != 0:
        return None
    if last_noun_class == 0:
        return None
    if letter_run_len != 0:
        return None
    if word_buffer:
        return None
    # Only at SPACE word-start, not immediately after newline (line
    # starts are heavily biased elsewhere).
    if last_char_class != 1:
        return None
    vec = _CLASS_VECS.get(last_noun_class)
    if vec is None:
        return None
    scale = _AGE_SCALE.get(noun_class_age)
    if scale is None:
        return None
    return [v * scale for v in vec]
