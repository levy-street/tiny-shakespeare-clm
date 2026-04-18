"""Anaphoric referent consumer.

Reads `state.referent_gender` and biases word-start first letters
toward pronouns matching the tracked discourse referent. Fires at
word-start outside speaker-label territory, scaled down by
staleness.

  REF_MALE   → boost "h" (he, him, his), small boost "H" at
               sentence-start; mild penalty on "s" (she) and "t"
               (they) — the referent is singular masculine.
  REF_FEMALE → boost "s" (she), small boost "h" for her/hers;
               penalty on "h" for him/his ambiguity is AVOIDED
               (both "her" and "his" start with "h"; instead
               boost "s" which is the less-ambiguous feminine
               starter).
  REF_NEUTER → boost "i" (it/its). Mild; "it" is short and
               frequently needed anyway.
  REF_PLURAL → boost "t" (they, them, their); mild.

Strength is proportional to (1 - staleness/20), so the bias decays.

We intentionally avoid firing at HAS_VERB or POST_OBJ slot if the
last word wasn't a punctuation/clause-break — the pronoun signal
is strongest at FRESH (after sentence-end) or post-conjunction
positions.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

REF_NONE = 0
REF_MALE = 1
REF_FEMALE = 2
REF_NEUTER = 3
REF_PLURAL = 4


def referent_start_bias(
    referent_gender: int,
    referent_staleness: int,
    clause_slot: int,
    speaker_label_state: int,
    letter_run_len: int,
    last_cls: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if referent_gender == REF_NONE:
        return None
    if letter_run_len != 0:
        return None
    # Fire strongest at FRESH slot (subject position); weaker elsewhere.
    if clause_slot == 0:  # FRESH
        slot_scale = 1.0
    elif clause_slot == 1:  # HAS_SUBJ — a second pronoun here is rare
        slot_scale = 0.25
    else:
        slot_scale = 0.55  # HAS_VERB / POST_OBJ — object pronouns fit

    staleness_scale = max(0.0, 1.0 - referent_staleness / 20.0)
    s = slot_scale * staleness_scale
    if s < 0.05:
        return None

    vec = [0.0] * VOCAB_SIZE

    if referent_gender == REF_MALE:
        if "h" in VOCAB_INDEX:
            vec[VOCAB_INDEX["h"]] += 0.06 * s
        if "H" in VOCAB_INDEX:
            vec[VOCAB_INDEX["H"]] += 0.04 * s
    elif referent_gender == REF_FEMALE:
        if "s" in VOCAB_INDEX:
            vec[VOCAB_INDEX["s"]] += 0.05 * s
        if "S" in VOCAB_INDEX:
            vec[VOCAB_INDEX["S"]] += 0.03 * s
    elif referent_gender == REF_NEUTER:
        if "i" in VOCAB_INDEX:
            vec[VOCAB_INDEX["i"]] += 0.04 * s
    elif referent_gender == REF_PLURAL:
        if "t" in VOCAB_INDEX:
            vec[VOCAB_INDEX["t"]] += 0.04 * s

    return vec
