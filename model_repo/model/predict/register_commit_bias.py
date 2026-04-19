"""Predict consumer for `state.thou_thee_commit` (× `state.case_slot`).

Once a speaker has committed to the T-form or V-form address register
within a turn (see pipeline/register_commit.py), bias word-start
letter choice to stay in-register. The commit is a TURN-LEVEL signal
that reinforces the clause-level verb_agreement signal across
sentence boundaries — verb_agreement resets every clause, but an
addressee-register commit persists.

Two firing levels:

1. Baseline (always when committed + at word-start):
     Small positive bias on committed register's 2ps leading letter;
     mild negative on the opposite form. Gentle so it only tilts ties.
       T: +t/T, -y/Y
       V: +y/Y, -t/T (milder)

2. Case-slot amplification (when case_slot is active):
     When the upcoming word is in a grammatical PRONOUN SLOT
     (SUBJ after sentence-start / conjunction, OBJ after preposition
     or transitive verb), the probability that it's actually a 2ps
     pronoun is much higher. Stack an EXTRA tilt on top of the
     baseline for the committed letter.
       SUBJ + COMMIT_T: extra +t/T (thou)
       OBJ  + COMMIT_T: extra +t   (thee, thy, thine)
       SUBJ + COMMIT_V: extra +y/Y (ye)  [rare but valid]
       OBJ  + COMMIT_V: extra +y/Y (you)

The case-slot amplification decays with case_wait_words (0 → full,
1 → 0.65x, 2 → 0.35x, 3+ → silenced) to match the case_slot
consumer's own decay.

Magnitudes are kept modest — stacks with case_slot, startword,
phrase_bigram, speaker_register.

No corpus statistics — this is Early Modern English grammar.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# --- Baseline vectors (always applied when committed) ---
def _build_t_vec() -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    if "t" in VOCAB_INDEX:
        vec[VOCAB_INDEX["t"]] += 0.14
    if "T" in VOCAB_INDEX:
        vec[VOCAB_INDEX["T"]] += 0.10
    if "y" in VOCAB_INDEX:
        vec[VOCAB_INDEX["y"]] -= 0.10
    if "Y" in VOCAB_INDEX:
        vec[VOCAB_INDEX["Y"]] -= 0.06
    return vec


def _build_v_vec() -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    if "y" in VOCAB_INDEX:
        vec[VOCAB_INDEX["y"]] += 0.14
    if "Y" in VOCAB_INDEX:
        vec[VOCAB_INDEX["Y"]] += 0.10
    if "t" in VOCAB_INDEX:
        vec[VOCAB_INDEX["t"]] -= 0.05
    if "T" in VOCAB_INDEX:
        vec[VOCAB_INDEX["T"]] -= 0.03
    return vec


_T_VEC = _build_t_vec()
_V_VEC = _build_v_vec()


# --- Case-slot amplification vectors ---
# When an active pronoun slot + register commit align, stack EXTRA
# magnitude on the committed 2ps pronoun leading letter.
def _build_subj_amp_t() -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    if "t" in VOCAB_INDEX:
        vec[VOCAB_INDEX["t"]] += 0.22  # thou
    if "T" in VOCAB_INDEX:
        vec[VOCAB_INDEX["T"]] += 0.18
    return vec


def _build_obj_amp_t() -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    if "t" in VOCAB_INDEX:
        vec[VOCAB_INDEX["t"]] += 0.28  # thee / thy / thine (accusative/poss)
    # Capitals rare in OBJ slot; still mild boost.
    if "T" in VOCAB_INDEX:
        vec[VOCAB_INDEX["T"]] += 0.12
    return vec


def _build_subj_amp_v() -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    if "y" in VOCAB_INDEX:
        vec[VOCAB_INDEX["y"]] += 0.18  # ye (nominative V-form)
    if "Y" in VOCAB_INDEX:
        vec[VOCAB_INDEX["Y"]] += 0.14
    return vec


def _build_obj_amp_v() -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    if "y" in VOCAB_INDEX:
        vec[VOCAB_INDEX["y"]] += 0.22  # you / your / yours
    if "Y" in VOCAB_INDEX:
        vec[VOCAB_INDEX["Y"]] += 0.14
    return vec


_SUBJ_T = _build_subj_amp_t()
_OBJ_T = _build_obj_amp_t()
_SUBJ_V = _build_subj_amp_v()
_OBJ_V = _build_obj_amp_v()


CASE_NONE = 0
CASE_SUBJ = 1
CASE_OBJ = 2


def register_commit_start_bias(
    thou_thee_commit: int,
    letter_run_len: int,
    speaker_label_state: int,
    case_slot: int = 0,
    case_wait_words: int = 0,
) -> list[float] | None:
    """Return a VOCAB-sized word-start bias toward the committed
    address-register, or None when no bias applies. Case-slot-aware
    amplification stacks on top when a pronoun slot is active.
    """
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None

    if thou_thee_commit == 1:
        base = _T_VEC
        subj_amp = _SUBJ_T
        obj_amp = _OBJ_T
    elif thou_thee_commit == 2:
        base = _V_VEC
        subj_amp = _SUBJ_V
        obj_amp = _OBJ_V
    else:
        return None

    # If no active case slot, return baseline.
    if case_slot == CASE_NONE or case_wait_words >= 3:
        return base

    # Decay factor matches case_slot consumer.
    if case_wait_words == 0:
        scale = 1.0
    elif case_wait_words == 1:
        scale = 0.65
    else:
        scale = 0.35

    # Stack baseline + amp×scale into a fresh vector.
    amp = subj_amp if case_slot == CASE_SUBJ else obj_amp
    vec = list(base)  # copy
    for i in range(VOCAB_SIZE):
        vec[i] += amp[i] * scale
    return vec
