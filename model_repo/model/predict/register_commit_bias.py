"""Predict consumer for `state.thou_thee_commit`.

Once a speaker has committed to the T-form or V-form address register
within a turn (see pipeline/register_commit.py), bias word-start
letter choice to stay in-register. The commit is a TURN-LEVEL signal
that reinforces the clause-level verb_agreement signal across
sentence boundaries — verb_agreement resets every clause, but an
addressee-register commit persists.

Behavior:
  - When T_COMMIT and at word-start outside a speaker-label:
      small positive bias on "t" (thou/thee/thy/thine/thyself)
      small positive bias on "T" (sentence-initial capitalized form)
      mild negative bias on "y" (you/your/yours/ye) — discourages mixing
  - When V_COMMIT and at word-start:
      small positive bias on "y"/"Y"
      mild negative bias on "t"/"T"  (CAREFUL: "t" is an extremely
      common word-start — we only nudge, we do not suppress; the
      negative term is small)

Gentle magnitudes — this stacks with startword / phrase_bigram /
speaker_register / context_class biases, all of which already shape
word-start vocabulary. The commit-bias's job is only to break ties
toward in-register forms.

No corpus statistics — this is grammar.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Precomputed per-commit vectors.
def _build_t_vec() -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    if "t" in VOCAB_INDEX:
        vec[VOCAB_INDEX["t"]] += 0.14
    if "T" in VOCAB_INDEX:
        vec[VOCAB_INDEX["T"]] += 0.10
    # Mild discouragement of the opposing V-form leading letter.
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
    # Mild discouragement of the opposing T-form leading letter.
    # Smaller magnitude than the reverse because "t" starts many
    # common non-pronoun words (the, to, that, this, there).
    if "t" in VOCAB_INDEX:
        vec[VOCAB_INDEX["t"]] -= 0.05
    if "T" in VOCAB_INDEX:
        vec[VOCAB_INDEX["T"]] -= 0.03
    return vec


_T_VEC = _build_t_vec()
_V_VEC = _build_v_vec()


def register_commit_start_bias(
    thou_thee_commit: int,
    letter_run_len: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a VOCAB-sized word-start bias toward the committed
    address-register, or None when no bias applies.
    """
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None
    if thou_thee_commit == 1:
        return _T_VEC
    if thou_thee_commit == 2:
        return _V_VEC
    return None
