"""Predict layer — negation-scope word-start bias.

Reads `state.negation_count`, `state.words_since_negation`, and
`state.last_negation_word` (set by `pipeline/negation.py`). Fires at
word-start outside speaker labels when a negation is active in the
current sentence and the negation is recent (wait <= 4).

Shakespeare's characteristic negation continuations:

  "nor X nor Y"       — once "nor" fires, another "nor" is very
                        likely in the next 1-4 words.
  "not X but Y"       — "not" often attracts a later "but".
  "neither X nor Y"   — "neither" almost always preludes a "nor".
  "never X, never Y"  — parallel "never" patterns.
  "no, nor..."        — answer-opener "no" often chains to "nor".

The bias boosts word-start letters:
  "n" — nor, never, no, nothing, naught, none (chained negation)
  "b" — but (not X but Y)
  "y" — yet (concessive after negation)

The strongest anchor is `last_negation_word`:
  - "neither"         → very strong "n" boost for the upcoming "nor"
  - "nor"             → strong "n" boost (chained "nor nor nor")
  - "not"             → moderate "b" + mild "n" boost
  - "never" / "no" / "none" / "naught" / "nothing"
                      → mild "n" + mild "b"
  - n't contraction  → mild "b" (most contracted negations chain to
                        "but")

Decay with words_since_negation: 0 (just fired) / 1 (next word) see
full strength; 2 sees 0.7x; 3 sees 0.45x; 4 sees 0.25x; 5+ → None.

All weights are hand-chosen from prior knowledge of Shakespeare's
negation patterns — no corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_EXPLICIT_NEG_WORDS: frozenset[str] = frozenset({
    "not", "no", "nay", "never", "none", "nothing", "naught",
    "nought", "nor", "neither", "cannot",
})


def negation_start_bias(
    negation_count: int,
    words_since_negation: int,
    last_negation_word: str,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if negation_count <= 0:
        return None
    if words_since_negation > 6:
        return None
    if not last_negation_word:
        return None

    # Recency decay.
    if words_since_negation <= 1:
        decay = 1.0
    elif words_since_negation == 2:
        decay = 0.80
    elif words_since_negation == 3:
        decay = 0.60
    elif words_since_negation == 4:
        decay = 0.40
    elif words_since_negation == 5:
        decay = 0.25
    else:  # 6
        decay = 0.15

    # Per-trigger strengths: (n_boost, b_boost, y_boost).
    lnw = last_negation_word
    if lnw == "neither":
        n_b, b_b, y_b = 2.00, 0.25, 0.05
    elif lnw == "nor":
        n_b, b_b, y_b = 1.20, 0.18, 0.05
    elif lnw == "not":
        n_b, b_b, y_b = 0.30, 0.60, 0.15
    elif lnw in ("never", "no", "none", "naught", "nought", "nothing", "nay", "cannot"):
        n_b, b_b, y_b = 0.25, 0.35, 0.10
    elif lnw.endswith("n't"):
        # Contracted negation (don't, hasn't, can't, ...)
        n_b, b_b, y_b = 0.15, 0.32, 0.06
    else:
        # Unknown negation form — should not happen.
        return None

    # Escalate when negation_count >= 2 (e.g., "nor X nor"): a third
    # coordinated "nor" is very likely.
    if negation_count >= 2:
        n_b *= 1.4

    n_b *= decay
    b_b *= decay
    y_b *= decay

    if n_b == 0.0 and b_b == 0.0 and y_b == 0.0:
        return None

    vec = [0.0] * VOCAB_SIZE
    if "n" in VOCAB_INDEX:
        vec[VOCAB_INDEX["n"]] += n_b
    if "b" in VOCAB_INDEX:
        vec[VOCAB_INDEX["b"]] += b_b
    if "y" in VOCAB_INDEX:
        vec[VOCAB_INDEX["y"]] += y_b
    return vec
