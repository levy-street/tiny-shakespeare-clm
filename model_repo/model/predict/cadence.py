"""Cadence-texture bias — staccato (-1) ↔ flowing (+1).

Reads `state.cadence`, a rolling flow-tier scalar updated in
pipeline/flow.py from recent word-length distribution and clausal
punctuation density. Positive means the text has been sweeping and
long-breathed ("The multitudinous seas incarnadine"); negative means
it has been tight and choppy ("Stay, villain, hold!").

At word-end positions (buffer is a complete on-trie word), this
layer shifts a small amount of mass across the comma/semicolon/
period/space axes:

  staccato (cadence < 0):
    - boost ","  — commas punctuate short bursts
    - boost ";"  — semicolons too
    - slight space boost is fine but secondary
    - slight " " penalty (to redirect mass to the commas)

  flowing (cadence > 0):
    - boost " "  — more words follow in-clause
    - gentle penalty to "," — we're not carving clauses
    - gentle penalty to ";"

Magnitudes are small (scaled by |cadence|, cap 1.0) so the layer
nudges rather than overrides the strong punctuation-placement layers
already in compose.py. Fires only outside speaker-label territory.

No corpus statistics — the bumps encode a general prior about how
short-phrase vs long-phrase rhythms pattern across comma/space/.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def cadence_wordend_bias(cadence: float) -> list[float] | None:
    """Bias vector at word-end (on-trie, complete) given the cadence
    scalar. Returns None for small |cadence| (noise)."""
    if cadence is None or abs(cadence) < 0.12:
        return None
    c = max(-1.0, min(1.0, cadence))
    vec = [0.0] * VOCAB_SIZE
    if c < 0.0:
        # Staccato: more commas / semicolons, slight space pulldown.
        mag = -c  # 0..1
        if "," in VOCAB_INDEX:
            vec[VOCAB_INDEX[","]] += 0.45 * mag
        if ";" in VOCAB_INDEX:
            vec[VOCAB_INDEX[";"]] += 0.22 * mag
        if " " in VOCAB_INDEX:
            vec[VOCAB_INDEX[" "]] -= 0.10 * mag
    else:
        # Flowing: more spaces / long clauses, fewer commas.
        mag = c
        if " " in VOCAB_INDEX:
            vec[VOCAB_INDEX[" "]] += 0.25 * mag
        if "," in VOCAB_INDEX:
            vec[VOCAB_INDEX[","]] -= 0.30 * mag
        if ";" in VOCAB_INDEX:
            vec[VOCAB_INDEX[";"]] -= 0.20 * mag
    return vec
