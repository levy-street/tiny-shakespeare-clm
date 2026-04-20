"""Rolodex-gated capital-letter word-start penalty.

Complements `proper_noun_memory.proper_noun_memory_start_bias` (which
BOOSTS rolodex first-letters at word-start). This layer PENALIZES
mid-sentence capital letters whose lowercase is NOT represented as
a first-letter in the scene rolodex.

The rationale: at PN_NONE mid-sentence with no title/vocative lead,
a capital letter word-start is almost always one of:
  (a) a recurring proper noun — it IS in the rolodex;
  (b) a first-mention proper noun — rare, typically with prior lead;
  (c) a phantom capital — the failure mode we want to suppress.

The existing `proper_noun.proper_noun_start_bias` applies a uniform
A-Z penalty of -0.04 to -0.08 at far-from-sentence-end PN_NONE. This
layer sharpens that: letters that match rolodex first-letters are
EXCLUDED from the penalty (they already got a small +boost from
proper_noun_memory), and non-rolodex letters get a STRONGER penalty.

Gates:
  * speaker_label_state == 0
  * letter_run_len == 0 and word_buffer == ""
  * last_char in (" ", "\\n") — word-boundary
  * consecutive_newlines == 0 — not at a turn boundary
  * not sentence_start_pending — sentence-initial cap is legitimate
  * proper_noun_slot == PN_NONE — no title/vocative/quote expectation
  * chars_since_sentence_end >= 12 — allow early-sentence rare caps
    (e.g. "Go, Hamlet!" → "Hamlet" at offset ~4 has a "," lead
     which already raises PN_STRONG, so this gate is conservative)

No corpus statistics — rolodex is populated from the state stream;
penalty magnitudes come from prior knowledge.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def cap_gate_start_bias(
    proper_nouns_seen: tuple[str, ...],
    proper_noun_slot: int,
    speaker_label_state: int,
    sentence_start_pending: bool,
    consecutive_newlines: int,
    chars_since_sentence_end: int,
    words_in_sentence: int,
    letter_run_len: int,
    word_buffer: str,
    last_char: str,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0 or word_buffer != "":
        return None
    if last_char not in (" ", "\n"):
        return None
    if consecutive_newlines > 0:
        return None
    if sentence_start_pending:
        return None
    # Only fire in PN_NONE. Other slot values indicate active PN
    # expectation — leave those decisions to other layers.
    if proper_noun_slot != 0:
        return None
    if chars_since_sentence_end < 12:
        return None
    if words_in_sentence < 2:
        return None

    # First-letter set from the rolodex (lowercased).
    rolodex_firsts: set[str] = set()
    for w in proper_nouns_seen:
        if w:
            rolodex_firsts.add(w[0].lower())

    # Penalty scales with depth into the sentence. Shallow mid-sentence
    # caps (chars_since_sentence_end 12-25) get a soft nudge; deep
    # caps (>= 45) get a firm push away.
    if chars_since_sentence_end < 25:
        pen_nonrolodex = -0.35
        pen_rolodex = -0.05  # still slightly negative, not zero
    elif chars_since_sentence_end < 45:
        pen_nonrolodex = -0.60
        pen_rolodex = -0.10
    else:
        pen_nonrolodex = -0.90
        pen_rolodex = -0.15

    vec = [0.0] * VOCAB_SIZE
    for ch in _UPPER:
        lower = ch.lower()
        penalty = pen_rolodex if lower in rolodex_firsts else pen_nonrolodex
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] = penalty
    return vec
