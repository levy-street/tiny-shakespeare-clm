"""Proper-noun slot predict layer.

Reads `state.proper_noun_slot` (maintained by pipeline/proper_noun.py)
and biases A-Z at word-start positions:

  PN_NONE mid-sentence (chars_since_sentence_end >= 25, words_in_sentence
    >= 3): apply a gentle blanket penalty to all capital letters, since
    a phantom capital is the main failure mode here. Prev-char "," / ";"
    / ":" filtering is already done in the pipeline stage (those raise
    PN_STRONG), so at PN_NONE we know no vocative signal is present.

  PN_STRONG / PN_QUOTE: mild positive boost on all A-Z, since a
    proper name is expected (a narrower list would require name-trie
    hooks; this gives a uniform nudge).

  PN_MILD: no penalty, no boost — ambiguous.

Small weights: the existing context / startword layers already bias
toward lowercase at mid-sentence word-starts. This layer adds a
targeted ~0.06 nudge, enough to resolve the occasional phantom-cap
without shadow-banning legitimate proper nouns.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


PN_NONE: int = 0
PN_MILD: int = 1
PN_STRONG: int = 2
PN_QUOTE: int = 3


def proper_noun_start_bias(
    proper_noun_slot: int,
    speaker_label_state: int,
    sentence_start_pending: bool,
    chars_since_sentence_end: int,
    words_in_sentence: int,
    consecutive_newlines: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if consecutive_newlines > 0:
        return None
    if sentence_start_pending:
        return None

    if proper_noun_slot == PN_NONE:
        # Phantom-cap guard at mid-sentence only.
        if chars_since_sentence_end < 25:
            return None
        if words_in_sentence < 3:
            return None
        # Gentle penalty.
        if chars_since_sentence_end < 45:
            penalty = -0.04
        else:
            penalty = -0.08
        vec = [0.0] * VOCAB_SIZE
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] = penalty
        return vec

    if proper_noun_slot in (PN_STRONG, PN_QUOTE):
        # Mild boost on capital starts; a proper name is plausibly
        # coming. Cap at a small bump so the existing signal still
        # decides which letter.
        bonus = 0.10 if proper_noun_slot == PN_STRONG else 0.08
        vec = [0.0] * VOCAB_SIZE
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] = bonus
        return vec

    # PN_MILD: no bias.
    return None
