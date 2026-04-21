"""Turn-scoped proper-noun first-letter bias.

Reads `state.turn_rolodex` (proper nouns introduced in the CURRENT
turn) and `state.proper_nouns_seen` (the broader 10-entry rolodex that
rolls across turns / scenes).

Purpose: suppress cross-scene proper-noun injection.

The existing `proper_noun_memory_start_bias` boosts first letters of
EVERY name in the 10-entry global rolodex. That rolodex rolls with the
stream, so when speaker turns switch plays (HAMLET's turn follows
TAMORA's turn in the raw text), the global rolodex still holds TAMORA
at its head — and a HAMLET turn then has a boost toward letters like
"T" that shouldn't be present.

This layer ADJUSTS the global cap boost:
  * At mid-sentence word-start (PN_NONE) where cap_gate would normally
    allow a rolodex-letter cap, add a small EXTRA boost to letters
    whose lowercase first-letter appears in `turn_rolodex` (within-
    turn consistency).
  * Apply a SMALL penalty to capital letters that correspond to a
    global rolodex head NOT also present in `turn_rolodex` (stale
    carryover from a previous scene).

The deltas are small (range roughly ±0.15) so legitimate first-
mentions in a turn can still happen; we're only preferring within-
turn consistency over cross-turn carryover when they compete for the
same letter.

Gates:
  * speaker_label_state == 0
  * letter_run_len == 0 and word_buffer == ""
  * last_char in (" ", "\\n")
  * consecutive_newlines == 0 (not at a fresh-turn boundary)
  * not sentence_start_pending
  * proper_noun_slot == 0 (PN_NONE)
  * chars_since_sentence_end >= 12 (match cap_gate gate)
  * words_in_sentence >= 2 (match cap_gate gate)

No corpus statistics — weights come from prior knowledge of scene-
level name consistency in Shakespeare.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def turn_rolodex_bias(
    turn_rolodex: tuple[str, ...],
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
    if proper_noun_slot != 0:
        return None
    if chars_since_sentence_end < 12:
        return None
    if words_in_sentence < 2:
        return None

    # Build first-letter sets.
    turn_firsts: set[str] = set()
    for w in turn_rolodex:
        if w:
            turn_firsts.add(w[0].lower())

    global_firsts: set[str] = set()
    for w in proper_nouns_seen:
        if w:
            global_firsts.add(w[0].lower())

    # "Stale" = in global rolodex but NOT in turn rolodex. These are
    # the cross-scene carryovers we want to dampen.
    stale_firsts = global_firsts - turn_firsts

    if not turn_firsts and not stale_firsts:
        return None

    vec = [0.0] * VOCAB_SIZE

    # Boost within-turn caps. Capped small — we don't want to force a
    # cap when none is legit.
    for ch in turn_firsts:
        up = ch.upper()
        if up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] += 0.18

    # Penalize stale-global caps that aren't also in-turn. Small —
    # the GLOBAL rolodex still gets its own positive boost from
    # proper_noun_memory; this is a mild downward adjustment.
    for ch in stale_firsts:
        up = ch.upper()
        if up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] -= 0.14

    return vec
