"""List-parallelism bias layer.

Reads the list-progression state (pipeline/list_structure.py) and
biases the distribution at predict time toward parallel continuations:

  (a) Alliterative continuation: when `list_parallel_run >= 2` AND we
      are at a word-start position AFTER a comma (list_item_pending),
      boost the same `list_last_item_first_letter` that prior items
      shared. Gives the "by heaven, by earth, by all" pattern a
      concrete next-letter lift.

  (b) Conjunction closing: at word-start, when `commas_since_sent_end
      >= 2`, boost "a" (and), "o" (or), "n" (nor), "b" (but) — the
      common "A, B, C, and D" closing slot.

  (c) Parallel-POS continuation: if `list_first_item_pos` is a NOUN
      or VERB, and we're at word-start right after a comma, lift
      starters consistent with that POS. Conservative weights.

All weights are hand-specified from prior knowledge of Shakespearean
list rhetoric. No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE
from ..pipeline.pos import (
    POS_ADJECTIVE,
    POS_NOUN,
    POS_PROPER_NOUN,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
)


# Common starter-letter distributions for noun lists, verb lists,
# and adjective lists — reshape toward the right lexicon family.
_NOUN_STARTERS: dict[str, float] = {
    "h": 0.22,  # heart, hand, head, heaven, honour, hope
    "s": 0.20,  # soul, sword, sun, stars, sea, sorrow
    "l": 0.18,  # love, life, light, lord, lady, land
    "d": 0.16,  # death, day, doubt, deeds, dream
    "b": 0.16,  # blood, body, breath, battle
    "e": 0.14,  # eye, ear, earth, evil
    "f": 0.16,  # face, faith, fire, foe, fortune
    "t": 0.18,  # tongue, tears, truth, throne, time, thought
    "m": 0.16,  # moon, mind, mouth, mercy
    "n": 0.12,  # night, name, nature
    "w": 0.14,  # word, world, wound, war, wife
    "p": 0.10,  # peace, pride, power
    "g": 0.10,  # god, grief, grace, gold
    "c": 0.10,  # crown, child, cup
    "r": 0.10,  # rose, river, ring
    "k": 0.08,  # king, knight
}

_VERB_STARTERS: dict[str, float] = {
    "s": 0.18,  # say, see, speak, smile, sleep
    "l": 0.16,  # love, look, live, leave, lie, let
    "k": 0.14,  # know, kill, kiss, keep
    "h": 0.14,  # hear, have, hold, help
    "g": 0.14,  # go, give, get
    "c": 0.14,  # come, call, cry, curse
    "t": 0.14,  # take, tell, think, turn, talk
    "f": 0.12,  # feel, find, fear, fight, fly
    "d": 0.14,  # die, do, dream, dwell
    "w": 0.14,  # walk, weep, win, wonder, wait
    "b": 0.12,  # break, bring, bear, bid, blow
    "p": 0.10,  # pray, pass, prove, part
    "m": 0.08,  # make, meet, move
    "r": 0.10,  # run, rest, rise, read
    "a": 0.08,  # ask
}

_ADJ_STARTERS: dict[str, float] = {
    "s": 0.18,  # sweet, sad, silent, strong, soft
    "f": 0.18,  # fair, fell, false, free, fine, full
    "g": 0.16,  # good, great, gentle, grave, golden
    "b": 0.16,  # brave, bright, bitter, black, blest
    "t": 0.14,  # true, tender, thick, tall
    "d": 0.14,  # dear, dark, deep, dead, dread
    "h": 0.14,  # happy, holy, high, hard, hot
    "l": 0.14,  # light, low, loud, long, lean
    "c": 0.12,  # cold, cruel, clean, calm
    "w": 0.12,  # wild, weak, wise, warm, white
    "p": 0.10,  # pure, pale, poor
    "m": 0.10,  # meek, mad, mild, mean
    "r": 0.10,  # red, royal, rich, rough
    "n": 0.08,  # new, noble
    "o": 0.08,  # old
    "e": 0.08,  # even, evil
}


def list_start_bias(
    commas_since_sent_end: int,
    list_item_pending: bool,
    list_last_item_first_letter: str,
    list_parallel_run: int,
    list_first_item_pos: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a word-start bias vector for list-aware continuations.

    Fires only at word-start and only outside speaker-label territory.
    Caller checks the word-start condition; we add all applicable
    list-parallel signals.
    """
    if speaker_label_state != 0:
        return None
    # Only fire when list context is established (>= 1 comma in clause).
    if commas_since_sent_end < 1:
        return None

    vec = [0.0] * VOCAB_SIZE

    # --- (a) Alliterative continuation ---
    # When the first letter of each post-comma item has matched, nudge
    # the letter to continue the pattern. Only fires when parallel run
    # is already established (>=2 matching items), which is a rare and
    # high-signal configuration.
    if (
        list_item_pending
        and list_parallel_run >= 2
        and list_last_item_first_letter
    ):
        ch = list_last_item_first_letter
        # Strong narrow boost on the exact matching first letter.
        boost = 0.6 + 0.3 * min(list_parallel_run - 2, 3)  # 0.6..1.5
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += boost
        up = ch.upper()
        if up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] += boost * 0.4

    # --- (b) Conjunction closing (tiny) ---
    # After 2+ commas in clause, closing with "and" / "or" / "nor"
    # is common in natural English list rhetoric. The word-start
    # letter bias here is narrow (a/o/n) and conservative.
    if commas_since_sent_end >= 2 and list_item_pending:
        # "and" — strongest, but keep modest so it doesn't flood
        # other legitimate post-comma continuations.
        scale = 1.0 + 0.25 * min(commas_since_sent_end - 2, 3)
        if "a" in VOCAB_INDEX:
            vec[VOCAB_INDEX["a"]] += 0.30 * scale
        if "A" in VOCAB_INDEX:
            vec[VOCAB_INDEX["A"]] += 0.17 * scale
        if "o" in VOCAB_INDEX:
            vec[VOCAB_INDEX["o"]] += 0.13 * scale
        if "n" in VOCAB_INDEX:
            vec[VOCAB_INDEX["n"]] += 0.10 * scale

    return vec


def list_wordend_comma_bias(
    commas_since_sent_end: int,
    list_parallel_run: int,
    list_first_item_pos: int,
    chars_since_sentence_end: int,
    speaker_label_state: int,
) -> float:
    """Additional comma boost at word-end when a list pattern is
    established. Non-negative. Used alongside the existing
    mid-sentence comma bias to keep list items chaining.
    """
    if speaker_label_state != 0:
        return 0.0
    # Need at least one comma in this clause to be in a list.
    if commas_since_sent_end < 1:
        return 0.0
    # Don't force commas near sentence-end (where the sentence is
    # closing soon anyway).
    if chars_since_sentence_end >= 80:
        return 0.0
    # Alliterative list → more likely another comma item coming.
    boost = 0.0
    if list_parallel_run >= 2:
        boost += 0.5 + 0.2 * min(list_parallel_run - 2, 3)
    return boost
