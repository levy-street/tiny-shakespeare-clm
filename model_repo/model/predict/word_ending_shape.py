"""Predict layer — off-trie gibberish catcher via ending-shape score.

Reads `state.word_ending_shape_score` (maintained by
`pipeline/word_ending_shape.py`). This is a 0/1/2 score measuring
whether terminating the current word right now would yield a
recognizable English word shape.

Fires the strongest pressure when the three-way conjunction holds:
  * we are deep enough into a word (letter_run_len >= 5),
  * the word is OFF the known-word trie (`on_word_trie == False`),
  * the ending shape score is 0 (no canonical English word-ending).

That conjunction is the signature of gibberish drift — the sampler is
generating letters whose local bigrams are legal enough to dodge the
existing phonotactic close-out, but the accumulated buffer has no
real-English landing.

The layer is intentionally OFF for on-trie long words (we trust the
trie's judgment that they're real) and OFF for tails that do match
a canonical ending (the word might be in-trie or just long-tail real).

Gated on `speaker_label_state == 0` — name territory has looser
morphology.

No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def word_ending_shape_bias(
    word_ending_shape_score: int,
    on_word_trie: bool,
    letter_run_len: int,
    letters_off_trie: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    # Require deep off-trie drift — not just 1 letter off. Real words
    # the trie doesn't know often depart by a single suffix char and
    # recover. Gibberish runs keep extending.
    if letters_off_trie < 4:
        return None
    if letter_run_len < 7:
        return None
    if on_word_trie:
        return None
    if word_ending_shape_score != 0:
        # Score 1 or 2 — tail already looks word-shaped; no
        # additional pressure needed.
        return None

    # Escalating scale with letter_run_len. The longer we drift, the
    # more certain we're in gibberish. Kept gentle early (many real
    # long words land their ending at 10+ chars and would be hurt by
    # an over-eager push before that point); escalates hard after 10.
    n = letter_run_len
    if n == 7:
        sc = 0.25
    elif n == 8:
        sc = 0.55
    elif n == 9:
        sc = 1.00
    elif n == 10:
        sc = 1.60
    elif n == 11:
        sc = 2.30
    elif n == 12:
        sc = 3.00
    else:  # 13+
        sc = 3.70 + 0.50 * (n - 13)

    vec = [0.0] * VOCAB_SIZE

    # Terminators — primary push.
    if " " in VOCAB_INDEX:
        vec[VOCAB_INDEX[" "]] += sc
    if "\n" in VOCAB_INDEX:
        vec[VOCAB_INDEX["\n"]] += sc * 0.40
    for ch, w in (
        (",", 0.55), (".", 0.45), (";", 0.32), (":", 0.18),
        ("!", 0.28), ("?", 0.26),
    ):
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += sc * w

    # Prefer word-ending letters if we MUST extend — these are the
    # letters that would most quickly land us on a valid ending shape
    # (score 1+): -ed, -ing, -er, -est, -ly, -ty, -ry, etc.
    end_letter_boost = sc * 0.18
    for ch in ("e", "d", "s", "n", "t", "r", "h", "y", "g"):
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += end_letter_boost

    # Suppress rare / gibberish-extending letters.
    rare_pen = -sc * 0.42
    for ch in ("j", "q", "x", "z", "v", "w", "k"):
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += rare_pen

    # Gentle suppression on all letters (to tilt mass toward
    # terminators).
    light_pen = -sc * 0.10
    for ch in "abcdefghijklmnopqrstuvwxyz":
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += light_pen

    # Apostrophe continues the word; if we're in gibberish, we don't
    # want to extend via apostrophe either.
    if "'" in VOCAB_INDEX:
        vec[VOCAB_INDEX["'"]] += -sc * 0.25

    return vec
