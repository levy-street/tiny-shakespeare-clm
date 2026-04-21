"""Predict layer — mid-word terminate pressure when line is in a
consecutive off-trie run.

Reads `state.line_offtrie_streak` (pipeline/line_offtrie_streak.py).
Existing `line_coherence_wordend_bias` escalates newline pressure at
word-end. This layer adds ORTHOGONAL pressure MID-WORD: when we're
inside yet another off-trie word while a streak of off-trie words is
already active on this line, gently push toward space / punctuation
(cut the current word short so the line-end newline pressure can
fire).

Gates:
  * speaker_label_state == 0
  * letter_run_len >= 3 (past initial word-shape)
  * letters_off_trie >= 2 (current word already drifted off-trie)
  * not on_word_trie (we're not in a known-word continuation)
  * line_offtrie_streak >= 2 (line is actively in a gibberish run)
  * consecutive_newlines == 0 (not immediately post-newline)

Cross-dimensional: word-level gibberish defenses (mid_departure,
offtrie_depart, word_ending_shape) fire on per-word axes only.
This layer conditions word-level behavior on a LINE-level coherence
signal, which is structurally new.

Scale is deliberately gentle (capped around the mid_departure scale)
so we don't force word-truncation when a rare legitimate archaic
word is being built.

No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_LOWER = "abcdefghijklmnopqrstuvwxyz"


def line_offtrie_streak_bias(
    line_offtrie_streak: int,
    letter_run_len: int,
    letters_off_trie: int,
    on_word_trie: bool,
    consecutive_newlines: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if consecutive_newlines != 0:
        return None
    if line_offtrie_streak < 2:
        return None
    if on_word_trie:
        return None
    if letter_run_len < 3:
        return None
    if letters_off_trie < 2:
        return None

    # Scale from streak length (2→weak, 3→moderate, 4+→stronger).
    streak = min(line_offtrie_streak, 5)
    if streak == 2:
        term_scale = 0.22
        letter_pen = -0.06
    elif streak == 3:
        term_scale = 0.40
        letter_pen = -0.10
    elif streak == 4:
        term_scale = 0.62
        letter_pen = -0.14
    else:  # 5+
        term_scale = 0.82
        letter_pen = -0.18

    # Further scale with how deeply this particular word has drifted.
    off_bonus = 1.0 + 0.12 * min(letters_off_trie - 2, 4)
    term_scale *= off_bonus
    letter_pen *= off_bonus

    vec = [0.0] * VOCAB_SIZE

    # Primary terminator push — space first (most common word-end),
    # then punctuation, then newline (line_coherence handles \n at
    # word boundary separately; we only nudge gently here).
    for ch, w in (
        (" ", 1.00),
        (",", 0.55),
        (".", 0.45),
        (";", 0.28),
        ("!", 0.25),
        ("?", 0.22),
        ("\n", 0.30),
    ):
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += term_scale * w

    # Gentle suppression of all letters to tilt mass toward
    # terminators.
    for ch in _LOWER:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += letter_pen

    return vec
