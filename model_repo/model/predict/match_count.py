"""Predict consumer for `state.trie_match_count`.

Graded companion to the binary `on_word_trie`. Emits three related
biases based on how many known words still plausibly complete from
the current buffer.

    count == 0  and prev_count > 0
        The letter just appended took us OFF the known-word space
        entirely. There is no real-English completion. Aggressively
        boost terminators and common word-ending letters on the NEXT
        step. Stronger than the static off-trie bias because we know
        the departure JUST happened.

    count == 1
        Exactly one known word still matches. At buffer lengths >= 3
        this is near-certainty about the word. Discourage terminating
        prematurely (unless the buffer itself is already a complete
        word — handled elsewhere) and let the trie-bias layer vote
        for the unique next letter.

    count in {2, 3}
        Narrow completion set. Moderate anti-termination nudge so
        the remaining options get a fair chance.

All weights are applied only outside speaker-label territory and
only when we have at least one letter in the buffer.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def match_count_bias(
    trie_match_count: int,
    prev_trie_match_count: int,
    letter_run_len: int,
    on_word_trie: bool,
    word_buffer_len: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if word_buffer_len < 1:
        return None

    vec = [0.0] * VOCAB_SIZE

    # Case A — the buffer JUST fell off the known-word space. Fires
    # only on LATE departures (buffer length >= 6): these are cases
    # where a long real-word prefix has been invented past. Earlier
    # departures are already handled by offtrie_depart_bias and
    # letter-n-gram biases.
    if trie_match_count == 0 and prev_trie_match_count >= 2 and letter_run_len >= 6:
        depth = min(letter_run_len - 5, 6)  # depth 1..6
        term_boost = 0.20 + 0.15 * depth  # 0.35 .. 1.10
        for ch, w in (
            (" ", 1.0), ("\n", 0.35), (",", 0.55),
            (".", 0.42), (";", 0.30), (":", 0.22),
            ("!", 0.30), ("?", 0.28),
        ):
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += term_boost * w
        # Let safe word-ending letters through if model still wants a letter.
        end_boost = term_boost * 0.12
        for ch in ("e", "s", "d", "t", "n", "h", "r", "y"):
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += end_boost
        # Penalize all other letters, especially rare gibberish-extenders.
        pen = -term_boost * 0.14
        for ch in "abcdefghijklmnopqrstuvwxyz":
            if ch not in ("e", "s", "d", "t", "n", "h", "r", "y"):
                if ch in VOCAB_INDEX:
                    vec[VOCAB_INDEX[ch]] += pen
        rare_pen = -term_boost * 0.35
        for ch in ("j", "q", "x", "z", "v", "w"):
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += rare_pen
        return vec

    # Case B — narrow completion set. Only fires when still on the trie.
    if not on_word_trie:
        return None

    return None
