"""Word-integrity predict layer.

Reads `word_integrity` (flow-tier [0.0, 1.0] score of how word-shaped
the current `word_buffer` is). When integrity collapses — i.e., we're
mid-word with a buffer that doesn't look like a real English word —
strongly boost terminator characters (space, comma, period, semicolon,
colon, !, ?, newline) to push the model to bail out of the gibberish
run.

When integrity is high (on-trie, or short, or regular), no bias.
When we're at letter_run_len < 4, no bias (too early to judge).
Inside a speaker label, no bias (labels have their own FSM).
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_TERMINATORS = (" ", ",", ".", ";", ":", "?", "!", "\n")
_LOWER_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_UPPER_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def word_integrity_bias(
    word_integrity: float,
    letter_run_len: int,
    on_word_trie: bool,
    speaker_label_state: int,
    buffer_consonant_run: int,
) -> list[float] | None:
    """Return a bias vec pushing toward termination when the current
    word buffer looks like gibberish.
    """
    if speaker_label_state != 0:
        return None
    if on_word_trie:
        return None
    if letter_run_len < 6:
        return None
    # Healthy integrity — no need to intervene.
    if word_integrity >= 0.7:
        return None

    # Compute escalating termination pressure.
    # integrity 0.0 → strongest push; 0.7 → zero.
    deficit = 0.7 - word_integrity  # in (0.0, 0.7]
    # Base scale 0 → ~1.4 as deficit grows.
    scale = deficit * 2.0
    # Escalate with letter_run_len — the longer we've been going off-
    # trie, the more urgent termination becomes.
    if letter_run_len >= 9:
        scale *= 1.8
    elif letter_run_len >= 7:
        scale *= 1.4
    elif letter_run_len >= 5:
        scale *= 1.15
    # Very long trailing consonant run → amplify further.
    if buffer_consonant_run >= 4:
        scale *= 1.3

    vec = [0.0] * VOCAB_SIZE
    # Boost terminators.
    term_boost = scale * 1.2
    for t in _TERMINATORS:
        if t in VOCAB_INDEX:
            vec[VOCAB_INDEX[t]] = term_boost
    # Slight penalty on all alphabetic continuations.
    letter_penalty = -scale * 0.35
    for ch in _LOWER_LETTERS:
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = letter_penalty
    # Stronger penalty on uppercase continuations (shouldn't happen
    # mid-word anyway, but doubly so in gibberish).
    upper_penalty = -scale * 0.5
    for ch in _UPPER_LETTERS:
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = upper_penalty
    # Apostrophe also mildly penalized (would extend the run).
    if "'" in VOCAB_INDEX:
        vec[VOCAB_INDEX["'"]] = -scale * 0.2
    return vec
