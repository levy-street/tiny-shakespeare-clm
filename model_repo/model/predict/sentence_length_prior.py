"""Predict consumer for `state.recent_sentence_lengths`.

Reads the rolling tuple of recently-closed sentence lengths (in words)
and, when predicting the token that would end the *current* word in
a sentence with its word-count already near the rolling average,
nudges the distribution toward sentence-enders (. ? !) vs continuation
markers (, ; :) vs word space.

Purpose: Shakespeare speakers have characteristic sentence-length
signatures — soliloquies run long, dialogue runs short, stichomythia
runs very short. The signal was tracked in state but was inert. Now
the current sentence's progress versus recent peers conditions the
termination bias.

Conditions:
  - `recent_sentence_lengths` has at least 2 entries (otherwise no
    prior to anchor on)
  - outside speaker-label territory
  - we are at a word's end — last char is a letter AND the bias is
    applied at the token AFTER the word's last letter. We approximate
    by firing when `letter_run_len >= 2` and the buffer is a complete
    word. Callers should additionally gate on on_word_trie.

Math: compute a running average of the last entries and compare to
current `words_in_sentence`. Three regimes:
  - words_in_sentence << avg: the sentence is still young; boost
    comma / word-space, discourage sentence-enders.
  - words_in_sentence ≈ avg: neutral with small sentence-ender boost.
  - words_in_sentence >> avg: sentence has gone long; strong boost
    sentence-enders (. ? !), discourage continuation markers.

No corpus statistics — uses only state.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def sentence_length_prior_bias(
    recent_sentence_lengths: tuple[int, ...],
    words_in_sentence: int,
    letter_run_len: int,
    on_word_trie: bool,
    word_buffer_is_complete: bool,
    sentence_type: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if len(recent_sentence_lengths) < 2:
        return None
    # Only fire at word-end positions — the buffer must be a real
    # complete word and at least 2 letters long.
    if letter_run_len < 2:
        return None
    if not on_word_trie:
        return None
    if not word_buffer_is_complete:
        return None

    # Compute average of the last 4 entries (already capped at 4
    # upstream). Float arithmetic is fine; floor to int bucket not
    # needed.
    n = len(recent_sentence_lengths)
    s = 0
    for ln in recent_sentence_lengths[:4]:
        s += ln
    avg = s / min(n, 4)

    # Clamp avg to plausible Shakespeare range: 5–30 words.
    if avg < 5.0:
        avg = 5.0
    elif avg > 30.0:
        avg = 30.0

    delta = words_in_sentence - avg

    vec = [0.0] * VOCAB_SIZE

    # Asymmetric — only bias sentence-ENDING push when we've run
    # significantly past the rolling average. Early in the sentence
    # there's no signal: text can still legitimately be heading toward
    # a very long or short close. This keeps the layer informative
    # only when confident.
    if delta < 5:
        return None
    if delta < 8:
        for ch in (".", "?", "!"):
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += 0.10
    else:
        for ch in (".", "?", "!"):
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += 0.18
        if ";" in VOCAB_INDEX:
            vec[VOCAB_INDEX[";"]] += 0.06

    return vec
