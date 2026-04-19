"""Predict consumer — turn-level content-word echo biases.

Reads `state.turn_content_cache` (up to 10 distinct content words
emitted in the current speaker turn, most-recent first). Applies two
kinds of biases:

  1. word_start_bias: at the first letter of a fresh word (outside
     speaker-label territory), boost the first letter of each cached
     word with a decaying weight. This captures the Shakespearean
     pull toward thematic words already invoked in the turn
     ("honour", "blood", "king") without rigidly repeating them.

  2. mid_word_bias: when word_buffer (lowercased) is a proper prefix
     of some cached word AND differs from the most-recently-said
     content word (to avoid immediate verbatim echo at the very next
     word boundary), boost the letter that would continue the buffer
     toward that cached completion. Scaled by cache position (more
     recent = stronger) and by letter_run_len (longer prefix = more
     discriminating, stronger boost).

Both biases are conditioned on cache size >= 2 and letter-run-len
stage; the word-start bias uses a uniform budget so the total
additive influence is bounded, not growing linearly with cache size.

No corpus statistics — all weights are hand-picked.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Per-position multiplier into the echo budget (most-recent first).
_DECAY = (1.00, 0.85, 0.70, 0.58, 0.48, 0.40, 0.33, 0.28, 0.23, 0.19)


def turn_content_echo_start_bias(
    turn_content_cache: tuple[str, ...],
    speaker_label_state: int,
) -> list[float] | None:
    """At letter_run_len == 0 (word-start), boost the first letter of
    each cached content word. Small-magnitude: this is a texture nudge,
    not a commitment.
    """
    if speaker_label_state != 0:
        return None
    if len(turn_content_cache) < 2:
        return None

    vec = [0.0] * VOCAB_SIZE
    # Collect per-first-letter weight, decayed by cache position.
    per_letter: dict[str, float] = {}
    for i, w in enumerate(turn_content_cache):
        if i >= len(_DECAY):
            break
        if not w:
            continue
        c = w[0]
        per_letter[c] = per_letter.get(c, 0.0) + _DECAY[i]

    # Normalize against a small per-word "budget" so effective boost
    # stays bounded.
    total = sum(per_letter.values())
    if total <= 0.0:
        return None
    # Magnitude scale — soft (same order as startword leans).
    budget = 0.50

    for ch, w in per_letter.items():
        frac = w / total  # in (0, 1]
        # Convert to an additive log-bias with a gentle center.
        # frac > 1/|alphabet| boosts; frac < that penalizes nothing
        # (we only add positive boosts here, to not interfere with
        # other layers' vetoes).
        lean = budget * (frac * 2.0)  # 0..budget*2 range
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += lean
        # Also lift the capital form at sentence-like positions. The
        # compose layer decides if capitals apply; we pre-load both.
        up = ch.upper() if ch.isalpha() else None
        if up and up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] += lean * 0.50

    return vec


def turn_content_echo_mid_bias(
    buffer: str,
    letter_run_len: int,
    turn_content_cache: tuple[str, ...],
    last_completed_word: str,
    on_word_trie: bool,
    speaker_label_state: int,
) -> list[float] | None:
    """Mid-word: when buffer is a proper prefix of a cached word,
    boost the continuing letter toward that cached completion.
    """
    if speaker_label_state != 0:
        return None
    if letter_run_len < 2:
        return None
    if not buffer:
        return None
    if len(turn_content_cache) < 2:
        return None
    # Only fire on-trie (off-trie we're already pushing terminators
    # hard; we don't want to extend a drifted word into a cached
    # word's letters and accidentally stabilize gibberish).
    if not on_word_trie:
        return None

    buf = buffer
    matches: list[tuple[int, str]] = []  # (cache_idx, next_char)
    for i, w in enumerate(turn_content_cache):
        if i >= len(_DECAY):
            break
        if len(w) <= len(buf):
            continue
        if not w.startswith(buf):
            continue
        # Don't fire on the very word we JUST said — it's an immediate
        # verbatim dup which is usually wrong unless anaphoric, and
        # other layers handle anaphora specifically.
        if last_completed_word and w == last_completed_word:
            continue
        nxt = w[len(buf)]
        matches.append((i, nxt))

    if not matches:
        return None

    # Accumulate by next-char, decayed by position.
    per_char: dict[str, float] = {}
    for i, ch in matches:
        per_char[ch] = per_char.get(ch, 0.0) + _DECAY[i]

    # Length-tuned magnitude: longer prefixes are more discriminating.
    if letter_run_len == 2:
        mag = 0.35
    elif letter_run_len == 3:
        mag = 0.55
    elif letter_run_len == 4:
        mag = 0.70
    else:
        mag = 0.80

    vec = [0.0] * VOCAB_SIZE
    total = sum(per_char.values())
    if total <= 0.0:
        return None
    for ch, w in per_char.items():
        frac = w / total
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += mag * frac
    return vec
