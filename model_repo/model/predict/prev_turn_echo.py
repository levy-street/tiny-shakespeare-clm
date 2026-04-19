"""Predict consumer — cross-turn content echo bias.

Reads `state.prev_turn_content_tail`: the last-speaker's thematic
content words (up to 6, most-recent first) captured at the moment
their turn closed. Used at the OPENING of the new speaker's turn
to bias their first words to lexically echo what was just said.

Classic Shakespeare adjacency dynamic:
    A: "Where is the king?"
    B: "The king is dead."
or:
    A: "Speak'st thou of love?"
    B: "Love is the only truth."

The echo signal is strongest at the very first content-word position
of the new turn, and decays rapidly with words_in_turn. By the third
or fourth word the new speaker has taken over and the prior turn's
vocabulary should not dominate — hence a steep age-based gate.

Two biases:
  * start_bias: word-initial letter boost from prev-turn content
    first-letters (decayed by cache position).
  * mid_bias: when the current buffer is a proper prefix of a
    prev-turn cached word (and >= 2 letters in), nudge the next
    letter along that completion path — echo the prior word mid-
    stream (e.g., speaker A said "honour", speaker B begins "ho"
    → bias "n").

Gates:
  - speaker_label_state == 0 (inside turn body)
  - words_in_turn <= 2 (only first two words of the new turn)
  - prev_turn_content_tail non-empty
  - chars_since_sentence_end <= 30 (still in the opening sentence)

No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Per-position decay (most-recent first). The most-recently-used word
# of the prior speaker has the strongest echo pull.
_DECAY = (1.00, 0.75, 0.55, 0.40, 0.28, 0.20)


# Words-in-turn age decay: strongest at word 0 (turn just opened),
# falls off fast.
_AGE_DECAY = {
    0: 1.00,
    1: 0.55,
    2: 0.25,
}


def prev_turn_echo_start_bias(
    prev_turn_content_tail: tuple[str, ...],
    speaker_label_state: int,
    words_in_turn: int,
    chars_since_sentence_end: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if not prev_turn_content_tail:
        return None
    if words_in_turn > 2:
        return None
    if chars_since_sentence_end > 30:
        return None
    age_mult = _AGE_DECAY.get(words_in_turn, 0.0)
    if age_mult <= 0.0:
        return None

    # Collect first-letter mass weighted by position decay.
    per_letter: dict[str, float] = {}
    for i, w in enumerate(prev_turn_content_tail):
        if i >= len(_DECAY):
            break
        if not w:
            continue
        c = w[0]
        per_letter[c] = per_letter.get(c, 0.0) + _DECAY[i]

    total = sum(per_letter.values())
    if total <= 0.0:
        return None

    vec = [0.0] * VOCAB_SIZE
    # Budget — soft nudge, not a commitment.
    budget = 0.45 * age_mult

    for ch, w in per_letter.items():
        frac = w / total  # (0, 1]
        lean = budget * (frac * 2.0)
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += lean
        # Also lift the capital form (sentence-start or line-start).
        if ch.isalpha():
            up = ch.upper()
            if up in VOCAB_INDEX:
                vec[VOCAB_INDEX[up]] += lean * 0.55

    return vec


def prev_turn_echo_mid_bias(
    buffer: str,
    letter_run_len: int,
    prev_turn_content_tail: tuple[str, ...],
    speaker_label_state: int,
    words_in_turn: int,
    chars_since_sentence_end: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if not prev_turn_content_tail:
        return None
    if words_in_turn > 2:
        return None
    if letter_run_len < 2 or letter_run_len > 6:
        return None
    if len(buffer) != letter_run_len:
        return None
    if chars_since_sentence_end > 30:
        return None
    age_mult = _AGE_DECAY.get(words_in_turn, 0.0)
    if age_mult <= 0.0:
        return None

    buf = buffer.lower()
    # Collect next-letter mass from cached words whose lowercased form
    # starts with buf and is strictly longer.
    per_letter: dict[str, float] = {}
    for i, w in enumerate(prev_turn_content_tail):
        if i >= len(_DECAY):
            break
        if not w:
            continue
        wl = w.lower()
        if len(wl) <= letter_run_len:
            continue
        if not wl.startswith(buf):
            continue
        nxt = wl[letter_run_len]
        per_letter[nxt] = per_letter.get(nxt, 0.0) + _DECAY[i]

    if not per_letter:
        return None
    total = sum(per_letter.values())
    if total <= 0.0:
        return None

    vec = [0.0] * VOCAB_SIZE
    # Scale grows with prefix length — longer prefix = narrower
    # plausible continuations = stronger commitment.
    scale = 0.25 + 0.10 * (letter_run_len - 1)
    scale *= age_mult

    for ch, w in per_letter.items():
        frac = w / total
        lean = scale * (frac * 2.0)
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += lean

    return vec
