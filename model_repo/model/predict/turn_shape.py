"""Stichomythia-mode bias — cross-turn rhythm consumer.

Reads `state.stichomythia_mode` (set by pipeline/turn_shape.py). Mild
biases applied at sentence-end-punctuation decisions within a turn:

  RAPID mode (recent 2+ turns were ≤ 2 lines — rapid exchange):
    After the first completed sentence of the current turn, boost the
    newline character — we likely want to close this turn quickly to
    keep the stichomythic rhythm. Also boosts terminal ".", "!", "?"
    at word-end to encourage sentence-close when mid-turn.

  SUSTAINED mode (last completed turn was >= 6 lines — declamatory):
    Within the first sentence of the current turn, suppress newline
    and terminator choices slightly — declamatory mode continues.

UNKNOWN mode: no bias.

Very mild magnitudes — this is texture, not a hard enforcer. The rule
only fires when structural conditions hold (word-end position for
terminator boosts, letter_run_len == 0 for newline boost).

No corpus statistics — magnitudes from prior knowledge of dialogue
rhythm in verse drama.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

UNKNOWN = 0
RAPID = 1
SUSTAINED = 2


def stichomythia_bias(
    stichomythia_mode: int,
    sentences_in_turn: int,
    letter_run_len: int,
    word_buffer_len: int,
    speaker_label_state: int,
    last_char_class: int,
) -> list[float] | None:
    if stichomythia_mode == UNKNOWN:
        return None
    if speaker_label_state != 0:
        return None

    vec = [0.0] * VOCAB_SIZE

    if stichomythia_mode == RAPID:
        # After the first sentence of current turn is done, boost
        # newline at word-start / at terminator positions. The turn
        # should close after 1-2 sentences in RAPID mode.
        if sentences_in_turn >= 1 and letter_run_len == 0:
            # We're at a word-start / just-past-terminator position.
            nl = VOCAB_INDEX.get("\n")
            if nl is not None:
                vec[nl] += 0.35
        # Also, at word-end positions in the first sentence, encourage
        # sentence terminators slightly. Approximate "word-end" by
        # letter_run_len >= 3 and a buffer of reasonable length; a
        # ". ! ?" boost here nudges the distribution but the existing
        # sentence_backbone / sentence_length layers already handle
        # the heavy lifting.
        if letter_run_len >= 3:
            for ch in (".", "!", "?"):
                idx = VOCAB_INDEX.get(ch)
                if idx is not None:
                    vec[idx] += 0.08

    elif stichomythia_mode == SUSTAINED:
        # Declamatory mode — first sentence tends to keep going.
        # Very mild suppression of newline at word-start in first
        # sentence.
        if sentences_in_turn == 0 and letter_run_len == 0:
            nl = VOCAB_INDEX.get("\n")
            if nl is not None:
                vec[nl] -= 0.12

    # If everything is zero, return None.
    if not any(v != 0.0 for v in vec):
        return None
    return vec
