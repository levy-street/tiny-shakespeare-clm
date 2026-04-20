"""Predict layer — sentence_pressure terminator bias.

Reads `state.sentence_pressure` (signed scalar, roughly [-2.5, 2.0]).

When pressure is strongly NEGATIVE (sentence is structurally open),
apply graduated penalties to terminators so the model continues the
sentence. Newline gets the strongest penalty because phantom mid-clause
\n is a known sample pain point (fragment lines, spurious speaker
labels from double-newlines).

When pressure is strongly POSITIVE (sentence is long and fully backed
by subject+verb with no open NP or subord), gently boost '.' to make
long run-ons close.

Fires only at word-end decision points so in-word letter decisions
are unaffected:
  - letter_run_len >= 1 (a word is actually being formed), OR
  - letter_run_len == 0 (just emitted a terminator — still a good
    moment to bias the NEXT character choice between more text and
    another terminator).

Gated to speaker_label_state == 0. No bias during speaker labels.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def sentence_pressure_bias(
    sentence_pressure: float,
    letter_run_len: int,
    speaker_label_state: int,
    words_in_sentence: int,
    chars_since_sentence_end: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if sentence_pressure == 0.0:
        return None

    vec = [0.0] * VOCAB_SIZE

    if sentence_pressure < 0.0:
        # Suppress terminators. Magnitude scales with |pressure| but
        # capped so we don't produce illegal distributions.
        mag = min(-sentence_pressure, 2.5)  # 0.0 .. 2.5
        # Only engage past a moderate threshold (structural fluff).
        if mag < 0.45:
            return None

        # Newline — strongest penalty. Mid-clause \n is the worst
        # sample-quality bug (fragment lines, phantom speaker labels).
        nl_pen = -0.50 * mag
        # Further amplify the newline penalty when we're not near a
        # natural line boundary.
        if letter_run_len >= 2:
            # Mid-word — newline would split a word. Doubly bad.
            nl_pen *= 1.25

        nl_idx = VOCAB_INDEX.get("\n")
        if nl_idx is not None:
            vec[nl_idx] += nl_pen

        # Sentence-terminal punctuation. Softer than newline because
        # a premature '.' is recoverable with a new sentence.
        sent_pen = -0.28 * mag
        for ch in (".", "!", "?"):
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += sent_pen

        # Semicolon/colon — also clause-terminal; softer.
        for ch in (";", ":"):
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += -0.16 * mag

    else:
        # Positive pressure — sentence is ready to close.
        mag = min(sentence_pressure, 2.0)
        if mag < 0.3:
            return None

        # Only fire when the sentence has already run some length to
        # avoid over-aggressive closing of tight replies.
        if chars_since_sentence_end < 40 and words_in_sentence < 6:
            return None

        # Small boost on period; question/exclamation stay context-driven.
        pd_idx = VOCAB_INDEX.get(".")
        if pd_idx is not None:
            vec[pd_idx] += 0.22 * mag

    if not any(v != 0.0 for v in vec):
        return None
    return vec
