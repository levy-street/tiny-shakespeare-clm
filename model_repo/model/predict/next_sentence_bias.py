"""Cross-sentence opener bias based on the previous sentence's type.

Consumes `state.prev_sentence_type` (set by `pipeline/sentence.py` at
sentence-end punctuation, preserved across the sentence boundary) and
returns a first-letter bias for the NEXT sentence's opener.

The signal is a discourse-level prior that the existing invocation /
n-gram / starter-letter layers don't see: Shakespeare's sentences come
in predictable reply-patterns.

  After ? (INTERROG)
    Common response openers:
      "Ay, ..."     → A
      "No, ..."     → N
      "Nay, ..."    → N
      "Yes, ..."    → Y
      "Why, ..."    → W (mild)
      "I ..."       → I  (self-reply)
      "Thou ..."    → T  (direct address)
      "Sir, ..."    → S
      "My ..."      → M
    Elevated: A, N, Y, I, T, S, M. Mildly suppress: another W/H aux-
    question opener (double questions chain less often than one might
    expect in dialogue).

  After ! (EXCLAM)
    Emotional momentum: another invocation opener is elevated.
      "O ..."       → O
      "Alas, ..."   → A
      "Ah, ..."     → A
      "Fie, ..."    → F
      "Heaven(s) ..."→ H
      "Hark, ..."   → H
      "My ..."      → M (e.g., "My lord!", "My heart!")
      "What ..."    → W (rhetorical chain)
      "Why ..."     → W
    Elevated: O, A, F, H, M, W.

  After . (DECL)
    Neutral. No bias — default priors dominate.

Fires only at true sentence-starts and only outside speaker labels.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Sentence-type constants (mirror pipeline/sentence.py without import cycle).
_SENT_UNKNOWN = 0
_SENT_DECL = 1
_SENT_INTERROG = 2
_SENT_EXCLAM = 3


# Post-question response openers. Capitals are what actually appear
# at sentence-start (after ". "/"? "/"! " or after a newline).
_POST_INTERROG_CAPS: dict[str, float] = {
    "A": 0.25,  # Ay, Alas, And, Ah — #1 response opener
    "N": 0.20,  # No, Nay, Never
    "Y": 0.12,  # Yes, Yea
    "I": 0.16,  # I, It
}

# Post-exclamation momentum openers. Invocation-adjacent.
_POST_EXCLAM_CAPS: dict[str, float] = {
    "O": 0.30,  # O, Oh — #1 momentum opener
    "A": 0.20,  # Alas, Ah, Ay
    "H": 0.14,  # Heavens, Hark, Hail
    "M": 0.14,  # My
}


def _build(weights: dict[str, float], scale: float) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch, w in weights.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
    return vec


def next_sentence_start_bias(
    prev_sentence_type: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a sentence-start bias vector keyed on prev_sentence_type.

    Returns None (no-op) for DECL / UNKNOWN or in speaker-label mode.
    Caller must gate on `is_sentence_start`.
    """
    if speaker_label_state != 0:
        return None
    if prev_sentence_type == _SENT_INTERROG:
        return _build(_POST_INTERROG_CAPS, 1.0)
    if prev_sentence_type == _SENT_EXCLAM:
        return _build(_POST_EXCLAM_CAPS, 1.0)
    return None
