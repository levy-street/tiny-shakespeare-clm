"""Predict consumer — turn-emphasis punctuation texture.

Reads `state.turn_exclam_count`, `state.turn_question_count`, and
`state.sentences_in_turn`. When the current speaker turn has shown a
consistent pattern of emphatic punctuation ("!"-heavy or "?"-heavy),
nudge the next sentence-end-punct choice toward that same mark.

This is a TEXTURE bias: Shakespeare turns aren't uniformly
distributed across the three sentence-enders — a character in a rage
stacks "!"s ("O villain! Most damned villain! Smiling villain!"),
and a character interrogating stacks "?"s ("Is it? Is it so? How
canst thou?"). The intra-turn autocorrelation of the sentence-end
punct class is a real regularity that no other layer sees.

Fires at word-end positions where the next char is plausibly a
sentence terminator (letter_run_len >= 2, on-trie with complete
buffer or off-trie with drift). Magnitude scales with the excess
"!" or "?" density beyond the baseline mix.

No corpus statistics — just the structural observation that turns
have characteristic emphatic textures that persist across sentences.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def turn_punct_texture_bias(
    turn_exclam_count: int,
    turn_question_count: int,
    sentences_in_turn: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return word-end bias favoring the turn's dominant end-punct.

    Fires only when the turn has enough history to be distinctive
    (sentences_in_turn >= 2). The signal is:
        exclam_ratio = turn_exclam_count / sentences_in_turn
        question_ratio = turn_question_count / sentences_in_turn

    At baseline, expect exclam_ratio ~ 0.20 and question_ratio ~ 0.15
    across Shakespeare (declaratives dominate). Excess over those is
    mapped to an additive bias on "!" and "?" respectively. A "."-only
    turn (both ratios near 0) gets a small "." nudge.

    Returns None outside speaker-label-free territory or when the turn
    is too young.
    """
    if speaker_label_state != 0:
        return None
    if sentences_in_turn < 2:
        return None

    excl = turn_exclam_count / sentences_in_turn
    ques = turn_question_count / sentences_in_turn
    # period_frac is implicit: 1 - excl - ques.

    # Baseline expectations — conservative so we only lift clear
    # excess, not noise.
    excl_excess = max(0.0, excl - 0.15)
    ques_excess = max(0.0, ques - 0.12)
    # Both-heavy: fall through to both boosts.

    # Cap contributions: very short turns (n=2) give ratios {0, 0.5, 1.0}
    # which are noisy. Scale down when the sample is tiny.
    if sentences_in_turn == 2:
        confidence = 0.45
    elif sentences_in_turn == 3:
        confidence = 0.70
    elif sentences_in_turn == 4:
        confidence = 0.85
    else:
        confidence = 1.0

    excl_boost = excl_excess * 2.4 * confidence
    ques_boost = ques_excess * 2.4 * confidence

    # A declarative-only turn should hold its ground: if we've seen
    # 3+ "." and zero "!"/"?", modestly nudge "." over its siblings.
    period_boost = 0.0
    if excl == 0.0 and ques == 0.0 and sentences_in_turn >= 3:
        period_boost = 0.20 * confidence

    if excl_boost == 0.0 and ques_boost == 0.0 and period_boost == 0.0:
        return None

    vec = [0.0] * VOCAB_SIZE
    if "!" in VOCAB_INDEX:
        vec[VOCAB_INDEX["!"]] += excl_boost
        # Cross-pen the rival terminators by a smaller amount so the
        # softmax shift is meaningful. Pull "." down; leave "?" less
        # since "?"-turns are a different regime.
        vec[VOCAB_INDEX["."]] -= excl_boost * 0.25
    if "?" in VOCAB_INDEX:
        vec[VOCAB_INDEX["?"]] += ques_boost
        vec[VOCAB_INDEX["."]] -= ques_boost * 0.25
    if period_boost > 0.0 and "." in VOCAB_INDEX:
        vec[VOCAB_INDEX["."]] += period_boost
        if "!" in VOCAB_INDEX:
            vec[VOCAB_INDEX["!"]] -= period_boost * 0.40
        if "?" in VOCAB_INDEX:
            vec[VOCAB_INDEX["?"]] -= period_boost * 0.40
    return vec


# Interjection-opener bias: at the START of a new sentence within an
# emphatic turn, boost "O"/"A"/"H" (exclamative openers) or
# WH-capitals (interrogative openers).
def turn_punct_texture_sentence_start_bias(
    turn_exclam_count: int,
    turn_question_count: int,
    sentences_in_turn: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if sentences_in_turn < 2:
        return None

    excl = turn_exclam_count / sentences_in_turn
    ques = turn_question_count / sentences_in_turn
    excl_excess = max(0.0, excl - 0.15)
    ques_excess = max(0.0, ques - 0.12)

    if sentences_in_turn == 2:
        confidence = 0.40
    elif sentences_in_turn == 3:
        confidence = 0.65
    else:
        confidence = 0.85

    if excl_excess == 0.0 and ques_excess == 0.0:
        return None

    vec = [0.0] * VOCAB_SIZE
    excl_boost = excl_excess * 1.6 * confidence
    ques_boost = ques_excess * 1.6 * confidence

    # Exclamative sentence openers (capitals, since a fresh sentence
    # after "! " will capitalize).
    for ch in ("O", "A", "H"):  # O, Alas/Ah, How/Hark/Hail
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += excl_boost
    # Also lowercase of same letters (where sentence boundary without
    # capital — a mid-sentence continuation after an emphatic clause).
    for ch in ("o", "a", "h"):
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += excl_boost * 0.40

    # Interrogative openers (WH-heavy).
    for ch in ("W", "H"):  # What, Why, Whither; How
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += ques_boost
    for ch in ("w", "h"):
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += ques_boost * 0.40
    # "C" for "Canst / Can", "I" for "Is"/"Is there"
    for ch in ("C", "I"):
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += ques_boost * 0.60

    return vec
