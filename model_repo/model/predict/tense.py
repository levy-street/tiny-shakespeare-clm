"""Predict layer — sentence tense register.

Reads `state.sentence_tense` and biases subsequent word selection
in two places:

  word-start: in a sentence with established tense, boost FIRST-LETTER
    of verbs consistent with that tense.
      PAST:     w (was, were, would), h (had, hath-NO→THAT'S_PRESENT),
                d (did), s (said, spoke→PAST), c (came), t (took,
                told, thought), k (knew), g (gave), f (found, fell,
                fought), b (bore, broke, began), r (rose).
      PRESENT:  i (is, am→NO starts with a), a (am, art, are), h
                (has, hath, have), d (do, does, doth), s (says,
                speaketh), l (loves, liveth).
      FUTURE:   (after will/shall, expecting BARE verb): b (be), d
                (do), s (see/say/speak), g (go/give), h (have),
                m (make), k (know), t (take/tell).

  mid-word (suffix shape): when buffer length >= 3 and off-trie or
    on-trie near a verb form, tilt letter choices:
      PAST: push 'd'/'e'→'d' to complete -ed past (walk→walked,
            love→loved, fear→feared). Penalize 's' that would create
            a present-tense -s.
      PRESENT: push 's'/'h' to complete -s/-eth (walk→walks, speak→
            speaketh). Penalize 'd' that would create past -ed.

Fires only when speaker_label_state == 0 and sentence_tense != 0.
Gates age — after 10+ words, stop firing; dependent clauses can
naturally shift tense.

Bias magnitudes small — this layer RIDES other biases and only
tilts when tense is established.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Word-start first-letter boosts per tense.
_PAST_START_BOOSTS: dict[str, float] = {
    "w": 0.04,  # was, were, would
    "h": 0.03,  # had
    "d": 0.03,  # did
    "s": 0.04,  # said, saw, spoke, spake, stood
    "c": 0.03,  # came, could
    "t": 0.03,  # took, told, thought, threw, tore
    "g": 0.03,  # gave, grew, got, gone
    "f": 0.03,  # found, fell, fought, felt
    "b": 0.03,  # bore, broke, began, bought, brought
    "r": 0.02,  # rose, ran, rang
}

_PRESENT_START_BOOSTS: dict[str, float] = {
    "i": 0.04,  # is
    "a": 0.04,  # am, art, are
    "h": 0.04,  # has, hath, have
    "d": 0.03,  # do, does, doth, dost
    "s": 0.03,  # says, saith, speaks
    "l": 0.02,  # loves, liveth
}

_FUTURE_START_BOOSTS: dict[str, float] = {
    # After will/shall, a bare verb follows. Common starters:
    "b": 0.04,  # be
    "d": 0.03,  # do
    "s": 0.03,  # see, say, speak
    "g": 0.03,  # go, give
    "h": 0.03,  # have, hear, hold
    "m": 0.02,  # make, meet
    "k": 0.02,  # know, kill, keep
    "t": 0.02,  # take, tell
    "c": 0.02,  # come
}


def _age_scale(age: int) -> float:
    if age <= 2:
        return 1.0
    if age <= 5:
        return 0.85
    if age <= 8:
        return 0.65
    if age <= 12:
        return 0.40
    if age <= 18:
        return 0.20
    return 0.0


def tense_start_bias(
    sentence_tense: int,
    sentence_tense_age: int,
    speaker_label_state: int,
    letter_run_len: int,
    word_buffer: str,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if sentence_tense == 0:
        return None
    if letter_run_len != 0 or word_buffer:
        return None
    scale = _age_scale(sentence_tense_age)
    if scale <= 0.0:
        return None

    if sentence_tense == 1:  # PAST
        table = _PAST_START_BOOSTS
    elif sentence_tense == 2:  # PRESENT
        table = _PRESENT_START_BOOSTS
    else:  # FUTURE
        table = _FUTURE_START_BOOSTS

    vec = [0.0] * VOCAB_SIZE
    for ch, w in table.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
        up = ch.upper()
        idx_u = VOCAB_INDEX.get(up)
        if idx_u is not None:
            vec[idx_u] += w * 0.7 * scale
    return vec


def tense_midword_bias(
    sentence_tense: int,
    sentence_tense_age: int,
    word_buffer: str,
    letter_run_len: int,
    speaker_label_state: int,
    on_word_trie: bool,
) -> list[float] | None:
    """When building a verb-shaped word, tilt toward tense-consistent
    suffix endings."""
    if speaker_label_state != 0:
        return None
    if sentence_tense == 0:
        return None
    if letter_run_len < 3:
        return None
    scale = _age_scale(sentence_tense_age)
    if scale <= 0.0:
        return None

    # Only fire when the buffer tail suggests a verb-suffix decision
    # is imminent. Look at last 1-2 letters:
    #   PAST:    last letter is a consonant before 'e' or 'd' → tilt toward -ed
    #   PRESENT: last letter suggests -s or -eth extension
    tail = word_buffer.lower()
    last = tail[-1]
    prev = tail[-2] if len(tail) >= 2 else ""

    vec = [0.0] * VOCAB_SIZE

    if sentence_tense == 1:  # PAST — favor -ed completion
        # If tail is "e" and prev is consonant → boost 'd' directly
        if last == "e" and prev and prev not in "aeiou":
            if "d" in VOCAB_INDEX:
                vec[VOCAB_INDEX["d"]] += 0.08 * scale
            if "s" in VOCAB_INDEX:
                vec[VOCAB_INDEX["s"]] -= 0.05 * scale
    elif sentence_tense == 2:  # PRESENT — favor -s / -eth / -s completion
        # After vowel 'e', could be -eth ending.
        if last == "e" and prev and prev not in "aeiou":
            if "t" in VOCAB_INDEX:
                vec[VOCAB_INDEX["t"]] += 0.06 * scale
            if "s" in VOCAB_INDEX:
                vec[VOCAB_INDEX["s"]] += 0.05 * scale
    # FUTURE: verbs are base-form after will/shall; no specific suffix
    # tilt — just rely on word-start layer.

    # Don't overapply when on_word_trie and a strong trie signal exists.
    if on_word_trie:
        for i in range(VOCAB_SIZE):
            vec[i] *= 0.5

    return vec
