"""Pentameter / iambic-meter bias at end-of-word in verse passages.

Shakespeare's verse is dominantly iambic pentameter — 10 syllables per
line, 11 for feminine endings, with 9 and 12 as occasional variants.
The existing newline-at-long-line bias uses character count (csn >= 20,
25, 30, ...) as a proxy. Syllable count is the *actual* metric
Shakespeare's ear was timing to.

`pipeline/prosody.py` maintains `syllables_in_line` (count of C->V
transitions since the last newline) and `prev_line_syllables` (last
non-empty line's count). This layer consumes both.

Fires at word-end positions in verse passages:
  - verse_score > 0 AND verse_line_run >= 1 (we're in a verse run)
  - speaker_label_state == 0
  - letter_run_len >= 2, on_word_trie, word_buffer a complete word
    (i.e., a real word-end where newline is legal)

Targets:
  - If prev_line_syllables in {9, 10, 11}: target = prev_line_syllables
    (match the just-established meter)
  - Else: target = 10 (pentameter default)

Bumps to newline at word-end:
  - syllables_in_line == target:     +1.8  (prime line-end)
  - syllables_in_line == target+1:   +1.0  (feminine ending plausible)
  - syllables_in_line == target-1:   +0.3  (slight — feminine inverse)
  - syllables_in_line >= target+2:   +2.5  (line overrunning — close it)
  - syllables_in_line <= target-2:   -0.6  (too short — don't close yet)

All bumps are gentle-to-moderate. They're additive on top of the
existing char-count newline biases — when both agree (e.g., csn>=30
AND syllables==10), the composite is strong but natural; when they
disagree (csn==30 but syllables==7), the syllable layer resists the
premature newline.

No corpus statistics — pentameter's 10-syllable target is a
well-known feature of Shakespeare's verse.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def pentameter_wordend_bias(
    syllables_in_line: int,
    prev_line_syllables: int,
    verse_score: float,
    verse_line_run: int,
    chars_since_newline: int,
) -> list[float] | None:
    """Return a bias vector for \n at word-end in verse passages.

    Caller must have already checked that we're at a legal word-end
    (letter_run_len >= 2, on_word_trie, word_buffer a complete word,
    speaker_label_state == 0).
    """
    # Fire in verse mode. Two activation regimes:
    #   (A) Calibrated: a prior pentameter-length line anchors the
    #       target exactly; requires verse_line_run >= 2 AND
    #       9 <= prev_line_syllables <= 11.
    #   (B) Default-target fallback: when verse_score is very strong
    #       but we lack a calibrated anchor (e.g., first line after
    #       a speaker label), use target=10 (iambic pentameter
    #       default). Gated more conservatively — needs verse_score
    #       >= 1.0 and verse_line_run >= 1.
    if verse_score < 0.6:
        return None
    if syllables_in_line < 5:
        return None
    if chars_since_newline < 18:
        return None

    if verse_line_run >= 2 and 9 <= prev_line_syllables <= 11:
        target = prev_line_syllables
    elif verse_score >= 1.0 and verse_line_run >= 1:
        # Default pentameter target when uncalibrated.
        target = 10
    else:
        return None

    diff = syllables_in_line - target

    vec = [0.0] * VOCAB_SIZE
    nl_idx = VOCAB_INDEX.get("\n")
    if nl_idx is None:
        return None

    if diff <= -3:
        # Line way too short; \n resistance.
        vec[nl_idx] -= 0.4
    elif diff == -2:
        # Two syllables short; gentle \n resistance.
        vec[nl_idx] -= 0.15
    elif diff >= 3:
        # Overrunning by 3+ syllables; \n nudge.
        vec[nl_idx] += 0.6
    elif diff == 2:
        # Over by 2 syllables; gentle \n nudge.
        vec[nl_idx] += 0.15
    else:
        return None

    return vec


# ---------------------------------------------------------------------
# Word-start bias from meter_confidence + expected_stress.
# ---------------------------------------------------------------------
# Reads the rolling meter_confidence, expected_stress (0 weak / 1
# strong for the NEXT syllable onset), and syllables_until_line_end
# maintained by pipeline/meter.py. Tilts the next word's opening
# letter toward content-word openers on the ictus and function-word
# openers on the offbeat.

# Content-word opener letters — strong metrical onsets (plosives,
# nasals, fricatives, liquids) and emotional content vowels.
_CONTENT_START: dict[str, float] = {
    "b": 0.42,  # blood, battle, breath, beauty, brave, bright
    "c": 0.35,  # come, courage, crown, care, cruel
    "d": 0.40,  # death, dear, doubt, dream, dark, deed
    "g": 0.32,  # grace, good, god, grave, grief, glory
    "k": 0.22,  # king, keep, knave, kiss, knee
    "p": 0.36,  # power, peace, prince, pride, pain, part
    "m": 0.30,  # man, maid, mercy, mother, might, mind
    "n": 0.24,  # noble, night, nature
    "f": 0.35,  # fear, fair, fool, faith, father, fire
    "s": 0.38,  # soul, sword, shame, silent, sleep, speak, son
    "v": 0.20,  # virtue, valour, victory, voice
    "w": 0.32,  # word, world, wife, war, wrath, wound, wonder
    "l": 0.36,  # love, life, lord, light, liberty, loss
    "r": 0.30,  # right, rage, reason, revenge, rose
    "h": 0.26,  # heart, heaven, hope, honour, hour, hate
    "t": 0.20,  # time, true, truth, treason (weaker — also function)
    # Emotional content-vowel openers (softer).
    "a": 0.16,  # anger, arms, angel, agony
    "e": 0.18,  # earth, enemy, eye, ear, England
    # Capital mirrors for line openers (stage directions / imperatives).
    "B": 0.30, "C": 0.24, "D": 0.28, "G": 0.24, "K": 0.16,
    "P": 0.26, "M": 0.22, "N": 0.18, "F": 0.26, "S": 0.28,
    "L": 0.26, "R": 0.22, "H": 0.20, "T": 0.14, "W": 0.26,
    "A": 0.14, "E": 0.14, "V": 0.16, "O": 0.12,
}

# Function-word opener letters — monosyllabic closed-class onsets.
_FUNCTION_START: dict[str, float] = {
    "t": 0.50,  # the, to, thy, thou, this, that, their, these, those
    "a": 0.40,  # a, an, and, at, as, all, any
    "o": 0.42,  # of, or, on, our, o'er, out
    "i": 0.38,  # in, is, if, it, into
    "b": 0.26,  # but, by, be, been
    "m": 0.30,  # my, me, may, must
    "w": 0.32,  # with, when, where, while, who, we, will, would
    "h": 0.28,  # his, her, have, hath, he, has, had
    "n": 0.24,  # not, now, no, nor, ne'er
    "s": 0.18,  # so, shall, she, should, some, such
    "y": 0.20,  # you, your, yet, ye
    "f": 0.22,  # for, from, 'fore
    "u": 0.16,  # unto, upon, under
    # Capital mirrors (weaker — line openers skew content/imperative).
    "T": 0.28, "A": 0.20, "O": 0.18, "I": 0.35,
    "M": 0.18, "W": 0.22, "H": 0.18, "Y": 0.16,
    "N": 0.16, "S": 0.12, "F": 0.14,
}


def _build_vec(src: dict[str, float]) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch, w in src.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w
    return vec


_CONTENT_VEC = _build_vec(_CONTENT_START)
_FUNCTION_VEC = _build_vec(_FUNCTION_START)


def meter_word_start_bias(
    meter_confidence: float,
    expected_stress: int,
    syllables_until_line_end: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a word-start bias vec from the iambic meter state, or
    None if confidence is below deadband or inside speaker label."""
    if speaker_label_state != 0:
        return None
    if meter_confidence < 0.30:
        return None

    mc = meter_confidence
    # Ramp: 0.30 → 0.32; 0.60 → 0.52; 1.0 → 0.70.
    if mc >= 0.60:
        scale = 0.52 + (mc - 0.60) * 0.45  # 0.52 .. 0.70
    else:
        scale = 0.32 + (mc - 0.30) * 0.67  # 0.32 .. 0.52

    src = _CONTENT_VEC if expected_stress == 1 else _FUNCTION_VEC

    # Near pentameter close, meter-stress tilt matters less than the
    # orthogonal line-end pressure. Halve the magnitude.
    if syllables_until_line_end <= 1:
        scale *= 0.50

    return [v * scale for v in src]
