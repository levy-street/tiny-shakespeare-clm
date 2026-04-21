"""Predict layer — gibberish-accumulation recovery.

When `turn_gibberish_count` or `recent_word_realities` show that the
model has been producing garbage words, intervene at two leverage
points:

  * WORD-START (letter_run_len == 0, last_char was a space):
      Bias the first letter of the new word toward "safe" common
      Shakespearean word-starts (function words, pronouns, common
      verbs/nouns), and penalize low-frequency letters that seed
      gibberish (j/q/x/z/k in initial position). This breaks the
      letter-ngram momentum that would otherwise extrapolate the
      last gibberish word's tail into another gibberish word.

  * MID-WORD (letter_run_len >= 3, off-trie): augment the existing
      gibberish-hardcap signal so the word closes even sooner when
      the surrounding context has been garbage. A sentence with two
      prior gibberish words is far more likely to produce a third —
      close fast and hope the next sentence resets cleanly.

Gates:
  * speaker_label_state == 0 — let speaker labels follow their own FSM.
  * Fires only when gibberish signal is non-trivial:
      turn_gibberish_count >= 2 OR
      2+ of the last 3 recent_word_realities were GIBBERISH (3).

Scale is modest; this is a nudge, not a lock. No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Shakespearean-common word-start letters, hand-graded by prior
# knowledge of how often each letter opens a common word in Early
# Modern English (function words, pronouns, verbs-to-be/do/have,
# common content words). Positive values = more common, boost up;
# negative = rarer starter, suppress.
_LOWER_SAFE_STARTS: dict[str, float] = {
    "t": 1.00,  # the, that, thou, thy, thee, to, then, them, this, these
    "a": 0.90,  # a, and, as, at, an, all, am, are, art, any
    "o": 0.75,  # of, or, on, O, our, one, out
    "i": 0.75,  # I, in, is, it, if
    "w": 0.85,  # with, what, who, will, we, where, when, why, would, well
    "h": 0.85,  # he, her, his, have, had, hath, his, here, how
    "s": 0.70,  # so, shall, she, some, such, say, said, see
    "b": 0.65,  # be, but, by, been, both, before
    "m": 0.70,  # my, me, more, most, man, much, mine, madam
    "n": 0.65,  # no, not, nor, now, ne'er, never, nothing
    "f": 0.55,  # for, from, first, faith, false, fair, father
    "l": 0.55,  # let, love, lord, lady, light, life
    "y": 0.55,  # you, your, yet, yes, young
    "d": 0.50,  # do, doth, did, dost, down, death, dear
    "c": 0.45,  # come, can, could, call, care
    "p": 0.40,  # pray, poor, part, place, prince
    "g": 0.35,  # go, good, great, gentle, God, give, grace
    "r": 0.30,  # right, rather, read, rise
    "e": 0.30,  # every, even, ever, else, ere, each
    "u": 0.15,  # up, until, unto, us
    # Mildly suppressed (rare word-starts)
    "k": -0.30,
    "v": -0.20,
    "z": -0.80,
    "j": -0.60,
    "q": -0.80,
    "x": -1.20,
}

# Capital versions inherit the same shape but at reduced magnitude,
# because capitals at word-start occur only after line-break / sentence-
# start / proper-noun contexts already handled elsewhere. Here we're
# inside a sentence after a space, so lowercase dominates.
_UPPER_SAFE_STARTS: dict[str, float] = {
    ch.upper(): v * 0.20 for ch, v in _LOWER_SAFE_STARTS.items()
}


def _build_word_start_vec(gibberish_signal: float) -> list[float] | None:
    """gibberish_signal in [0, 1]; scales the bias magnitude."""
    if gibberish_signal <= 0.0:
        return None
    vec = [0.0] * VOCAB_SIZE
    base = 0.18 * gibberish_signal  # cap at ~0.18 nats for strongest letter
    for ch, w in _LOWER_SAFE_STARTS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += base * w
    for ch, w in _UPPER_SAFE_STARTS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += base * w
    return vec


def _compute_signal(
    turn_gibberish_count: int,
    turn_real_count: int,
    sentence_gibberish_count: int,
    recent_word_realities: tuple[int, ...],
) -> float:
    """Return a [0, 1] signal — how much gibberish recovery pressure?"""
    # Fast-path: no gibberish at all.
    if (
        turn_gibberish_count == 0
        and sentence_gibberish_count == 0
        and not any(r == 3 for r in recent_word_realities)
    ):
        return 0.0

    # Count recent gibberish in the last 3 words.
    recent = recent_word_realities[:3]
    recent_gib = sum(1 for r in recent if r == 3)

    # Primary signal: recent short window matters most.
    # recent_gib / len(recent) gives 0/0.33/0.67/1.00.
    if len(recent) == 0:
        short = 0.0
    else:
        short = recent_gib / len(recent)

    # Sentence-level contribution: sentence_gibberish_count >= 2 is
    # a strong signal that the current sentence is polluted.
    if sentence_gibberish_count >= 3:
        sent = 1.0
    elif sentence_gibberish_count == 2:
        sent = 0.7
    elif sentence_gibberish_count == 1:
        sent = 0.3
    else:
        sent = 0.0

    # Turn-level contribution — milder because it may reflect an old
    # patch of garbage that we've recovered from.
    if turn_gibberish_count >= 4:
        turn = 0.9
    elif turn_gibberish_count >= 3:
        turn = 0.6
    elif turn_gibberish_count >= 2:
        turn = 0.4
    else:
        turn = 0.0

    # Contrast: if the turn has produced MANY real words too, discount.
    # A turn with 10 real + 2 gibberish is in-distribution; a turn with
    # 2 real + 2 gibberish is actively pathological.
    if turn_real_count >= 8:
        turn *= 0.4
        sent *= 0.7
    elif turn_real_count >= 4:
        turn *= 0.7
        sent *= 0.85

    # Combine — recent short window gets the most weight; sentence
    # second; turn as background.
    signal = 0.55 * short + 0.30 * sent + 0.15 * turn
    if signal > 1.0:
        signal = 1.0
    if signal < 0.0:
        signal = 0.0
    return signal


def word_start_safe_bias(
    letter_run_len: int,
    last_char_class: int,
    speaker_label_state: int,
    turn_gibberish_count: int,
    turn_real_count: int,
    sentence_gibberish_count: int,
    recent_word_realities: tuple[int, ...],
) -> list[float] | None:
    """Bias the first letter of a new word toward safe word-starts
    when recent context has been gibberish-heavy."""
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None
    # Only at a position where a letter might plausibly follow.
    # last_char_class SPACE=1 or PUNCT_MID=7 (after ", " etc.).
    # We allow both; primary target is post-space.
    if last_char_class not in (1, 7):
        return None
    signal = _compute_signal(
        turn_gibberish_count,
        turn_real_count,
        sentence_gibberish_count,
        recent_word_realities,
    )
    if signal < 0.2:
        return None
    return _build_word_start_vec(signal)


def mid_word_close_boost(
    letter_run_len: int,
    on_word_trie: bool,
    letters_off_trie: int,
    speaker_label_state: int,
    turn_gibberish_count: int,
    sentence_gibberish_count: int,
    recent_word_realities: tuple[int, ...],
    word_red_flags: int,
    bad_bigram_count: int,
) -> list[float] | None:
    """When mid-word and off-trie AND the surrounding context has been
    gibberish-heavy, boost word-terminators to close this word FAST
    (before it becomes another gibberish entry)."""
    if speaker_label_state != 0:
        return None
    if on_word_trie:
        return None
    if letters_off_trie < 1:
        return None
    if letter_run_len < 3:
        return None

    # Require BOTH (1) historical gibberish context and (2) current-word
    # trouble signals to fire. Either alone is already covered by
    # existing layers (word_integrity_bias, gibberish_hardcap_bias,
    # offtrie_depart_bias). The novelty here is the CONJUNCTION: when
    # we have both "this turn has been garbage" AND "this word is
    # going off phonotactically", the termination pressure should
    # stack multiplicatively.
    recent = recent_word_realities[:3]
    recent_gib = sum(1 for r in recent if r == 3)

    # Historical signal (0..1).
    hist = 0.0
    if recent_gib >= 2:
        hist = 1.0
    elif recent_gib == 1:
        hist = 0.5
    if sentence_gibberish_count >= 3:
        hist = max(hist, 1.0)
    elif sentence_gibberish_count == 2:
        hist = max(hist, 0.7)
    elif sentence_gibberish_count == 1:
        hist = max(hist, 0.3)
    if turn_gibberish_count >= 4:
        hist = max(hist, 0.6)

    # Current-word trouble signal (0..1).
    cur = 0.0
    if bad_bigram_count >= 1:
        cur = max(cur, 0.7)
    if word_red_flags >= 2:
        cur = max(cur, 0.6)
    if word_red_flags >= 3:
        cur = 1.0

    # Fire only when BOTH signals are non-trivial.
    if hist < 0.3 or cur < 0.5:
        return None

    # Conjunction scale: small base times both signals.
    scale = 0.80 * hist * cur

    # Escalate with letter_run_len — the longer this word already is,
    # the more urgent closure.
    if letter_run_len >= 8:
        scale *= 1.7
    elif letter_run_len >= 6:
        scale *= 1.35
    elif letter_run_len >= 5:
        scale *= 1.15

    if scale <= 0.0:
        return None

    vec = [0.0] * VOCAB_SIZE
    if " " in VOCAB_INDEX:
        vec[VOCAB_INDEX[" "]] += scale * 1.0
    if "," in VOCAB_INDEX:
        vec[VOCAB_INDEX[","]] += scale * 0.55
    if "." in VOCAB_INDEX:
        vec[VOCAB_INDEX["."]] += scale * 0.50
    if ";" in VOCAB_INDEX:
        vec[VOCAB_INDEX[";"]] += scale * 0.30
    if "!" in VOCAB_INDEX:
        vec[VOCAB_INDEX["!"]] += scale * 0.30
    if "?" in VOCAB_INDEX:
        vec[VOCAB_INDEX["?"]] += scale * 0.25
    if "\n" in VOCAB_INDEX:
        vec[VOCAB_INDEX["\n"]] += scale * 0.40
    # Mild suppression of rare letters that would extend a gibberish run.
    for ch in ("j", "q", "x", "z", "v", "k"):
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] -= scale * 0.25
    return vec
