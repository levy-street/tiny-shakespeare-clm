"""Word-length cadence word-start bias.

Unexplored axis: prosodic rhythm of recent word lengths.

Shakespeare (and natural English) alternates between short and long
words with characteristic rhythm. Two opposite failure modes can
emerge if we don't track it:

  * A runaway monosyllabic chain ("be or of to in a is at", as can
    happen when function-word priors over-fire) — stylistically flat
    and semantically empty.
  * A runaway polysyllabic chain ("multitudinous declamatory
    incarnadine magnificent") — real Shakespeare has bursts of
    this, but not 6 in a row.

This layer reads `state.recent_word_lengths` (rolling tuple, most-
recent LAST, capped at 6) and at word-start applies a SMALL first-
letter bias tilting toward the kind of word that balances the recent
cadence.

Signals computed:

  mean_len   — mean of recent_word_lengths
  trailing3  — mean of last 3 (if available)
  mono_run   — count of trailing consecutive 1-2 letter words
  poly_run   — count of trailing consecutive 7+ letter words

Bias logic:

  mono_run >= 3  →  next word should be longer. Boost CONTENT-word
                    first-letters (l/s/b/d/k/c/g/h/f/p/r/m — opens
                    nouns/verbs/adjectives), suppress function-word
                    shorts (a/A, o/O solo, I — but keep t because
                    many content words open with t).

  poly_run >= 2  →  next word is very likely short. Boost function-
                    word first-letters (t/a/i/o/h/w/b/f/m), mildly
                    suppress polysyllable-opening clusters.

  mean_len very high (>= 6.0) with trailing3 >= 6 → same as poly_run.

  mean_len very low (<= 2.6) with trailing3 <= 2.6 → same as mono_run.

  Otherwise → no bias (we're in the healthy alternating zone).

Amplitude is deliberately small (0.10-0.22) — this is a GENTLE
rhythm tilt, not a hard constraint. The goal is to make samples
feel paced correctly, not to override stronger lexical signals.

No corpus statistics — thresholds and letter families come from
prior knowledge of English word-length distribution and Shakespeare's
rhetorical rhythm.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Content-word first-letters: letters that predominantly open nouns,
# verbs, adjectives — semantically weighty words typically polysyllabic
# OR monosyllabic-but-stressed.
_CONTENT_BOOST: dict[str, float] = {
    "l": 0.35,   # love, lord, life, light, look, leave, live, lie
    "s": 0.30,   # sword, sun, soul, sweet, seek, speak, stand
    "b": 0.30,   # blood, beauty, bear, break, bring, beat, bid
    "d": 0.30,   # death, day, dear, do, die, draw, dread
    "k": 0.25,   # king, knight, kill, know, kiss, kneel
    "c": 0.30,   # crown, come, call, cry, curse, carry, cold
    "g": 0.25,   # god, good, great, grace, give, grant
    "h": 0.25,   # heart, heaven, hand, head, home, hold, hell
    "f": 0.25,   # fair, friend, father, fire, find, feel, fall
    "p": 0.25,   # pride, praise, peace, power, pray, poor
    "r": 0.22,   # rose, royal, right, rise, run, rage
    "m": 0.22,   # mind, man, make, mother, meet, mourn
    "n": 0.18,   # night, noble, name, need, near
    "v": 0.18,   # voice, virtue, vengeance
    "j": 0.15,   # joy, judge, journey
}

# Function-word first-letters: letters that predominantly open short
# closed-class words (articles, prepositions, conjunctions, pronouns,
# auxiliaries).
_FUNCTION_BOOST: dict[str, float] = {
    "t": 0.35,   # the, to, that, thou, thee, thy, though, this
    "a": 0.30,   # a, and, as, at, an, am, art
    "i": 0.28,   # in, is, it, if, I
    "o": 0.28,   # of, or, on, O, oh, our
    "h": 0.20,   # he, his, her, him, have, hath
    "w": 0.25,   # with, when, which, what, we, will, would
    "b": 0.18,   # by, be, but, been
    "m": 0.18,   # my, me
    "f": 0.15,   # for, from
    "y": 0.15,   # you, your, ye, yet
    "n": 0.15,   # nor, not, no
}

# Letters we mildly suppress in the OPPOSITE direction: under mono_run
# we reduce solo-monosyllable starters; under poly_run we reduce
# polysyllable-prone starters.
_MONO_RUN_DAMP: dict[str, float] = {
    # Solo-word starters (single-letter words or ultra-short words)
    # that further a monosyllabic run when we want to break out.
    "A": -0.18,  # A, An
    "a": -0.12,  # a, at, an, am
    "I": -0.20,  # I (standalone pronoun)
    "O": -0.18,  # O (invocation — standalone)
    "o": -0.10,  # o'er, on, of
    # Also dampen very short aux/pronoun starters mildly.
    "t": -0.05,  # to, the (but many content t-words too — mild)
}

_POLY_RUN_DAMP: dict[str, float] = {
    # Letters that disproportionately open Latinate polysyllables.
    "c": -0.12,  # compassion, conspiracy, circumstance
    "p": -0.10,  # perpetual, providential
    "d": -0.08,  # devotion, determination
    "m": -0.08,  # magnificent, miserable
    "i": -0.06,  # ignominious, importunate
    "s": -0.06,  # supposition, subjugation
    "v": -0.10,  # voluminous, vitiating
    "j": -0.05,
    "e": -0.08,  # excellent, exorbitant
    "u": -0.08,  # ungovernable, undulatory
}


def _build_vec(which: str) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    if which == "content":
        for ch, w in _CONTENT_BOOST.items():
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += w
            up = VOCAB_INDEX.get(ch.upper())
            if up is not None:
                vec[up] += w * 0.55
        for ch, w in _MONO_RUN_DAMP.items():
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += w  # w is negative
    elif which == "function":
        for ch, w in _FUNCTION_BOOST.items():
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += w
            up = VOCAB_INDEX.get(ch.upper())
            if up is not None:
                vec[up] += w * 0.55
        for ch, w in _POLY_RUN_DAMP.items():
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += w
    return vec


_CONTENT_VEC = _build_vec("content")
_FUNCTION_VEC = _build_vec("function")


def word_length_cadence_bias(
    recent_word_lengths: tuple[int, ...],
    speaker_label_state: int,
) -> list[float] | None:
    """Return a small first-letter bias at word-start based on the
    shape of recent_word_lengths. None when not confident enough to
    fire. Caller gates on letter_run_len == 0."""
    if speaker_label_state != 0:
        return None
    n = len(recent_word_lengths)
    if n < 3:
        return None

    # Trailing runs of extreme-length words.
    mono_run = 0
    for L in reversed(recent_word_lengths):
        if L <= 2:
            mono_run += 1
        else:
            break
    poly_run = 0
    for L in reversed(recent_word_lengths):
        if L >= 7:
            poly_run += 1
        else:
            break

    # Rolling means.
    trailing3 = sum(recent_word_lengths[-3:]) / 3.0
    mean_len = sum(recent_word_lengths) / n

    # Strong mono signal: break out with a content-word letter.
    if mono_run >= 3:
        scale = min(1.6, 0.7 + 0.3 * (mono_run - 2))
        return [v * scale for v in _CONTENT_VEC]

    # Strong poly signal: rebalance with a function-word letter.
    if poly_run >= 2:
        scale = min(1.6, 0.8 + 0.35 * (poly_run - 1))
        return [v * scale for v in _FUNCTION_VEC]

    # Soft signals: sustained means beyond the comfortable zone.
    if mean_len <= 2.7 and trailing3 <= 2.7:
        return [v * 0.70 for v in _CONTENT_VEC]
    if mean_len >= 6.2 and trailing3 >= 6.0:
        return [v * 0.70 for v in _FUNCTION_VEC]

    return None
