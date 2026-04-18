"""Iambic-foot word-start bias.

Unexplored axis: metrical stress position within the iambic line.
Shakespeare's verse is dominantly iambic pentameter — positions 1,3,
5,7,9 are unstressed (weak beats), positions 2,4,6,8,10 are stressed
(strong beats). Function words (articles, prepositions, possessives,
conjunctions) overwhelmingly land on UNSTRESSED beats; content words
(nouns, verbs, adjectives with full lexical semantics) land on
STRESSED beats.

Existing state tracks `syllables_in_line` (a syllable-position
counter) but no layer conditions word choice on the expected stress
of the NEXT syllable. This layer reads syllables_in_line and, at
word-start in confident iambic contexts, biases the first letter
toward either function-word or content-word starter families based
on parity of the next syllable position.

Fires only when:
  - verse_score is high (>= 0.6, confident verse)
  - verse_line_run >= 2 (established verse run)
  - prev_line_syllables in {9, 10, 11} (anchor line was pentameter)
  - speaker_label_state == 0
  - We're at a word-start (caller guards)

All weights from prior knowledge of English prosody — no corpus
statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# Function-word starter letters — favored on UNSTRESSED positions.
# These letters open monosyllabic function words that typically sit
# on weak beats of iambic verse: the/to/that/a/and/of/in/is/it/with/
# for/from/my/his/her/he/by/be/but/upon/unto/yet/no/nor/so/as.
_FUNC_STARTERS: dict[str, float] = {
    "t": 0.8,   # the, to, that, this, thou, thee, thy, though
    "a": 0.7,   # a, and, as, at, an, after, am, art, all
    "o": 0.6,   # of, on, o'er, or, O, oh, our
    "i": 0.6,   # in, is, it, if, I
    "w": 0.55,  # with, when, which, what, we, was, were, will, would
    "f": 0.45,  # for, from
    "m": 0.45,  # my, me
    "h": 0.55,  # his, her, he, him, hath, have, has, how, here
    "b": 0.40,  # by, be, but, been, being, before
    "u": 0.30,  # upon, us, unto, under
    "y": 0.25,  # you, your, yet, ye
    "n": 0.25,  # nor, not, no
    "s": 0.25,  # so, shall, should, she
}

# Content-word starter letters — favored on STRESSED positions.
# These open polysyllabic or semantically-weighty monosyllables
# (nouns, verbs, adjectives) that typically sit on strong beats.
_CONTENT_STARTERS: dict[str, float] = {
    "l": 0.7,   # lord, love, life, light, look, leave, live, lie
    "s": 0.55,  # sword, sun, soul, sweet, seek, speak, stand, strike
    "k": 0.55,  # king, knight, kill, know, kiss, kneel
    "g": 0.50,  # god, good, great, grace, go, give, grant
    "f": 0.45,  # fair, friend, father, fire, find, feel, fall, fear
    "b": 0.55,  # blood, beauty, bear, break, bring, beat, bid
    "h": 0.45,  # heart, heaven, hand, head, home, hold, hell, hate
    "d": 0.50,  # death, day, dear, do, die, draw, dread
    "r": 0.40,  # rose, royal, right, rise, run, rage
    "c": 0.50,  # crown, come, call, cry, carry, cold, curse
    "p": 0.45,  # pride, praise, peace, power, pray, poor
    "w": 0.40,  # war, word, woman, wife, world, weep, wake
    "m": 0.35,  # mind, man, make, mother, meet, mourn
    "n": 0.30,  # night, noble, name, need, near
    "t": 0.35,  # tears, throne, truth, tread, trust (content t-words)
    "v": 0.30,  # voice, virtue, vengeance
    "j": 0.25,  # joy, judge, journey
}


def _build_pair() -> tuple[list[float], list[float]]:
    """Return (unstressed_bias, stressed_bias) vectors."""
    SCALE = 0.25
    ANTI_SCALE = 0.08
    unstressed = [0.0] * VOCAB_SIZE
    stressed = [0.0] * VOCAB_SIZE

    for ch, w in _FUNC_STARTERS.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            unstressed[idx] += SCALE * w
            stressed[idx] -= ANTI_SCALE * w
        up = ch.upper()
        up_idx = VOCAB_INDEX.get(up)
        if up_idx is not None:
            unstressed[up_idx] += SCALE * w * 0.6
            stressed[up_idx] -= ANTI_SCALE * w * 0.6

    for ch, w in _CONTENT_STARTERS.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            stressed[idx] += SCALE * w
            unstressed[idx] -= ANTI_SCALE * w
        up = ch.upper()
        up_idx = VOCAB_INDEX.get(up)
        if up_idx is not None:
            stressed[up_idx] += SCALE * w * 0.6
            unstressed[up_idx] -= ANTI_SCALE * w * 0.6

    return unstressed, stressed


_UNSTRESSED_BIAS, _STRESSED_BIAS = _build_pair()


def iambic_word_start_bias(
    syllables_in_line: int,
    verse_score: float,
    verse_line_run: int,
    prev_line_syllables: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a bias vector for first-letter at word-start in verse.

    Returns None when we're not in a confident iambic context.
    """
    if speaker_label_state != 0:
        return None
    # Need confident verse signal.
    if verse_score < 0.6:
        return None
    if verse_line_run < 2:
        return None
    # Need a pentameter-anchor line (9-11 syllables).
    if not (9 <= prev_line_syllables <= 11):
        return None
    # Line-opening (syllables_in_line == 0): trochaic inversion is
    # common at line start ("Shall I compare thee..." opens trochaic
    # DUM-da). Don't bias at position 0 — the prior is too uncertain.
    if syllables_in_line == 0:
        return None
    # Late-line positions (7+): the signal weakens as the line
    # approaches its end where natural closure dominates.
    if syllables_in_line >= 9:
        return None

    # Next syllable is at position (syllables_in_line + 1).
    # Iambic: position-1=unstressed, position-2=stressed, ...
    next_pos = syllables_in_line + 1
    if next_pos % 2 == 1:
        # Next beat is unstressed — prefer function-word starters.
        return _UNSTRESSED_BIAS
    else:
        # Next beat is stressed — prefer content-word starters.
        return _STRESSED_BIAS
