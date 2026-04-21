"""Predict layer — emotional-valence word-start bias.

Reads `state.emotional_valence` in [-1.0, +1.0]. When the recent
content-word diction has drifted strongly positive (valence >= +0.25)
or negative (valence <= -0.25), this layer biases the NEXT word's
first letter toward first letters typical of valence-matching
vocabulary, and mildly suppresses opposite-polarity openers.

Gates:
  * speaker_label_state == 0
  * letter_run_len == 0 (word-start position)
  * last_char_class in (1, 7)  — post-space / mid-punct
  * |emotional_valence| >= 0.25

No corpus statistics — the first-letter weights are hand-authored
from prior knowledge of Early Modern English moral/affective
vocabulary.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# First letters that tilt POSITIVE in Shakespeare: letters whose
# common content-word starters (both lowercase and capitalized) land
# more often in positive diction than negative. Hand-graded magnitudes
# reflect how clear the tilt is.
#
# 'h' — honour, heaven, happy, hope, holy (strong positive; "hate"
#       is a single negative word against many positives).
# 'g' — grace, good, glad, gentle, gold, glory, gracious (positive
#       dominates, despite "grief", "guilt").
# 'k' — kind, kindness (clean positive; "kill" is a common verb
#       but doesn't carry strong negative valence by itself — we
#       tag it via "murder" / "slay" / "blood" stems).
# 'l' — love, lovely, loyal, loyalty (positive dominates; "lie"
#       ambiguous, already excluded).
# 'b' — blessed, bliss, bright, brave, beauty (positive leaning
#       despite "base", "blood", "betray").
# 's' — sweet, saint, smile, sure, sincere (positive lean; "sin",
#       "shame", "scorn" negative — net slightly positive).
# 'p' — peace, pure, praise, pity (positive lean).
# 'm' — mercy, meek, merry, mild (positive; "murder" is strong but
#       "mercy" and "mild" are both frequent).
# 'v' — virtue, valiant, victor (positive; "vice", "vile" — mixed,
#       use caution).
# 't' — true, truth, truly, trust, tender, triumph (positive).
_POSITIVE_STARTERS: dict[str, float] = {
    # Clearly-positive tilts only. Cut 's' (sin/shame), 'b' (base/
    # blood), 'm' (murder), 't' (treachery) which had strong negative
    # counterparts that polluted the tilt.
    "h": 0.45,   # honour, heaven, happy, hope, holy — 'hate' is the
                 # only strong negative; net strongly positive.
    "g": 0.45,   # grace, good, glad, gentle, gold, glory — 'grief',
                 # 'guilt' lighter; net positive.
    "k": 0.40,   # kind, kindly — very clean.
    "l": 0.30,   # love, loyal, lovely — 'lie' excluded already.
    "p": 0.35,   # peace, pure, praise, pity — 'pain', 'poison'
                 # negative but less frequent.
}

# First letters that tilt NEGATIVE: letters whose common content
# starters in Shakespeare land more often in negative / morally
# heavy vocabulary.
#
# 'f' — foul, false, filth, fiend, fear (strong negative; "fair"
#       excluded as ambiguous).
# 'c' — cruel, corrupt, cursed, coward (negative leaning).
# 'w' — wretched, wicked, woe, wrong, weep (strongly negative;
#       "wise", "wonder" positive but less frequent).
# 'r' — rude, rank, rotten, rage (negative lean; some neutral).
# 'd' — damned, dark, dreadful, devil, death (strong negative;
#       despite "dear", "divine" positive).
# 'v' — vile, vice, venom, villain (negative; overlaps with positive
#       tier — handle via smaller magnitude).
_NEGATIVE_STARTERS: dict[str, float] = {
    # Clearly-negative tilts only. Cut 'c' (courage/care) and 'v'
    # (virtue/valiant) which had strong positive counterparts.
    "f": 0.35,   # foul, false, filth, fiend, fear — 'fair' excluded.
    "w": 0.40,   # wretched, wicked, woe, wrong, weep — 'wise',
                 # 'wonder' lighter; net negative.
    "d": 0.30,   # damned, dark, dreadful, devil, death — 'dear',
                 # 'divine' positive but less frequent in negative
                 # passages.
}


def valence_start_bias(
    emotional_valence: float,
    letter_run_len: int,
    last_char_class: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None
    # 1 = SPACE, 7 = PUNCT_MID.
    if last_char_class not in (1, 7):
        return None
    v = emotional_valence
    if abs(v) < 0.25:
        return None

    # Scale by valence magnitude with saturation.
    mag = min(abs(v), 1.0)
    # scale ramps from ~0.15 at |v|=0.25 to ~0.50 at |v|=1.0.
    # Conservative so the bias registers without swamping strong
    # per-letter POS / trie / phonotactic priors.
    scale = 0.15 + 0.35 * ((mag - 0.25) / 0.75)

    vec = [0.0] * VOCAB_SIZE
    if v > 0:
        primary = _POSITIVE_STARTERS
        rival = _NEGATIVE_STARTERS
    else:
        primary = _NEGATIVE_STARTERS
        rival = _POSITIVE_STARTERS

    for ch, w in primary.items():
        for c in (ch, ch.upper()):
            idx = VOCAB_INDEX.get(c)
            if idx is not None:
                vec[idx] += scale * w

    # Mild suppression of opposite-polarity openers. Keep penalties
    # small — word-start already has many strong priors; we don't
    # want to swamp trie / POS / phonotactic signals.
    for ch, w in rival.items():
        for c in (ch, ch.upper()):
            idx = VOCAB_INDEX.get(c)
            if idx is not None:
                vec[idx] -= scale * w * 0.5

    return vec
