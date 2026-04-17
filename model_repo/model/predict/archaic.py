"""Archaic-register bias layer.

Reads `state.archaic_density` (a rolling [0, 1] float maintained by
the flow pipeline that rises on archaic markers like "thou/hath/doth/
prithee/anon/forsooth" and decays over tokens) and, at word-start,
biases the first-letter distribution toward archaic-leading words.
Also provides mid-word disambiguation when the buffer is a prefix of
both an archaic word and a modern word.

This is a flow/texture layer: the signal is continuous mood, not a
fixed table keyed on literal context. A speaker who just said
"prithee, my lord, 'tis" has a much higher probability of saying
"thou" next than one who said "I, my lord, will". That register
bleed-through is what this layer captures.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Per-letter lean at word-start, normalized by density. A value of +X
# means: at density=1.0, add +X log-bias to this letter at word-start.
# Values are soft (0.2-0.7 range) because density <= 1.0 keeps this
# gentle vs. stacked on top of startword / next_word / phrase_bigram.
_ARCHAIC_START_LEAN: dict[str, float] = {
    # Only the very strongest leaners — and small. Most archaic words
    # are on the existing trie already; this layer only nudges the
    # ambiguous boundary cases.
    "t": 0.25,   # thou, thee, thy, thine, 'tis, 'twas
    "h": 0.18,   # hath, hast, hither
    "w": 0.10,   # wherefore, whence, whither, wilt
    # Pull down "y" (modern "you/your") very slightly.
    "y": -0.08,
}

# Archaic letter-bias also applies to the capital forms (line-start
# / sentence-start positions where capitals are expected). We mirror
# at 0.6x the lowercase lean.
_CAPITAL_SCALE = 0.6


def archaic_start_bias(density: float) -> list[float]:
    """Return a VOCAB_SIZE bias vector weighted by density."""
    if density <= 0.0:
        return [0.0] * VOCAB_SIZE
    vec = [0.0] * VOCAB_SIZE
    d = max(0.0, min(1.0, density))
    for ch, lean in _ARCHAIC_START_LEAN.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = d * lean
        up = ch.upper()
        if up != ch and up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] = d * lean * _CAPITAL_SCALE
    return vec


# Mid-word archaic disambiguation. When word_buffer matches one of
# these prefixes AND archaic_density is high, bias toward the archaic
# completion character. These are cases where both an archaic word
# and a modern word share the same prefix.
#
# Keys: lowercase word_buffer prefix.
# Values: { char: logit_bump_at_density_1.0 }.
_ARCHAIC_MIDWORD: dict[str, dict[str, float]] = {
    # Archaic-specific completions. These fire in proportion to
    # archaic_density so modern registers are unaffected. Scaled
    # conservatively because some prefixes are compatible with both
    # archaic and modern continuations.
    "tho": {"u": 0.55},  # thou (nudge over those/though)
    "hit": {"h": 0.30},  # hither (over "hit")
    "met": {"h": 0.40},  # methinks
    "pri": {"t": 0.40},  # prithee
    "fors": {"o": 0.60},  # forsooth
    "quo": {"t": 0.45},  # quoth
    "ano": {"n": 0.35},  # anon
    "whi": {"l": 0.25},  # whilst
    "whe": {"n": 0.10, "r": 0.10},
    "pri'": {"t": 0.35},  # 'prithee-like
    "thi": {"n": 0.12},  # thine / this — gentle
    "alac": {"k": 0.50},  # alack
    "fie": {},  # no extension
    "sirra": {"h": 0.60},  # sirrah
    "met'": {"h": 0.45},  # methinks alt
}


def archaic_midword_bias(word_buffer: str, density: float) -> list[float] | None:
    """Return a VOCAB_SIZE bias vector for archaic mid-word completion,
    or None if the buffer is not in our table or density is 0."""
    if density <= 0.0 or not word_buffer:
        return None
    wb = word_buffer.lower()
    entry = _ARCHAIC_MIDWORD.get(wb)
    if entry is None:
        return None
    d = max(0.0, min(1.0, density))
    vec = [0.0] * VOCAB_SIZE
    for ch, bump in entry.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = d * bump
    return vec
