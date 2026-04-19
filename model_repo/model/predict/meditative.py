"""Meditative-register bias layer.

Reads `state.meditative_register` (a rolling [0, 1] float maintained by
the flow pipeline, rising on philosophical / inward-gazing vocabulary
like "think, soul, mind, doubt, dream, nature, reason, conscience,
memory, wonder, truth" and decaying slowly per word), and at word-start
biases the first-letter distribution toward meditative-leading words.

This is a flow/texture layer: the signal is continuous scene-level
mood, not a fixed table keyed on literal context. A speaker two words
into "To be, or not to be — that is the question; whether 'tis nobler
in the mind to suffer ..." sits in deep meditative register. The next
content word is far more likely to be abstract ("arrows", "fortune",
"sea", "troubles") than concrete-battlefield ("sword", "dagger",
"blood"). That bleed-through is what this layer captures.

Distinct from archaic_density (which captures early-modern grammar)
and emotional_intensity (outward exclamation). Meditative mode is
the soliloquy texture.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Per-letter lean at word-start, normalized by register. A value of
# +X means at register=1.0 we add +X log-bias to this letter at
# word-start. Values stay small (0.1-0.25) because register can sit
# at 0.5+ for long stretches and this stacks on top of startword,
# next_word, phrase_bigram, and archaic biases.
#
# Leans toward letters that begin the *meditative* lexicon: mental
# states (think/thought), reflective subjunctive framings (seems/
# suppose), the subjunctive/conditional triangle (if/whether/though).
_MEDITATIVE_START_LEAN: dict[str, float] = {
    "t": 0.14,   # think, thought, truth, thus, though
    "m": 0.18,   # mind, memory, meditate, methinks, muse, mortal, might
    "s": 0.10,   # soul, spirit, seems, self, suppose
    "d": 0.10,   # dream, doubt, destiny, death, deep
    "w": 0.12,   # wonder, wisdom, whether, were, would
    "r": 0.08,   # reason, reflect, remember
    "p": 0.10,   # ponder, perhaps, perceive, providence, philosophy
    "f": 0.07,   # fate, fortune, fool, folly
    "n": 0.06,   # nature, nothing, noble
    "i": 0.08,   # if, imagine, idea
    "h": 0.05,   # haply, heaven, how, hope
    # Gently pull down letters that open concrete-action words
    # (not absent from meditative prose, but less dominant).
    "b": -0.05,  # blood, battle, bear (concrete)
    "k": -0.05,  # kill, knock
    "g": -0.03,  # grim, grip (concrete)
}

# Lean mirrored at 0.55x for capital forms (sentence-start).
_CAPITAL_SCALE = 0.55


def meditative_start_bias(register: float) -> list[float]:
    """Return a VOCAB_SIZE bias vector weighted by register.

    Returns an all-zero vector when register <= 0.
    """
    if register <= 0.0:
        return [0.0] * VOCAB_SIZE
    vec = [0.0] * VOCAB_SIZE
    r = max(0.0, min(1.0, register))
    for ch, lean in _MEDITATIVE_START_LEAN.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = r * lean
        up = ch.upper()
        if up != ch and up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] = r * lean * _CAPITAL_SCALE
    return vec


# Mid-word disambiguation. When word_buffer matches one of these
# prefixes AND meditative_register is high, bias toward the
# meditative completion letter. Cases where a meditative word and
# a non-meditative word share a prefix.
#
# Scaled conservatively. Only fires when register > 0.
_MEDITATIVE_MIDWORD: dict[str, dict[str, float]] = {
    # "thou"/"though"/"thought": after "thou" the next char closes
    # (space/punct); but "though" takes "g", "thought" takes "g"
    # then "h", "t". After "tho" a meditative speaker may want
    # "u" (thou) OR "u" followed by "g" (though/thought). Keep gentle.
    "thoug": {"h": 0.35},   # though, thought — vs. thouge(nonsense)
    "thoh": {"t": 0.0},     # placeholder (harmless no-op)
    "thin": {"k": 0.20},    # think vs. thine (register decides)
    "mus": {"i": 0.15, "e": 0.10},  # musing / muse
    "pon": {"d": 0.25},     # ponder vs. pond/pone
    "per": {"h": 0.18, "c": 0.10},  # perhaps / perceive
    "hap": {"l": 0.25},     # haply vs. happy/happen
    "wond": {"e": 0.30},    # wonder / wondrous
    "medi": {"t": 0.35},    # meditate / meditation
    "refle": {"c": 0.30},   # reflect / reflection
    "nat": {"u": 0.20},     # nature vs. native (gentle)
    "cons": {"c": 0.25, "i": 0.15},  # conscience / consider
    "phi": {"l": 0.30},     # philosophy / philosopher
    "dou": {"b": 0.20},     # doubt / doubtful (vs. double)
    "des": {"t": 0.15},     # destiny vs. descend
    "etern": {"a": 0.35, "i": 0.25},  # eternal / eternity
}


def meditative_midword_bias(
    word_buffer: str, register: float
) -> list[float] | None:
    """Return a VOCAB_SIZE bias vector for meditative mid-word, or None.

    Returns None when the buffer is not in our table or register is 0.
    """
    if register <= 0.0 or not word_buffer:
        return None
    wb = word_buffer.lower()
    entry = _MEDITATIVE_MIDWORD.get(wb)
    if entry is None:
        return None
    r = max(0.0, min(1.0, register))
    vec = [0.0] * VOCAB_SIZE
    for ch, bump in entry.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = r * bump
    return vec
