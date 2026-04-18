"""Tonal-texture next-word first-letter bias.

Consumes `state.tonal_weight` (a rolling [-1, +1] float maintained by
the flow pipeline that captures whether the recent lexicon is
dark/heavy or light/hopeful) and, at word-start, biases the first
letter of the next word toward lexicon consistent with the register.

A tonal register in Shakespeare has strong lexical coherence: a
scene that has just said "blood", "death", and "grief" is far
more likely to say "dagger", "murder", "rage", "hell" next than
"love", "joy", "sweet". Conversely, once "love", "sweet", "joy"
appear, their continuation clusters too. This layer captures the
feel — not a hard constraint, a gentle drift proportional to the
rolling tone.

The bias is applied only at word-start positions (after space or
single newline), outside speaker-label territory, and scales with
|tonal_weight|.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Letters that commonly start DARK lexicon (blood/battle/death/grief/
# curse/night/hell/rage/dread/sorrow/villain...).
_DARK_LETTERS: dict[str, float] = {
    "b": 0.45,  # blood, battle, black, bleak, bitter, bone, bury, broken
    "d": 0.50,  # death, dead, dark, dagger, dreadful, doom, dismal
    "g": 0.35,  # grief, grave, gore, ghost, groan
    "h": 0.30,  # hell, hate, hatred, horror, heavy
    "m": 0.25,  # murder, mourn, monster, misery, malice
    "s": 0.25,  # sword, sorrow, slay, slaughter, sin, shame, sick, slain
    "w": 0.22,  # wound, woe, weep, weary, wrath, wretch
    "c": 0.30,  # curse, cruel, cold, corpse, coffin, crime
    "r": 0.22,  # rage, rend, rot, ruin, revenge
    "p": 0.25,  # pain, poison, plague, perish, pale
    "t": 0.18,  # tears, torment, tyrant, traitor, terror, tomb
    "f": 0.18,  # fear, foul, fatal, fiend, fall
    # Light-register letters are gently penalized when dark.
    "l": -0.18,  # love, light, laugh, lovely
    "j": -0.15,  # joy
    # Neutral letters: no bias.
}


# Letters that commonly start LIGHT lexicon (love/joy/fair/sweet/smile/
# gentle/pure/bless/heaven/grace/hope...).
_LIGHT_LETTERS: dict[str, float] = {
    "l": 0.45,   # love, light, laugh, lovely, lord (affection context)
    "j": 0.35,   # joy, joyful
    "s": 0.32,   # sweet, smile, soft, sing, sunshine
    "f": 0.32,   # fair, fond, friend, faith, festival, flower
    "g": 0.30,   # gentle, grace, good, golden, glad, gift
    "h": 0.25,   # heaven, heart, happy, hope
    "b": 0.22,   # bless, beauty, bright, bliss
    "m": 0.18,   # mirth, merry, music, mercy
    "p": 0.18,   # pure, peace, pretty, play
    "c": 0.15,   # cheer, calm, courteous
    "k": 0.15,   # kind, kindness, king (benevolent)
    # Dark-register letters are gently penalized when light.
    "d": -0.20,  # death, dark
    "w": -0.12,  # woe, wound
    "r": -0.08,  # rage, ruin
}


def _build_bias(letters: dict[str, float]) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch, lean in letters.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = lean
        up = ch.upper()
        if up in VOCAB_INDEX and up != ch:
            # Capitalized forms at sentence-start positions get 0.6x.
            vec[VOCAB_INDEX[up]] = lean * 0.6
    return vec


_DARK_BIAS: list[float] = _build_bias(_DARK_LETTERS)
_LIGHT_BIAS: list[float] = _build_bias(_LIGHT_LETTERS)


# Overall multiplier on |tonal_weight| → bias strength.
_SCALE: float = 1.0


def tonal_start_bias(tonal_weight: float) -> list[float] | None:
    """Return a VOCAB_SIZE bias vector at word-start, scaled by the
    rolling tonal_weight. None when |weight| is small (neutral).

    tonal_weight ∈ [-1, +1]: negative → dark bias; positive → light bias.
    """
    if tonal_weight == 0.0:
        return None
    w = max(-1.0, min(1.0, tonal_weight))
    if w <= 0.0:
        # Dark register: apply dark bias scaled by |w|.
        s = -w * _SCALE
        if s < 0.02:
            return None
        return [s * x for x in _DARK_BIAS]
    else:
        s = w * _SCALE
        if s < 0.02:
            return None
        return [s * x for x in _LIGHT_BIAS]
