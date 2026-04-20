"""Confessional-intimacy ↔ public-declamation register word-start bias.

Reads `state.confessional_intimacy` (flow-tier scalar in [-1, +1])
and applies a first-letter bias at word-start outside speaker labels.

The bias is symmetric — one vector direction for each pole — with
deadband near zero to avoid spurious nudges when the register is not
committed.

Bias magnitude ramps with |intensity|:
  |intensity| < 0.25 → no bias (deadband)
  0.25–0.6          → small tilt (scale 0.35 …  0.80)
  0.6–1.0           → confident tilt (scale 0.80 … 1.10)

No corpus statistics — the letter families come from prior knowledge
of English / Shakespeare lexicon commonly opening words in each
register.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Confessional-register starter letters — favored when intimacy > 0.
# These open the confessional lexicon: I/me/my, thou/thee/thy, heart/
# hope/happy, soul/sorrow/sigh, dream/doubt/death-intime, remember,
# feel/fear, alas/ah/oh.
_CONF_BOOST: dict[str, float] = {
    # Pronouns & intimate address — strongest.
    "I": 0.55,  # I (standalone)
    "i": 0.35,  # ish, it (weaker, lowercase less specific)
    "m": 0.45,  # my, me, mine, myself
    "t": 0.40,  # thou, thee, thy, thine, think, thought
    # Interior-noun lexicon.
    "h": 0.40,  # heart, hope, hollow, heaven
    "s": 0.38,  # soul, sorrow, sigh, silent, soft
    "d": 0.35,  # dream, doubt, death, dear, deep
    "r": 0.28,  # remember, rest, repent, rue
    "w": 0.28,  # weep, wish, wonder, woe
    "f": 0.25,  # fear, feel, felt, fond
    # Sigh / exclamation openers.
    "O": 0.45,  # O (intimate sigh opener)
    "A": 0.35,  # Alas, Ah
    "a": 0.20,  # alas, ah (lowercase after punct)
    # Prayerful / soft.
    "p": 0.20,  # pray, prithee, pity
}

# Public-register starter letters — favored when intimacy < 0.
# These open the public / oratorical / ceremonial lexicon: we/our,
# you/your/ye, lords/friends/gentlemen, hear/hark/behold, majesty/
# grace/highness/lord, march/fight/strike.
_PUB_BOOST: dict[str, float] = {
    "w": 0.45,  # we, our (no but our starts o), will (plural), witness
    "y": 0.50,  # you, your, ye
    "o": 0.35,  # our, ourselves, once (ceremonial open)
    "l": 0.40,  # lords, liege, lord
    "f": 0.40,  # friends, fellows, fight
    "g": 0.35,  # gentlemen, grace
    "h": 0.30,  # hearken, hear, highness, honour
    "m": 0.30,  # majesty, march, masters, mark
    "b": 0.30,  # brethren, brothers, behold
    "c": 0.30,  # countrymen, citizens, command, charge
    "s": 0.30,  # sirs, soldiers, subjects, speak, strike
    "n": 0.25,  # now, noble
    "k": 0.22,  # king, know (proclamatory)
    # Capital versions for line openers.
    "W": 0.35, "Y": 0.35, "O": 0.25, "L": 0.30, "F": 0.30,
    "G": 0.25, "H": 0.22, "M": 0.22, "B": 0.22, "C": 0.22,
    "S": 0.22, "N": 0.20, "K": 0.18,
}


def _build_vec(pole: str) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    src = _CONF_BOOST if pole == "conf" else _PUB_BOOST
    anti = _PUB_BOOST if pole == "conf" else _CONF_BOOST
    for ch, w in src.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w
        # Upper-case mirror unless the key is already upper.
        if ch.isalpha() and ch.islower():
            up = VOCAB_INDEX.get(ch.upper())
            if up is not None:
                vec[up] += w * 0.5
    # No anti-bias — many letters (t/m/s/h) legitimately open words
    # in both registers. The positive boost on the in-register letters
    # is enough tilt; penalizing opposite letters was over-reaching.
    return vec


_CONF_VEC = _build_vec("conf")
_PUB_VEC = _build_vec("pub")


def confessional_word_start_bias(
    confessional_intimacy: float,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a word-start bias vec tilting toward the committed
    register pole. None if register is in deadband or inside speaker
    label FSM."""
    if speaker_label_state != 0:
        return None
    ci = confessional_intimacy
    mag = abs(ci)
    if mag < 0.25:
        return None

    # Ramp: 0.25 → scale 0.35; 0.6 → scale 0.80; 1.0 → scale 1.10.
    if mag >= 0.6:
        scale = 0.32 + (mag - 0.6) * 0.35  # up to ~0.46
    else:
        scale = 0.10 + (mag - 0.25) * 0.63  # 0.10 → 0.32

    if ci > 0:
        return [v * scale for v in _CONF_VEC]
    else:
        return [v * scale for v in _PUB_VEC]
