"""Tier 3 flow: gravitas register — moral / philosophical weight.

Maintains `state.gravitas_register` ∈ [0, 1]. Rises on abstract-noun
moral/cosmic lexicon (honour, virtue, soul, duty, heaven, fate,
conscience, truth, justice); falls on concrete/quotidian lexicon
(drink, bread, bed, jest, fool). Distinct from lament (grief) and
tenderness (love): gravitas is the *seriousness* of the thought, not
its valence.

Consumed by predict/gravitas.py.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


# Gravitas-core — strongest abstract-moral signal.
_GRAVITAS_CORE: frozenset[str] = frozenset({
    "honour", "honor", "honours", "honors", "honoured", "honored",
    "virtue", "virtues", "virtuous", "virtuously",
    "soul", "souls",
    "duty", "duties", "dutiful", "dutifully",
    "conscience", "consciences",
    "truth", "truths", "truly",
    "justice", "unjust", "just",
    "reason", "reasons", "unreason",
    "fate", "fates", "fated",
    "doom", "doomed",
    "mortal", "mortals", "mortality",
    "immortal", "immortality",
    "heaven", "heavens", "heavenly",
    "earth", "earthly",
    "nature",
    "god", "gods", "godly",
    "divine", "divinity",
    "eternal", "eternally", "eternity",
    "perpetual", "perpetually",
    "everlasting",
})

# Gravitas-halo — softer abstract texture.
_GRAVITAS_HALO: frozenset[str] = frozenset({
    "honest", "honesty", "honestly",
    "faith", "faithful", "faithless", "faithfully",
    "sin", "sins", "sinful", "sinless",
    "holy", "sacred", "sanctify",
    "blest", "blessed", "blessing", "bless",
    "curse", "cursed", "cursing",
    "grace", "graces", "gracious", "graciously",
    "mercy", "merciful", "merciless",
    "pity", "pitiful", "piteous",
    "shame", "shameful", "shamed",
    "glory", "glorious",
    "power", "powers", "powerful",
    "spirit", "spirits", "spiritual",
    "right", "rights", "wrong", "wrongs",
    "deed", "deeds",
    "life", "lives", "death", "dying", "dead",
    "world", "worlds", "worldly",
    "time", "times",
    "will",
    "peace", "peaceful",
    "war", "wars",
    "crown", "crowns", "crowned",
})

# Anti-gravitas — concrete / quotidian / frivolous lexicon.
_ANTI_GRAVITAS: frozenset[str] = frozenset({
    "drink", "drinks", "drank", "drunk",
    "cup", "cups",
    "ale", "beer", "wine", "wines",
    "meat", "meats",
    "bread",
    "bed", "beds",
    "sleep", "slept",
    "laugh", "laughs", "laughed", "laughing",
    "jest", "jests", "jested", "jesting",
    "fool", "fools", "foolish",
    "merry", "merrily",
    "sport", "sports",
})

_CORE_BUMP = 0.18
_HALO_BUMP = 0.09
_ANTI_BUMP = -0.06
_DECAY = 0.95
_TURN_SCALE = 0.55


def update_gravitas(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    grav = state.gravitas_register

    # Speaker-turn boundary: partial reset (new speaker may inherit).
    if ch == "\n" and state.consecutive_newlines >= 2:
        new_grav = grav * _TURN_SCALE
        if abs(new_grav - grav) > 1e-6 or grav > 0.0:
            return state.model_copy(update={"gravitas_register": new_grav})
        return state

    # On just_finished_word: apply bump + decay.
    if state.just_finished_word and state.last_completed_word:
        word = state.last_completed_word.lower().strip("'")
        bump = 0.0
        if word in _GRAVITAS_CORE:
            bump = _CORE_BUMP
        elif word in _GRAVITAS_HALO:
            bump = _HALO_BUMP
        elif word in _ANTI_GRAVITAS:
            bump = _ANTI_BUMP

        new_grav = grav * _DECAY + bump
        if new_grav < 0.0:
            new_grav = 0.0
        elif new_grav > 1.0:
            new_grav = 1.0

        if abs(new_grav - grav) > 1e-6:
            return state.model_copy(update={"gravitas_register": new_grav})

    return state
