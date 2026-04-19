"""Tier 3 flow: tenderness register — rolling love/romance texture.

Maintains `state.tenderness_register` ∈ [0, 1]. Rises on love/sweet/
fair/gentle/dear lexicon; falls on war/blood/rage lexicon; mildly
anti-correlated with grief lexicon.

Distinct from tonal_weight (generic valence), imagery_density
(sensory concreteness), and lament_register (grief, opposite pole).

Consumed by predict/tenderness.py.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


# Tender-core — strongest romance signal.
_TENDER_CORE: frozenset[str] = frozenset({
    "love", "loves", "loved", "loving", "lover", "lovers", "beloved",
    "sweet", "sweets", "sweetly", "sweeter", "sweetest",
    "dear", "dearest",
    "darling",
    "fair", "fairer", "fairest",
    "beauty", "beauteous",
    "kiss", "kisses", "kissed", "kissing",
    "gentle", "gently", "gentler",
    "mild", "mildly",
    "tender", "tenderly", "tenderness",
    "soft", "softly", "softer",
    "fond", "fondly",
    "kind", "kindly", "kindness",
    "charming", "charm",
    "angel", "angels", "angelic",
    "flower", "flowers",
    "rose", "roses",
    "bosom",
    "mistress",
})

# Tender-halo — softer atmosphere.
_TENDER_HALO: frozenset[str] = frozenset({
    "cheek", "cheeks",
    "eye", "eyes",
    "lip", "lips",
    "heart",
    "bright",
    "blossom", "blossoms",
    "delight", "delights", "delightful",
    "grace", "graces", "gracious",
    "heaven", "heavens",
    "true",
    "mine",
    "divine",
})

# Anti-tender (violence/hate) — strong negative.
_ANTI_TENDER: frozenset[str] = frozenset({
    "war", "wars",
    "blood", "bloody",
    "sword", "swords",
    "arms",
    "battle", "battles",
    "slain", "slay", "slays", "slew",
    "kill", "kills", "killed", "killing",
    "strike", "strikes", "struck", "striking",
    "rage", "raging",
    "hate", "hates", "hated",
    "wrath", "wrathful",
    "fury", "furious",
    "curse", "cursed", "curs'd", "cursing",
    "foe", "foes",
    "enemy", "enemies",
    "foul",
    "rotten",
    "venom", "poison",
})

# Anti-tender-mild (grief lexicon) — opposite pole from lament.
_ANTI_TENDER_MILD: frozenset[str] = frozenset({
    "death", "dying", "dead",
    "dread",
    "grief", "griefs",
    "woe",
    "sorrow", "sorrows",
    "tears",
})

_CORE_BUMP = 0.15
_HALO_BUMP = 0.08
_ANTI_BUMP = -0.12
_ANTI_MILD_BUMP = -0.04
_DECAY = 0.93
_TURN_SCALE = 0.40


def update_tenderness(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    ten = state.tenderness_register

    # Speaker-turn boundary.
    if ch == "\n" and state.consecutive_newlines >= 2:
        new_ten = ten * _TURN_SCALE
        if abs(new_ten - ten) > 1e-6 or ten > 0.0:
            return state.model_copy(update={"tenderness_register": new_ten})
        return state

    # On just_finished_word: apply bump + decay.
    if state.just_finished_word and state.last_completed_word:
        word = state.last_completed_word.lower().strip("'")
        bump = 0.0
        if word in _TENDER_CORE:
            bump = _CORE_BUMP
        elif word in _TENDER_HALO:
            bump = _HALO_BUMP
        elif word in _ANTI_TENDER:
            bump = _ANTI_BUMP
        elif word in _ANTI_TENDER_MILD:
            bump = _ANTI_MILD_BUMP

        new_ten = ten * _DECAY + bump
        if new_ten < 0.0:
            new_ten = 0.0
        elif new_ten > 1.0:
            new_ten = 1.0

        if abs(new_ten - ten) > 1e-6:
            return state.model_copy(update={"tenderness_register": new_ten})

    return state
