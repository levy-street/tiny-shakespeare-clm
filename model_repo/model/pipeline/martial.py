"""Tier 3 — martial / battlefield register charge.

Shakespeare's battlefield, war-scene and crisis speeches have a
distinctive lexicon — swords, blood, wounds, arms, soldiers, drums —
that CLUSTERS: once a scene goes martial, it stays martial for a
while. Similarly, pastoral / peaceful scenes (Midsummer wood,
chamber scenes, benedictions) cluster in the opposite direction.

Maintains `state.martial_charge`, a rolling float in roughly
[-2.0, +3.0]. On each completed content word:
  - If the word is in the curated MARTIAL vocabulary, add +0.70.
  - If the word is in the curated PEACEFUL / PASTORAL vocabulary,
    add -0.40.
  - All words decay the charge by ×0.93 so influence fades.
  - Clamp to [-2.0, +3.0].
  - Reset to 0 on speaker-turn boundary.

Runs late enough that `just_finished_word` and `last_completed_word`
are current (after update_linguistic). No corpus statistics — the
vocabularies are prior-knowledge curated word lists.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


# Curated martial / battlefield / arms lexicon. Lowercased.
_MARTIAL: frozenset[str] = frozenset({
    # Weapons
    "sword", "swords", "blade", "blades", "dagger", "daggers",
    "spear", "spears", "pike", "pikes", "lance", "lances",
    "arrow", "arrows", "bow", "bows", "shield", "shields",
    "steel", "iron", "gun", "guns", "cannon", "cannons",
    "musket", "poniard", "rapier", "axe", "halberd",
    # Armor / gear
    "armor", "armour", "helm", "helmet", "helmets", "corslet",
    "mail", "gauntlet", "vizor", "banner", "banners", "standard",
    "ensign", "trumpet", "trumpets", "drum", "drums", "fife",
    # Wounds / blood / violence
    "wound", "wounds", "wounded", "blood", "bleeding", "bleed",
    "bloody", "gore", "gash", "scar", "scars", "slain",
    "kill", "killed", "kills", "killing", "slay", "slew",
    "slaughter", "slain", "murder", "murdered",
    "strike", "struck", "strikes", "smite", "smote", "smitten",
    "stab", "stabbed", "pierce", "pierced", "cleave", "cleft",
    # War actions
    "fight", "fights", "fought", "fighting", "battle", "battles",
    "war", "wars", "warrior", "warriors", "soldier", "soldiers",
    "captain", "captains", "general", "colonel", "marshal",
    "knight", "knights", "squire", "champion",
    "enemy", "enemies", "foe", "foes", "foeman",
    "siege", "sieges", "besiege", "besieged", "breach", "breaches",
    "assault", "charge", "charges", "charged", "attack",
    "march", "marched", "marches", "retreat", "retire",
    "conquer", "conquered", "conquest", "vanquish", "vanquished",
    "triumph", "victory", "defeat", "defeated",
    # Battle scene / mounts
    "horse", "horses", "steed", "steeds", "coursers", "coursers",
    "rider", "riders", "cavalry",
    # Abstract martial
    "arm", "arms", "armed", "armeth", "force", "forces",
    "power", "powers", "host", "hosts", "legion", "legions",
    "troop", "troops", "ranks", "files", "platoon",
    "fray", "frays", "combat", "combats", "duel", "duels",
    "fought", "fight", "engage", "engaged",
})

# Curated peaceful / pastoral lexicon.
_PEACEFUL: frozenset[str] = frozenset({
    "peace", "peaceful", "quiet", "quietly", "still", "calm",
    "rest", "resting", "sleep", "sleeping", "slumber", "slumbered",
    "love", "lover", "lovers", "beloved", "sweet", "sweetness",
    "gentle", "gentleness", "kind", "kindness", "mild", "mildness",
    "soft", "softly", "tender", "tenderly",
    "home", "hearth", "bed", "bread", "cup", "wine",
    "song", "songs", "music", "melody", "lute", "pipe",
    "flower", "flowers", "rose", "roses", "violet", "lily",
    "garden", "grove", "meadow", "bower", "orchard", "wood",
    "dove", "doves", "lamb", "lambs", "bird", "birds", "nightingale",
    "friend", "friends", "friendship",
    "smile", "smiled", "laugh", "laughed", "laughter",
    "joy", "joys", "joyful", "delight", "pleasure",
    "kiss", "kisses", "kissed",
    "bless", "blessed", "blessing", "benediction",
    "hope", "hopes", "heal", "healed", "healer",
    "light", "dawn", "morning", "spring", "summer",
})


_DECAY: float = 0.93
_MARTIAL_BUMP: float = 0.70
_PEACEFUL_BUMP: float = -0.40
_MIN: float = -2.0
_MAX: float = 3.0


def update_martial(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-turn boundary reset.
    if ch == "\n" and state.consecutive_newlines >= 2:
        if state.martial_charge != 0.0:
            return state.model_copy(update={"martial_charge": 0.0})
        return state

    if not state.just_finished_word:
        return state
    word = state.last_completed_word
    if not word:
        return state

    charge = state.martial_charge * _DECAY
    if word in _MARTIAL:
        charge += _MARTIAL_BUMP
    elif word in _PEACEFUL:
        charge += _PEACEFUL_BUMP
    # Clamp.
    if charge > _MAX:
        charge = _MAX
    elif charge < _MIN:
        charge = _MIN
    # Snap near-zero to zero (numeric quiet).
    if abs(charge) < 1e-4:
        charge = 0.0
    if charge == state.martial_charge:
        return state
    return state.model_copy(update={"martial_charge": charge})
