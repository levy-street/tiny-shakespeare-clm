"""Tier 3 flow: fury register — rage / wrath / curse texture.

Maintains `state.fury_register` ∈ [0, 1]. Rises on rage-class lexicon
and exclamation-marked speech; cooled by tender / peaceful lexicon;
decays per completed word; mostly reset across speaker turns.

Distinct from:
  - tonal_weight  (scene dark vs light — external events)
  - gravitas      (moral / philosophical weight — controlled, not hot)
  - lament        (grief — mournful passive, not aggressive)
  - tenderness    (love — opposite polarity)

Fury is ANGRY SPEECH: curses, threats, insults, imprecations.
Characteristic moments: Lear's storm, Timon's misanthropy, Iago's
asides, Mercutio's "a plague o' both your houses".

No corpus statistics — the fury lexicon is hand-curated from prior
knowledge of Shakespearean invective.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


# Strongest anger markers — curse-words, invective nouns, imperatives
# of violence.
_STRONG_FURY: frozenset[str] = frozenset({
    # Direct anger nouns
    "rage", "rages", "raging",
    "wrath", "wrathful",
    "fury", "furies", "furious", "furiously",
    "anger", "angers", "angered", "angrily",
    # Curse/imprecation
    "damn", "damned", "damnation",
    "curse", "cursed", "curses", "cursing",
    "plague", "plagues", "plaguey",
    "hell", "hellish", "hellfire",
    # Invective nouns (direct insult targets)
    "villain", "villains", "villainous", "villainy",
    "knave", "knaves", "knavery", "knavish",
    "traitor", "traitors", "traitorous",
    "slave", "slaves",  # as insult ("thou slave!")
    "wretch", "wretches", "wretched",
    "fiend", "fiends", "fiendish",
    "devil", "devils", "devilish",
    "viper", "vipers",
    "poison", "poisons", "poisoned", "poisonous",
    "venom", "venomous",
    # Descriptive insults
    "vile", "vilely",
    "foul", "foully",
    "rotten", "rot",
    "putrid",
    "base",  # as in "base slave"
})


# Milder anger markers.
_MILD_FURY: frozenset[str] = frozenset({
    "hate", "hated", "hates", "hateful", "hating", "hatred",
    "strike", "strikes", "struck", "smite", "smote", "smitten",
    "slay", "slays", "slew", "slain",
    "kill", "kills", "killed",
    "murder", "murders", "murdered", "murderous",
    "scorn", "scorns", "scorned", "scornful",
    "spite", "spites", "spiteful",
    "shame", "shameful", "ashamed",
    "bastard", "bastards",
    "rascal", "rascals", "rascally",
    "rogue", "rogues", "roguish",
    "cur", "curs",
    "beast", "beasts", "beastly",
    "dog", "dogs",  # as insult; neutral elsewhere — accept the noise
    "hag", "hags",
    "witch",  # Lear to Goneril
    "fool",  # as insult
    "liar", "liars", "lies", "lying",
    "false", "falsely",
    "monstrous", "monster", "monsters",
    "bloody",  # bloody deeds
    "burn", "burns", "burning", "burnt",
    "avenge", "avenged", "vengeance",
    "revenge", "revenged",
})


# Cooling — tender/peaceful lexicon lowers fury.
_COUNTER_FURY: frozenset[str] = frozenset({
    "peace", "peaceful",
    "love", "loves", "loved", "loving", "lovely",
    "sweet", "sweetly", "sweetness",
    "gentle", "gently", "gentleness",
    "kind", "kindly", "kindness",
    "fair", "fairly", "fairness",
    "soft", "softly",
    "calm", "calmly",
    "mercy", "merciful", "mercies",
    "forgive", "forgives", "forgiven", "forgiving",
    "pardon", "pardons", "pardoned",
    "grace", "graces", "gracious",
    "pity", "pities", "pitied",
})


_STRONG_BUMP = 0.22
_MILD_BUMP = 0.10
_COUNTER_BUMP = -0.06
_EXCLAM_BUMP = 0.08  # "!" token boosts when fury already > 0
_DECAY = 0.94
_TURN_SCALE = 0.20


def update_fury(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    fury = state.fury_register

    # Speaker-turn boundary: partial reset.
    if ch == "\n" and state.consecutive_newlines >= 2:
        new_fury = fury * _TURN_SCALE
        if abs(new_fury - fury) > 1e-6 or fury > 0.0:
            return state.model_copy(update={"fury_register": new_fury})
        return state

    # Exclamation mark at sentence-end — amplifies but only if there's
    # already some anger present. Prevents neutral exclamations
    # ("O brave new world!") from triggering fury.
    if ch == "!" and fury > 0.05:
        new_fury = fury + _EXCLAM_BUMP
        if new_fury > 1.0:
            new_fury = 1.0
        if abs(new_fury - fury) > 1e-6:
            return state.model_copy(update={"fury_register": new_fury})
        return state

    # On just_finished_word: apply bump + decay.
    if state.just_finished_word and state.last_completed_word:
        word = state.last_completed_word.lower().strip("'")
        bump = 0.0
        if word in _STRONG_FURY:
            bump = _STRONG_BUMP
        elif word in _MILD_FURY:
            bump = _MILD_BUMP
        elif word in _COUNTER_FURY:
            bump = _COUNTER_BUMP

        new_fury = fury * _DECAY + bump
        if new_fury < 0.0:
            new_fury = 0.0
        elif new_fury > 1.0:
            new_fury = 1.0

        if abs(new_fury - fury) > 1e-6:
            return state.model_copy(update={"fury_register": new_fury})

    return state
