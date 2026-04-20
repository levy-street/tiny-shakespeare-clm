"""Tier 3 flow: mirth register — rolling comic/merry texture.

Maintains `state.mirth_register` ∈ [0, 1]. Rises on merry/laugh/jest/
fool/feast/revel/song lexicon; falls on grief/wrath/death lexicon.

Distinct from tenderness (love/romance), tonal_weight (generic
dark/light), imagery_density (sensory concreteness), and gravitas
(moral weight). Comic scenes — fools, clowns, drinking songs,
rustic wedding feasts — thicken this register. Orthogonal to
tenderness because love scenes can be somber and comic scenes
are often rough/crude (sack, ale, belly).

Consumed by predict/mirth.py to tilt word-start letters toward
mirth-lexicon starters when the register is elevated.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


# Mirth-core — strongest merry/comic signal.
_MIRTH_CORE: frozenset[str] = frozenset({
    "mirth", "mirthful",
    "merry", "merrily", "merrier", "merriest", "merriment",
    "laugh", "laughs", "laughed", "laughing", "laughter",
    "jest", "jests", "jested", "jester", "jesting",
    "joy", "joys", "joyful", "joyous",
    "glad", "gladly", "gladness", "gladder",
    "smile", "smiles", "smiled", "smiling",
    "fool", "fools", "fooling", "foolish", "folly",
    "clown", "clowns",
    "sport", "sports", "sporting",
    "frolic", "frolics",
    "festive", "festival", "festivals",
    "feast", "feasts", "feasted", "feasting",
    "revel", "revels", "revelry",
    "pleasant", "pleasantly",
    "happy", "happier", "happiest", "happiness",
    "cheer", "cheers", "cheerful", "cheerly", "cheering",
    "blithe", "blithely",
    "jolly",
    "gay", "gaily",
    "wit", "wits", "witty",
    "jape", "japes",
})

# Mirth-halo — softer comic atmosphere.
_MIRTH_HALO: frozenset[str] = frozenset({
    "play", "plays", "played", "playing",
    "tune", "tunes",
    "dance", "dances", "dancing",
    "pipe", "pipes",
    "bell", "bells",
    "wine",
    "ale",
    "cup", "cups",
    "drink", "drinks", "drinking",
    "fiddle",
    "drum", "drums",
    "morris",
    "tabor",
    "music",
    "song", "songs",
    "sing", "sings", "sang", "sung", "singing",
    "celebrate",
    "wedding", "weddings",
    "bride",
    "holiday",
    "sack",
    "belly",
    "jig",
    "caper",
    "masque",
    "carol",
})

# Anti-mirth (grief/wrath) — strong negative pull.
_ANTI_MIRTH: frozenset[str] = frozenset({
    "grief", "griefs",
    "sorrow", "sorrows",
    "woe", "woes",
    "mourn", "mourns", "mourned", "mourning",
    "weep", "weeps", "wept", "weeping",
    "tears",
    "moan", "moans", "moaning",
    "lament", "laments",
    "sigh", "sighs", "sighing",
    "death", "deaths", "dying",
    "grave", "graves",
    "tomb", "tombs",
    "doom",
    "hell",
    "curse", "curses", "cursed", "curs'd",
    "wrath", "wrathful",
    "rage", "rages", "raging",
    "fury", "furious",
    "despair",
    "ruin",
    "horror",
    "dread",
    "terror",
})

# Anti-mirth-mild — subtler negative.
_ANTI_MIRTH_MILD: frozenset[str] = frozenset({
    "dead",
    "fear", "fears", "feared",
    "sad", "sadly", "sadness",
    "heavy",
    "bleak",
    "cold",
    "dark",
})


_CORE_BUMP = 0.15
_HALO_BUMP = 0.08
_ANTI_BUMP = -0.12
_ANTI_MILD_BUMP = -0.04
_DECAY = 0.93
_TURN_SCALE = 0.40


def update_mirth(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    mir = state.mirth_register

    # Speaker-turn boundary: collapse toward zero. Comic register is
    # local to a speaker; a new turn doesn't inherit it fully.
    if ch == "\n" and state.consecutive_newlines >= 2:
        new_mir = mir * _TURN_SCALE
        if abs(new_mir - mir) > 1e-6 or mir > 0.0:
            return state.model_copy(update={"mirth_register": new_mir})
        return state

    # On just_finished_word: apply bump + decay.
    if state.just_finished_word and state.last_completed_word:
        word = state.last_completed_word.lower().strip("'")
        bump = 0.0
        if word in _MIRTH_CORE:
            bump = _CORE_BUMP
        elif word in _MIRTH_HALO:
            bump = _HALO_BUMP
        elif word in _ANTI_MIRTH:
            bump = _ANTI_BUMP
        elif word in _ANTI_MIRTH_MILD:
            bump = _ANTI_MILD_BUMP

        new_mir = mir * _DECAY + bump
        if new_mir < 0.0:
            new_mir = 0.0
        elif new_mir > 1.0:
            new_mir = 1.0

        if abs(new_mir - mir) > 1e-6:
            return state.model_copy(update={"mirth_register": new_mir})

    return state
