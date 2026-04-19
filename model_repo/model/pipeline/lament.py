"""Tier 3 flow: lament register — rolling grief/plaintive texture.

Maintains `state.lament_register` ∈ [0, 1] as a rolling float that
rises on grief-lexicon completions and falls on joy/action
completions. Distinct from tonal_weight (which captures generic
dark/light valence) by targeting the *specific* lexicon of lament —
moaning, mourning, sorrow, weeping, loss.

Decay: 0.92 per completed word. Speaker-turn: *0.35. Sentence-end
"!": -0.02 (lament moans rather than shouts). All weights are
hand-chosen from Shakespeare's lament passages — no corpus statistics.

Consumed by predict/lament.py.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


# Lament-core words: strongest grief signal.
_LAMENT_CORE: frozenset[str] = frozenset({
    "alas", "alack", "woe", "sorrow", "sorrows", "grief", "griefs",
    "weep", "weeps", "wept", "weeping",
    "sigh", "sighs", "sighed", "sighing",
    "tears", "teardrop", "tear",
    "mourn", "mourns", "mourned", "mourning",
    "lament", "laments", "lamented",
    "pity", "pitied", "piteous",
    "wretched", "wretch", "wretches",
    "forlorn", "doleful", "sorrowful", "sad", "sadly",
    "dirge",
})

# Lament-halo: softer grief atmosphere.
_LAMENT_HALO: frozenset[str] = frozenset({
    "heart", "hearts", "heavy",
    "dead", "death", "deaths", "dying", "die", "dies",
    "lost", "loss",
    "poor",
    "cursed", "curs'd", "curse", "cursing",
    "dread", "dreaded", "dreadful",
    "pain", "pains", "pained",
    "loss", "losses",
    "tomb", "grave", "graves",
})

# Anti-lament (joy) — strong negative signal.
_JOY_WORDS: frozenset[str] = frozenset({
    "joy", "joys", "joyful",
    "mirth", "mirthful",
    "merry", "merrily",
    "glad", "gladness", "gladly",
    "laugh", "laughs", "laughed", "laughing", "laughter",
    "smile", "smiles", "smiled", "smiling",
    "happy", "happily",
    "gay", "sport", "sports", "revel", "revels", "revelry",
})

# Anti-lament-mild (action verbs) — lament is contemplative.
_ACTION_WORDS: frozenset[str] = frozenset({
    "strike", "struck", "striking",
    "march", "marched", "marching",
    "charge", "charged", "charging",
    "fight", "fought", "fighting",
    "seize", "seized", "seizing",
    "slay", "slew", "slain", "slaying",
    "run", "ran", "running",
    "leap", "leapt", "leaped",
    "spur", "spurred",
})

_CORE_BUMP = 0.18
_HALO_BUMP = 0.10
_JOY_BUMP = -0.15
_ACTION_BUMP = -0.05
_BANG_BUMP = -0.02
_DECAY = 0.92
_TURN_SCALE = 0.35


def update_lament(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    lam = state.lament_register

    # Speaker-turn boundary: big attenuation.
    if ch == "\n" and state.consecutive_newlines >= 2:
        new_lam = lam * _TURN_SCALE
        if abs(new_lam - lam) > 1e-6 or lam > 0.0:
            return state.model_copy(update={"lament_register": new_lam})
        return state

    # Sentence-end "!" decay: lament moans rather than shouts.
    if ch == "!":
        new_lam = lam + _BANG_BUMP
        if new_lam < 0.0:
            new_lam = 0.0
        if abs(new_lam - lam) > 1e-6:
            return state.model_copy(update={"lament_register": new_lam})
        return state

    # On just_finished_word: apply bumps + decay.
    if state.just_finished_word and state.last_completed_word:
        word = state.last_completed_word.lower().strip("'")
        bump = 0.0
        if word in _LAMENT_CORE:
            bump = _CORE_BUMP
        elif word in _LAMENT_HALO:
            bump = _HALO_BUMP
        elif word in _JOY_WORDS:
            bump = _JOY_BUMP
        elif word in _ACTION_WORDS:
            bump = _ACTION_BUMP

        # Decay first, then apply bump.
        new_lam = lam * _DECAY + bump
        if new_lam < 0.0:
            new_lam = 0.0
        elif new_lam > 1.0:
            new_lam = 1.0

        if abs(new_lam - lam) > 1e-6:
            return state.model_copy(update={"lament_register": new_lam})

    return state
