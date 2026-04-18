"""Doubt register pipeline stage.

Maintains `doubt_register`, a rolling [-1, +1] float tracking whether
the emerging text is in doubt mode (+) vs assertion mode (-). Updated
at word completion by matching the completed word against two lists
and at sentence-end punctuation by reacting to "?"/"!".

Decay is applied at every completed word so the register fades when
no epistemic markers appear.

Runs after update_pos (so last_completed_word is fresh) and after
update_linguistic (so last_char and last_char_class reflect the
terminal punctuation).
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB
from .linguistic import PUNCT_END


# Doubt words — weight +0.14 when they complete as a word.
_DOUBT_STRONG: frozenset[str] = frozenset({
    "perhaps", "perchance", "peradventure", "belike",
    "methinks", "methought",
    "haply",
    "may", "might", "maybe", "mayhap",
    "seem", "seems", "seemed", "seeming",
    "wonder", "wondering",
    "doubt", "doubts", "doubtful",
    "suspect", "suspects", "suspected",
    "unsure", "uncertain",
})

# Lighter doubt markers — weight +0.05.
_DOUBT_LIGHT: frozenset[str] = frozenset({
    "if", "whether", "or",
})

# Certainty/assertion words — weight -0.14.
_CERTAIN_STRONG: frozenset[str] = frozenset({
    "verily", "surely", "truly",
    "indeed", "certes", "certain", "certainly",
    "doubtless",
    "assured", "assuredly",
    "forsooth",
    "aye",
})

# Knowledge-assertion — weight -0.08.
_KNOW_MARKERS: frozenset[str] = frozenset({
    "know", "knows", "known", "knew", "knowest", "knowing",
    "see", "sees", "seen", "saw",
    "is", "art", "am", "was", "were",
})

# Imperative-ish verbs (when at sentence start). Weight -0.06.
_IMPERATIVE_VERBS: frozenset[str] = frozenset({
    "go", "come", "speak", "hear", "look", "stay", "stop",
    "hold", "hark", "see", "behold", "strike", "away",
    "silence", "peace",
})


_DECAY = 0.93
_CLIP = 1.0
_TURN_MULT = 0.30


def _clip(v: float) -> float:
    if v > _CLIP:
        return _CLIP
    if v < -_CLIP:
        return -_CLIP
    return v


def update_doubt(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-turn boundary: dampen toward 0.
    if state.consecutive_newlines >= 2 and ch == "\n":
        new_val = state.doubt_register * _TURN_MULT
        if abs(new_val - state.doubt_register) < 1e-9:
            return state
        return state.model_copy(update={"doubt_register": new_val})

    # Sentence-end punctuation: "?" = doubt spike, "!" = assertion spike.
    if state.last_char_class == PUNCT_END:
        bump = 0.0
        if ch == "?":
            bump = +0.06
        elif ch == "!":
            bump = -0.06
        if bump != 0.0:
            new_val = _clip(state.doubt_register + bump)
            if abs(new_val - state.doubt_register) < 1e-9:
                return state
            return state.model_copy(update={"doubt_register": new_val})
        return state

    # Word-completion updates.
    if not state.just_finished_word:
        return state
    if state.speaker_label_state != 0:
        return state

    word = state.last_completed_word
    if not word:
        return state
    lookup = word.lstrip("'")

    bump = 0.0
    if lookup in _DOUBT_STRONG:
        bump = +0.14
    elif lookup in _CERTAIN_STRONG:
        bump = -0.14
    elif lookup in _DOUBT_LIGHT:
        bump = +0.05
    elif lookup in _KNOW_MARKERS:
        bump = -0.08
    elif (
        state.words_in_sentence == 1  # first word of sentence just completed
        and lookup in _IMPERATIVE_VERBS
    ):
        bump = -0.06

    # Decay-then-bump (order doesn't matter mathematically for small
    # values; use decay-then-bump so decay happens every word).
    new_val = _clip(state.doubt_register * _DECAY + bump)
    if abs(new_val - state.doubt_register) < 1e-9:
        return state
    return state.model_copy(update={"doubt_register": new_val})
