"""Tier 2 — thou/you register commit.

Tracks whether the current speaker, within the current turn, has
committed to the T-form (thou/thee/thy/thine/thyself and -st auxiliaries)
or the V-form (you/your/yours/ye) for addressing their listener. In
Early Modern English these are not interchangeable: once committed,
a speaker stays in-register for the rest of the turn. Switching is
linguistically jarring ("Thou art mad. You look pale." is wrong).

Transitions on word completion:
  - UNCOMMITTED (0): if last_completed_word ∈ T-set, → T_COMMIT (1)
                    else if last_completed_word ∈ V-set, → V_COMMIT (2)
                    else stay UNCOMMITTED
  - T_COMMIT / V_COMMIT: sticky within the turn; do not flip (we don't
    penalize the rare legitimate in-turn switch, we just hold).

Reset:
  - On turn boundary (consecutive_newlines >= 2) → UNCOMMITTED.

The "thou-set" also includes auxiliary verb forms that inflect with
2nd-singular morphology only when addressing a T-form subject.
These are a STRONG signal of thou-register even when the pronoun
itself has elided ("Art thou mad?" → "Art, indeed.").

No corpus statistics — the T/V pronoun families and -st forms are
Early Modern English grammatical knowledge.
"""

from __future__ import annotations

from ..state import ModelState

COMMIT_NONE = 0
COMMIT_T = 1
COMMIT_V = 2

# T-form markers: 2ps pronouns + inflected auxiliaries whose -st ending
# is a dead giveaway of thou-register.
_T_WORDS: frozenset[str] = frozenset({
    # Pronouns proper
    "thou", "thee", "thy", "thine", "thyself",
    # -st auxiliaries (thou art / hast / didst / wast / wert / wilt /
    # shalt / canst / couldst / wouldst / shouldst / mayst / mayest /
    # dost / doth-is-3sg-so-skip / hadst / durst)
    "art", "hast", "hadst",
    "didst", "dost", "doest",
    "wast", "wert", "wilt", "willst",
    "shalt", "shouldst", "shouldest",
    "canst", "couldst", "couldest",
    "wouldst", "wouldest",
    "mayst", "mayest",
    "durst",
    # Other thou-verbs: knowest, seest, sayest, hearst, lovest...
    # These end in -est/-st; we don't enumerate — pipeline can infer
    # from suffix, but to keep this stage deterministic and cheap we
    # list only the most frequent.
    "knowest", "seest", "sayest", "speakest", "hearest",
    "lovest", "lovedst", "givest", "takest", "makest",
    "liest", "diest", "thinkest", "seemest",
    "prithee",  # pray-thee contraction
})

# V-form markers.
_V_WORDS: frozenset[str] = frozenset({
    "you", "your", "yours", "yourself", "yourselves",
    "ye",
})


def update_register_commit(state: ModelState, token_id: int) -> ModelState:
    # Turn boundary reset. Use consecutive_newlines as the canonical
    # "between turns" signal (matches speaker_memory / words_in_turn).
    if state.consecutive_newlines >= 2:
        if state.thou_thee_commit != COMMIT_NONE:
            return state.model_copy(update={"thou_thee_commit": COMMIT_NONE})
        return state

    # Only transition on word completion.
    if not (state.just_finished_word and state.last_completed_word):
        return state

    # Skip speaker-label territory.
    if state.speaker_label_state != 0:
        return state

    # Already committed — sticky.
    if state.thou_thee_commit != COMMIT_NONE:
        return state

    w = state.last_completed_word.lower()
    if w in _T_WORDS:
        return state.model_copy(update={"thou_thee_commit": COMMIT_T})
    if w in _V_WORDS:
        return state.model_copy(update={"thou_thee_commit": COMMIT_V})
    return state
