"""Vocative-expectation pipeline stage.

Sets `state.vocative_expectation = True` when the recent word pattern
looks like a Shakespearean vocative construction in progress — a
possessive-like or interjectional lead ("my", "thy", "O", "sweet",
"good") followed by a vocative-adjective ("dear", "gentle", "noble",
"fair", "poor"). When True, the predict layer biases the first letter
of the next word toward vocative-noun starters: l (lord/lady/liege),
s (sir/son/sister), m (madam/master/mistress), f (friend/father),
p (prince), b (brother), c (cousin/captain), k (king), q (queen).

Reset on sentence-end punctuation, verb-like POS, or any word that
breaks the (LEAD)(ADJ) shape.

This is a narrow structural texture field: it doesn't try to capture
every vocative (a bare "lord!" is also a vocative). It captures the
distinctive two-word lead-in that is nearly diagnostic of the
vocative construction in Early Modern English:
  - "my dear lord"   / "my good sir"     / "my sweet madam"
  - "thy noble friend" / "thy gentle brother"
  - "O dear father"  / "O sweet mistress"
  - "good my lord"   / "sweet my master"  (inverted)
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB
from .pos import (
    POS_AUX_VERB,
    POS_MODAL,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
)

# Words that can LEAD a vocative: possessives, interjections, and
# vocative-adjectives that pair up with other vocative-adjectives.
_VOCATIVE_LEAD: frozenset[str] = frozenset({
    "my", "thy", "mine", "thine", "our", "your",
    "o", "oh", "ah",
    "good", "sweet", "dear", "gentle", "fair", "poor", "noble",
    "kind", "honest", "worthy", "gracious", "mighty", "royal",
    "brave",
})

# Vocative-adjectives that, when appearing after a LEAD, trigger the
# expectation of a vocative noun next.
_VOCATIVE_ADJ: frozenset[str] = frozenset({
    "good", "sweet", "dear", "gentle", "fair", "poor", "noble",
    "kind", "honest", "true", "brave", "worthy", "gracious",
    "mighty", "royal", "most",
})

_VERB_POS: frozenset[int] = frozenset({
    POS_AUX_VERB, POS_MODAL, POS_VERB, POS_VERB_ING, POS_VERB_ED,
})


def update_vocative(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Hard reset on sentence-end.
    if ch in ".?!":
        if state.vocative_expectation:
            return state.model_copy(update={"vocative_expectation": False})
        return state

    # A clause-internal comma/semicolon preserves the flag if it was set
    # (e.g. "my dear, dear lord") but doesn't affect it.
    if not state.just_finished_word or not state.last_completed_word:
        return state

    w = state.last_completed_word
    pos = state.last_word_pos
    prev = state.prev_completed_word

    # A verb-like word clears expectation: vocative constructions are
    # verbless noun phrases.
    if pos in _VERB_POS:
        if state.vocative_expectation:
            return state.model_copy(update={"vocative_expectation": False})
        return state

    # Trigger: current word is a vocative-adj AND previous completed
    # word is a vocative-lead. This is the diagnostic two-word pattern.
    should_set = (w in _VOCATIVE_ADJ) and (prev in _VOCATIVE_LEAD)

    if should_set != state.vocative_expectation:
        return state.model_copy(update={"vocative_expectation": should_set})
    return state
