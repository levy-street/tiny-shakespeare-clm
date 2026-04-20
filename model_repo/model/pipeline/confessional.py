"""Tier 3 — confessional-intimacy ↔ public-declamation register.

Updates `state.confessional_intimacy` (float in [-1.0, +1.0]) based
on which completed word just closed.

The axis captures a textural register distinct from every existing
emotional register (fury, tenderness, gravitas, lament, doubt), from
addressing_register (thou vs. you), and from archaic_density. It
answers a different question: *who is the speaker talking to?*

  +1 end — confessional monologue / intimate address. A soliloquy, a
           deathbed confession, a whispered aside, a private love-
           speech. Lexicon: interior verbs, 1sg pronouns, tender
           nouns, sighs.

  -1 end — public declamation. A battlefield oration, a coronation
           speech, a trial address, a proclamation. Lexicon: plural
           pronouns, vocative plurals, imperatives, titles, honorifics.

Shakespeare switches between these registers consciously and the
switch stays stable for multiple lines. Once Hamlet begins "To be
or not to be…" the confessional register locks in until the scene
shifts. Once Henry V begins "Once more unto the breach, dear
friends…" the public register locks in.

The pipeline stage bumps the score at word completion by a small
increment keyed to the completed word's class. Decay is per-word
(0.93 multiplier) so the register naturally cools off absent
reinforcement. On speaker-turn boundary the score is damped (×0.25)
but not zeroed — the new speaker enters with some register
expectation from the scene context, which the first words will
either confirm or flip.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


# Words that push toward confessional / intimate register (+).
_CONFESSIONAL_WORDS: frozenset[str] = frozenset({
    # First-person singulars (core confessional marker).
    "i", "me", "my", "mine", "myself",
    # Interior-state verbs.
    "think", "thought", "feel", "felt", "fear",
    "feared", "hope", "hoped", "doubt", "doubted", "wish",
    "wished", "wonder", "wondered", "dream", "dreamed", "dreamt",
    "remember", "suspect", "suspected", "suppose",
    "believe", "imagine", "pray", "prithee", "mourn",
    "love", "loved", "hate", "hated", "long",
    "weep", "sigh", "grieve", "despair", "lament",
    # Intimate 2sg address.
    "thou", "thee", "thy", "thine", "thyself",
    # Tender-interior nouns.
    "heart", "soul", "breath", "bosom", "mind",
    "conscience", "memory", "dream", "solitude", "silence",
    "grief", "sorrow", "woe", "tear", "tears", "sigh", "sighs",
    # Sigh-interjections (strong signal).
    "alas", "alack", "ah", "oh", "fie",
    # Confessional aux verbs (soft).
    "am", "was", "were",
})

# Words that push toward public / declamatory register (-).
_PUBLIC_WORDS: frozenset[str] = frozenset({
    # First-person plurals.
    "we", "us", "our", "ours", "ourselves",
    # Plural 2-person / 2pl address.
    "ye", "your", "yours",
    # Plural vocatives / addresses.
    "lords", "friends", "gentlemen", "countrymen",
    "masters", "sirs", "brothers", "soldiers", "fellows",
    "citizens", "subjects", "people", "brethren",
    # Imperative / attention-getting verbs.
    "hear", "hearken", "behold", "mark", "attend",
    "witness", "speak", "silence", "proclaim",
    # Ceremonial / honorific nouns.
    "majesty", "grace", "highness", "excellence", "honour",
    "honor", "sovereign", "liege", "queen", "king",
    "prince", "duke", "lord",
    # Proclamatory / assertive openers.
    "know", "now", "therefore", "wherefore", "behold",
    "lo", "see",
    # Public-action verbs.
    "command", "decree", "declare", "pronounce", "proclaim",
    "march", "fight", "charge", "strike", "advance",
})


# Per-word bumps (small; decay is gentle too).
_POS_BUMP = 0.12
_POS_BUMP_STRONG = 0.20  # pronouns / sigh interjections
_NEG_BUMP = -0.12
_NEG_BUMP_STRONG = -0.20  # plurals / vocatives

_STRONG_POS: frozenset[str] = frozenset({
    "i", "me", "my", "mine", "myself",
    "thou", "thee", "thy", "thine", "thyself",
    "alas", "alack", "ah", "oh",
    "heart", "soul",
})

_STRONG_NEG: frozenset[str] = frozenset({
    "we", "us", "our", "ours",
    "lords", "friends", "gentlemen", "countrymen",
    "masters", "sirs", "brothers", "soldiers",
    "citizens", "subjects", "brethren",
    "hear", "hearken", "behold", "mark", "attend",
    "majesty", "grace", "highness", "excellence",
})

# Per-word decay toward 0.
_DECAY = 0.93
# Damping on speaker-turn boundary.
_TURN_DAMP = 0.25


def _classify(word: str) -> float:
    w = word.lower()
    # Strip a single leading/trailing apostrophe ('tis / ne'er edges).
    if w.startswith("'"):
        w = w[1:]
    if w.endswith("'"):
        w = w[:-1]
    if not w:
        return 0.0
    if w in _STRONG_POS:
        return _POS_BUMP_STRONG
    if w in _STRONG_NEG:
        return _NEG_BUMP_STRONG
    if w in _CONFESSIONAL_WORDS:
        return _POS_BUMP
    if w in _PUBLIC_WORDS:
        return _NEG_BUMP
    return 0.0


def update_confessional(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-turn boundary: damp the register rather than zero it.
    if ch == "\n" and state.consecutive_newlines >= 2:
        new_val = state.confessional_intimacy * _TURN_DAMP
        if abs(new_val) < 1e-4:
            new_val = 0.0
        if abs(new_val - state.confessional_intimacy) > 1e-6:
            return state.model_copy(update={"confessional_intimacy": new_val})
        return state

    # Only act on word-completion events.
    if not state.just_finished_word or not state.last_completed_word:
        return state

    bump = _classify(state.last_completed_word)
    # Always decay (even on zero-bump words) so the register cools
    # absent reinforcement.
    new_val = state.confessional_intimacy * _DECAY + bump

    # Clamp.
    if new_val > 1.0:
        new_val = 1.0
    elif new_val < -1.0:
        new_val = -1.0

    if abs(new_val) < 1e-4:
        new_val = 0.0

    if abs(new_val - state.confessional_intimacy) > 1e-6:
        return state.model_copy(update={"confessional_intimacy": new_val})
    return state
