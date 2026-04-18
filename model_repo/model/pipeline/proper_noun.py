"""Proper-noun expectation pipeline stage.

Maintains `state.proper_noun_slot`, an integer register encoding how
strongly a capitalized proper noun is expected at the NEXT word-start:

  0 PN_NONE   — no signal (default)
  1 PN_MILD   — after vocative-lead / possessive-adjective / sentence
                 punctuation boundary. A capital is plausible but not
                 required (e.g. common noun also fine).
  2 PN_STRONG — after a title noun ("lord", "sir", "king", "queen",
                 "saint", ...) or vocative-delimiter punctuation
                 (comma / colon / dash). A capital proper noun is
                 strongly expected.
  3 PN_QUOTE  — after a reported-speech lead ("said", "quoth",
                 "cried", ...). The next word begins direct speech
                 and is typically capitalized.

Set at word-completion or at sentence-ending punctuation. Decays:
  - If the slot was triggered on a prior word and NO capital came on
    the next word-start, it clears to PN_NONE.
  - On any punctuation other than a vocative-delimiter, it clears.
  - On speaker-turn boundary, it clears.

Consumed by predict/proper_noun.py at word-start:
  - PN_NONE mid-sentence → penalize A-Z lightly (phantom-cap guard).
  - PN_MILD or higher → no penalty, mild boost for title-compatible
    capitals.

Runs after update_pos (needs last_completed_word and POS) and after
update_linguistic (needs sentence_start_pending / chars_since_space).
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


# Proper-noun-slot values.
PN_NONE: int = 0
PN_MILD: int = 1
PN_STRONG: int = 2
PN_QUOTE: int = 3


# Title nouns that typically precede a proper name ("lord Henry",
# "saint Crispin", "queen Mab", "sir John"). Lower-cased.
_TITLE_NOUNS: frozenset[str] = frozenset({
    "lord", "lady", "sir", "madam", "master", "mistress",
    "saint", "st",
    "king", "queen", "prince", "princess",
    "duke", "duchess", "count", "earl", "baron",
    "captain", "general", "doctor", "friar",
    "brother", "sister", "father", "mother",
    "son", "daughter", "cousin", "uncle", "aunt", "nephew", "niece",
})


# Vocative-lead adjectives / possessives — a proper-name vocative
# may follow but so may a common noun. Signal is PN_MILD.
_VOCATIVE_LEADS: frozenset[str] = frozenset({
    "good", "sweet", "noble", "gentle", "fair", "dear", "poor",
    "brave", "young", "old",
    "my", "thy", "thine", "our", "your", "his", "her",
    "mine",
    "o", "oh",  # "O Hamlet," "Oh Cassio,"
})


# Reported-speech leads — direct quote (typically capitalized) often
# follows.
_QUOTE_LEADS: frozenset[str] = frozenset({
    "said", "saith", "says", "quoth", "quod", "cried", "cries",
    "answered", "replied", "spoke", "speaketh",
})


def update_proper_noun(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-turn boundary: clear.
    if state.consecutive_newlines >= 2 and ch == "\n":
        if state.proper_noun_slot != PN_NONE:
            return state.model_copy(update={"proper_noun_slot": PN_NONE})
        return state

    # Sentence-end punctuation: clear (next word starts a new
    # sentence; the start-of-sentence capital signal is handled by
    # sentence_start_pending).
    if ch in (".", "?", "!"):
        if state.proper_noun_slot != PN_NONE:
            return state.model_copy(update={"proper_noun_slot": PN_NONE})
        return state

    # Vocative-delimiter punctuation: PN_STRONG.
    if ch in (",", ";", ":"):
        if state.speaker_label_state != 0:
            return state
        if state.proper_noun_slot != PN_STRONG:
            return state.model_copy(update={"proper_noun_slot": PN_STRONG})
        return state

    # Word completion: set based on the completed word.
    if not state.just_finished_word:
        # If we just started emitting a new word's letters, the slot
        # was "consumed" (or not). Clear on the first letter of a new
        # word so the signal is one-shot.
        if ch.isalpha() and state.letter_run_len == 1:
            if state.proper_noun_slot != PN_NONE:
                return state.model_copy(update={"proper_noun_slot": PN_NONE})
        return state

    if state.speaker_label_state != 0:
        return state

    word = state.last_completed_word
    if not word:
        return state
    lookup = word.lstrip("'")

    new_slot = PN_NONE
    if lookup in _TITLE_NOUNS:
        new_slot = PN_STRONG
    elif lookup in _VOCATIVE_LEADS:
        new_slot = PN_MILD
    elif lookup in _QUOTE_LEADS:
        new_slot = PN_QUOTE

    if new_slot != state.proper_noun_slot:
        return state.model_copy(update={"proper_noun_slot": new_slot})
    return state
