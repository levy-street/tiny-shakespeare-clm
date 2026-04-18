"""Addressee / vocative-memory stage.

Records which vocative noun the current speaker has been using to
address their interlocutor. Updates `state.last_vocative` (the word,
lowercased) and `state.turn_vocative_count` (how many times).

Reset rules:
  - Speaker-turn change (consecutive_newlines >= 2): clear both.
  - (No reset on sentence-end — a speaker can address the same
    person across several sentences within one turn.)

Update rule at word completion:
  - If the completed word is a recognized vocative noun AND
    state.prev_completed_word is a vocative-lead token ("my",
    "thy", "good", "sweet", "dear", "gentle", "fair", "O",
    "noble", "kind", "worthy"), OR
  - the completed word is a vocative noun at sentence-start
    (words_in_sentence <= 1) — catches bare "Lord, ..." openings.
  Then set last_vocative to the completed word (lowercased) and
  increment turn_vocative_count.

This is not a counter of corpus occurrences — it's the model's
internal memory of what IT has committed to during this turn.
"""

from __future__ import annotations

from ..state import ModelState

# Canonical vocative nouns. Hand-picked from prior knowledge of
# Shakespearean address forms (not from corpus statistics).
_VOCATIVE_NOUNS: frozenset[str] = frozenset({
    # Noble titles
    "lord", "lords", "lady", "ladies", "liege", "sire", "majesty",
    "grace", "highness", "prince", "princess", "queen", "king",
    "duke", "duchess", "earl", "baron", "count",
    # Formal address
    "sir", "madam", "master", "mistress", "signior", "signor",
    "mesdames", "messieurs",
    # Kin
    "father", "mother", "brother", "sister", "son", "daughter",
    "cousin", "uncle", "aunt", "husband", "wife", "child",
    "kinsman", "kinsmen", "kinswoman",
    # Peer / friend
    "friend", "friends", "fellow", "fellows", "companion",
    "good", "gentle",
    # Hostile / pejorative
    "villain", "villains", "wretch", "slave", "traitor", "traitors",
    "rogue", "knave", "knaves", "rascal", "coward", "dog", "beast",
    # Juvenile / servant
    "boy", "lad", "girl", "maid", "page", "servant", "sirrah",
    "varlet",
    # Religious / clerical
    "priest", "father", "brother",
    # Military / office
    "captain", "lieutenant", "soldier", "soldiers", "guard", "guards",
    "knight", "knights",
    # Affectionate
    "love", "heart", "soul", "sweet", "dear", "dearest", "darling",
    "beloved", "fair",
    # Abstract vocative (Shakespearean)
    "fool", "fools", "gentleman", "gentlemen", "gentlewoman",
})

# Words that can LEAD a vocative construction — any of these
# preceding a vocative noun confirms the vocative reading.
_VOCATIVE_LEAD: frozenset[str] = frozenset({
    "my", "thy", "mine", "thine", "our", "your",
    "o", "oh", "ah",
    "good", "sweet", "dear", "gentle", "fair", "poor", "noble",
    "kind", "honest", "worthy", "gracious", "mighty", "royal",
    "brave", "true", "most",
})


def update_addressee(state: ModelState, token_id: int) -> ModelState:
    # Reset on speaker-turn boundary.
    if state.consecutive_newlines >= 2:
        if state.last_vocative or state.turn_vocative_count:
            return state.model_copy(update={
                "last_vocative": "",
                "turn_vocative_count": 0,
            })
        return state

    # Only inspect at word completion.
    if not state.just_finished_word or not state.last_completed_word:
        return state

    w = state.last_completed_word
    if w not in _VOCATIVE_NOUNS:
        return state

    prev = state.prev_completed_word
    is_after_lead = prev in _VOCATIVE_LEAD
    # Also accept bare sentence-start vocatives: "Lord, ...", "Friends,
    # Romans, countrymen ...".
    is_sentence_start = state.words_in_sentence <= 1

    if not (is_after_lead or is_sentence_start):
        return state

    # Don't re-register the same vocative repeatedly in a row —
    # increment count but don't change last_vocative.
    if w == state.last_vocative:
        return state.model_copy(update={
            "turn_vocative_count": state.turn_vocative_count + 1,
        })

    return state.model_copy(update={
        "last_vocative": w,
        "turn_vocative_count": state.turn_vocative_count + 1,
    })
