"""Anaphoric referent tracking pipeline stage.

Maintains `state.referent_gender` and `state.referent_staleness`:
after a gender-significant noun or proper noun enters the discourse,
we remember its gender so the predict layer can bias subsequent
pronouns toward the matching form.

Transitions (on word completion):

  - A proper noun (POS_PROPER_NOUN) → try to match a small table
    of Shakespeare character names to male/female; unknown proper
    names default to REF_MALE (Shakespeare's lexicon is male-
    dominated but this is a weak prior — only bumps H a little).

  - A role noun ("king", "queen", "lord", "lady", etc.) → set
    REF_MALE / REF_FEMALE / REF_NEUTER based on the noun.

  - A pronoun matching the current referent ("he" after REF_MALE,
    "she" after REF_FEMALE) → refresh, set staleness to 0.

  - A conflicting pronoun ("she" after REF_MALE) → switch the
    tracked gender, set staleness to 0.

  - Any other word completion → increment staleness.

Staleness >= 20 → clear (REF_NONE).

Speaker-turn change (new last_speaker_label): clear.

This stage runs AFTER update_speaker_memory so it can see the
current speaker context, but BEFORE flow (it's Tier 2 linguistic).
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB
from .pos import POS_PROPER_NOUN, POS_PRONOUN, POS_NOUN

REF_NONE = 0
REF_MALE = 1
REF_FEMALE = 2
REF_NEUTER = 3
REF_PLURAL = 4


# Role nouns whose grammatical gender is strongly predictable in
# Early Modern English.
_MALE_NOUNS: frozenset[str] = frozenset({
    "king", "lord", "duke", "sir", "master", "father", "son",
    "brother", "prince", "knight", "boy", "man", "husband",
    "friar", "priest", "monk", "soldier", "gentleman", "earl",
    "count", "baron", "page", "servant", "villain", "knave",
    "fool", "clown", "gravedigger", "friar", "ghost", "lad",
    "fellow", "uncle", "grandsire", "nephew", "sire",
    "emperor", "pope", "bishop", "squire", "lord",
})
_FEMALE_NOUNS: frozenset[str] = frozenset({
    "queen", "lady", "madam", "sister", "mother", "daughter",
    "wife", "maid", "nurse", "mistress", "dame", "girl",
    "princess", "duchess", "countess", "gentlewoman", "aunt",
    "niece", "woman", "widow", "matron", "dame",
})
_NEUTER_NOUNS: frozenset[str] = frozenset({
    "heart", "soul", "sword", "crown", "throne", "love", "death",
    "life", "fortune", "time", "night", "day", "sun", "moon",
    "stars", "star", "sea", "heaven", "hell", "word", "tongue",
    "mind", "breath", "tear", "blood", "body", "face", "eye",
    "hand", "head", "law", "truth", "honour", "fame", "virtue",
    "peace", "war", "battle", "storm", "rain", "wind", "fire",
    "earth", "flower", "rose", "tree", "bird", "horse", "dog",
    "lion", "book", "letter", "paper", "ring", "cup", "bread",
    "money", "gold",
})
_PLURAL_NOUNS: frozenset[str] = frozenset({
    "lords", "ladies", "gentlemen", "friends", "soldiers",
    "knights", "men", "women", "children", "brothers", "sisters",
    "sons", "daughters", "citizens", "princes", "kings", "queens",
    "dukes", "masters", "servants", "commoners", "nobles",
    "heavens", "stars", "fates", "gods",
})
# Common Shakespeare character-name genders (lowercased proper
# nouns). This is a hand-curated subset of major characters.
_MALE_NAMES: frozenset[str] = frozenset({
    "hamlet", "horatio", "claudius", "polonius", "laertes",
    "fortinbras", "osric", "reynaldo",
    "lear", "gloucester", "edgar", "edmund", "kent", "oswald",
    "albany", "cornwall", "fool",
    "macbeth", "banquo", "duncan", "malcolm", "donalbain",
    "macduff", "ross", "lennox",
    "othello", "iago", "cassio", "roderigo", "brabantio",
    "montano", "lodovico",
    "romeo", "mercutio", "benvolio", "tybalt", "paris", "friar",
    "capulet", "montague", "escalus",
    "caesar", "brutus", "cassius", "antony", "octavius", "casca",
    "cinna", "pompey",
    "henry", "hal", "falstaff", "hotspur", "bolingbroke",
    "richard", "bardolph", "pistol", "nym",
    "lysander", "demetrius", "oberon", "puck", "theseus",
    "bottom", "quince",
    "shylock", "antonio", "bassanio", "lorenzo", "gratiano",
    "launcelot", "tubal",
    "orlando", "jacques", "oliver", "duke", "touchstone",
    "prospero", "ferdinand", "caliban", "ariel", "stephano",
    "trinculo", "gonzalo", "alonso", "sebastian",
    "troilus", "pandarus", "hector", "priam", "agamemnon",
    "achilles", "ajax", "ulysses",
    "coriolanus", "menenius", "aufidius",
    "timon", "apemantus", "alcibiades",
    "lear", "lear",
    "john", "arthur", "philip", "lewis", "hubert", "robert",
    "petruchio", "lucentio", "grumio", "hortensio", "tranio",
    "gremio", "baptista", "vincentio",
    "benedick", "claudio", "don", "leonato", "borachio",
})
_FEMALE_NAMES: frozenset[str] = frozenset({
    "ophelia", "gertrude",
    "cordelia", "goneril", "regan",
    "lady", "hecate",
    "desdemona", "emilia", "bianca",
    "juliet", "nurse",
    "portia", "calpurnia",
    "katharine", "doll", "mistress", "quickly", "nell",
    "hermia", "helena", "hippolyta", "titania",
    "jessica", "nerissa",
    "rosalind", "celia", "phoebe", "audrey",
    "miranda",
    "cressida", "helen", "cassandra", "andromache",
    "volumnia", "virgilia",
    "constance", "blanch", "eleanor",
    "kate", "bianca",
    "beatrice", "hero", "ursula",
    "viola", "olivia", "maria",
    "isabella", "mariana", "juliet",
    "imogen", "innogen",
    "perdita", "hermione", "paulina",
    "marina", "thaisa",
    "cleopatra", "charmian", "iras", "octavia",
    "queen", "princess", "dido",
    "tamora", "lavinia",
})


# Pronouns and their associated referent-gender.
_PRONOUN_REFERENCE: dict[str, int] = {
    "he": REF_MALE, "him": REF_MALE, "his": REF_MALE, "himself": REF_MALE,
    "she": REF_FEMALE, "her": REF_FEMALE, "hers": REF_FEMALE, "herself": REF_FEMALE,
    "it": REF_NEUTER, "its": REF_NEUTER, "itself": REF_NEUTER,
    "they": REF_PLURAL, "them": REF_PLURAL, "their": REF_PLURAL,
    "theirs": REF_PLURAL, "themselves": REF_PLURAL,
}


def _classify(word: str, pos: int) -> int:
    """Return referent-gender class for a word, or REF_NONE if unknown."""
    w = word.lower()
    if w in _MALE_NOUNS:
        return REF_MALE
    if w in _FEMALE_NOUNS:
        return REF_FEMALE
    if w in _NEUTER_NOUNS:
        return REF_NEUTER
    if w in _PLURAL_NOUNS:
        return REF_PLURAL
    if pos == POS_PROPER_NOUN:
        if w in _MALE_NAMES:
            return REF_MALE
        if w in _FEMALE_NAMES:
            return REF_FEMALE
        # Unknown proper noun: return NONE (don't guess gender).
        return REF_NONE
    return REF_NONE


def update_referent(state: ModelState, token_id: int) -> ModelState:
    # Speaker-turn change: clear.
    # (This is detected later in pipeline but we observe
    # last_speaker_label which is set on ":" closing the label.
    # Instead, rely on staleness for cross-turn decay.)

    # Only update on word completion.
    if not (state.just_finished_word and state.last_completed_word):
        return state

    w = state.last_completed_word.lower()
    pos = state.last_word_pos
    cur = state.referent_gender
    stale = state.referent_staleness

    # Pronoun that matches current referent → refresh.
    pronoun_ref = _PRONOUN_REFERENCE.get(w)
    if pronoun_ref is not None:
        if pronoun_ref == cur and cur != REF_NONE:
            if stale != 0:
                return state.model_copy(update={"referent_staleness": 0})
            return state
        # Conflicting pronoun: switch to the pronoun's gender.
        return state.model_copy(
            update={
                "referent_gender": pronoun_ref,
                "referent_staleness": 0,
            }
        )

    # Try to classify as a gendered referent.
    new = _classify(w, pos)
    if new != REF_NONE:
        return state.model_copy(
            update={
                "referent_gender": new,
                "referent_staleness": 0,
            }
        )

    # Otherwise, stale the current referent.
    if cur == REF_NONE:
        return state
    new_stale = stale + 1
    if new_stale >= 20:
        return state.model_copy(
            update={
                "referent_gender": REF_NONE,
                "referent_staleness": 0,
            }
        )
    return state.model_copy(update={"referent_staleness": new_stale})
