"""Word-bigram continuation bias.

Extends `next_word.py` beyond the first letter. `next_word` biases only
the *first* character of the new word given the previous completed
word. But conditioning on the previous word also sharpens letters 2, 3,
4 of the new word: after "to ", if buffer is "b", the next letter is
almost certainly "e" (to be) rather than "u" (to but) or "y" (to by).

Given `last_completed_word` and the current `word_buffer`, we consult
a prev-word-specific mini-trie of expected continuations and boost the
next letter along those paths. The bias scales with the number of
matching continuations and with how specific the prefix has become.

All data comes from prior knowledge of Shakespearean idiom —
no corpus statistics.
"""

from __future__ import annotations

import math

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# prev_word -> list of (expected next word, integer weight).
# Weights come from prior knowledge — common Shakespeare word pairs.
_EXPECT: dict[str, tuple[tuple[str, int], ...]] = {
    # Infinitive marker
    "to": (
        ("be", 14), ("have", 7), ("see", 7), ("do", 6), ("make", 5),
        ("take", 5), ("give", 4), ("speak", 5), ("hear", 4), ("tell", 4),
        ("come", 4), ("go", 4), ("know", 4), ("think", 3), ("find", 3),
        ("keep", 3), ("hold", 3), ("leave", 3), ("meet", 3), ("serve", 3),
        ("play", 3), ("bring", 3), ("bear", 3), ("live", 3), ("die", 3),
        ("love", 3), ("the", 10), ("my", 6), ("thy", 5), ("thee", 4),
        ("him", 4), ("her", 4), ("me", 4), ("us", 3), ("them", 3),
        ("a", 5), ("this", 3), ("that", 3), ("his", 4), ("your", 3),
        ("whom", 2), ("what", 2), ("no", 2), ("any", 2), ("such", 2),
        ("none", 1), ("thine", 2), ("our", 2),
    ),
    # First-person pronoun
    "i": (
        ("am", 14), ("will", 10), ("have", 9), ("do", 7), ("shall", 7),
        ("would", 6), ("could", 4), ("must", 6), ("may", 5), ("might", 4),
        ("pray", 6), ("did", 5), ("know", 5), ("think", 4), ("say", 3),
        ("saw", 3), ("said", 3), ("see", 4), ("fear", 3), ("love", 3),
        ("hope", 3), ("hear", 3), ("shall", 6), ("cannot", 3), ("come", 3),
        ("go", 2), ("beseech", 3), ("dare", 3), ("hold", 2), ("have", 8),
        ("thank", 3), ("crave", 2), ("charge", 2), ("swear", 2),
        ("had", 3), ("was", 3), ("like", 2), ("should", 2),
    ),
    # Second-person archaic
    "thou": (
        ("art", 12), ("hast", 10), ("dost", 7), ("doth", 3), ("didst", 5),
        ("shalt", 6), ("wilt", 6), ("wouldst", 4), ("couldst", 4),
        ("shouldst", 3), ("canst", 4), ("mayst", 3), ("knowest", 3),
        ("seest", 3), ("hearest", 2), ("speakest", 2), ("givest", 2),
        ("sayest", 2), ("lovest", 2), ("art", 8),
        ("and", 3), ("my", 3), ("thy", 2),
    ),
    # Articles/determiners
    "the": (
        ("king", 6), ("lord", 5), ("queen", 4), ("duke", 4), ("prince", 4),
        ("man", 5), ("men", 3), ("world", 4), ("night", 4), ("day", 4),
        ("sun", 3), ("moon", 3), ("stars", 2), ("sea", 3), ("earth", 3),
        ("heavens", 3), ("gods", 2), ("heart", 4), ("soul", 3), ("mind", 3),
        ("eye", 3), ("eyes", 3), ("hand", 3), ("face", 3), ("blood", 3),
        ("death", 3), ("life", 3), ("love", 3), ("time", 3), ("sword", 3),
        ("crown", 3), ("court", 3), ("queen", 3), ("house", 3), ("field", 3),
        ("gate", 2), ("battle", 2), ("war", 3), ("peace", 3), ("cause", 3),
        ("fault", 2), ("noble", 3), ("people", 3), ("truth", 3),
        ("same", 3), ("other", 3), ("first", 2), ("last", 2), ("next", 2),
        ("last", 2), ("very", 2), ("more", 2), ("most", 2), ("poor", 3),
        ("good", 3), ("great", 3), ("best", 2), ("worst", 2), ("worthy", 2),
        ("dead", 2), ("living", 2), ("fair", 3), ("true", 2),
        ("old", 2), ("young", 2), ("wise", 2),
    ),
    "a": (
        ("man", 4), ("woman", 2), ("fool", 3), ("king", 2), ("lord", 2),
        ("knight", 2), ("soldier", 2), ("beggar", 2), ("friend", 3),
        ("word", 3), ("sword", 3), ("horse", 2), ("letter", 2),
        ("tale", 3), ("name", 2), ("ring", 2), ("day", 2), ("night", 2),
        ("moment", 2), ("plague", 2), ("good", 3), ("great", 3),
        ("noble", 3), ("poor", 3), ("true", 2), ("brave", 2), ("wise", 2),
        ("little", 3), ("thousand", 2), ("hundred", 2), ("most", 2),
        ("little", 2), ("fair", 3), ("worthy", 2), ("fine", 2),
    ),
    "my": (
        ("lord", 12), ("liege", 4), ("lady", 5), ("dear", 4), ("love", 5),
        ("heart", 5), ("son", 3), ("father", 3), ("mother", 3),
        ("brother", 3), ("sister", 3), ("queen", 3), ("king", 3),
        ("soul", 4), ("life", 3), ("blood", 3), ("lord", 10), ("honour", 3),
        ("good", 3), ("own", 3), ("sweet", 3), ("gracious", 3),
        ("noble", 3), ("fair", 3), ("poor", 3), ("dear", 4), ("true", 2),
        ("hand", 3), ("eyes", 3), ("mind", 3), ("friend", 3),
    ),
    "thy": (
        ("father", 3), ("mother", 3), ("son", 3), ("lord", 3),
        ("hand", 3), ("heart", 4), ("soul", 3), ("love", 3), ("mind", 2),
        ("eyes", 3), ("face", 3), ("tongue", 2), ("name", 3), ("self", 3),
        ("wife", 2), ("daughter", 2), ("friend", 2), ("brother", 2),
        ("own", 3), ("good", 2), ("noble", 2), ("sweet", 2), ("dear", 2),
        ("fair", 2), ("wisdom", 2), ("anger", 2),
    ),
    "your": (
        ("grace", 6), ("lordship", 4), ("honour", 4), ("majesty", 5),
        ("highness", 3), ("worship", 3), ("good", 3), ("own", 3),
        ("dear", 3), ("noble", 3), ("gracious", 3), ("son", 2),
        ("father", 2), ("hand", 2), ("eyes", 2), ("heart", 2),
        ("daughter", 2), ("husband", 2), ("wife", 2), ("friend", 2),
    ),
    "his": (
        ("grace", 3), ("honour", 3), ("lordship", 3), ("father", 3),
        ("son", 3), ("mother", 2), ("hand", 3), ("heart", 3),
        ("face", 2), ("eyes", 2), ("brother", 2), ("wife", 2),
        ("sword", 2), ("horse", 2), ("name", 2), ("own", 3),
        ("good", 2), ("noble", 2), ("side", 2), ("foot", 2),
    ),
    "her": (
        ("grace", 3), ("majesty", 2), ("honour", 2), ("father", 3),
        ("son", 2), ("mother", 2), ("hand", 2), ("heart", 3),
        ("face", 2), ("eyes", 2), ("husband", 3), ("brother", 2),
        ("sweet", 2), ("fair", 2), ("own", 2), ("good", 2),
    ),
    # Prepositions
    "of": (
        ("the", 12), ("my", 6), ("his", 5), ("her", 4), ("your", 4),
        ("thy", 4), ("our", 4), ("their", 3), ("that", 3), ("this", 3),
        ("these", 2), ("those", 2), ("all", 3), ("any", 2), ("such", 3),
        ("mine", 2), ("thine", 2), ("love", 3), ("death", 3), ("life", 3),
        ("man", 2), ("men", 2), ("gods", 2), ("war", 2), ("peace", 2),
        ("england", 2), ("rome", 2), ("france", 2), ("honour", 2),
        ("old", 2), ("youth", 2), ("nothing", 2), ("no", 2),
    ),
    "in": (
        ("the", 10), ("my", 5), ("his", 5), ("her", 4), ("your", 3),
        ("thy", 3), ("our", 3), ("their", 3), ("this", 4), ("that", 3),
        ("these", 2), ("mine", 2), ("thine", 2), ("all", 3), ("any", 2),
        ("such", 2), ("sooth", 2), ("faith", 3), ("truth", 3), ("time", 2),
        ("heaven", 2), ("hell", 2), ("love", 2), ("peace", 2), ("war", 2),
        ("arms", 2), ("vain", 2),
    ),
    "on": (
        ("the", 8), ("my", 4), ("his", 4), ("her", 3), ("your", 3),
        ("thy", 3), ("our", 2), ("their", 2), ("this", 3), ("that", 3),
        ("these", 2), ("all", 2), ("such", 2), ("me", 3), ("us", 2),
        ("him", 3), ("her", 2), ("them", 2), ("you", 3), ("thee", 3),
        ("earth", 3),
    ),
    "with": (
        ("the", 7), ("my", 6), ("his", 5), ("her", 5), ("your", 4),
        ("thy", 4), ("our", 4), ("their", 3), ("me", 4), ("us", 3),
        ("him", 4), ("her", 3), ("them", 3), ("you", 4), ("thee", 4),
        ("all", 2), ("such", 2), ("more", 2), ("love", 2), ("honour", 2),
        ("tears", 2), ("such", 2), ("a", 3), ("an", 2), ("no", 2),
        ("what", 2), ("those", 2),
    ),
    "for": (
        ("the", 7), ("my", 5), ("his", 4), ("her", 4), ("your", 3),
        ("thy", 3), ("our", 3), ("their", 3), ("me", 4), ("us", 3),
        ("him", 4), ("her", 3), ("them", 3), ("you", 4), ("thee", 4),
        ("a", 4), ("an", 2), ("all", 3), ("such", 2), ("what", 2),
        ("ever", 2), ("shame", 2), ("god", 2), ("love", 2), ("this", 3),
        ("that", 3), ("these", 2), ("those", 2),
    ),
    "by": (
        ("the", 6), ("my", 4), ("his", 4), ("her", 3), ("your", 3),
        ("thy", 3), ("our", 2), ("their", 2), ("this", 3), ("that", 3),
        ("heaven", 3), ("god", 3), ("all", 2), ("me", 2), ("them", 2),
        ("which", 2), ("what", 2), ("himself", 2), ("herself", 2),
    ),
    "at": (
        ("the", 6), ("my", 3), ("his", 3), ("her", 3), ("your", 2),
        ("thy", 2), ("our", 2), ("their", 2), ("this", 2), ("that", 2),
        ("once", 3), ("length", 2), ("home", 2), ("last", 2),
        ("hand", 2), ("heart", 2), ("all", 2),
    ),
    "as": (
        ("the", 5), ("my", 3), ("his", 3), ("her", 3), ("your", 2),
        ("thy", 3), ("our", 2), ("their", 2), ("this", 2), ("that", 2),
        ("it", 4), ("thou", 3), ("you", 3), ("he", 3), ("she", 3),
        ("we", 3), ("they", 2), ("i", 4), ("much", 2), ("many", 2),
        ("such", 2), ("well", 3), ("if", 3), ("though", 2), ("true", 2),
        ("good", 2), ("far", 2), ("soon", 2),
    ),
    # Conjunctions / discourse openers
    "and": (
        ("the", 6), ("my", 4), ("his", 3), ("her", 3), ("your", 3),
        ("thy", 3), ("our", 3), ("their", 2), ("this", 3), ("that", 3),
        ("i", 5), ("he", 3), ("she", 3), ("we", 3), ("they", 2),
        ("thou", 3), ("you", 3), ("if", 3), ("when", 3), ("then", 3),
        ("so", 3), ("yet", 3), ("let", 3), ("leave", 2), ("make", 2),
        ("bid", 2), ("see", 2), ("know", 2), ("therefore", 2),
        ("all", 2), ("more", 2), ("now", 3),
    ),
    "but": (
        ("the", 4), ("my", 3), ("his", 2), ("her", 2), ("your", 2),
        ("thy", 2), ("our", 2), ("their", 2), ("this", 2), ("that", 3),
        ("i", 5), ("he", 3), ("she", 3), ("we", 3), ("they", 2),
        ("thou", 3), ("you", 3), ("if", 2), ("when", 2), ("now", 2),
        ("yet", 3), ("soft", 2), ("hark", 2), ("see", 2), ("hold", 2),
        ("stay", 2), ("what", 2), ("where", 2), ("who", 2),
        ("nothing", 2), ("no", 2), ("one", 2),
    ),
    "or": (
        ("the", 3), ("my", 2), ("his", 2), ("her", 2), ("this", 2),
        ("that", 2), ("no", 3), ("not", 2), ("if", 2), ("when", 2),
        ("thou", 2), ("i", 3), ("we", 2), ("he", 2), ("she", 2),
        ("any", 2), ("some", 2), ("else", 3), ("other", 2),
    ),
    # Modals/aux — what usually follows
    "shall": (
        ("i", 5), ("we", 4), ("he", 3), ("she", 2), ("they", 2),
        ("you", 3), ("thou", 3), ("be", 5), ("have", 4), ("not", 4),
        ("do", 3), ("make", 2), ("see", 2), ("go", 2), ("come", 2),
        ("find", 2), ("give", 2), ("take", 2), ("the", 2), ("my", 2),
        ("speak", 2), ("know", 2),
    ),
    "will": (
        ("i", 2), ("we", 2), ("not", 4), ("be", 4), ("have", 3),
        ("do", 3), ("make", 2), ("see", 2), ("go", 2), ("come", 2),
        ("give", 2), ("take", 2), ("speak", 2), ("tell", 2), ("it", 3),
        ("you", 2), ("thee", 2), ("the", 2), ("my", 2),
    ),
    "do": (
        ("not", 6), ("i", 3), ("we", 2), ("you", 3), ("thee", 3),
        ("me", 3), ("him", 2), ("her", 2), ("them", 2), ("it", 3),
        ("so", 3), ("well", 2), ("this", 2), ("that", 2), ("the", 2),
        ("my", 2), ("pray", 2), ("beseech", 2), ("entreat", 2),
        ("love", 2), ("say", 2), ("know", 2), ("think", 2),
    ),
    "have": (
        ("i", 4), ("we", 3), ("you", 3), ("thou", 3), ("not", 4),
        ("been", 4), ("seen", 3), ("said", 3), ("heard", 3), ("done", 4),
        ("made", 3), ("found", 3), ("given", 2), ("taken", 2),
        ("had", 2), ("no", 3), ("a", 3), ("the", 2), ("my", 2),
        ("thee", 3), ("me", 3), ("it", 3), ("some", 2), ("much", 2),
    ),
    "am": (
        ("i", 2), ("not", 4), ("a", 4), ("the", 2), ("no", 3),
        ("thy", 2), ("your", 2), ("sure", 3), ("come", 2), ("glad", 2),
        ("sorry", 2), ("afraid", 2), ("content", 2), ("none", 2),
        ("bound", 2),
    ),
    "art": (
        ("thou", 4), ("a", 4), ("the", 2), ("not", 4), ("no", 2),
        ("my", 3), ("our", 2), ("mine", 2), ("come", 2), ("sure", 2),
        ("so", 2), ("too", 2), ("more", 2), ("most", 2), ("but", 2),
    ),
    "is": (
        ("the", 5), ("a", 5), ("an", 2), ("my", 3), ("his", 3),
        ("her", 3), ("your", 3), ("thy", 3), ("our", 2), ("their", 2),
        ("this", 3), ("that", 3), ("no", 3), ("not", 3), ("it", 3),
        ("she", 2), ("he", 2), ("true", 3), ("good", 2), ("well", 2),
        ("dead", 2), ("gone", 2), ("come", 2), ("here", 2), ("there", 2),
        ("ever", 2), ("most", 2), ("more", 2), ("one", 2), ("any", 2),
    ),
    "was": (
        ("a", 4), ("the", 4), ("an", 2), ("my", 3), ("his", 3),
        ("her", 3), ("your", 2), ("thy", 2), ("our", 2), ("their", 2),
        ("this", 2), ("that", 2), ("no", 2), ("not", 3), ("ever", 2),
        ("never", 2), ("it", 3), ("he", 2), ("she", 2), ("one", 2),
        ("more", 2), ("most", 2),
    ),
    "are": (
        ("the", 4), ("you", 3), ("we", 3), ("they", 2), ("my", 2),
        ("his", 2), ("her", 2), ("your", 2), ("thy", 2), ("our", 2),
        ("their", 2), ("not", 3), ("no", 2), ("all", 3), ("these", 2),
        ("those", 2), ("come", 2), ("gone", 2), ("ever", 2), ("most", 2),
    ),
    "be": (
        ("not", 4), ("a", 3), ("the", 2), ("my", 2), ("no", 3),
        ("so", 3), ("it", 2), ("gone", 2), ("done", 2), ("as", 2),
        ("thou", 2), ("you", 2), ("more", 2), ("most", 2), ("sure", 2),
    ),
    # Negations / interjections
    "o": (
        ("my", 5), ("thou", 4), ("thee", 3), ("lord", 4), ("heaven", 4),
        ("god", 3), ("gods", 3), ("what", 4), ("why", 3), ("how", 3),
        ("where", 3), ("when", 2), ("woe", 3), ("most", 2), ("that", 3),
        ("sweet", 3), ("dear", 3), ("good", 3), ("noble", 3), ("fair", 3),
        ("sir", 3), ("gentle", 2),
    ),
    "no": (
        ("more", 4), ("less", 2), ("other", 2), ("man", 3), ("king", 2),
        ("one", 3), ("doubt", 2), ("matter", 3), ("sir", 3), ("my", 3),
        ("i", 3), ("not", 3), ("no", 2), ("such", 2), ("shame", 2),
        ("words", 2), ("harm", 2), ("words", 2), ("he", 2), ("she", 2),
    ),
    "not": (
        ("so", 4), ("to", 3), ("a", 3), ("the", 3), ("my", 3),
        ("his", 2), ("her", 2), ("your", 2), ("thy", 2), ("for", 2),
        ("in", 2), ("with", 2), ("be", 3), ("have", 2), ("i", 3),
        ("you", 2), ("he", 2), ("she", 2), ("we", 2), ("they", 2),
        ("one", 2), ("any", 2), ("worth", 2), ("now", 2),
    ),
    # Pronoun-based
    "he": (
        ("is", 5), ("was", 5), ("hath", 5), ("had", 4), ("will", 3),
        ("would", 3), ("shall", 3), ("doth", 3), ("did", 3), ("does", 2),
        ("has", 2), ("must", 2), ("may", 2), ("might", 2), ("cannot", 2),
        ("comes", 2), ("goes", 2), ("knows", 2), ("speaks", 2),
        ("stands", 2), ("holds", 2), ("lives", 2), ("loves", 2),
        ("that", 3), ("who", 3), ("which", 2),
    ),
    "she": (
        ("is", 5), ("was", 5), ("hath", 4), ("had", 3), ("will", 3),
        ("would", 3), ("shall", 3), ("doth", 3), ("did", 3), ("does", 2),
        ("has", 2), ("must", 2), ("may", 2), ("might", 2), ("cannot", 2),
        ("comes", 2), ("speaks", 2), ("loves", 2), ("that", 2),
    ),
    "we": (
        ("are", 4), ("were", 3), ("have", 4), ("had", 3), ("will", 4),
        ("would", 3), ("shall", 4), ("must", 3), ("may", 2), ("might", 2),
        ("do", 3), ("did", 3), ("know", 2), ("shall", 3), ("cannot", 2),
        ("come", 2), ("go", 2), ("have", 3), ("three", 2), ("two", 2),
        ("both", 2),
    ),
    "they": (
        ("are", 4), ("were", 3), ("have", 3), ("had", 3), ("will", 3),
        ("would", 3), ("shall", 2), ("do", 3), ("did", 3),
        ("come", 2), ("go", 2), ("say", 2), ("know", 2), ("must", 2),
    ),
    "you": (
        ("are", 4), ("have", 4), ("will", 4), ("shall", 3), ("must", 3),
        ("do", 3), ("did", 3), ("may", 2), ("might", 2), ("know", 3),
        ("see", 2), ("say", 2), ("speak", 2), ("think", 2), ("had", 2),
        ("were", 2), ("cannot", 2),
    ),
    "me": (
        ("to", 5), ("not", 4), ("the", 3), ("a", 2), ("in", 3),
        ("with", 3), ("of", 3), ("from", 2), ("and", 3), ("my", 2),
        ("on", 2), ("thy", 2), ("no", 2), ("be", 2), ("more", 2),
        ("how", 2), ("what", 2), ("where", 2), ("why", 2),
    ),
    "him": (
        ("to", 4), ("not", 3), ("the", 2), ("a", 2), ("in", 2),
        ("with", 2), ("of", 2), ("and", 3), ("that", 2), ("who", 2),
        ("for", 2), ("on", 2), ("from", 2), ("into", 2), ("home", 2),
        ("hither", 2),
    ),
    "her": (
        ("to", 3), ("not", 3), ("the", 2), ("a", 2), ("in", 2),
        ("with", 2), ("and", 3), ("that", 2), ("who", 2), ("for", 2),
        ("on", 2), ("hand", 2), ("heart", 2), ("eyes", 2), ("own", 2),
    ),
    # Verse/discourse openers
    "when": (
        ("i", 4), ("thou", 3), ("he", 3), ("she", 2), ("we", 3),
        ("they", 2), ("you", 2), ("the", 3), ("my", 2), ("this", 2),
        ("that", 2), ("first", 2), ("last", 2),
    ),
    "if": (
        ("i", 4), ("thou", 4), ("he", 3), ("she", 2), ("we", 3),
        ("they", 2), ("you", 3), ("the", 3), ("this", 2), ("that", 2),
        ("it", 3), ("any", 2), ("not", 2), ("ever", 2),
    ),
    "that": (
        ("i", 4), ("thou", 3), ("he", 3), ("she", 2), ("we", 3),
        ("they", 2), ("you", 3), ("is", 3), ("was", 3), ("were", 2),
        ("which", 3), ("the", 2), ("my", 2), ("this", 2), ("hath", 2),
        ("had", 2), ("will", 2), ("would", 2), ("shall", 2),
    ),
    "which": (
        ("i", 3), ("he", 3), ("she", 2), ("we", 3), ("they", 2),
        ("you", 3), ("is", 3), ("was", 3), ("were", 2), ("hath", 2),
        ("the", 2), ("my", 2), ("thou", 2), ("being", 2),
    ),
    "what": (
        ("is", 4), ("was", 3), ("are", 3), ("were", 2), ("i", 3),
        ("thou", 3), ("he", 3), ("she", 2), ("we", 2), ("they", 2),
        ("you", 3), ("a", 3), ("the", 2), ("my", 2), ("more", 2),
        ("say", 2), ("meant", 2), ("said", 2), ("do", 2), ("hath", 2),
        ("shall", 2), ("will", 2), ("should", 2), ("would", 2),
        ("news", 2), ("news", 2), ("cheer", 2), ("ho", 2), ("man", 2),
        ("say", 2), ("sir", 2),
    ),
    "then": (
        ("i", 4), ("thou", 3), ("he", 3), ("she", 2), ("we", 3),
        ("they", 2), ("you", 3), ("the", 3), ("my", 2), ("this", 2),
        ("that", 2), ("let", 3), ("come", 2), ("go", 2), ("speak", 2),
        ("know", 2), ("up", 2), ("hear", 2),
    ),
    "for": (
        ("the", 7), ("my", 5), ("his", 4), ("her", 4), ("your", 3),
        ("thy", 3), ("our", 3), ("their", 3), ("me", 4), ("us", 3),
        ("him", 4), ("her", 3), ("them", 3), ("you", 4), ("thee", 4),
        ("a", 4), ("an", 2), ("all", 3), ("such", 2), ("what", 2),
        ("ever", 2), ("shame", 2), ("god", 2), ("love", 2), ("this", 3),
        ("that", 3), ("these", 2), ("those", 2),
    ),
    # Possessive / existentials
    "there": (
        ("is", 5), ("was", 4), ("are", 3), ("were", 3), ("be", 3),
        ("shall", 2), ("will", 2), ("was", 3), ("lies", 2), ("stands", 2),
        ("hath", 2), ("comes", 2),
    ),
    "here": (
        ("is", 4), ("comes", 3), ("hath", 2), ("was", 2), ("stands", 2),
        ("lies", 2), ("be", 2), ("we", 3), ("i", 3), ("thou", 2),
    ),
    "this": (
        ("is", 5), ("was", 3), ("day", 3), ("night", 3), ("man", 3),
        ("lord", 3), ("king", 2), ("the", 2), ("my", 2), ("that", 2),
        ("side", 2), ("way", 2), ("hour", 2), ("time", 2), ("house", 2),
        ("world", 2), ("land", 2), ("business", 2), ("matter", 2),
    ),
    "these": (
        ("are", 3), ("were", 2), ("my", 3), ("thy", 2), ("his", 2),
        ("her", 2), ("your", 2), ("our", 2), ("their", 2), ("eyes", 2),
        ("words", 2), ("tears", 2), ("men", 2), ("days", 2), ("hands", 2),
    ),
    # Frequently used content openers
    "my": (
        ("lord", 12), ("liege", 4), ("lady", 5), ("dear", 4), ("love", 5),
        ("heart", 5), ("son", 3), ("father", 3), ("mother", 3),
        ("brother", 3), ("sister", 3), ("queen", 3), ("king", 3),
        ("soul", 4), ("life", 3), ("blood", 3), ("lord", 10), ("honour", 3),
        ("good", 3), ("own", 3), ("sweet", 3), ("gracious", 3),
        ("noble", 3), ("fair", 3), ("poor", 3), ("dear", 4), ("true", 2),
        ("hand", 3), ("eyes", 3), ("mind", 3), ("friend", 3),
    ),
    "good": (
        ("my", 3), ("lord", 5), ("sir", 4), ("friend", 3), ("night", 4),
        ("morrow", 3), ("faith", 2), ("fellow", 2), ("master", 2),
        ("man", 2), ("lady", 2), ("madam", 2), ("brother", 2), ("son", 2),
    ),
    "old": (
        ("man", 2), ("father", 2), ("friend", 2), ("time", 2), ("age", 1),
    ),
    "young": (
        ("lord", 2), ("prince", 2), ("man", 2), ("sir", 2), ("knight", 2),
    ),
    "hath": (
        ("been", 4), ("seen", 3), ("said", 3), ("done", 3), ("made", 3),
        ("found", 2), ("taken", 2), ("given", 2), ("not", 3),
        ("a", 3), ("the", 2), ("my", 2), ("his", 2), ("her", 2),
        ("no", 2), ("so", 2), ("never", 2),
    ),
    "hast": (
        ("been", 3), ("seen", 3), ("said", 2), ("done", 3), ("made", 2),
        ("found", 2), ("taken", 2), ("given", 2), ("not", 3), ("thou", 2),
        ("a", 3), ("the", 2), ("my", 2), ("no", 2),
    ),
}


def _build_continuation_tries() -> dict[str, dict[str, dict[str, int]]]:
    """For each prev_word, build a trie of expected next-word continuations.

    Structure: prev_word -> prefix -> {next_char: weight}.
    """
    out: dict[str, dict[str, dict[str, int]]] = {}
    for prev, entries in _EXPECT.items():
        trie: dict[str, dict[str, int]] = {}
        for word, w in entries:
            word = word.lower()
            for i in range(len(word) + 1):
                prefix = word[:i]
                trie.setdefault(prefix, {})
                if i < len(word):
                    nxt = word[i]
                    trie[prefix][nxt] = trie[prefix].get(nxt, 0) + w
                else:
                    # Word terminator — boost space/punct to close cleanly.
                    for term in (" ", ",", ".", ";", ":", "?", "!", "\n"):
                        trie[prefix][term] = trie[prefix].get(term, 0) + w
        out[prev] = trie
    return out


_CONT_TRIES: dict[str, dict[str, dict[str, int]]] = _build_continuation_tries()


def _build_vector(nexts: dict[str, int], prefix_len: int) -> list[float]:
    """Build a VOCAB bias vector from a next-char weight map."""
    vec = [0.0] * VOCAB_SIZE
    total = sum(nexts.values())
    if total <= 0:
        return vec
    # Scale grows with prefix length — longer prefix = more informative.
    scale = min(0.35 + 0.2 * prefix_len, 1.1)
    # Small negative bump on all unlisted letters so unseen branches
    # are softly penalized.
    negative = -0.4 * scale
    for ch in "abcdefghijklmnopqrstuvwxyz":
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = negative
    for ch, w in nexts.items():
        if ch not in VOCAB_INDEX:
            continue
        frac = w / total
        bias = scale * math.log((frac + 0.03) / 0.05)
        vec[VOCAB_INDEX[ch]] = bias
        if ch.isalpha() and ch != ch.upper():
            up = ch.upper()
            if up in VOCAB_INDEX:
                vec[VOCAB_INDEX[up]] = bias * 0.5
    return vec


def _build_prefix_bias() -> dict[tuple[str, str], list[float]]:
    """Precompute bias vectors keyed by (prev_word, prefix)."""
    out: dict[tuple[str, str], list[float]] = {}
    for prev, trie in _CONT_TRIES.items():
        for prefix, nexts in trie.items():
            if not prefix:
                continue  # first-letter handled by next_word already
            out[(prev, prefix)] = _build_vector(nexts, len(prefix))
    return out


_PREFIX_BIAS: dict[tuple[str, str], list[float]] = _build_prefix_bias()


def word_bigram_continue_bias(
    last_completed_word: str,
    word_buffer: str,
    letter_run_len: int,
    speaker_label_state: int,
    on_word_trie: bool,
) -> list[float] | None:
    """Return bias vector for the next-char given the previous word and
    the current word buffer. Active only when:
      - We're NOT inside a speaker label.
      - The current buffer is 1-4 letters long.
      - last_completed_word is in our expectation dict.
      - The current buffer is a known prefix of some expected continuation.
    """
    if speaker_label_state != 0:
        return None
    if not last_completed_word:
        return None
    if not word_buffer:
        return None
    if letter_run_len < 1 or letter_run_len > 8:
        return None
    if len(word_buffer) != letter_run_len:
        return None
    prev = last_completed_word.lower()
    trie = _CONT_TRIES.get(prev)
    if trie is None:
        return None
    key = word_buffer.lower()
    return _PREFIX_BIAS.get((prev, key))
