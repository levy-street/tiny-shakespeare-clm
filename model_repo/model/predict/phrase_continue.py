"""Two-word phrase continuation bias.

Extends `phrase_bigram.py` beyond the first letter of the next word.
Given the last TWO completed words (prev_completed_word and
last_completed_word), bias the full mid-word trajectory (letters 1-6)
of the current word toward canonical continuations of that phrase:

  "I am" + "n" → "o" (not), "e" (nearly? no — use weights)
  "I have" + "b" → "e" (been)
  "I pray" + "t" → "h" (thee)
  "thou hast" + "b" → "e" (been)
  "my lord" + "i" → " " (often word-alone) / some common follow
  "of the" + "k" → "i" (king)
  "in the" + "w" → "o" (world)
  "is a" + "m" → "a" (man/matter)
  "do not" + "s" → "a" (say), "p" (speak), "e" (see)

Runs alongside `word_bigram_continue` (1-word conditioning). When both
fire, they stack — 2-word signal is stronger. Active only outside
speaker labels, at buffers 1-6 letters long. All entries from prior
knowledge of Shakespeare idiom.
"""

from __future__ import annotations

import math

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_EXPECT: dict[tuple[str, str], tuple[tuple[str, int], ...]] = {
    # --- I + X ---
    ("i", "am"): (
        ("not", 6), ("a", 4), ("the", 3), ("no", 3), ("sure", 4),
        ("content", 3), ("glad", 3), ("sorry", 3), ("afraid", 3),
        ("come", 3), ("thy", 3), ("your", 2), ("bound", 2), ("none", 2),
        ("one", 2), ("much", 2), ("more", 2), ("his", 2), ("so", 3),
        ("in", 2), ("out", 2), ("but", 2), ("as", 2), ("of", 2),
    ),
    ("i", "have"): (
        ("been", 6), ("seen", 4), ("said", 3), ("done", 4), ("made", 3),
        ("found", 3), ("heard", 3), ("given", 2), ("taken", 2),
        ("no", 4), ("not", 3), ("a", 3), ("the", 2), ("my", 2),
        ("thee", 3), ("it", 3), ("you", 2), ("some", 2), ("much", 2),
        ("such", 2), ("enough", 2),
    ),
    ("i", "had"): (
        ("rather", 3), ("thought", 2), ("been", 3), ("seen", 2),
        ("heard", 2), ("done", 2), ("a", 3), ("no", 3), ("not", 2),
        ("the", 2), ("my", 2), ("forgot", 2),
    ),
    ("i", "will"): (
        ("not", 5), ("be", 4), ("have", 3), ("do", 3), ("make", 2),
        ("see", 2), ("go", 3), ("come", 2), ("give", 2), ("take", 2),
        ("speak", 3), ("tell", 3), ("hear", 2), ("keep", 2), ("stay", 2),
        ("stand", 2), ("bring", 2), ("send", 2), ("find", 2), ("look", 2),
        ("unto", 2), ("you", 2), ("with", 2), ("to", 2),
    ),
    ("i", "do"): (
        ("not", 6), ("beseech", 4), ("entreat", 3), ("love", 3),
        ("pray", 3), ("remember", 2), ("think", 3), ("fear", 3),
        ("know", 3), ("see", 2), ("assure", 2), ("protest", 2),
        ("confess", 2),
    ),
    ("i", "did"): (
        ("not", 5), ("never", 2), ("think", 3), ("hear", 3), ("see", 3),
        ("know", 2), ("love", 2), ("speak", 2), ("stand", 2), ("tell", 2),
    ),
    ("i", "would"): (
        ("not", 4), ("fain", 3), ("have", 3), ("speak", 3), ("be", 3),
        ("rather", 3), ("to", 2), ("the", 2), ("a", 2), ("my", 2),
        ("hear", 2), ("see", 2), ("know", 2), ("say", 2), ("thou", 2),
        ("you", 2),
    ),
    ("i", "could"): (
        ("not", 4), ("have", 2), ("never", 2), ("speak", 2), ("say", 2),
        ("wish", 2), ("find", 2), ("tell", 2), ("hear", 2), ("see", 2),
        ("the", 2),
    ),
    ("i", "shall"): (
        ("not", 4), ("be", 4), ("have", 3), ("do", 2), ("make", 2),
        ("see", 2), ("go", 2), ("come", 2), ("speak", 2), ("tell", 2),
        ("live", 2), ("die", 2), ("know", 2), ("find", 2),
    ),
    ("i", "must"): (
        ("be", 3), ("have", 2), ("not", 3), ("needs", 3), ("to", 2),
        ("confess", 2), ("speak", 2), ("go", 2), ("tell", 2), ("look", 2),
    ),
    ("i", "may"): (
        ("not", 3), ("be", 3), ("do", 2), ("have", 2), ("speak", 2),
        ("see", 2), ("know", 2), ("say", 2),
    ),
    ("i", "pray"): (
        ("thee", 10), ("you", 8), ("sir", 4), ("my", 3), ("thy", 2),
        ("god", 2), ("heaven", 2), ("for", 2), ("to", 2), ("me", 2),
        ("tell", 2), ("speak", 2),
    ),
    ("i", "beseech"): (
        ("thee", 8), ("you", 6), ("your", 4), ("my", 3), ("the", 2),
        ("for", 2), ("god", 2), ("heaven", 2),
    ),
    ("i", "see"): (
        ("the", 3), ("no", 3), ("not", 2), ("it", 3), ("thee", 2),
        ("you", 2), ("him", 2), ("her", 2), ("them", 2), ("my", 2),
        ("thy", 2), ("how", 2), ("what", 2),
    ),
    ("i", "know"): (
        ("not", 5), ("the", 3), ("my", 2), ("thy", 2), ("him", 2),
        ("her", 2), ("them", 2), ("you", 2), ("it", 3), ("no", 2),
        ("what", 2), ("who", 2), ("thee", 2), ("thou", 2), ("he", 2),
        ("she", 2), ("that", 2),
    ),
    ("i", "hear"): (
        ("the", 3), ("you", 2), ("him", 2), ("her", 2), ("them", 2),
        ("thee", 2), ("thou", 2), ("no", 2), ("not", 2), ("a", 2),
    ),
    ("i", "think"): (
        ("he", 2), ("she", 2), ("thou", 2), ("you", 2), ("it", 3),
        ("not", 2), ("the", 2), ("my", 2), ("no", 2), ("so", 3),
        ("this", 2), ("that", 2),
    ),
    ("i", "thank"): (
        ("thee", 4), ("you", 4), ("the", 2), ("my", 2), ("god", 2),
        ("heaven", 2), ("your", 2),
    ),
    ("i", "was"): (
        ("a", 3), ("the", 2), ("not", 3), ("no", 2), ("born", 2),
        ("ever", 2), ("never", 2), ("in", 2), ("of", 2), ("too", 2),
    ),
    ("i", "dare"): (
        ("not", 3), ("swear", 2), ("say", 2), ("tell", 2), ("be", 2),
        ("speak", 2), ("it", 2), ("no", 2),
    ),
    # --- thou + X ---
    ("thou", "art"): (
        ("a", 4), ("the", 3), ("my", 3), ("thy", 2), ("not", 4),
        ("no", 2), ("mine", 2), ("our", 2), ("come", 2), ("sure", 2),
        ("so", 3), ("too", 2), ("more", 2), ("most", 2), ("but", 2),
        ("as", 2), ("in", 2), ("of", 2),
    ),
    ("thou", "hast"): (
        ("been", 4), ("seen", 3), ("said", 2), ("done", 3), ("made", 2),
        ("found", 2), ("taken", 2), ("given", 2), ("not", 4), ("a", 3),
        ("the", 2), ("my", 2), ("no", 2), ("thy", 2), ("thee", 2),
    ),
    ("thou", "dost"): (
        ("not", 3), ("love", 2), ("speak", 2), ("know", 2), ("see", 2),
        ("hear", 2), ("make", 2), ("take", 2), ("me", 2),
    ),
    ("thou", "wilt"): (
        ("not", 3), ("be", 3), ("have", 2), ("do", 2), ("see", 2),
        ("say", 2), ("find", 2), ("go", 2), ("come", 2), ("make", 2),
        ("prove", 2),
    ),
    ("thou", "shalt"): (
        ("not", 4), ("be", 3), ("have", 2), ("find", 2), ("see", 2),
        ("go", 2), ("know", 2), ("hear", 2), ("die", 2), ("live", 2),
    ),
    ("thou", "didst"): (
        ("not", 3), ("speak", 2), ("say", 2), ("see", 2), ("hear", 2),
        ("love", 2), ("promise", 2), ("know", 2),
    ),
    # --- my + X ---
    ("my", "lord"): (
        ("i", 4), ("is", 2), ("it", 3), ("what", 3), ("how", 2),
        ("where", 2), ("when", 2), ("why", 2), ("you", 2), ("he", 2),
        ("she", 2), ("we", 2), ("they", 2), ("the", 2), ("my", 2),
        ("thy", 2), ("this", 2), ("that", 2), ("speak", 2), ("hear", 2),
        ("no", 2), ("ay", 2),
    ),
    ("my", "good"): (
        ("lord", 6), ("sir", 3), ("friend", 3), ("master", 2), ("lady", 2),
        ("madam", 2), ("prince", 2), ("cousin", 2),
    ),
    ("my", "noble"): (
        ("lord", 4), ("father", 2), ("friend", 2), ("son", 2),
        ("brother", 2), ("prince", 2), ("duke", 2), ("cousin", 2),
    ),
    ("my", "dear"): (
        ("lord", 3), ("friend", 3), ("son", 3), ("father", 2),
        ("mother", 2), ("brother", 2), ("sister", 2), ("heart", 2),
        ("love", 2), ("cousin", 2), ("queen", 2),
    ),
    ("my", "sweet"): (
        ("lord", 3), ("love", 3), ("lady", 2), ("heart", 2), ("soul", 2),
        ("son", 2), ("queen", 2),
    ),
    ("my", "gracious"): (
        ("lord", 5), ("sovereign", 3), ("liege", 3), ("queen", 2),
        ("king", 2),
    ),
    # --- good + X ---
    ("good", "my"): (
        ("lord", 6), ("liege", 4), ("lady", 3), ("dear", 2), ("love", 2),
    ),
    ("good", "sir"): (
        ("i", 3), ("what", 2), ("how", 2), ("the", 2), ("my", 2),
        ("why", 2), ("it", 2), ("you", 2), ("thou", 2),
    ),
    # --- of + X ---
    ("of", "the"): (
        ("king", 4), ("world", 3), ("house", 3), ("queen", 2), ("duke", 2),
        ("prince", 2), ("lord", 2), ("people", 2), ("men", 2), ("gods", 2),
        ("day", 2), ("night", 2), ("sun", 2), ("moon", 2), ("sea", 2),
        ("earth", 2), ("land", 2), ("court", 2), ("state", 2), ("time", 2),
        ("same", 2), ("other", 2), ("first", 2), ("last", 2), ("dead", 2),
        ("noble", 2), ("best", 2), ("most", 2), ("worst", 2), ("poor", 2),
        ("father", 2), ("son", 2), ("mother", 2), ("holy", 2),
    ),
    ("of", "my"): (
        ("lord", 4), ("love", 3), ("heart", 3), ("soul", 2), ("life", 2),
        ("father", 2), ("son", 2), ("mother", 2), ("blood", 2), ("own", 2),
        ("own", 2), ("mind", 2), ("master", 2),
    ),
    ("of", "his"): (
        ("grace", 3), ("father", 2), ("son", 2), ("brother", 2),
        ("sword", 2), ("hand", 2), ("head", 2), ("death", 2), ("life", 2),
        ("honour", 2), ("heart", 2), ("soul", 2),
    ),
    ("of", "all"): (
        ("men", 3), ("the", 2), ("my", 2), ("his", 2), ("his", 2),
        ("these", 2), ("those", 2), ("our", 2), ("their", 2), ("your", 2),
        ("thy", 2), ("this", 2),
    ),
    ("of", "your"): (
        ("grace", 3), ("lordship", 2), ("honour", 2), ("majesty", 2),
        ("father", 2), ("son", 2), ("love", 2), ("own", 2),
    ),
    # --- in + X ---
    ("in", "the"): (
        ("world", 3), ("name", 3), ("end", 2), ("morning", 2), ("night", 2),
        ("day", 2), ("first", 2), ("last", 2), ("midst", 2), ("king", 2),
        ("queen", 2), ("city", 2), ("house", 2), ("court", 2), ("sea", 2),
        ("same", 2), ("field", 2), ("state", 2), ("streets", 2),
    ),
    ("in", "my"): (
        ("heart", 3), ("soul", 2), ("mind", 2), ("sight", 2), ("life", 2),
        ("youth", 2), ("hands", 2), ("own", 2), ("house", 2), ("dream", 2),
        ("ear", 2), ("name", 2), ("bosom", 2),
    ),
    ("in", "this"): (
        ("world", 3), ("place", 3), ("house", 2), ("kind", 2), ("case", 2),
        ("matter", 2), ("cause", 2), ("wise", 2), ("business", 2),
        ("court", 2), ("state", 2),
    ),
    # --- on + X ---
    ("on", "the"): (
        ("earth", 3), ("ground", 2), ("king", 2), ("duke", 2), ("queen", 2),
        ("prince", 2), ("same", 2), ("other", 2), ("morrow", 2), ("head", 2),
        ("sea", 2), ("face", 2), ("floor", 2), ("wall", 2), ("field", 2),
    ),
    # --- with + X ---
    ("with", "the"): (
        ("king", 3), ("duke", 2), ("queen", 2), ("rest", 2), ("others", 2),
        ("lord", 2), ("prince", 2), ("sword", 2), ("eye", 2), ("same", 2),
    ),
    ("with", "my"): (
        ("sword", 2), ("hand", 2), ("heart", 2), ("life", 2), ("love", 2),
        ("soul", 2), ("own", 2), ("lord", 2),
    ),
    # --- to + X ---
    ("to", "the"): (
        ("king", 4), ("queen", 2), ("duke", 2), ("court", 3), ("world", 2),
        ("death", 3), ("ground", 2), ("same", 2), ("other", 2), ("first", 2),
        ("last", 2), ("end", 2), ("gates", 2), ("house", 2), ("field", 2),
        ("people", 2), ("streets", 2), ("prince", 2),
    ),
    ("to", "my"): (
        ("lord", 4), ("heart", 3), ("soul", 2), ("father", 2), ("love", 2),
        ("mother", 2), ("son", 2), ("brother", 2), ("wife", 2), ("house", 2),
        ("grave", 2),
    ),
    ("to", "thee"): (
        ("i", 3), ("a", 2), ("the", 2), ("my", 2), ("is", 2),
        ("as", 2), ("for", 2), ("from", 2), ("that", 2), ("and", 2),
    ),
    ("to", "be"): (
        ("or", 3), ("a", 3), ("the", 2), ("my", 2), ("his", 2),
        ("not", 2), ("no", 2), ("so", 3), ("done", 2), ("known", 2),
        ("king", 2), ("queen", 2), ("thy", 2), ("gone", 2), ("good", 2),
        ("true", 2), ("wise", 2), ("free", 2), ("an", 2), ("rid", 2),
    ),
    # --- by + X ---
    ("by", "the"): (
        ("king", 2), ("gods", 2), ("lord", 2), ("mass", 2), ("body", 2),
        ("heart", 2), ("sword", 2), ("way", 2), ("hand", 2),
    ),
    ("by", "my"): (
        ("troth", 3), ("faith", 3), ("soul", 2), ("hand", 2), ("life", 2),
        ("honour", 2), ("love", 2),
    ),
    # --- at + X ---
    ("at", "the"): (
        ("court", 2), ("gate", 2), ("king", 2), ("door", 2), ("same", 2),
        ("last", 2), ("first", 2), ("end", 2), ("foot", 2), ("heart", 2),
        ("head", 2),
    ),
    # --- for + X ---
    ("for", "the"): (
        ("king", 3), ("love", 2), ("world", 2), ("same", 2), ("first", 2),
        ("last", 2), ("most", 2), ("time", 2), ("rest", 2), ("love", 2),
    ),
    ("for", "my"): (
        ("lord", 2), ("father", 2), ("part", 3), ("own", 2), ("love", 2),
        ("son", 2), ("sake", 2), ("life", 2), ("soul", 2), ("heart", 2),
    ),
    # --- is + X ---
    ("is", "the"): (
        ("king", 3), ("day", 2), ("night", 2), ("world", 2), ("man", 2),
        ("son", 2), ("queen", 2), ("duke", 2), ("cause", 2), ("fault", 2),
        ("matter", 2), ("best", 2), ("most", 2), ("same", 2), ("other", 2),
        ("house", 2), ("time", 2), ("hour", 2),
    ),
    ("is", "a"): (
        ("man", 3), ("king", 2), ("woman", 2), ("thing", 2), ("tale", 2),
        ("matter", 2), ("good", 2), ("great", 2), ("fool", 2), ("fair", 2),
        ("worthy", 2), ("noble", 2), ("word", 2),
    ),
    ("is", "not"): (
        ("so", 3), ("the", 2), ("a", 2), ("to", 2), ("my", 2),
        ("in", 2), ("for", 2), ("worth", 2), ("yet", 2), ("but", 2),
        ("here", 2), ("this", 2), ("that", 2), ("enough", 2),
    ),
    ("is", "my"): (
        ("lord", 3), ("son", 2), ("father", 2), ("mother", 2), ("love", 2),
        ("heart", 2), ("life", 2), ("friend", 2),
    ),
    # --- do + X ---
    ("do", "not"): (
        ("say", 3), ("speak", 3), ("think", 3), ("know", 2), ("hear", 2),
        ("see", 2), ("love", 2), ("tell", 2), ("make", 2), ("take", 2),
        ("you", 3), ("thou", 2), ("let", 2), ("pray", 2), ("weep", 2),
        ("fear", 2), ("doubt", 2), ("forget", 2),
    ),
    ("do", "you"): (
        ("know", 3), ("hear", 2), ("see", 2), ("say", 2), ("think", 2),
        ("speak", 2), ("love", 2), ("mean", 2), ("mock", 2),
    ),
    ("do", "thee"): (
        ("good", 2), ("no", 2), ("wrong", 2), ("service", 2),
    ),
    # --- will + X ---
    ("will", "not"): (
        ("be", 3), ("have", 2), ("do", 2), ("say", 2), ("go", 2),
        ("come", 2), ("see", 2), ("speak", 2), ("hear", 2), ("let", 2),
        ("it", 2), ("stay", 2),
    ),
    ("will", "be"): (
        ("a", 2), ("the", 2), ("as", 2), ("so", 2), ("no", 2),
        ("gone", 2), ("done", 2), ("revenged", 2), ("there", 2),
    ),
    # --- shall + X ---
    ("shall", "not"): (
        ("be", 3), ("have", 2), ("do", 2), ("see", 2), ("find", 2),
        ("go", 2), ("come", 2), ("speak", 2), ("hear", 2), ("die", 2),
        ("know", 2), ("wear", 2), ("match", 2), ("win", 2), ("live", 2),
    ),
    ("shall", "be"): (
        ("a", 2), ("the", 2), ("as", 2), ("so", 2), ("no", 2),
        ("done", 2), ("gone", 2), ("king", 2), ("thine", 2), ("mine", 2),
    ),
    # --- been + X ---
    ("have", "been"): (
        ("a", 3), ("in", 2), ("at", 2), ("so", 2), ("to", 2),
        ("with", 2), ("here", 2), ("there", 2), ("thus", 2), ("more", 2),
        ("of", 2), ("ever", 2), ("never", 2),
    ),
    ("had", "been"): (
        ("a", 2), ("in", 2), ("so", 2), ("with", 2), ("better", 2),
        ("more", 2),
    ),
    # --- let + X ---
    ("let", "us"): (
        ("go", 3), ("be", 2), ("have", 2), ("hear", 2), ("see", 2),
        ("speak", 2), ("stay", 2), ("know", 2), ("take", 2), ("not", 2),
        ("come", 2), ("make", 2),
    ),
    ("let", "me"): (
        ("see", 3), ("hear", 2), ("speak", 3), ("know", 2), ("have", 2),
        ("go", 2), ("take", 2), ("make", 2), ("tell", 2), ("kiss", 2),
        ("embrace", 2), ("die", 2), ("live", 2), ("be", 2), ("not", 2),
        ("alone", 2),
    ),
    ("let", "him"): (
        ("go", 2), ("be", 2), ("come", 2), ("speak", 2), ("hear", 2),
        ("see", 2), ("have", 2), ("take", 2), ("not", 2), ("die", 2),
    ),
    ("let", "them"): (
        ("go", 2), ("be", 2), ("come", 2), ("hear", 2), ("speak", 2),
        ("not", 2), ("take", 2), ("have", 2), ("die", 2),
    ),
    ("let", "thy"): (
        ("heart", 2), ("mind", 2), ("tongue", 2), ("hand", 2), ("eye", 2),
        ("words", 2),
    ),
    # --- O + X ---
    ("o", "my"): (
        ("lord", 5), ("love", 3), ("heart", 3), ("soul", 2), ("god", 2),
        ("son", 2), ("father", 2), ("mother", 2), ("dear", 2), ("sweet", 2),
        ("liege", 2), ("master", 2),
    ),
    ("o", "thou"): (
        ("art", 3), ("that", 3), ("hast", 2), ("shalt", 2), ("most", 2),
        ("sweet", 2), ("fair", 2), ("dear", 2), ("noble", 2), ("true", 2),
    ),
    ("o", "heaven"): (
        ("s", 2), ("ly", 2), ("forbid", 2),
    ),
    ("o", "sweet"): (
        ("lord", 2), ("love", 2), ("sir", 2), ("friend", 2), ("maid", 2),
    ),
    # --- post-negation / assertions ---
    ("no", "more"): (
        ("of", 3), ("than", 3), ("to", 2), ("words", 2), ("i", 2),
        ("but", 2), ("then", 2), ("the", 2), ("a", 2), ("can", 2),
        ("shall", 2),
    ),
    ("no", "man"): (
        ("is", 2), ("can", 2), ("shall", 2), ("hath", 2), ("ever", 2),
        ("that", 2), ("but", 2),
    ),
    # --- we + X ---
    ("we", "are"): (
        ("not", 3), ("all", 2), ("men", 2), ("the", 2), ("a", 2),
        ("come", 2), ("gone", 2), ("too", 2), ("now", 2), ("here", 2),
        ("but", 2),
    ),
    ("we", "have"): (
        ("seen", 2), ("heard", 2), ("said", 2), ("done", 2), ("made", 2),
        ("found", 2), ("a", 2), ("the", 2), ("our", 2), ("no", 2),
        ("not", 2),
    ),
    ("we", "will"): (
        ("not", 3), ("be", 2), ("have", 2), ("do", 2), ("make", 2),
        ("go", 2), ("come", 2), ("see", 2), ("speak", 2), ("find", 2),
        ("hear", 2),
    ),
    ("we", "shall"): (
        ("not", 2), ("be", 2), ("have", 2), ("see", 2), ("find", 2),
        ("go", 2), ("come", 2), ("know", 2), ("meet", 2),
    ),
    # --- he/she/it + X ---
    ("he", "is"): (
        ("a", 3), ("the", 2), ("my", 2), ("not", 3), ("no", 2),
        ("dead", 3), ("come", 2), ("gone", 2), ("here", 2), ("there", 2),
        ("so", 2), ("very", 2), ("as", 2),
    ),
    ("she", "is"): (
        ("a", 3), ("the", 2), ("my", 2), ("not", 3), ("no", 2),
        ("dead", 2), ("come", 2), ("gone", 2), ("here", 2), ("there", 2),
        ("so", 2), ("fair", 2), ("my", 2),
    ),
    ("it", "is"): (
        ("a", 4), ("the", 3), ("my", 2), ("not", 3), ("no", 3),
        ("so", 3), ("true", 3), ("most", 2), ("too", 2), ("as", 2),
        ("but", 2), ("enough", 2), ("only", 2),
    ),
    ("it", "was"): (
        ("a", 4), ("the", 2), ("my", 2), ("not", 2), ("no", 2),
        ("so", 2), ("never", 2), ("ever", 2), ("but", 2),
    ),
    # --- compound prepositions ---
    ("out", "of"): (
        ("the", 3), ("my", 2), ("his", 2), ("her", 2), ("your", 2),
        ("thy", 2), ("this", 2), ("that", 2), ("sight", 2), ("doors", 2),
        ("town", 2), ("hand", 2),
    ),
    ("one", "of"): (
        ("the", 3), ("my", 2), ("his", 2), ("her", 2), ("your", 2),
        ("these", 2), ("those", 2), ("them", 2), ("us", 2), ("you", 2),
        ("our", 2),
    ),
    ("all", "the"): (
        ("king", 2), ("world", 2), ("men", 2), ("day", 2), ("night", 2),
        ("ways", 2), ("time", 2), ("rest", 2), ("same", 2), ("while", 2),
    ),
    # --- common phrasings ---
    ("there", "is"): (
        ("a", 3), ("the", 2), ("my", 2), ("no", 3), ("not", 2),
        ("none", 2), ("one", 2), ("some", 2), ("but", 2),
    ),
    ("there", "was"): (
        ("a", 3), ("the", 2), ("no", 2), ("none", 2),
    ),
    ("here", "is"): (
        ("a", 3), ("the", 2), ("my", 2), ("no", 2), ("not", 2),
        ("one", 2), ("some", 2),
    ),
    ("here", "comes"): (
        ("the", 3), ("my", 2), ("a", 2), ("his", 2), ("her", 2),
    ),
    ("what", "is"): (
        ("this", 3), ("the", 2), ("he", 2), ("she", 2), ("it", 3),
        ("thy", 2), ("your", 2), ("his", 2), ("her", 2), ("a", 2),
        ("more", 2),
    ),
    ("what", "a"): (
        ("man", 2), ("woman", 2), ("thing", 2), ("fool", 2), ("piece", 2),
        ("deal", 2),
    ),
    # --- after prepositions leading to pronouns ---
    ("unto", "the"): (
        ("king", 2), ("house", 2), ("court", 2), ("world", 2), ("end", 2),
        ("same", 2),
    ),
}


def _build_continuation_tries() -> dict[tuple[str, str], dict[str, dict[str, int]]]:
    """For each (prev_prev, prev) pair, build a trie of expected continuations."""
    out: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
    for key, entries in _EXPECT.items():
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
                    for term in (" ", ",", ".", ";", ":", "?", "!", "\n"):
                        trie[prefix][term] = trie[prefix].get(term, 0) + w
        out[key] = trie
    return out


_CONT_TRIES: dict[tuple[str, str], dict[str, dict[str, int]]] = _build_continuation_tries()


def _build_vector(nexts: dict[str, int], prefix_len: int) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    total = sum(nexts.values())
    if total <= 0:
        return vec
    # Scale stronger than 1-word because the 2-word context is more
    # informative.
    scale = min(0.55 + 0.25 * prefix_len, 1.5)
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


def _build_prefix_bias() -> dict[tuple[str, str, str], list[float]]:
    out: dict[tuple[str, str, str], list[float]] = {}
    for key, trie in _CONT_TRIES.items():
        for prefix, nexts in trie.items():
            if not prefix:
                continue
            out[(key[0], key[1], prefix)] = _build_vector(nexts, len(prefix))
    return out


_PREFIX_BIAS: dict[tuple[str, str, str], list[float]] = _build_prefix_bias()


def phrase_continue_bias(
    prev_completed_word: str,
    last_completed_word: str,
    word_buffer: str,
    letter_run_len: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if not prev_completed_word or not last_completed_word:
        return None
    if not word_buffer:
        return None
    if letter_run_len < 1 or letter_run_len > 8:
        return None
    if len(word_buffer) != letter_run_len:
        return None
    p2 = prev_completed_word.lower()
    p1 = last_completed_word.lower()
    key = word_buffer.lower()
    return _PREFIX_BIAS.get((p2, p1, key))
