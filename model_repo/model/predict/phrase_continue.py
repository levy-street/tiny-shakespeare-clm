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
    # --- expanded I + verb combinations ---
    ("i", "love"): (
        ("thee", 5), ("her", 3), ("him", 2), ("my", 2), ("the", 2), ("thy", 2),
        ("you", 2), ("not", 2), ("none", 1),
    ),
    ("i", "saw"): (
        ("him", 3), ("her", 3), ("the", 3), ("my", 2), ("a", 2), ("thee", 2),
        ("it", 2), ("his", 2), ("no", 1),
    ),
    ("i", "said"): (
        ("i", 3), ("so", 3), ("to", 3), ("unto", 2), ("the", 2), ("my", 2),
        ("it", 2),
    ),
    ("i", "told"): (
        ("thee", 3), ("him", 3), ("her", 3), ("you", 3), ("my", 2), ("the", 2),
    ),
    ("i", "heard"): (
        ("the", 3), ("a", 2), ("him", 2), ("her", 2), ("it", 2), ("my", 2),
        ("thee", 2),
    ),
    ("i", "found"): (
        ("him", 3), ("her", 3), ("the", 2), ("my", 2), ("a", 2), ("it", 2),
        ("thee", 2),
    ),
    ("i", "fear"): (
        ("not", 4), ("me", 3), ("the", 2), ("thee", 2), ("him", 2),
        ("thy", 1), ("my", 1),
    ),
    ("i", "speak"): (
        ("not", 3), ("of", 3), ("to", 3), ("the", 2), ("it", 2), ("no", 2),
    ),
    ("i", "come"): (
        ("to", 4), ("not", 3), ("from", 2), ("hither", 2), ("with", 2),
        ("for", 2), ("no", 1),
    ),
    ("i", "go"): (
        ("to", 3), ("not", 2), ("with", 2), ("forth", 2), ("and", 2),
    ),
    ("i", "swear"): (
        ("to", 3), ("by", 3), ("it", 2), ("thee", 2),
    ),
    ("i", "wish"): (
        ("thee", 3), ("him", 2), ("her", 2), ("you", 2), ("to", 2), ("it", 2),
        ("thy", 2), ("my", 2), ("no", 2),
    ),
    ("i", "mean"): (
        ("to", 4), ("not", 3), ("no", 2), ("the", 2), ("thee", 2), ("him", 1),
    ),
    # --- my + noun (very common starting phrases) ---
    ("my", "heart"): (
        ("is", 3), ("doth", 2), ("hath", 2), ("and", 2), ("to", 2), ("of", 2),
    ),
    ("my", "soul"): (
        ("is", 3), ("doth", 2), ("hath", 2), ("and", 2), ("to", 2),
    ),
    ("my", "father"): (
        ("is", 3), ("and", 3), ("hath", 2), ("doth", 2), ("was", 2), ("the", 1),
    ),
    ("my", "mother"): (
        ("is", 3), ("and", 3), ("hath", 2), ("doth", 2), ("was", 2),
    ),
    # --- the + ... ---
    ("the", "king"): (
        ("is", 3), ("hath", 3), ("doth", 2), ("of", 2), ("and", 2), ("his", 2),
        ("my", 1), ("shall", 2), ("will", 2),
    ),
    ("the", "queen"): (
        ("is", 3), ("hath", 3), ("doth", 2), ("of", 2), ("and", 2), ("his", 2),
    ),
    ("the", "duke"): (
        ("of", 5), ("is", 2), ("hath", 2), ("and", 2), ("doth", 2),
    ),
    ("the", "lord"): (
        ("of", 4), ("hath", 2), ("is", 2), ("and", 2),
    ),
    ("the", "world"): (
        ("is", 3), ("of", 2), ("and", 2), ("hath", 2), ("doth", 2),
        ("to", 2),
    ),
    ("the", "sun"): (
        ("is", 3), ("doth", 2), ("hath", 2), ("of", 2), ("and", 2),
        ("shall", 1),
    ),
    ("the", "time"): (
        ("is", 3), ("of", 3), ("when", 2), ("to", 2), ("hath", 2), ("and", 2),
        ("shall", 1),
    ),
    # --- after "upon" / "from" ---
    ("upon", "my"): (
        ("soul", 3), ("life", 3), ("word", 3), ("honour", 2), ("head", 2),
        ("knee", 2), ("knees", 1), ("faith", 2),
    ),
    ("upon", "his"): (
        ("head", 3), ("soul", 2), ("knee", 2), ("knees", 2), ("life", 2),
        ("breast", 2), ("face", 2),
    ),
    ("from", "the"): (
        ("king", 2), ("court", 2), ("world", 2), ("sea", 2), ("earth", 2),
        ("ground", 2), ("time", 2), ("east", 1), ("west", 1), ("north", 1),
    ),
    ("from", "my"): (
        ("soul", 3), ("heart", 3), ("lord", 2), ("lady", 2), ("father", 2),
        ("mother", 2), ("eyes", 2),
    ),
    # --- "be" phrases ---
    ("may", "be"): (
        ("a", 3), ("the", 2), ("so", 2), ("done", 2), ("my", 2),
    ),
    ("must", "be"): (
        ("a", 3), ("so", 2), ("the", 2), ("done", 2), ("my", 2),
    ),
    ("would", "be"): (
        ("a", 3), ("the", 2), ("so", 2), ("done", 2), ("my", 2),
    ),
    # --- "have/hath" chains ---
    ("hath", "done"): (
        ("it", 2), ("the", 2), ("me", 2), ("thee", 2), ("so", 2), ("no", 1),
    ),
    ("hath", "made"): (
        ("me", 3), ("thee", 2), ("him", 2), ("her", 2), ("a", 2), ("the", 2),
    ),
    ("have", "done"): (
        ("it", 3), ("so", 2), ("the", 2), ("no", 2), ("thee", 2), ("with", 2),
    ),
    ("have", "made"): (
        ("me", 3), ("a", 2), ("thee", 2), ("him", 2), ("her", 2), ("the", 2),
    ),
    ("have", "seen"): (
        ("the", 3), ("him", 2), ("her", 2), ("my", 2), ("a", 2), ("thee", 2),
    ),
    # --- "give/take/make/let" chains ---
    ("give", "me"): (
        ("thy", 3), ("your", 2), ("the", 2), ("a", 2), ("my", 2), ("leave", 3),
        ("some", 2),
    ),
    ("make", "me"): (
        ("a", 3), ("not", 2), ("the", 2), ("to", 2), ("thy", 1),
    ),
    ("tell", "me"): (
        ("not", 3), ("the", 3), ("thy", 3), ("what", 2), ("how", 2), ("why", 2),
        ("of", 2), ("a", 1),
    ),
    # --- pronoun + verb ---
    ("he", "hath"): (
        ("done", 3), ("made", 2), ("said", 2), ("a", 2), ("the", 2), ("no", 2),
        ("been", 2), ("sworn", 1),
    ),
    ("she", "hath"): (
        ("done", 3), ("made", 2), ("said", 2), ("a", 2), ("no", 2),
        ("been", 2),
    ),
    ("they", "are"): (
        ("the", 2), ("not", 3), ("all", 2), ("but", 2), ("my", 2), ("gone", 2),
        ("here", 2), ("come", 2),
    ),
    ("you", "are"): (
        ("a", 3), ("not", 3), ("the", 2), ("my", 2), ("too", 2), ("so", 2),
        ("welcome", 2), ("come", 2),
    ),
    ("we", "must"): (
        ("not", 3), ("be", 2), ("have", 2), ("go", 2), ("do", 2), ("needs", 1),
    ),
    # --- "if" conditionals ---
    ("if", "thou"): (
        ("wilt", 2), ("art", 2), ("hast", 2), ("dost", 2), ("didst", 2),
        ("shalt", 2), ("be", 2), ("have", 1), ("wert", 1),
    ),
    ("if", "i"): (
        ("be", 3), ("have", 2), ("had", 2), ("were", 3), ("could", 2),
        ("should", 2), ("must", 2), ("do", 2),
    ),
    ("if", "he"): (
        ("be", 3), ("have", 2), ("had", 2), ("were", 2), ("come", 2),
        ("doth", 2), ("hath", 2),
    ),
    ("if", "the"): (
        ("king", 2), ("duke", 2), ("queen", 2), ("lord", 2), ("gods", 2),
        ("world", 2), ("time", 2),
    ),
    # --- "but" + pronoun/determiner ---
    ("but", "i"): (
        ("am", 3), ("will", 2), ("have", 2), ("do", 2), ("must", 2),
        ("know", 2), ("say", 2), ("fear", 2), ("pray", 2),
    ),
    ("but", "thou"): (
        ("art", 2), ("hast", 2), ("wilt", 2), ("dost", 2), ("shalt", 1),
    ),
    ("but", "he"): (
        ("is", 3), ("was", 2), ("hath", 2), ("doth", 2), ("will", 2),
        ("shall", 1),
    ),
    ("but", "she"): (
        ("is", 3), ("was", 2), ("hath", 2), ("doth", 2), ("will", 2),
    ),
    ("but", "the"): (
        ("king", 2), ("queen", 2), ("duke", 2), ("lord", 2), ("time", 2),
        ("world", 2),
    ),
    # --- "who" / "whose" ---
    ("who", "is"): (
        ("there", 3), ("this", 2), ("he", 2), ("she", 2), ("the", 2),
        ("it", 2),
    ),
    ("whose", "name"): (
        ("is", 3), ("shall", 2), ("was", 2), ("doth", 2), ("hath", 2),
    ),
    # --- relative pronouns ---
    ("that", "i"): (
        ("am", 3), ("have", 3), ("will", 2), ("did", 2), ("do", 2),
        ("know", 2), ("may", 2), ("should", 2), ("must", 2), ("should", 1),
    ),
    ("that", "he"): (
        ("is", 3), ("hath", 2), ("doth", 2), ("was", 2), ("will", 2),
        ("shall", 2),
    ),
    ("that", "thou"): (
        ("art", 2), ("hast", 2), ("dost", 2), ("wilt", 2), ("shalt", 1),
        ("didst", 1),
    ),
    ("which", "i"): (
        ("have", 3), ("did", 2), ("do", 2), ("would", 2), ("will", 2),
        ("must", 2),
    ),
    # --- "O" invocations ---
    ("o", "god"): (
        ("of", 3), ("the", 2), ("i", 2), ("my", 2),
    ),
    ("o", "lord"): (
        ("of", 3), ("the", 2), ("my", 2), ("i", 2),
    ),
    # --- "all" combinations ---
    ("all", "my"): (
        ("life", 2), ("love", 2), ("heart", 2), ("soul", 2), ("days", 2),
        ("friends", 2), ("hopes", 2),
    ),
    # --- "most" / "more" ---
    ("more", "than"): (
        ("a", 3), ("i", 2), ("all", 2), ("the", 2), ("my", 2), ("thou", 2),
        ("he", 2), ("ever", 2),
    ),
    ("most", "noble"): (
        ("lord", 3), ("sir", 2), ("prince", 2), ("and", 2), ("king", 1),
    ),
    # --- "thy" nouns ---
    ("thy", "love"): (
        ("is", 3), ("and", 2), ("to", 2), ("shall", 2), ("will", 2),
    ),
    ("thy", "name"): (
        ("is", 3), ("shall", 2), ("and", 2), ("from", 2), ("to", 2),
    ),
    ("thy", "heart"): (
        ("is", 3), ("and", 2), ("to", 2), ("shall", 1),
    ),
    # --- "good" ---
    ("good", "night"): (
        ("my", 3), ("sir", 2), ("to", 2), ("sweet", 2), ("and", 2),
    ),
    ("good", "morrow"): (
        ("my", 3), ("to", 3), ("sir", 2), ("sweet", 2), ("and", 2),
    ),
    # --- "and" conjunctions (very common) ---
    ("and", "i"): (
        ("will", 3), ("shall", 2), ("have", 2), ("do", 2), ("am", 3),
        ("must", 2), ("know", 2), ("say", 2), ("think", 2), ("fear", 2),
        ("pray", 2),
    ),
    ("and", "thou"): (
        ("art", 3), ("shalt", 2), ("wilt", 2), ("hast", 2), ("dost", 2),
        ("didst", 2),
    ),
    ("and", "he"): (
        ("is", 3), ("was", 2), ("hath", 3), ("doth", 2), ("will", 2),
        ("shall", 2), ("said", 2),
    ),
    ("and", "she"): (
        ("is", 3), ("was", 2), ("hath", 3), ("doth", 2), ("will", 2),
        ("shall", 2),
    ),
    ("and", "you"): (
        ("shall", 2), ("will", 2), ("are", 3), ("have", 2), ("too", 2),
        ("my", 2),
    ),
    ("and", "we"): (
        ("will", 3), ("shall", 3), ("are", 2), ("have", 2), ("must", 2),
    ),
    ("and", "they"): (
        ("are", 3), ("shall", 2), ("will", 2), ("have", 2), ("were", 2),
    ),
    ("and", "the"): (
        ("king", 2), ("queen", 2), ("lord", 2), ("rest", 2), ("rest", 2),
        ("world", 2), ("gods", 2), ("day", 2), ("night", 2), ("sun", 2),
        ("moon", 2),
    ),
    ("and", "my"): (
        ("lord", 3), ("love", 2), ("heart", 2), ("soul", 2), ("life", 2),
        ("father", 2), ("mother", 2), ("friend", 2),
    ),
    ("and", "thy"): (
        ("love", 2), ("heart", 2), ("soul", 2), ("name", 2), ("father", 2),
    ),
    ("and", "all"): (
        ("the", 3), ("my", 2), ("his", 2), ("her", 2), ("thy", 2),
        ("your", 2), ("our", 2), ("their", 2),
    ),
    ("and", "so"): (
        ("i", 3), ("it", 2), ("he", 2), ("she", 2), ("we", 2), ("the", 2),
        ("farewell", 2), ("you", 2),
    ),
    ("and", "yet"): (
        ("i", 3), ("the", 2), ("he", 2), ("not", 2), ("it", 2), ("my", 2),
    ),
    # --- "am" ---
    ("am", "not"): (
        ("a", 3), ("the", 2), ("so", 3), ("yet", 2), ("afraid", 2),
        ("such", 2), ("mad", 2), ("i", 2),
    ),
    ("am", "a"): (
        ("man", 3), ("woman", 2), ("fool", 2), ("gentleman", 2),
        ("stranger", 2), ("soldier", 2), ("true", 2),
    ),
    ("am", "the"): (
        ("king", 2), ("queen", 2), ("man", 2), ("very", 2), ("same", 2),
    ),
    ("am", "thy"): (
        ("father", 2), ("son", 2), ("friend", 2), ("servant", 2), ("lord", 2),
    ),
    ("am", "no"): (
        ("more", 2), ("man", 2), ("traitor", 2), ("fool", 2), ("coward", 2),
    ),
    # --- "be" phrases (not just auxiliary) ---
    ("be", "not"): (
        ("so", 3), ("a", 2), ("too", 2), ("afraid", 2), ("angry", 2),
        ("the", 2), ("my", 2),
    ),
    ("be", "the"): (
        ("king", 2), ("man", 2), ("cause", 2), ("case", 2), ("death", 2),
        ("first", 2), ("last", 2),
    ),
    ("be", "so"): (
        (",", 3), (".", 2), ("bold", 2), ("kind", 2), ("good", 2),
    ),
    ("be", "a"): (
        ("man", 3), ("villain", 2), ("king", 2), ("fool", 2), ("friend", 2),
        ("traitor", 2), ("noble", 2),
    ),
    # --- "not" ---
    ("not", "to"): (
        ("be", 5), ("speak", 2), ("see", 2), ("do", 2), ("have", 2),
        ("die", 2), ("go", 2), ("say", 2),
    ),
    ("not", "the"): (
        ("king", 2), ("man", 2), ("lord", 2), ("thing", 2), ("less", 2),
        ("world", 2), ("cause", 2),
    ),
    ("not", "a"): (
        ("word", 3), ("man", 2), ("whit", 2), ("jot", 2), ("soul", 2),
        ("drop", 2),
    ),
    # --- conjunctions/prepositions ---
    ("or", "the"): (
        ("king", 2), ("queen", 2), ("lord", 2), ("duke", 2), ("world", 2),
    ),
    ("or", "not"): (
        ("to", 4), (",", 2), (".", 1),
    ),
    ("or", "i"): (
        ("will", 2), ("shall", 2), ("have", 2), ("must", 2), ("am", 2),
    ),
    ("as", "i"): (
        ("am", 3), ("have", 2), ("said", 2), ("did", 2), ("do", 2),
        ("know", 2), ("think", 2),
    ),
    ("as", "thou"): (
        ("art", 3), ("hast", 2), ("wilt", 2), ("shalt", 2), ("dost", 2),
    ),
    ("as", "the"): (
        ("king", 2), ("queen", 2), ("sun", 2), ("lord", 2), ("world", 2),
        ("day", 2), ("night", 2), ("time", 2),
    ),
    ("as", "a"): (
        ("man", 2), ("friend", 2), ("king", 2), ("woman", 2), ("child", 2),
        ("stranger", 2),
    ),
    ("as", "my"): (
        ("lord", 2), ("soul", 2), ("heart", 2), ("life", 2), ("love", 2),
        ("friend", 2),
    ),
    ("when", "i"): (
        ("was", 3), ("am", 2), ("shall", 2), ("will", 2), ("have", 2),
        ("had", 2), ("saw", 2),
    ),
    ("when", "thou"): (
        ("art", 3), ("wast", 2), ("hast", 2), ("wilt", 2), ("shalt", 2),
    ),
    ("when", "he"): (
        ("is", 2), ("was", 2), ("hath", 2), ("doth", 2), ("shall", 2),
        ("comes", 2),
    ),
    ("when", "the"): (
        ("king", 2), ("queen", 2), ("sun", 2), ("day", 2), ("night", 2),
        ("world", 2),
    ),
    # --- "where" / "whence" / "whither" ---
    ("where", "is"): (
        ("the", 3), ("my", 2), ("he", 2), ("she", 2), ("thy", 2), ("your", 2),
    ),
    ("where", "art"): (
        ("thou", 6),
    ),
    # --- "sweet"/"dear"/"fair"/"noble" adjectives + noun ---
    ("sweet", "lord"): (
        (",", 3), ("!", 2), (".", 2),
    ),
    ("dear", "lord"): (
        (",", 3), ("!", 2), (".", 2),
    ),
    ("fair", "lady"): (
        (",", 3), ("!", 2), (".", 2),
    ),
    ("fair", "maid"): (
        (",", 2), ("!", 1), (".", 1),
    ),
    ("noble", "lord"): (
        (",", 3), ("!", 2), (".", 2),
    ),
    ("noble", "prince"): (
        (",", 3), ("!", 1), (".", 2),
    ),
    ("gentle", "lord"): (
        (",", 3), ("!", 1), (".", 2),
    ),
    ("good", "friend"): (
        (",", 3), ("!", 2), (".", 2),
    ),
    ("dear", "friend"): (
        (",", 3), ("!", 2), (".", 2),
    ),
    ("old", "man"): (
        (",", 3), (".", 2), ("!", 1),
    ),
    # --- "the" + common nouns (extras) ---
    ("the", "day"): (
        ("of", 2), ("is", 2), ("was", 2), ("shall", 2), ("and", 2),
    ),
    ("the", "night"): (
        ("is", 2), ("of", 2), ("was", 2), ("and", 2), ("hath", 2),
    ),
    ("the", "gods"): (
        ("of", 2), ("have", 2), ("hath", 1), ("and", 2), ("shall", 2),
    ),
    ("the", "man"): (
        ("is", 2), ("that", 2), ("who", 2), ("hath", 2), ("of", 2),
    ),
    ("the", "cause"): (
        ("of", 3), ("is", 2), ("why", 2), ("and", 2),
    ),
    ("the", "house"): (
        ("of", 3), ("is", 2), ("and", 2), ("was", 2),
    ),
    ("the", "earth"): (
        (",", 2), ("and", 2), ("is", 2), ("hath", 1), ("shall", 1),
    ),
    ("the", "heavens"): (
        ("are", 2), (",", 2), ("and", 2), ("have", 2),
    ),
    # --- emotive openers / verbs ---
    ("speak", "not"): (
        (",", 2), ("of", 2), ("to", 2), ("a", 2), ("the", 2), ("so", 2),
    ),
    ("speak", "to"): (
        ("me", 3), ("him", 2), ("her", 2), ("the", 2), ("my", 2),
    ),
    ("go", "to"): (
        (",", 3), ("the", 2), ("my", 2), ("thy", 2), ("him", 2), ("her", 2),
    ),
    ("come", "to"): (
        ("me", 3), ("the", 2), ("my", 2), ("thy", 2), ("him", 2), ("thee", 2),
    ),
    ("come", "hither"): (
        (",", 4), (".", 2), ("!", 1),
    ),
    ("come", "forth"): (
        (",", 3), (".", 2), ("!", 1),
    ),
    ("stand", "back"): (
        (",", 3), (".", 2), ("!", 1),
    ),
    ("stand", "forth"): (
        (",", 3), (".", 2),
    ),
    ("farewell", "my"): (
        ("lord", 3), ("good", 2), ("sweet", 2), ("dear", 2), ("love", 2),
        ("friend", 2), ("son", 2),
    ),
    ("farewell", "farewell"): (
        (",", 3), ("!", 2), (".", 2),
    ),
    # --- questions ---
    ("what", "hath"): (
        ("he", 3), ("she", 2), ("thou", 2), ("the", 2), ("my", 2),
        ("this", 2),
    ),
    ("what", "dost"): (
        ("thou", 6),
    ),
    ("what", "say"): (
        ("you", 4), ("thou", 3), ("i", 2),
    ),
    ("how", "now"): (
        (",", 4), ("?", 2), ("!", 1),
    ),
    ("how", "do"): (
        ("you", 3), ("thou", 2), ("i", 2),
    ),
    ("how", "dost"): (
        ("thou", 6),
    ),
    # --- modals ---
    ("thou", "mayst"): (
        ("be", 2), ("have", 2), ("see", 2), ("do", 2), ("not", 2),
    ),
    ("thou", "couldst"): (
        ("not", 2), ("have", 2), ("be", 2), ("see", 2),
    ),
    ("thou", "wert"): (
        ("a", 2), ("the", 2), ("not", 2), ("but", 2), ("best", 2),
    ),
    # --- cannot / must not ---
    ("must", "not"): (
        ("be", 3), ("do", 2), ("go", 2), ("have", 2), ("speak", 2),
    ),
    ("can", "not"): (
        ("be", 3), ("tell", 2), ("speak", 2), ("go", 2), ("do", 2),
    ),
    ("may", "not"): (
        ("be", 3), ("have", 2), ("go", 2), ("do", 2),
    ),
    # --- "by" emphatic oaths (prev = "by") ---
    ("by", "heaven"): (
        (",", 3), ("!", 2), (".", 1),
    ),
    ("by", "god"): (
        (",", 3), ("!", 2),
    ),
    # --- New high-value 2-word contexts ---
    # Auxiliary/modal + pronoun frames
    ("have", "i"): (
        ("not", 3), ("done", 2), ("seen", 2), ("said", 2), ("heard", 2),
        ("been", 2), ("a", 2), ("any", 2), ("no", 2),
    ),
    ("will", "i"): (
        ("not", 3), ("go", 2), ("do", 2), ("be", 2), ("speak", 2),
    ),
    ("hast", "thou"): (
        ("not", 3), ("seen", 2), ("done", 2), ("heard", 2), ("any", 2),
        ("no", 2), ("a", 2), ("made", 2),
    ),
    ("dost", "thou"): (
        ("not", 3), ("love", 2), ("know", 2), ("speak", 2), ("think", 2),
        ("hear", 2), ("mean", 2), ("see", 2),
    ),
    ("art", "thou"): (
        ("not", 4), ("come", 3), ("there", 2), ("mad", 2), ("a", 2),
        ("so", 2), ("my", 2), ("the", 2),
    ),
    ("wilt", "thou"): (
        ("not", 3), ("go", 2), ("be", 2), ("have", 2), ("do", 2),
        ("speak", 2), ("stay", 2), ("hear", 2),
    ),
    ("shalt", "thou"): (
        ("not", 4), ("be", 2), ("have", 2), ("see", 2), ("find", 2),
    ),
    # Common sentence-initial vocatives and exclamatives
    ("o", "thou"): (
        ("most", 2), ("blessed", 2), ("wretched", 2), ("that", 2),
        ("heaven", 2), ("god", 2),
    ),
    ("o", "my"): (
        ("lord", 4), ("love", 3), ("god", 3), ("heart", 3), ("soul", 3),
        ("father", 2), ("mother", 2), ("friend", 2), ("dear", 2),
        ("good", 2), ("son", 2), ("liege", 2),
    ),
    ("o", "god"): (
        ("!", 3), (",", 2), ("of", 3), ("the", 2),
    ),
    ("o", "heaven"): (
        ("!", 2), (",", 2), ("s", 2),
    ),
    ("o", "sweet"): (
        ("lord", 2), ("lady", 2), ("friend", 2), ("juliet", 1), ("soul", 2),
    ),
    # "my good" + noun
    ("my", "good"): (
        ("lord", 4), ("friend", 3), ("lady", 2), ("liege", 2), ("master", 2),
        ("sir", 2), ("man", 2),
    ),
    ("my", "dear"): (
        ("lord", 3), ("friend", 3), ("love", 2), ("father", 2),
        ("mother", 2), ("son", 2),
    ),
    ("my", "noble"): (
        ("lord", 4), ("friend", 3), ("liege", 2), ("master", 2), ("prince", 2),
    ),
    ("my", "sweet"): (
        ("lord", 3), ("lady", 3), ("friend", 2), ("queen", 2), ("love", 2),
    ),
    # "in my" / "in his" / "in her"
    ("in", "my"): (
        ("heart", 3), ("soul", 2), ("eyes", 2), ("hand", 2), ("mind", 2),
        ("life", 2), ("house", 2), ("bosom", 2), ("sight", 2),
    ),
    ("in", "his"): (
        ("heart", 2), ("hand", 2), ("eyes", 2), ("face", 2), ("bed", 2),
        ("youth", 2), ("life", 2),
    ),
    ("in", "her"): (
        ("heart", 2), ("hand", 2), ("eyes", 2), ("face", 2), ("ear", 2),
        ("breast", 2),
    ),
    ("in", "thy"): (
        ("heart", 2), ("eyes", 2), ("hand", 2), ("face", 2), ("youth", 2),
    ),
    # "to my/thy/his/her"
    ("to", "my"): (
        ("lord", 3), ("father", 2), ("mother", 2), ("friend", 2),
        ("love", 2), ("heart", 2), ("soul", 2), ("brother", 2),
    ),
    ("to", "thy"): (
        ("father", 2), ("love", 2), ("heart", 2), ("master", 2), ("soul", 2),
    ),
    ("to", "his"): (
        ("father", 2), ("mother", 2), ("majesty", 2), ("highness", 2),
        ("grace", 2), ("lord", 2),
    ),
    ("to", "her"): (
        ("lord", 2), ("husband", 2), ("father", 2), ("love", 2),
    ),
    # "with my/thy/his/her"
    ("with", "my"): (
        ("lord", 2), ("love", 2), ("hand", 2), ("sword", 2), ("friend", 2),
        ("life", 2), ("heart", 2),
    ),
    ("with", "his"): (
        ("hand", 2), ("sword", 2), ("eyes", 2), ("father", 2), ("men", 2),
    ),
    ("with", "her"): (
        ("hand", 2), ("lord", 2), ("husband", 2), ("love", 2),
    ),
    # Speech-act frames
    ("i", "beseech"): (
        ("you", 4), ("thee", 3), ("your", 2),
    ),
    ("pray", "you"): (
        (",", 3), (".", 2), ("sir", 3), ("tell", 2), ("let", 2),
        ("speak", 2), ("good", 2),
    ),
    ("tell", "me"): (
        (",", 2), (".", 2), ("not", 3), ("what", 2), ("how", 2),
        ("why", 2), ("where", 2), ("who", 2), ("of", 2), ("the", 2),
    ),
    ("tell", "thee"): (
        (",", 2), (".", 2), ("what", 2), ("how", 2), ("of", 2),
    ),
    # Interrogative word + auxiliary
    ("what", "shall"): (
        ("i", 3), ("we", 2), ("he", 2), ("be", 2),
    ),
    ("what", "will"): (
        ("you", 2), ("thou", 2), ("he", 2), ("she", 2), ("become", 2),
    ),
    ("what", "is"): (
        ("the", 3), ("this", 2), ("that", 2), ("he", 2), ("your", 2),
        ("thy", 2), ("it", 2), ("your", 2), ("a", 2),
    ),
    ("what", "art"): (
        ("thou", 4),
    ),
    ("what", "thou"): (
        ("art", 2), ("dost", 2), ("hast", 2), ("wilt", 2), ("sayest", 2),
    ),
    ("where", "the"): (
        ("king", 2), ("devil", 2), ("duke", 2), ("lord", 2), ("sun", 2),
    ),
    ("where", "thou"): (
        ("art", 3), ("dost", 2), ("wilt", 2), ("liest", 2),
    ),
    ("why", "dost"): (
        ("thou", 4),
    ),
    ("why", "then"): (
        (",", 2), ("i", 2), ("we", 2), ("let", 2),
    ),
    # Noun-noun common chains
    ("good", "my"): (
        ("lord", 5), ("lady", 2), ("liege", 2), ("master", 2), ("friend", 2),
    ),
    ("sweet", "my"): (
        ("lord", 3), ("lady", 2),
    ),
    ("my", "liege"): (
        (",", 3), (".", 2), ("!", 2), (" ", 2),
    ),
    ("your", "grace"): (
        (",", 3), (".", 2), ("!", 2), ("is", 2), ("shall", 2),
    ),
    ("your", "majesty"): (
        (",", 3), (".", 2), ("is", 2), ("shall", 2),
    ),
    ("your", "highness"): (
        (",", 3), (".", 2),
    ),
    ("your", "honour"): (
        (",", 2), (".", 2),
    ),
    # "at the/my/his"
    ("at", "the"): (
        ("door", 2), ("king", 2), ("last", 2), ("court", 2), ("gates", 2),
        ("court", 2), ("sight", 2), ("time", 2),
    ),
    ("at", "my"): (
        ("heart", 2), ("feet", 2), ("hand", 2), ("request", 2), ("lord", 2),
    ),
    # "this is" + X
    ("this", "is"): (
        ("the", 3), ("a", 3), ("my", 2), ("no", 2), ("true", 2), ("not", 2),
        ("your", 2), ("thy", 2), ("he", 2), ("that", 2),
    ),
    ("that", "is"): (
        ("the", 3), ("a", 3), ("my", 2), ("no", 2), ("true", 2), ("not", 2),
        ("most", 2), ("well", 2), ("gone", 2), ("done", 2),
    ),
    # "he/she hath" and "he/she is"
    ("he", "hath"): (
        ("done", 2), ("said", 2), ("made", 2), ("been", 2), ("no", 2),
        ("a", 2), ("not", 2), ("sent", 2),
    ),
    ("she", "hath"): (
        ("done", 2), ("said", 2), ("a", 2), ("no", 2), ("not", 2),
        ("made", 2),
    ),
    ("he", "was"): (
        ("a", 3), ("the", 2), ("not", 2), ("no", 2), ("so", 2),
        ("my", 2), ("born", 2), ("slain", 2), ("ever", 2),
    ),
    ("she", "was"): (
        ("a", 3), ("the", 2), ("not", 2), ("no", 2), ("so", 2),
        ("my", 2),
    ),
    # Sentence openers
    ("and", "yet"): (
        (",", 2), ("i", 2), ("he", 2), ("she", 2), ("the", 2),
    ),
    ("but", "yet"): (
        (",", 2), ("i", 2), ("the", 2),
    ),
    ("and", "so"): (
        (",", 2), ("i", 2), ("he", 2), ("she", 2), ("we", 2), ("they", 2),
        ("it", 2), ("the", 2), ("farewell", 2), ("be", 2),
    ),
    ("and", "now"): (
        (",", 2), ("i", 2), ("he", 2), ("she", 2), ("the", 2),
    ),
    # Time/place adverbials
    ("this", "day"): (
        (",", 2), (".", 2), ("!", 1), ("shall", 2),
    ),
    ("this", "night"): (
        (",", 2), (".", 2), ("!", 1), ("shall", 2),
    ),
    ("the", "sun"): (
        ("is", 2), (",", 2), ("doth", 2), ("shines", 2), ("was", 2),
    ),
    ("the", "moon"): (
        ("is", 2), (",", 2), ("doth", 2), ("shines", 2),
    ),
    ("the", "duke"): (
        ("of", 3), ("is", 2), ("hath", 2), ("was", 2), ("shall", 2),
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
