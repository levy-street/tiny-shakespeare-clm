"""Three-word phrase continuation bias.

Extends `phrase_continue.py` with one more word of lookback. Given the
three most recent completed words (prev_prev_completed_word,
prev_completed_word, last_completed_word), bias the full mid-word
trajectory of the current buffer toward canonical continuations:

  "I have been" + "g" → "o" (gone)
  "thou shalt not" + "d" → "i" (die), "o" (do)
  "to be or" + "n" → "o" (not)
  "I know not" + "w" → "h" (what, where, why)
  "out of my" + "s" → "i" (sight), "o" (soul)
  "let us go" + "t" → "o" (to)
  "I do not" + "k" → "n" (know)
  "I shall not" + "b" → "e" (be)
  "if I were" + "a" → " " (a)
  "God save the" + "k" → "i" (king)

Much sparser than 2-word context, but the entries we do hit are very
high-confidence. Scale is stronger than phrase_continue because 3-word
context is more informative. All entries from prior Shakespeare idiom —
no corpus statistics.
"""

from __future__ import annotations

import math

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_EXPECT: dict[tuple[str, str, str], tuple[tuple[str, int], ...]] = {
    # --- "I have ..." chains ---
    ("i", "have", "been"): (
        ("a", 3), ("so", 3), ("here", 2), ("there", 2), ("the", 2),
        ("with", 2), ("in", 2), ("to", 2), ("wronged", 2), ("deceived", 2),
        ("too", 2), ("much", 2), ("as", 2),
    ),
    ("i", "have", "seen"): (
        ("the", 3), ("him", 3), ("her", 3), ("my", 2), ("a", 3), ("thee", 2),
        ("no", 2),
    ),
    ("i", "have", "done"): (
        ("it", 3), ("no", 2), ("the", 2), ("thee", 2), ("my", 2), ("with", 2),
        ("so", 2),
    ),
    ("i", "have", "heard"): (
        ("the", 3), ("it", 2), ("a", 2), ("him", 2), ("her", 2), ("of", 3),
        ("my", 2),
    ),
    ("i", "have", "said"): (
        ("it", 2), ("so", 2), ("to", 2), ("the", 2), ("my", 2),
    ),
    ("i", "have", "sworn"): (
        ("to", 3), ("by", 2), ("it", 2),
    ),
    ("i", "have", "not"): (
        ("seen", 3), ("heard", 2), ("known", 2), ("been", 3), ("done", 2),
        ("a", 2), ("the", 2), ("slept", 1),
    ),
    # --- "I do/did/am not" ---
    ("i", "do", "not"): (
        ("know", 3), ("think", 2), ("love", 2), ("see", 2), ("hear", 2),
        ("fear", 2), ("like", 2), ("wish", 2), ("care", 2),
    ),
    ("i", "did", "not"): (
        ("know", 3), ("think", 2), ("see", 2), ("hear", 2), ("mean", 2),
    ),
    ("i", "am", "not"): (
        ("a", 3), ("the", 2), ("so", 2), ("yet", 2), ("afraid", 2),
        ("such", 2), ("mad", 2), ("well", 2),
    ),
    ("i", "will", "not"): (
        ("be", 3), ("do", 2), ("go", 2), ("say", 2), ("speak", 2),
        ("stay", 2), ("have", 2), ("hear", 2),
    ),
    ("i", "shall", "not"): (
        ("be", 3), ("do", 2), ("go", 2), ("see", 2), ("need", 2),
    ),
    ("i", "would", "not"): (
        ("have", 3), ("be", 2), ("do", 2), ("for", 2), ("so", 2),
    ),
    ("i", "could", "not"): (
        ("have", 2), ("be", 2), ("speak", 2), ("stay", 2),
    ),
    ("i", "know", "not"): (
        ("what", 3), ("how", 2), ("why", 2), ("where", 2), ("the", 2),
        ("whether", 2), ("when", 2), ("who", 2),
    ),
    ("i", "see", "not"): (
        ("the", 2), ("what", 2), ("how", 2), ("why", 2),
    ),
    ("i", "fear", "not"): (
        ("thee", 2), ("death", 2), ("the", 2), ("him", 2),
    ),
    # --- "thou shalt/wilt/art ..." ---
    ("thou", "shalt", "not"): (
        ("die", 3), ("be", 3), ("have", 2), ("see", 2), ("find", 2),
        ("go", 1), ("kill", 2), ("bear", 2), ("want", 1), ("need", 1),
    ),
    ("thou", "wilt", "not"): (
        ("be", 3), ("have", 2), ("go", 2), ("die", 2), ("come", 2),
        ("speak", 2),
    ),
    ("thou", "art", "a"): (
        ("fool", 2), ("man", 2), ("knave", 2), ("villain", 2), ("noble", 2),
        ("true", 2), ("good", 2), ("sweet", 2), ("very", 2),
    ),
    ("thou", "art", "not"): (
        ("a", 2), ("the", 2), ("so", 3), ("yet", 2), ("alone", 2),
        ("such", 2), ("what", 2), ("mad", 2),
    ),
    ("thou", "hast", "not"): (
        ("done", 2), ("been", 3), ("said", 2), ("heard", 2), ("seen", 2),
        ("a", 2), ("the", 2),
    ),
    ("thou", "hast", "done"): (
        ("it", 2), ("well", 3), ("me", 2), ("a", 2), ("the", 2),
    ),
    ("thou", "dost", "not"): (
        ("know", 3), ("love", 2), ("see", 2), ("hear", 2), ("fear", 2),
    ),
    ("thou", "didst", "not"): (
        ("know", 2), ("see", 2), ("hear", 2), ("swear", 2), ("love", 2),
    ),
    # --- "to be" chains ---
    ("to", "be", "or"): (
        ("not", 8),
    ),
    ("to", "be", "a"): (
        ("man", 2), ("king", 2), ("queen", 2), ("lord", 2), ("fool", 2),
        ("traitor", 2), ("friend", 2), ("father", 2), ("true", 2),
    ),
    ("or", "not", "to"): (
        ("be", 8),
    ),
    ("not", "to", "be"): (
        ("the", 2), ("a", 2), ("so", 2), ("gone", 2), ("done", 2),
    ),
    # --- "let us" chains ---
    ("let", "us", "go"): (
        ("to", 3), ("and", 2), ("hence", 2), ("forth", 2), ("in", 2),
    ),
    ("let", "us", "be"): (
        ("gone", 3), ("merry", 2), ("not", 2), ("together", 2),
    ),
    ("let", "us", "have"): (
        ("a", 2), ("no", 2), ("it", 2), ("some", 2),
    ),
    # --- "out of my/his/the" ---
    ("out", "of", "my"): (
        ("sight", 3), ("way", 2), ("mind", 2), ("life", 2), ("heart", 2),
        ("wits", 2), ("sight,", 1), ("soul", 2),
    ),
    ("out", "of", "his"): (
        ("mind", 2), ("sight", 2), ("wits", 2), ("way", 2), ("grave", 2),
    ),
    ("out", "of", "the"): (
        ("way", 3), ("world", 2), ("question", 2), ("sight", 2), ("house", 2),
        ("gate", 2), ("field", 2),
    ),
    # --- "God / heaven save" ---
    ("god", "save", "the"): (
        ("king", 6), ("queen", 3), ("prince", 2),
    ),
    ("heaven", "save", "the"): (
        ("king", 4), ("queen", 2),
    ),
    # --- "the love / grace of" ---
    ("for", "the", "love"): (
        ("of", 5),
    ),
    ("by", "the", "mass"): (
        (",", 2), (".", 2),
    ),
    # --- "I pray thee" chains ---
    ("i", "pray", "thee"): (
        (",", 3), ("tell", 3), ("speak", 2), ("come", 2), ("go", 2),
        ("take", 2), ("do", 2), ("sir", 2),
    ),
    ("i", "pray", "you"): (
        (",", 3), ("tell", 2), ("speak", 2), ("sir", 3), ("come", 2),
    ),
    ("i", "beseech", "you"): (
        (",", 3), ("sir", 3), ("tell", 2),
    ),
    ("i", "beseech", "thee"): (
        (",", 3), ("good", 2), ("tell", 2),
    ),
    # --- vocative openers ---
    ("o", "my", "lord"): (
        (",", 3), ("!", 3), ("?", 2),
    ),
    ("good", "my", "lord"): (
        (",", 3), ("!", 2), ("?", 2),
    ),
    ("my", "good", "lord"): (
        (",", 3), ("!", 2), ("?", 2),
    ),
    ("my", "noble", "lord"): (
        (",", 3), ("!", 2), ("?", 2),
    ),
    ("my", "dear", "lord"): (
        (",", 3), ("!", 2),
    ),
    ("my", "gracious", "lord"): (
        (",", 3), ("!", 2),
    ),
    # --- "if X were/had" ---
    ("if", "i", "were"): (
        ("a", 2), ("the", 2), ("not", 2), ("but", 2), ("to", 2),
    ),
    ("if", "thou", "wert"): (
        ("a", 2), ("but", 2), ("not", 2),
    ),
    ("if", "he", "had"): (
        ("been", 2), ("not", 2), ("a", 2), ("the", 2),
    ),
    ("if", "i", "had"): (
        ("been", 2), ("not", 2), ("a", 2), ("but", 2),
    ),
    # --- "no more" chains ---
    ("no", "more", "of"): (
        ("this", 3), ("that", 2), ("it", 2), ("him", 2), ("thy", 2),
    ),
    ("no", "more", "words"): (
        (",", 3), (".", 2),
    ),
    # --- "it is a" ---
    ("it", "is", "a"): (
        ("man", 2), ("woman", 2), ("good", 2), ("great", 2), ("noble", 2),
        ("thing", 2), ("true", 2), ("villain", 2), ("fool", 2), ("fair", 2),
    ),
    ("it", "is", "not"): (
        ("a", 2), ("so", 2), ("the", 2), ("enough", 2), ("my", 2),
        ("yet", 2),
    ),
    ("it", "is", "the"): (
        ("king", 2), ("lord", 2), ("queen", 2), ("hour", 2), ("time", 2),
        ("way", 2),
    ),
    # --- "what is" ---
    ("what", "is", "this"): (
        ("?", 3), ("that", 2),
    ),
    ("what", "is", "thy"): (
        ("name", 3), ("will", 2), ("pleasure", 2), ("cause", 2),
    ),
    ("what", "is", "your"): (
        ("name", 2), ("will", 2), ("pleasure", 2), ("grace", 2),
    ),
    ("what", "is", "the"): (
        ("matter", 3), ("news", 3), ("cause", 2), ("reason", 2), ("time", 2),
    ),
    # --- "who is" ---
    ("who", "is", "there"): (
        ("?", 4), (",", 2),
    ),
    # --- "shall I" ---
    ("what", "shall", "i"): (
        ("do", 3), ("say", 2), ("think", 2),
    ),
    # --- "he/she is" ---
    ("he", "is", "a"): (
        ("good", 2), ("noble", 2), ("man", 2), ("true", 2), ("worthy", 2),
        ("villain", 2),
    ),
    ("she", "is", "a"): (
        ("good", 2), ("fair", 2), ("woman", 2), ("true", 2), ("noble", 2),
    ),
    ("he", "is", "not"): (
        ("a", 2), ("so", 2), ("the", 2), ("here", 2), ("yet", 2),
    ),
    # --- "tell me" chains ---
    ("tell", "me", "what"): (
        ("thou", 2), ("you", 2),
    ),
    ("tell", "me", "how"): (
        ("thou", 2), ("you", 2), ("it", 2),
    ),
    ("tell", "me", "why"): (
        ("thou", 2), ("you", 2),
    ),
    # --- "come, come" ---
    ("come", ",", "come"): (
        (",", 3), ("sir", 2),
    ),
    # --- "by my" oaths ---
    ("by", "my", "troth"): (
        (",", 3), ("!", 1),
    ),
    ("by", "my", "soul"): (
        (",", 3), ("!", 1),
    ),
    ("by", "my", "faith"): (
        (",", 3), ("!", 1),
    ),
    ("upon", "my", "life"): (
        (",", 3), (".", 2),
    ),
    ("upon", "my", "word"): (
        (",", 3), (".", 2),
    ),
    ("upon", "my", "soul"): (
        (",", 3), ("!", 2),
    ),
    # --- "hath done" etc ---
    ("hath", "done", "me"): (
        ("a", 2), ("wrong", 2), ("good", 2),
    ),
    # --- "you are" ---
    ("you", "are", "a"): (
        ("good", 2), ("noble", 2), ("man", 2), ("villain", 2), ("true", 2),
        ("fool", 2), ("worthy", 2),
    ),
    ("you", "are", "welcome"): (
        (",", 3), (".", 2), ("to", 2),
    ),
    # --- "farewell" closures ---
    ("farewell", ",", "my"): (
        ("lord", 3), ("good", 2), ("sweet", 2), ("dear", 2), ("love", 2),
    ),
    # --- "and so" ---
    ("and", "so", "i"): (
        ("am", 2), ("will", 2), ("do", 2), ("did", 2), ("have", 2),
        ("leave", 2), ("depart", 2),
    ),
    # --- "I will go" ---
    ("i", "will", "go"): (
        ("to", 3), ("with", 2), ("and", 2), ("hence", 2),
    ),
    ("i", "will", "be"): (
        ("a", 2), ("the", 2), ("so", 2), ("gone", 2), ("thine", 2),
    ),
    # --- "I am a" ---
    ("i", "am", "a"): (
        ("man", 2), ("woman", 2), ("soldier", 2), ("gentleman", 2),
        ("stranger", 2), ("true", 2), ("fool", 2),
    ),
    ("i", "am", "the"): (
        ("king", 2), ("queen", 2), ("man", 2), ("son", 2), ("father", 2),
    ),
    ("i", "am", "thy"): (
        ("father", 2), ("son", 2), ("friend", 2), ("servant", 2),
    ),
    # --- further 3-word patterns from common Shakespearean idiom ---
    ("there", "is", "no"): (
        ("more", 3), ("end", 2), ("help", 2), ("harm", 2), ("cause", 2),
        ("doubt", 2), ("man", 2), ("such", 2),
    ),
    ("there", "is", "a"): (
        ("man", 2), ("thing", 2), ("lady", 2), ("woman", 2), ("kind", 2),
        ("time", 2), ("way", 2), ("world", 2),
    ),
    ("there", "is", "not"): (
        ("a", 2), ("so", 2), ("one", 2), ("such", 2),
    ),
    ("there", "is", "the"): (
        ("point", 2), ("man", 2), ("question", 2), ("lord", 2),
    ),
    ("i", "am", "sorry"): (
        ("for", 3), ("to", 3), (",", 2), ("sir", 2), ("that", 2),
    ),
    ("i", "do", "beseech"): (
        ("you", 4), ("thee", 3), ("your", 2),
    ),
    ("i", "do", "love"): (
        ("thee", 3), ("her", 2), ("him", 2), ("you", 2), ("my", 2),
    ),
    ("i", "do", "believe"): (
        ("it", 2), ("the", 2), ("thee", 2), ("him", 2),
    ),
    ("i", "do", "remember"): (
        ("me", 2), ("him", 2), (",", 2), ("a", 2), ("the", 2),
    ),
    ("what", "say", "you"): (
        ("?", 4), (",", 2), ("to", 3), ("of", 2),
    ),
    ("what", "say", "thou"): (
        ("?", 4), (",", 2), ("to", 3),
    ),
    ("how", "now", ","): (
        ("my", 2), ("good", 2), ("sir", 2), ("what", 2), ("daughter", 2),
        ("madam", 2), ("sweet", 2),
    ),
    ("how", "do", "you"): (
        ("do", 3), ("?", 3), (",", 2), ("know", 2),
    ),
    ("how", "dost", "thou"): (
        ("?", 3), (",", 2), ("now", 2), ("do", 2),
    ),
    ("i", "prithee", "tell"): (
        ("me", 4), ("us", 2),
    ),
    ("i", "prithee", ","): (
        ("sir", 2), ("speak", 2), ("come", 2), ("tell", 2), ("good", 2),
    ),
    ("where", "is", "my"): (
        ("lord", 3), ("father", 3), ("son", 2), ("sword", 2), ("daughter", 2),
        ("wife", 2), ("friend", 2),
    ),
    ("where", "is", "thy"): (
        ("father", 3), ("sword", 2), ("lord", 2), ("mother", 2),
    ),
    ("where", "is", "he"): (
        ("?", 4), (",", 2), ("gone", 2), ("now", 2),
    ),
    ("where", "art", "thou"): (
        ("?", 5), (",", 2), ("gone", 1),
    ),
    ("who", "art", "thou"): (
        ("?", 5), (",", 2),
    ),
    ("who", "are", "you"): (
        ("?", 4), (",", 2),
    ),
    ("when", "i", "was"): (
        ("a", 3), ("young", 2), ("but", 2), ("in", 2), ("with", 2),
    ),
    ("in", "the", "world"): (
        (",", 3), (".", 2), ("?", 2),
    ),
    ("in", "the", "name"): (
        ("of", 5),
    ),
    ("all", "the", "world"): (
        ("'s", 2), ("is", 2), (",", 2), ("?", 2),
    ),
    ("from", "my", "heart"): (
        (",", 3), (".", 2), ("!", 1),
    ),
    ("with", "all", "my"): (
        ("heart", 4), ("soul", 2), ("might", 2),
    ),
    ("by", "this", "hand"): (
        (",", 4), (".", 2), ("!", 1),
    ),
    ("by", "this", "light"): (
        (",", 3), (".", 2),
    ),
    ("by", "heaven", ","): (
        ("i", 2), ("he", 2), ("it", 2), ("a", 2), ("thou", 2),
    ),
    ("i", "warrant", "you"): (
        (",", 3), (".", 2), ("!", 1),
    ),
    ("i", "warrant", "thee"): (
        (",", 3), (".", 2),
    ),
    ("i", "thank", "you"): (
        (",", 3), ("sir", 2), ("for", 2), (".", 2),
    ),
    ("i", "thank", "thee"): (
        (",", 3), ("for", 2), (".", 2),
    ),
    ("farewell", ",", "good"): (
        ("my", 2), ("sir", 2), ("friend", 2), ("madam", 2), ("cousin", 2),
        ("brother", 2),
    ),
    ("good", "night", ","): (
        ("my", 2), ("sweet", 2), ("good", 2), ("sir", 2),
    ),
    ("good", "morrow", ","): (
        ("my", 2), ("sir", 2), ("good", 2), ("sweet", 2),
    ),
    ("what", "a", "piece"): (
        ("of", 5),
    ),
    ("i", "have", "a"): (
        ("mind", 2), ("heart", 2), ("tongue", 2), ("son", 2), ("word", 2),
        ("brother", 2), ("friend", 2), ("daughter", 2),
    ),
    ("you", "shall", "not"): (
        ("have", 2), ("be", 3), ("go", 2), ("speak", 2), ("see", 2),
    ),
    ("you", "have", "done"): (
        ("me", 2), ("well", 3), ("a", 2), ("it", 2),
    ),
    ("he", "hath", "done"): (
        ("me", 2), ("it", 2), ("no", 2), ("the", 2),
    ),
    ("hath", "he", "not"): (
        ("?", 3), ("done", 2), ("been", 2),
    ),
    ("is", "it", "not"): (
        ("?", 4), ("so", 2), ("a", 2),
    ),
    ("i", "cannot", "tell"): (
        (",", 3), (".", 2), ("what", 2), ("how", 2), ("why", 2),
    ),
    ("i", "cannot", "speak"): (
        ("it", 2), ("to", 2), (",", 2),
    ),
    ("do", "you", "hear"): (
        ("?", 4), (",", 2), ("me", 2),
    ),
    ("do", "you", "know"): (
        ("?", 3), ("me", 2), ("what", 2), ("him", 2), ("her", 2),
    ),
    ("did", "you", "not"): (
        ("?", 3), ("hear", 2), ("see", 2), ("know", 2),
    ),
    ("take", "my", "leave"): (
        (",", 3), (".", 3), ("of", 2),
    ),
    ("take", "heed", ","): (
        ("my", 2), ("sir", 2), ("lest", 2),
    ),
    ("come", "hither", ","): (
        ("sir", 2), ("my", 2), ("good", 2), ("sweet", 2),
    ),
    ("come", ",", "sir"): (
        (",", 3), (".", 2), ("!", 1),
    ),
    ("so", "help", "me"): (
        ("god", 3), ("heaven", 2), (",", 2),
    ),
    ("as", "i", "am"): (
        ("a", 3), ("the", 2), ("thy", 2), ("true", 2),
    ),
    ("as", "thou", "art"): (
        ("a", 3), ("the", 2), ("my", 2), ("true", 2),
    ),
    ("if", "it", "be"): (
        ("so", 3), ("true", 2), ("not", 2), ("the", 2),
    ),
    ("if", "you", "will"): (
        (",", 2), ("not", 2), ("have", 2), ("go", 2),
    ),
    ("if", "thou", "wilt"): (
        (",", 2), ("not", 2), ("have", 2), ("go", 2),
    ),
    ("in", "good", "faith"): (
        (",", 3), (".", 2),
    ),
    ("in", "good", "sooth"): (
        (",", 3), (".", 2),
    ),
    ("i", "tell", "thee"): (
        (",", 3), ("what", 2), ("no", 2), (".", 2),
    ),
    ("i", "tell", "you"): (
        (",", 3), ("what", 2), ("sir", 2), (".", 2),
    ),
    ("mark", "me", ","): (
        ("sir", 2), ("my", 2), ("now", 2),
    ),
    ("mark", "you", ","): (
        ("sir", 2), ("my", 2), ("now", 2),
    ),
}


def _build_tries() -> dict[tuple[str, str, str], dict[str, dict[str, int]]]:
    """Build a trie per 3-word context."""
    out: dict[tuple[str, str, str], dict[str, dict[str, int]]] = {}
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
                    # Word ended — terminal chars.
                    for term in (" ", ",", ".", ";", ":", "?", "!", "\n"):
                        trie[prefix][term] = trie[prefix].get(term, 0) + w
        out[key] = trie
    return out


_TRIES: dict[tuple[str, str, str], dict[str, dict[str, int]]] = _build_tries()


def _build_vector(nexts: dict[str, int], prefix_len: int) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    total = sum(nexts.values())
    if total <= 0:
        return vec
    # Scale stronger than 2-word context — 3-word is the sharpest cue.
    scale = min(0.7 + 0.3 * prefix_len, 1.8)
    negative = -0.45 * scale
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


def _build_prefix_bias() -> dict[tuple[str, str, str, str], list[float]]:
    out: dict[tuple[str, str, str, str], list[float]] = {}
    for key, trie in _TRIES.items():
        for prefix, nexts in trie.items():
            if not prefix:
                continue
            out[(key[0], key[1], key[2], prefix)] = _build_vector(
                nexts, len(prefix)
            )
    return out


_PREFIX_BIAS: dict[tuple[str, str, str, str], list[float]] = _build_prefix_bias()


def phrase_trigram_continue_bias(
    prev_prev_completed_word: str,
    prev_completed_word: str,
    last_completed_word: str,
    word_buffer: str,
    letter_run_len: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if (
        not prev_prev_completed_word
        or not prev_completed_word
        or not last_completed_word
    ):
        return None
    if not word_buffer:
        return None
    if letter_run_len < 1 or letter_run_len > 8:
        return None
    if len(word_buffer) != letter_run_len:
        return None
    p3 = prev_prev_completed_word.lower()
    p2 = prev_completed_word.lower()
    p1 = last_completed_word.lower()
    key = word_buffer.lower()
    return _PREFIX_BIAS.get((p3, p2, p1, key))
