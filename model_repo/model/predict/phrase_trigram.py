"""Word-level trigram word-start bias layer.

Keyed on the three most recent completed words (prev_prev, prev, last),
this layer biases the first letter of the NEXT word. It's a
true 3-word lookback — strictly more specific than phrase_bigram
(which only sees the last 2).

Shakespeare's writing has a strong tail of multi-word formulas that
3-word context disambiguates far better than 2-word context:

   "i pray thee"  -> "tell"/"speak" (both start "t"/"s"; bigram after
                     "pray thee" is too broad — "pray thee" alone can
                     begin many continuations. With "i pray thee"
                     we strongly expect "tell/speak/hear/stay/do".)
   "to be or"     -> "not" (n)
   "or not to"    -> "be" (b)
   "not to be"    -> "that" (t)
   "by my troth"  -> "i" (i)
   "o my dear"    -> "lord/friend/madam" (l/f/m)
   "good my lord" -> punctuation (, / .) — ends a vocative.
   "i do beseech" -> "thee/you" (t/y)
   "an it please" -> "thee/you/your" (t/y)

Layer is applied at word-start (space/single-newline precedes) when
we have three non-empty words in context. Returns None when no
entry matches.

All entries come from prior knowledge of early-modern English
formulas — no corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# (prev_prev, prev, last) -> { first_letter: bias }
# Bias is a log-odds nudge added to logits; 1.0-2.5 is typical.
_PT: dict[tuple[str, str, str], dict[str, float]] = {
    # "I pray thee ..." → tell, speak, stay, hear, do, let, say, come,
    # look, go, what, how, now.
    ("i", "pray", "thee"): {"t": 2.2, "s": 1.6, "h": 1.4, "d": 1.4,
                             "l": 1.2, "c": 1.2, "g": 1.0, "b": 0.9,
                             "n": 1.0, "w": 1.2, "m": 0.7, "r": 0.6},
    ("pray", "thee", "tell"): {"m": 2.0, "u": 1.2, "h": 1.0, "t": 0.8},
    # "to be or" → not
    ("to", "be", "or"): {"n": 3.0},
    # "or not to" → be
    ("or", "not", "to"): {"b": 2.8},
    # "not to be" → that / ,  (famous quotation or a pause)
    ("not", "to", "be"): {"t": 2.0, ",": 1.2, ".": 0.6},
    # "i do beseech" / "i do beseech you"
    ("i", "do", "beseech"): {"t": 2.4, "y": 2.0},
    ("do", "beseech", "you"): {",": 1.6, "t": 1.0, "m": 0.8, "s": 0.7},
    # "by my troth" → i / , / .
    ("by", "my", "troth"): {",": 2.0, ".": 1.2, "i": 1.4, "t": 0.8},
    # "upon my honour" / "upon my soul" / "upon my word"
    ("upon", "my", "honour"): {",": 2.0, ".": 1.2, "i": 0.8},
    ("upon", "my", "soul"): {",": 2.0, "i": 1.4, "t": 0.8},
    ("upon", "my", "word"): {",": 2.0, ".": 1.2, "i": 1.0},
    # "as you like" → it
    ("as", "you", "like"): {"i": 2.8, "t": 1.0},
    # "what do you" → want, mean, say, do, think, know
    ("what", "do", "you"): {"w": 1.6, "m": 1.6, "s": 1.4, "d": 1.2,
                             "t": 1.4, "k": 1.4, "a": 0.9},
    # "how do you" → do, fare, find, mean
    ("how", "do", "you"): {"d": 2.2, "f": 1.6, "m": 1.2, "l": 0.9},
    # "when i was" → young, a (boy/child), your (age), there, there (here)
    ("when", "i", "was"): {"a": 1.4, "y": 1.4, "i": 0.6, "t": 1.0},
    # "my lord i" → have, am, do, pray, beseech, will, would, shall,
    # know, cannot
    ("my", "lord", "i"): {"h": 1.6, "a": 1.4, "d": 1.2, "p": 1.4,
                           "b": 1.2, "w": 1.4, "s": 1.0, "k": 0.9,
                           "c": 0.9, "t": 0.8},
    ("my", "gracious", "lord"): {",": 1.8, "i": 1.0, ".": 0.8, "w": 0.6},
    ("my", "noble", "lord"): {",": 1.8, "i": 1.0, ".": 0.8, "w": 0.6},
    ("my", "good", "lord"): {",": 1.8, "i": 1.2, "w": 0.6},
    ("good", "my", "lord"): {",": 1.8, "i": 1.2, ".": 0.8, "w": 0.6},
    # "in good time" → ,
    ("in", "good", "time"): {",": 1.8, ".": 1.0},
    # "i have been" → a, at, so, in, with, the, here, there, to
    ("i", "have", "been"): {"s": 1.4, "a": 1.4, "h": 1.2, "t": 1.2,
                             "w": 1.0, "i": 0.8, "o": 0.6, "n": 0.6,
                             "b": 0.6, "f": 0.6},
    # "i will not" → be, have, do, let, tell, go, stay, stand, stir,
    # suffer, yield, give
    ("i", "will", "not"): {"b": 1.4, "h": 1.2, "d": 1.2, "l": 1.2,
                            "t": 1.2, "g": 1.2, "s": 1.4, "y": 1.0,
                            "a": 0.8},
    # "i do not" → know, think, understand, mean, like, love, care,
    # doubt, dare, fear
    ("i", "do", "not"): {"k": 1.6, "t": 1.4, "u": 1.0, "m": 1.2,
                          "l": 1.2, "c": 1.0, "d": 1.4, "f": 1.0,
                          "w": 0.8, "b": 0.8, "s": 0.6},
    # "i cannot tell" → ,/./how/what/why
    ("i", "cannot", "tell"): {",": 1.6, ".": 0.8, "h": 1.2, "w": 1.4,
                               "y": 1.0, ";": 0.6},
    # "god save the" → king / queen / duke
    ("god", "save", "the"): {"k": 2.4, "q": 1.6, "d": 1.0, "c": 0.8,
                              "e": 0.6},
    ("long", "live", "the"): {"k": 2.4, "q": 1.6, "d": 1.0, "p": 1.2},
    # "hail to thee" / "hail to you" — close with ,
    ("hail", "to", "thee"): {",": 1.6, ".": 0.8, "f": 0.8},
    # "the more i" → see/know/think/love/look/hear/consider
    ("the", "more", "i"): {"s": 1.4, "k": 1.2, "t": 1.0, "l": 1.2,
                            "h": 1.0, "c": 0.8},
    # "as i am" → a (man/woman), an (honest), no (saint), your,
    # so, true, here
    ("as", "i", "am"): {"a": 1.4, "s": 1.0, "n": 1.0, "y": 0.8,
                         "t": 0.9, "h": 0.8},
    # "what is the" → matter, news, reason, cause, meaning, time
    ("what", "is", "the"): {"m": 1.8, "n": 1.4, "r": 1.4, "c": 1.4,
                             "t": 0.8, "b": 0.6},
    # "this is the" → man, day, way, sword, house, time, hour, place
    ("this", "is", "the"): {"m": 1.4, "d": 1.2, "w": 1.2, "s": 1.0,
                             "h": 1.0, "t": 1.0, "p": 1.0, "k": 0.6,
                             "b": 0.6, "c": 0.6},
    # "here is the" / "there is the"
    ("here", "is", "the"): {"m": 1.4, "k": 1.2, "q": 1.0, "l": 1.0,
                             "n": 1.0, "t": 0.8, "c": 0.8},
    # "in the name" → of
    ("in", "the", "name"): {"o": 3.0},
    # "for the love" → of
    ("for", "the", "love"): {"o": 2.6},
    # "the name of" → god / heaven / (a proper noun)
    ("the", "name", "of"): {"g": 1.6, "h": 1.4, "t": 0.8, "j": 0.8,
                             "c": 0.6, "l": 0.5},
    # "never more shall" → /  "shall never more"
    ("never", "more", "shall"): {"i": 0.8, "b": 1.0, "s": 0.8},
    # "now is the" → winter / time / hour
    ("now", "is", "the"): {"w": 2.2, "t": 1.8, "h": 1.2, "d": 0.8},
    # "winter of our" → discontent (d)
    ("winter", "of", "our"): {"d": 3.0},
    # "the king is" → dead / here / gone / coming / here / angry
    ("the", "king", "is"): {"d": 1.8, "h": 1.2, "g": 1.2, "c": 1.0,
                             "a": 1.0, "m": 0.6, "n": 0.8},
    # "the queen is" — same shape
    ("the", "queen", "is"): {"d": 1.8, "h": 1.2, "g": 1.2, "c": 1.0,
                              "w": 0.8, "a": 0.9},
    # "come hither" / "come forth"
    ("come", "hither", "and"): {"s": 1.2, "t": 1.2, "l": 1.0, "b": 1.0,
                                 "h": 0.8, "k": 0.6},
    # "out of my" → sight, way, mind, house, presence
    ("out", "of", "my"): {"s": 1.8, "w": 1.4, "m": 1.2, "h": 1.0,
                           "p": 1.0, "h": 0.8},
    # "heart of my" → heart/soul/love
    ("heart", "of", "my"): {"h": 1.2, "l": 1.0, "s": 0.8},
    # "if it please" → you / your / thee / thy
    ("if", "it", "please"): {"y": 2.0, "t": 1.6, "g": 0.8, "h": 0.6},
    ("an", "it", "please"): {"y": 2.0, "t": 1.6, "g": 0.8},
    # "give me your" → hand, word, leave, help, hand, gold
    ("give", "me", "your"): {"h": 1.8, "w": 1.4, "l": 1.2, "g": 0.8,
                              "p": 0.6},
    # "take my" / "give my"
    ("take", "my", "word"): {",": 1.2, ".": 0.8, "f": 0.8, "t": 0.8,
                              "a": 0.6, "i": 0.6},
    # "why then i" → will / may / must / shall / would
    ("why", "then", "i"): {"w": 1.6, "m": 1.4, "s": 1.2, "a": 1.0,
                            "h": 0.8, "d": 0.6},
    # "fare thee well" → ,/./; (closing)
    ("fare", "thee", "well"): {",": 1.8, ".": 1.2, ";": 0.8, "!": 0.8},
    ("fare", "you", "well"): {",": 1.8, ".": 1.2, ";": 0.8, "!": 0.8},
    # "hath given me" → a, the, his, her, this, such
    ("hath", "given", "me"): {"a": 1.4, "t": 1.0, "h": 1.0, "s": 1.0,
                               "l": 0.6},
    # "i am a" → man, gentleman, soldier, ..., poor
    ("i", "am", "a"): {"m": 1.4, "g": 1.2, "s": 1.2, "p": 1.4, "w": 0.8,
                        "f": 1.0, "k": 0.8, "t": 0.6, "c": 0.6},
    # "but i am" → a/no/so/not/one/the/as/your
    ("but", "i", "am"): {"a": 1.2, "n": 1.4, "s": 1.2, "o": 0.8,
                          "t": 0.8, "y": 0.6, "h": 0.6},
    # "but i will" → ..
    ("but", "i", "will"): {"b": 1.0, "n": 1.4, "h": 1.0, "d": 1.0,
                            "s": 1.0, "g": 0.8, "m": 0.8, "t": 0.8,
                            "l": 0.6},
    # "speak to me" → ,/./!/of/again
    ("speak", "to", "me"): {",": 1.2, ".": 0.8, "!": 0.8, "o": 1.2,
                             "a": 1.0},
    # "let us go" → hence / together / to / and
    ("let", "us", "go"): {"h": 1.4, "t": 1.2, "a": 1.0, "b": 0.8},
    # "let us be" → gone / friends / merry / quiet
    ("let", "us", "be"): {"g": 1.4, "f": 1.2, "m": 1.0, "q": 0.8,
                           "r": 0.6},
    # "i know not" → how / why / what / where / when / whether
    ("i", "know", "not"): {"h": 1.4, "w": 2.0, ".": 1.0, ",": 0.8},
    # "i see it" → not / now / in / in my / well
    ("i", "see", "it"): {"n": 1.6, "w": 1.2, "i": 1.0, ".": 0.8,
                          ",": 0.8},
    # "tell me true" — final word often
    ("tell", "me", "true"): {",": 1.6, ".": 1.0, "!": 0.8},
    # "give me leave" → to
    ("give", "me", "leave"): {"t": 2.4, ",": 1.0},
    # "thou hast not" → ..
    ("thou", "hast", "not"): {"s": 1.0, "d": 1.0, "a": 1.0, "y": 0.8,
                               "t": 0.8},
    # "thou shalt not" → ..(commandment-like)
    ("thou", "shalt", "not"): {"l": 0.6, "k": 0.6, "s": 0.8, "f": 0.8,
                                "h": 0.6, "b": 0.8, "d": 0.8, "m": 0.6},
    # "in such a" → case/manner/sort
    ("in", "such", "a"): {"c": 1.4, "m": 1.2, "s": 1.2, "t": 0.8,
                           "p": 0.6, "w": 0.6, "n": 0.6, "k": 0.6},
    # "by the gods" → ,
    ("by", "the", "gods"): {",": 1.8, ".": 0.8, "!": 0.6},
    ("by", "the", "lord"): {",": 1.8, ".": 0.8, "!": 0.6},
    # "for god sake" / "for god's sake"
    ("for", "god's", "sake"): {",": 1.6, ".": 0.8, "!": 0.8},
    # "by god's will" / ..
    ("by", "god's", "will"): {",": 1.2, "i": 0.8, "!": 0.6},
    # generic vocative closers
    ("my", "dear", "lord"): {",": 1.8, ".": 0.8, "!": 0.6},
    ("my", "dear", "friend"): {",": 1.6, ".": 0.8, "!": 0.6},
    ("o", "my", "lord"): {",": 1.8, ".": 0.8, "!": 1.2},
    ("o", "my", "liege"): {",": 1.8, ".": 0.8, "!": 1.2},
    ("o", "dear", "my"): {"l": 1.8, "f": 1.0, "s": 0.8},
    # Reply openers: "ay / yes my lord" constructions
    ("ay", "my", "lord"): {",": 1.4, ".": 0.8, ";": 0.6, "!": 0.6},
    ("no", "my", "lord"): {",": 1.4, ".": 0.8, ";": 0.6, "!": 0.6},
    ("yes", "my", "lord"): {",": 1.4, ".": 0.8, ";": 0.6, "!": 0.6},
    # "as it were" → ,/./(a (then))
    ("as", "it", "were"): {",": 1.6, ".": 0.8, ";": 0.6},
    # "what say you" → to/of/,/?
    ("what", "say", "you"): {"t": 1.4, "o": 1.2, ",": 0.8, "?": 0.6},
    # "will you go" / "will you stay"
    ("will", "you", "go"): {"w": 1.4, "?": 1.0, ".": 0.6},
    # "i beseech you" → , /./sir/my
    ("i", "beseech", "you"): {",": 1.4, "s": 0.8, "m": 0.8},
    # "what will you" → say/do/have/be
    ("what", "will", "you"): {"s": 1.4, "d": 1.4, "h": 1.0, "b": 0.8,
                               "g": 0.6},
    # "you are a" → fool / villain / traitor / man / good / brave
    ("you", "are", "a"): {"f": 1.4, "v": 1.4, "t": 1.2, "m": 1.0,
                           "g": 1.0, "b": 1.0, "k": 0.8, "s": 0.8},
    # "and yet i" → love/fear/know/must/cannot/will
    ("and", "yet", "i"): {"l": 1.2, "f": 1.2, "k": 1.2, "m": 1.2,
                           "c": 1.2, "w": 1.0, "h": 0.8, "d": 0.8},
    # "i am not" → ..  (mad/afraid/well/a coward)
    ("i", "am", "not"): {"m": 1.2, "a": 1.4, "w": 1.2, "y": 0.8,
                          "s": 1.0, "t": 0.8, "f": 1.0},
    # "am i not" → a / your / thy / the / thus / old
    ("am", "i", "not"): {"a": 1.2, "y": 1.2, "t": 1.4, "w": 0.8,
                          "o": 0.8},
    # "to see the" → king/queen/lord/face/light
    ("to", "see", "the"): {"k": 1.4, "q": 1.2, "l": 1.2, "f": 1.4,
                            "d": 0.8, "s": 0.8},
}


def _build_vectors() -> dict[tuple[str, str, str], list[float]]:
    out: dict[tuple[str, str, str], list[float]] = {}
    for key, entries in _PT.items():
        vec = [0.0] * VOCAB_SIZE
        for nxt, bias in entries.items():
            if nxt in VOCAB_INDEX:
                vec[VOCAB_INDEX[nxt]] = bias
                if nxt.isalpha() and nxt.lower() == nxt:
                    up = nxt.upper()
                    if up in VOCAB_INDEX:
                        vec[VOCAB_INDEX[up]] = bias * 0.6
        out[key] = vec
    return out


PHRASE_TRIGRAM_BIAS: dict[tuple[str, str, str], list[float]] = _build_vectors()

_GLOBAL_SCALE = 1.0


def phrase_trigram_bias(
    prev_prev: str, prev: str, last: str
) -> list[float] | None:
    """First-letter bias for the next word given the last 3 completed
    words (oldest first). Returns None if no entry exists."""
    if not prev_prev or not prev or not last:
        return None
    key = (prev_prev.lower(), prev.lower(), last.lower())
    v = PHRASE_TRIGRAM_BIAS.get(key)
    if v is None:
        return None
    if _GLOBAL_SCALE == 1.0:
        return v
    return [x * _GLOBAL_SCALE for x in v]
