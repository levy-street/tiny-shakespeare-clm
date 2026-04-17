"""Two-word phrase bigram bias layer.

Given the previous TWO completed words (prev_completed_word and
last_completed_word), bias the first letter of the next (i.e., third)
word. This captures 3-gram word-level formulas that are extremely
frequent in Shakespeare — "I pray thee", "O my lord", "I have been",
"by my troth", "to be or not", "let me speak", "I do beseech", etc.

Active only at word-start positions (after a space or single newline).
Stacks on top of next_word_bias (which uses only last_completed_word)
and pos_next_bias. Entries below are hand-chosen from prior knowledge
of Shakespearean idiom — no corpus statistics.
"""

from __future__ import annotations

import math

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# (prev, last) -> { first_letter_of_next_word: weight }.
# Weights are small positive integers; they're log-normalized into biases.
# Unlisted letters get a small negative bump.
_PHRASE_NEXT: dict[tuple[str, str], dict[str, int]] = {
    # "I am" + ...
    ("i", "am"): {"n": 5, "s": 4, "a": 4, "m": 3, "t": 3, "y": 3, "h": 3,
                  "g": 3, "w": 3, "b": 3, "f": 3, "l": 3, "c": 2, "d": 2,
                  "p": 2, "r": 2, "o": 2},
    ("thou", "art"): {"a": 5, "n": 4, "s": 4, "t": 3, "m": 3, "h": 3, "w": 3,
                      "b": 3, "d": 2, "g": 2, "p": 2, "c": 2, "f": 2, "r": 2,
                      "i": 2, "o": 2, "l": 2},
    ("i", "have"): {"n": 6, "s": 5, "b": 5, "d": 4, "h": 4, "t": 3, "l": 3,
                    "m": 3, "f": 3, "a": 3, "g": 3, "p": 2, "c": 2, "r": 2,
                    "w": 2, "o": 2},
    ("thou", "hast"): {"n": 5, "s": 4, "b": 4, "d": 4, "h": 3, "t": 3, "l": 3,
                       "m": 3, "f": 3, "a": 3, "g": 2, "p": 2, "c": 2},
    ("he", "hath"): {"n": 5, "s": 4, "b": 4, "d": 4, "m": 3, "t": 3, "f": 3,
                     "l": 3, "h": 3, "g": 2, "p": 2, "a": 2, "c": 2, "w": 2},
    ("she", "hath"): {"n": 5, "s": 4, "b": 4, "d": 4, "m": 3, "t": 3, "f": 3,
                      "l": 3, "h": 3, "g": 2, "p": 2, "a": 2, "c": 2, "w": 2},
    ("i", "will"): {"n": 5, "d": 4, "b": 4, "t": 4, "g": 3, "s": 3, "m": 3,
                    "h": 3, "a": 3, "f": 3, "p": 3, "l": 3, "c": 2, "r": 2,
                    "w": 2, "e": 2},
    ("we", "will"): {"n": 4, "d": 4, "b": 4, "t": 4, "g": 3, "s": 3, "m": 3,
                     "h": 3, "a": 3, "f": 3, "p": 3, "l": 3, "c": 2, "r": 2,
                     "w": 2},
    ("thou", "wilt"): {"n": 4, "d": 3, "b": 3, "t": 3, "g": 3, "s": 3, "m": 3,
                       "h": 3, "a": 3, "f": 3, "p": 2, "l": 2, "c": 2},
    ("i", "do"): {"n": 5, "b": 4, "s": 3, "t": 3, "p": 3, "l": 3, "h": 3,
                  "k": 3, "a": 3, "m": 3, "f": 3, "g": 2, "c": 2, "r": 2,
                  "w": 2, "e": 2},
    ("i", "did"): {"n": 4, "s": 3, "t": 3, "b": 3, "h": 3, "l": 3, "m": 3,
                   "a": 3, "f": 3, "g": 2, "p": 2, "c": 2, "r": 2, "w": 2},
    ("i", "would"): {"n": 4, "s": 3, "t": 3, "b": 3, "h": 3, "l": 3, "m": 3,
                     "a": 3, "f": 3, "g": 2, "p": 2, "c": 2, "r": 2, "w": 2,
                     "y": 2},
    ("i", "could"): {"n": 4, "s": 3, "t": 3, "b": 3, "h": 3, "l": 3, "m": 3,
                     "a": 3, "f": 3, "g": 2, "p": 2, "c": 2, "r": 2, "w": 2},
    ("i", "shall"): {"n": 4, "s": 3, "t": 3, "b": 3, "h": 3, "l": 3, "m": 3,
                     "a": 3, "f": 3, "g": 2, "p": 2, "c": 2, "r": 2, "w": 2},
    ("we", "shall"): {"n": 4, "s": 3, "t": 3, "b": 3, "h": 3, "l": 3, "m": 3,
                      "a": 3, "f": 3, "g": 2, "p": 2, "c": 2, "r": 2, "w": 2},
    ("i", "must"): {"n": 4, "s": 3, "t": 3, "b": 3, "h": 3, "l": 3, "m": 3,
                    "a": 3, "f": 3, "g": 2, "p": 2, "c": 2, "r": 2, "w": 2,
                    "d": 2},
    ("i", "may"): {"n": 4, "s": 3, "t": 3, "b": 3, "h": 3, "l": 3, "m": 3,
                   "a": 3, "f": 3, "g": 2, "p": 2, "c": 2, "r": 2, "w": 2,
                   "d": 2},
    ("it", "is"): {"n": 5, "s": 4, "t": 4, "a": 3, "m": 3, "b": 3, "i": 3,
                   "p": 3, "y": 3, "o": 2, "h": 2, "g": 2, "f": 2, "l": 2,
                   "r": 2, "c": 2, "d": 2, "w": 2, "e": 2},
    ("this", "is"): {"n": 4, "s": 3, "t": 3, "a": 3, "m": 3, "b": 3, "i": 3,
                     "p": 3, "y": 2, "o": 2, "h": 2, "g": 2, "f": 2, "l": 2,
                     "r": 2, "c": 2, "d": 2, "w": 2, "e": 2},
    ("that", "is"): {"n": 4, "s": 3, "t": 3, "a": 3, "m": 3, "b": 3, "i": 3,
                     "p": 3, "y": 2, "o": 2, "h": 2, "g": 2, "f": 2, "l": 2,
                     "r": 2, "c": 2, "d": 2, "w": 2},
    ("what", "is"): {"t": 4, "i": 3, "y": 3, "h": 3, "s": 3, "m": 3, "a": 3,
                     "b": 2, "d": 2, "f": 2, "g": 2, "l": 2, "o": 2, "p": 2,
                     "r": 2, "w": 2, "n": 2},
    ("you", "are"): {"n": 4, "s": 3, "t": 3, "a": 3, "m": 3, "b": 3, "i": 3,
                     "p": 3, "y": 2, "o": 2, "h": 2, "g": 2, "f": 2, "l": 2,
                     "r": 2, "c": 2, "d": 2, "w": 2},
    ("we", "are"): {"n": 4, "s": 3, "t": 3, "a": 3, "m": 3, "b": 3, "i": 3,
                    "p": 3, "y": 2, "o": 2, "h": 2, "g": 2, "f": 2, "l": 2,
                    "r": 2, "c": 2, "d": 2, "w": 2},
    ("they", "are"): {"n": 4, "s": 3, "t": 3, "a": 3, "m": 3, "b": 3, "i": 3,
                      "p": 3, "y": 2, "o": 2, "h": 2, "g": 2, "f": 2, "l": 2,
                      "r": 2, "c": 2, "d": 2, "w": 2},
    # Interjections / vocatives.
    ("o", "my"): {"l": 7, "g": 5, "d": 4, "s": 4, "f": 4, "h": 3, "n": 3,
                  "b": 3, "m": 3, "p": 3, "c": 2, "t": 2, "w": 2, "r": 2},
    ("o", "thou"): {"d": 4, "f": 4, "g": 4, "h": 4, "l": 3, "m": 3, "n": 3,
                    "s": 3, "t": 3, "b": 3, "w": 3, "a": 3, "c": 2, "p": 2},
    ("o", "sweet"): {"l": 4, "g": 3, "m": 3, "p": 3, "s": 3, "f": 3, "h": 3,
                     "b": 2, "c": 2, "d": 2, "n": 2, "r": 2, "t": 2, "w": 2},
    ("o", "god"): {"a": 3, "o": 3, "i": 3, "m": 3, "p": 3, "t": 3, "s": 3,
                   "h": 3, "b": 2, "d": 2, "f": 2, "g": 2, "l": 2, "n": 2,
                   "r": 2, "w": 2},
    # Formulaic greetings / oaths.
    ("by", "my"): {"t": 5, "f": 5, "s": 4, "l": 3, "h": 3, "m": 3, "b": 2,
                   "c": 2, "d": 2, "g": 2, "n": 2, "p": 2, "r": 2, "w": 2},
    ("good", "my"): {"l": 6, "g": 3, "f": 2, "b": 2, "d": 2, "h": 2, "m": 2,
                     "p": 2, "s": 2, "c": 2},
    ("my", "good"): {"l": 6, "m": 3, "s": 3, "f": 3, "h": 3, "b": 2, "c": 2,
                     "d": 2, "g": 2, "n": 2, "p": 2, "r": 2, "t": 2, "w": 2},
    ("i", "pray"): {"t": 6, "y": 4, "s": 3, "l": 2, "m": 2, "n": 2, "h": 2,
                    "g": 2, "b": 2, "d": 2, "c": 2, "f": 2, "p": 2, "r": 2,
                    "w": 2},
    ("pray", "you"): {"t": 5, "y": 2, "a": 2, "g": 2, "h": 2, "l": 2, "m": 2,
                      "n": 2, "s": 2, "b": 2, "c": 2, "d": 2, "f": 2, "p": 2,
                      "r": 2, "w": 2},
    ("i", "beseech"): {"y": 6, "t": 5, "s": 3, "h": 2, "m": 2, "g": 2, "b": 2,
                       "d": 2, "f": 2, "l": 2, "n": 2, "p": 2, "r": 2, "w": 2},
    # "Let me" + verb
    ("let", "me"): {"n": 4, "s": 4, "h": 4, "g": 3, "b": 3, "t": 3, "l": 3,
                    "f": 3, "m": 3, "a": 3, "k": 3, "p": 3, "r": 2, "w": 2,
                    "c": 2, "d": 2, "e": 2, "y": 2},
    ("let", "us"): {"n": 4, "s": 4, "h": 4, "g": 3, "b": 3, "t": 3, "l": 3,
                    "f": 3, "m": 3, "a": 3, "k": 3, "p": 3, "r": 2, "w": 2,
                    "c": 2, "d": 2, "e": 2, "y": 2},
    # After prepositions + pronouns.
    ("to", "be"): {"a": 4, "o": 4, "t": 4, "s": 3, "n": 3, "i": 3, "m": 3,
                   "h": 3, "w": 3, "b": 2, "c": 2, "d": 2, "f": 2, "g": 2,
                   "l": 2, "p": 2, "r": 2, "y": 2, "e": 2},
    ("to", "be,"): {"o": 5, "t": 4, "a": 3, "s": 3, "i": 3},  # not really
    ("must", "be"): {"a": 4, "o": 3, "t": 3, "s": 3, "n": 3, "i": 3, "m": 3,
                     "h": 3, "w": 3, "b": 2, "c": 2, "d": 2, "f": 2, "g": 2,
                     "l": 2, "p": 2, "r": 2, "y": 2, "e": 2},
    ("shall", "be"): {"a": 4, "o": 3, "t": 3, "s": 3, "n": 3, "i": 3, "m": 3,
                      "h": 3, "w": 3, "b": 2, "c": 2, "d": 2, "f": 2, "g": 2,
                      "l": 2, "p": 2, "r": 2, "y": 2},
    ("will", "be"): {"a": 4, "o": 3, "t": 3, "s": 3, "n": 3, "i": 3, "m": 3,
                     "h": 3, "w": 3, "b": 2, "c": 2, "d": 2, "f": 2, "g": 2,
                     "l": 2, "p": 2, "r": 2, "y": 2},
    # Common linking bigrams.
    ("for", "i"): {"a": 4, "h": 4, "w": 3, "m": 3, "s": 3, "k": 3, "c": 3,
                   "d": 3, "f": 2, "l": 2, "n": 2, "p": 2, "r": 2, "t": 2,
                   "b": 2, "g": 2},
    ("and", "i"): {"a": 4, "h": 4, "w": 3, "m": 3, "s": 3, "k": 3, "c": 3,
                   "d": 3, "f": 2, "l": 2, "n": 2, "p": 2, "r": 2, "t": 2,
                   "b": 2, "g": 2},
    ("that", "i"): {"a": 4, "h": 4, "w": 3, "m": 3, "s": 3, "k": 3, "c": 3,
                    "d": 3, "f": 2, "l": 2, "n": 2, "p": 2, "r": 2, "t": 2,
                    "b": 2, "g": 2},
    # "fare thee well".
    ("fare", "thee"): {"w": 8, "n": 2, "s": 2, "g": 2, "a": 1, "b": 1, "c": 1,
                       "d": 1, "f": 1, "h": 1, "l": 1, "m": 1, "p": 1, "r": 1,
                       "t": 1},
    # "God save the..."
    ("god", "save"): {"t": 6, "u": 5, "y": 4, "m": 2, "h": 2, "s": 2, "a": 2},
    # "my lord, ..." / "my lord; ..." — the comma case. At that
    # word-start, the third word is often "I" (capital I after speaker
    # addressing). Common follow-ons: "I", "what", "say", "sir", etc.
    ("my", "lord"): {"i": 4, "w": 3, "s": 3, "a": 3, "t": 3, "h": 3, "y": 3,
                     "n": 2, "m": 2, "b": 2, "c": 2, "d": 2, "f": 2, "g": 2,
                     "l": 2, "p": 2, "r": 2},
    ("good", "lord"): {"i": 3, "w": 3, "s": 3, "a": 3, "t": 3, "h": 3, "y": 3,
                       "n": 2, "m": 2, "b": 2, "c": 2, "d": 2, "f": 2, "g": 2,
                       "l": 2, "p": 2, "r": 2},
    # "what say you" / "what say'st"
    ("what", "say"): {"y": 5, "t": 4, "i": 3, "a": 2, "b": 2, "c": 2, "d": 2,
                      "e": 2, "f": 2, "g": 2, "h": 2, "l": 2, "m": 2, "n": 2,
                      "o": 2, "p": 2, "r": 2, "s": 2, "u": 2, "w": 2},
    # Preposition/conjunction + "the" → noun first letter. Diverse
    # consonant-initial (l/s/m/f/k/w/h/c/n/b/d/p/g/r/t), vowel-initial rare.
    ("to", "the"): {"l": 5, "s": 5, "m": 4, "f": 4, "w": 4, "h": 4, "c": 4,
                    "k": 4, "n": 4, "b": 4, "d": 4, "p": 4, "t": 3, "r": 3,
                    "g": 3, "e": 3, "o": 3, "i": 2, "a": 2, "y": 2, "v": 2,
                    "u": 1, "q": 1, "j": 1},
    ("of", "the"): {"l": 5, "s": 5, "m": 4, "f": 4, "w": 4, "h": 4, "c": 4,
                    "k": 4, "n": 4, "b": 4, "d": 4, "p": 4, "t": 3, "r": 3,
                    "g": 3, "e": 3, "o": 3, "i": 2, "a": 2, "y": 2, "v": 2,
                    "u": 1, "q": 1, "j": 1},
    ("in", "the"): {"l": 5, "s": 5, "m": 4, "f": 4, "w": 4, "h": 4, "c": 4,
                    "k": 4, "n": 4, "b": 4, "d": 4, "p": 4, "t": 3, "r": 3,
                    "g": 3, "e": 3, "o": 3, "i": 2, "a": 2, "y": 2, "v": 2},
    ("on", "the"): {"l": 5, "s": 5, "m": 4, "f": 4, "w": 4, "h": 4, "c": 4,
                    "k": 4, "n": 4, "b": 4, "d": 4, "p": 4, "t": 3, "r": 3,
                    "g": 3, "e": 3, "o": 3, "i": 2, "a": 2, "y": 2, "v": 2},
    ("for", "the"): {"l": 5, "s": 5, "m": 4, "f": 4, "w": 4, "h": 4, "c": 4,
                     "k": 4, "n": 4, "b": 4, "d": 4, "p": 4, "t": 3, "r": 3,
                     "g": 3, "e": 3, "o": 3, "i": 2, "a": 2, "y": 2, "v": 2},
    ("by", "the"): {"l": 5, "s": 5, "m": 4, "f": 4, "w": 4, "h": 4, "c": 4,
                    "k": 4, "n": 4, "b": 4, "d": 4, "p": 4, "t": 3, "r": 3,
                    "g": 3, "e": 3, "o": 3, "i": 2, "a": 2, "y": 2, "v": 2},
    ("with", "the"): {"l": 5, "s": 5, "m": 4, "f": 4, "w": 4, "h": 4, "c": 4,
                      "k": 4, "n": 4, "b": 4, "d": 4, "p": 4, "t": 3, "r": 3,
                      "g": 3, "e": 3, "o": 3, "i": 2, "a": 2, "y": 2, "v": 2},
    ("and", "the"): {"l": 5, "s": 5, "m": 4, "f": 4, "w": 4, "h": 4, "c": 4,
                     "k": 4, "n": 4, "b": 4, "d": 4, "p": 4, "t": 3, "r": 3,
                     "g": 3, "e": 3, "o": 3, "i": 2, "a": 2, "y": 2, "v": 2},
    ("all", "the"): {"l": 5, "s": 5, "m": 4, "f": 4, "w": 4, "h": 4, "c": 4,
                     "k": 4, "n": 4, "b": 4, "d": 4, "p": 4, "t": 3, "r": 3,
                     "g": 3, "e": 3, "o": 3, "i": 2, "a": 2, "y": 2, "v": 2},
    ("from", "the"): {"l": 5, "s": 5, "m": 4, "f": 4, "w": 4, "h": 4, "c": 4,
                      "k": 4, "n": 4, "b": 4, "d": 4, "p": 4, "t": 3, "r": 3,
                      "g": 3, "e": 3, "o": 3, "i": 2, "a": 2, "y": 2, "v": 2},
    ("at", "the"): {"l": 4, "s": 4, "m": 3, "f": 3, "w": 3, "h": 3, "c": 3,
                    "k": 3, "n": 3, "b": 3, "d": 3, "p": 3, "t": 3, "r": 3,
                    "g": 3, "e": 2, "o": 2, "i": 2, "a": 2, "y": 2, "v": 2},
    ("as", "the"): {"l": 4, "s": 4, "m": 3, "f": 3, "w": 3, "h": 3, "c": 3,
                    "k": 3, "n": 3, "b": 3, "d": 3, "p": 3, "t": 3, "r": 3,
                    "g": 3, "e": 2, "o": 2, "i": 2, "a": 2, "y": 2, "v": 2},
    ("o'", "the"): {"l": 4, "s": 4, "m": 3, "f": 3, "w": 3, "h": 3, "c": 3,
                    "k": 3, "n": 3, "b": 3, "d": 3, "p": 3, "t": 3, "r": 3,
                    "g": 3, "e": 2, "o": 2, "i": 2, "a": 2, "y": 2, "v": 2},
    ("i'", "the"): {"l": 4, "s": 4, "m": 3, "f": 3, "w": 3, "h": 3, "c": 3,
                    "k": 3, "n": 3, "b": 3, "d": 3, "p": 3, "t": 3, "r": 3,
                    "g": 3, "e": 2, "o": 2, "i": 2, "a": 2, "y": 2, "v": 2},
    # Pronoun + aux verb → similar to ("i", "have") patterns.
    ("you", "have"): {"n": 6, "s": 5, "b": 5, "d": 4, "h": 4, "t": 3, "l": 3,
                      "m": 3, "f": 3, "a": 3, "g": 3, "p": 2, "c": 2, "r": 2,
                      "w": 2, "o": 2},
    ("we", "have"): {"n": 6, "s": 5, "b": 5, "d": 4, "h": 4, "t": 3, "l": 3,
                     "m": 3, "f": 3, "a": 3, "g": 3, "p": 2, "c": 2, "r": 2,
                     "w": 2, "o": 2},
    ("they", "have"): {"n": 5, "s": 4, "b": 4, "d": 3, "h": 3, "t": 3, "l": 3,
                       "m": 3, "f": 3, "a": 3, "g": 2, "p": 2, "c": 2, "r": 2,
                       "w": 2, "o": 2},
    ("he", "is"): {"n": 5, "s": 4, "t": 4, "a": 3, "m": 3, "b": 3, "i": 3,
                   "p": 3, "y": 3, "o": 2, "h": 2, "g": 2, "f": 2, "l": 2,
                   "r": 2, "c": 2, "d": 2, "w": 2},
    ("she", "is"): {"n": 5, "s": 4, "t": 4, "a": 3, "m": 3, "b": 3, "i": 3,
                    "p": 3, "y": 3, "o": 2, "h": 2, "g": 2, "f": 2, "l": 2,
                    "r": 2, "c": 2, "d": 2, "w": 2},
    ("he", "was"): {"n": 4, "s": 3, "t": 3, "a": 3, "m": 3, "b": 3, "i": 3,
                    "p": 3, "y": 2, "o": 2, "h": 2, "g": 2, "f": 2, "l": 2,
                    "r": 2, "c": 2, "d": 2, "w": 2},
    # Preposition/prep-like + pronoun → verb
    ("of", "your"): {"l": 4, "h": 4, "s": 3, "f": 3, "m": 3, "b": 3, "g": 3,
                     "d": 3, "p": 2, "t": 2, "c": 2, "n": 2, "e": 2, "o": 2,
                     "a": 2, "r": 2, "v": 2, "k": 2, "i": 2, "u": 2, "w": 2,
                     "y": 2},
    ("of", "my"): {"l": 4, "h": 4, "s": 3, "f": 3, "m": 3, "b": 3, "g": 3,
                   "d": 3, "p": 2, "t": 2, "c": 2, "n": 2, "e": 2, "o": 2,
                   "a": 2, "r": 2, "v": 2, "k": 2, "i": 2, "u": 2, "w": 2,
                   "y": 2},
    ("in", "my"): {"l": 4, "h": 4, "s": 3, "f": 3, "m": 3, "b": 3, "g": 3,
                   "d": 3, "p": 2, "t": 2, "c": 2, "n": 2, "e": 2, "o": 2,
                   "a": 2, "r": 2, "v": 2, "k": 2, "i": 2, "u": 2, "w": 2,
                   "y": 2},
    # Conjunction + "to"
    ("and", "to"): {"t": 5, "b": 4, "s": 4, "h": 4, "m": 4, "d": 4, "g": 4,
                    "l": 3, "c": 3, "f": 3, "p": 3, "r": 3, "k": 3, "w": 3,
                    "n": 3, "a": 2, "e": 2, "i": 2, "o": 2, "y": 2, "v": 2},
    ("but", "to"): {"t": 5, "b": 4, "s": 4, "h": 4, "m": 4, "d": 4, "g": 4,
                    "l": 3, "c": 3, "f": 3, "p": 3, "r": 3, "k": 3, "w": 3,
                    "n": 3, "a": 2, "e": 2, "i": 2, "o": 2, "y": 2, "v": 2},
    # "like a"
    ("like", "a"): {"m": 4, "l": 4, "k": 4, "f": 4, "w": 3, "g": 3, "s": 4,
                    "n": 3, "b": 4, "c": 3, "d": 3, "p": 3, "t": 3, "h": 3,
                    "r": 3, "y": 2, "v": 2, "q": 1},
    # "if you"
    ("if", "you"): {"w": 4, "h": 4, "d": 4, "b": 3, "m": 3, "c": 3, "s": 3,
                    "l": 3, "k": 3, "g": 3, "a": 3, "p": 3, "t": 3, "r": 2,
                    "n": 2, "f": 2, "e": 2, "o": 2, "i": 2, "y": 2, "v": 2},
    ("if", "i"): {"a": 4, "h": 4, "w": 3, "m": 3, "s": 3, "k": 3, "c": 3,
                  "d": 3, "f": 2, "l": 2, "n": 2, "p": 2, "r": 2, "t": 2,
                  "b": 2, "g": 2},
    # "so much", "so great" etc. — nothing specific.
}


_GLOBAL_SCALE = 1.0


def _build_vectors() -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = {}
    for key, nexts in _PHRASE_NEXT.items():
        if not nexts:
            continue
        vec = [0.0] * VOCAB_SIZE
        listed = set(nexts.keys())
        # Gentle negative on unlisted lowercase letters.
        for ch in "abcdefghijklmnopqrstuvwxyz":
            if ch not in listed and ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] = -0.25 * _GLOBAL_SCALE
        total = sum(nexts.values())
        for ch, w in nexts.items():
            if ch not in VOCAB_INDEX:
                continue
            frac = w / total
            bias = _GLOBAL_SCALE * math.log((frac + 0.02) / 0.05)
            vec[VOCAB_INDEX[ch]] = bias
            up = ch.upper()
            if up in VOCAB_INDEX:
                vec[VOCAB_INDEX[up]] = bias * 0.6
        out[key] = vec
    return out


_PHRASE_BIAS: dict[tuple[str, str], list[float]] = _build_vectors()


def phrase_bigram_bias(prev_word: str, last_word: str) -> list[float] | None:
    """Return bias vector for the next-word first-letter given the
    previous two completed words, or None if no entry."""
    if not prev_word or not last_word:
        return None
    key = (prev_word, last_word)
    return _PHRASE_BIAS.get(key)
