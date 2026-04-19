"""Morphological-suffix completion bias — off-trie words.

When the word_buffer has drifted off the word_trie but its tail matches
the early part of a productive English / Early-Modern-English suffix,
bias the next letter toward the letter that COMPLETES that suffix, and
(once the suffix is complete) softly bias toward the word-terminator.

This layer targets a specific pain point visible in samples: off-trie
words extending past plausible endings into nonsense
("oophey", "saniln", "Blonnte"). Many of the off-trie words that
actually do occur in Shakespeare are archaic verb forms and regular
morphological derivations — `-eth`, `-est`, `-edst`, `-ing`, `-ly`,
`-ness`, `-ment`, `-ous`, `-ful`, `-less`, `-able`, `-tion`. By
short-circuiting their completions, we give the model credit for
those real forms and make off-trie extension more likely to land on a
morphologically sensible word.

Fires only:
  - outside speaker-label territory (state 0)
  - when the buffer is off the word_trie (trie-covered words are
    handled by word_trie_bias itself)
  - when letter_run_len >= 3 (short prefixes are too ambiguous)

All weights are from prior knowledge of English morphology; no corpus
statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# (tail match, next-letter, weight) tuples. Tail is checked as a suffix
# of word_buffer (case-insensitive). Weight is the log-bias bump added
# to the VOCAB slot for `next_letter`.
#
# The table is structured so that at each successive letter in a
# productive suffix there's a progression entry: e.g. "in" → "g",
# then "ing" → " ". The space entries serve as "this suffix is
# complete — let the word end".
_PROGRESSIONS: tuple[tuple[str, str, float], ...] = (
    # -ing (present participle / gerund)
    ("in", "g", 0.55),

    # -eth (3rd singular archaic: "speaketh", "hath", "doth")
    ("et", "h", 0.70),
    # -est (2nd singular archaic: "speakest", "knowest") and superlative
    ("es", "t", 0.45),

    # -edst / -edst — rare archaic 2sg past (didst, couldst -style);
    # do NOT force an unusual transition.
    # -ed (past tense / past participle)
    # "e" → "d" is too frequent a mid-word step in general; skip.

    # -ly (adverb)
    # bare "l" → "y" is too ambiguous; require "el" or "al".
    ("el", "y", 0.35),
    ("al", "y", 0.25),
    ("il", "y", 0.25),
    ("ul", "y", 0.20),

    # -ness (noun from adj: "kindness", "wildness", "darkness")
    ("nes", "s", 0.80),
    ("ne", "s", 0.20),  # weaker; "ne" can also be a stem

    # -ment (noun from verb: "judgement", "argument", "commandment")
    ("men", "t", 0.55),

    # -ous (adjective: "grievous", "piteous", "glorious")
    ("ou", "s", 0.25),  # weak; "ou" → "s" competes with "ou" → "r"/"n"/"g"
    ("iou", "s", 0.70),
    ("eou", "s", 0.60),

    # -ful (adjective: "fearful", "beautiful", "awful")
    ("fu", "l", 0.60),

    # -less (adjective: "fearless", "bottomless")
    ("les", "s", 0.55),

    # -able / -ible (adjective: "terrible", "lovable")
    ("abl", "e", 0.70),
    ("ibl", "e", 0.70),

    # -tion / -sion (noun)
    ("ati", "o", 0.55),
    ("tio", "n", 0.90),
    ("sio", "n", 0.85),
    ("atio", "n", 0.95),

    # -ance / -ence (noun)
    ("anc", "e", 0.65),
    ("enc", "e", 0.60),

    # -ward (direction)
    ("war", "d", 0.35),

    # -ward / -wards
    # -hood (noun: "knighthood", "manhood")
    ("hoo", "d", 0.45),
)

# Suffix completions — once the buffer ends in one of these, the word
# is semantically complete and can terminate. Boost space modestly.
_COMPLETE_SUFFIXES: tuple[tuple[str, float], ...] = (
    ("ing", 0.40),
    ("eth", 0.60),
    ("est", 0.30),   # also comparative; can extend
    ("ly", 0.35),
    ("ness", 0.70),
    ("ment", 0.60),
    ("ous", 0.50),
    ("ful", 0.50),
    ("less", 0.55),
    ("able", 0.55),
    ("ible", 0.55),
    ("tion", 0.70),
    ("sion", 0.70),
    ("ance", 0.55),
    ("ence", 0.55),
    ("ward", 0.35),
    ("hood", 0.45),
)


def suffix_completion_bias(
    word_buffer: str,
    letter_run_len: int,
    on_word_trie: bool,
    letters_off_trie: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a bias vector nudging toward suffix-progression letters.

    Active only off-trie, outside speaker labels, letter_run_len >= 3.
    """
    if speaker_label_state != 0:
        return None
    if on_word_trie:
        return None
    if letters_off_trie < 1:
        return None
    if letter_run_len < 3:
        return None
    if not word_buffer:
        return None

    wb = word_buffer.lower()
    vec = [0.0] * VOCAB_SIZE
    nonzero = False

    # Progression entries: tail-match then next-letter bias.
    for tail, nxt, w in _PROGRESSIONS:
        if wb.endswith(tail):
            idx = VOCAB_INDEX.get(nxt)
            if idx is not None:
                vec[idx] += w
                nonzero = True

    # Completion entries: suffix already complete, gentle terminator.
    for suf, w in _COMPLETE_SUFFIXES:
        if wb.endswith(suf):
            sp = VOCAB_INDEX.get(" ")
            if sp is not None:
                vec[sp] += w
                nonzero = True
            cm = VOCAB_INDEX.get(",")
            if cm is not None:
                vec[cm] += w * 0.40
                nonzero = True
            pd = VOCAB_INDEX.get(".")
            if pd is not None:
                vec[pd] += w * 0.25
                nonzero = True

    if not nonzero:
        return None
    return vec
