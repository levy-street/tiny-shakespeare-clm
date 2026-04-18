"""Word-end bigram plausibility bias.

Given `word_buffer` with letter_run_len >= 3, examine the last two
letters and decide whether they form a plausible English word-ending
bigram (boost " ") or an implausible one (push for continuation).

This is a targeted, fine-grained terminator signal that complements
`trie_recovery_bias` (which escalates on letters_past_complete / off)
by looking at the ACTUAL suffix shape. `trie_recovery` says "you've
drifted N letters — close soon"; this layer says "your last two
letters make a real English word-end — close NOW (or not-now)".

Fires only off the word_trie (where trie_recovery is active too),
because on-trie the word_trie itself encodes legitimate extensions.

All weights are from prior knowledge of English orthography; no
corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# Common English word-ending bigrams. Weights roughly reflect
# closure-confidence: higher = the word is more confidently done
# if we stop here. Many-word endings overlap across values.
_COMMON_END_BIGRAMS: dict[str, float] = {
    # Vowel + common consonant (very common word endings)
    "ed": 1.2, "er": 1.2, "es": 1.0, "en": 0.9, "in": 0.6,
    "on": 0.9, "an": 0.7, "un": 0.4, "or": 0.9, "ar": 0.7,
    "ir": 0.4, "ur": 0.5, "ad": 0.5, "id": 0.5, "od": 0.5,
    "ud": 0.4, "ap": 0.4, "op": 0.4, "ip": 0.4, "up": 0.3,
    "at": 0.7, "it": 0.8, "ot": 0.5, "ut": 0.5, "et": 0.8,
    "am": 0.5, "em": 0.5, "im": 0.5, "om": 0.5, "um": 0.5,
    "ag": 0.4, "eg": 0.2, "ig": 0.2, "og": 0.4, "ug": 0.3,
    "ab": 0.3, "eb": 0.2, "ib": 0.2, "ob": 0.3, "ub": 0.2,
    "ac": 0.3, "ic": 0.7, "oc": 0.3, "uc": 0.2,
    "af": 0.2, "ef": 0.3, "if": 0.4, "of": 0.8, "uf": 0.2,
    "ah": 0.4, "eh": 0.2, "oh": 0.4, "ow": 0.8, "aw": 0.6, "ew": 0.6,
    "ay": 0.9, "ey": 0.8, "oy": 0.5, "uy": 0.2,
    "ax": 0.3, "ex": 0.4, "ix": 0.3, "ox": 0.3,
    "az": 0.2, "ez": 0.2,
    # -e endings (silent e, very common)
    "me": 0.8, "be": 0.5, "we": 0.3, "he": 0.8, "re": 1.0,
    "se": 0.8, "te": 0.8, "ne": 0.8, "ce": 0.9, "de": 0.8,
    "ge": 0.7, "le": 1.2, "ke": 0.6, "ve": 0.8, "pe": 0.5,
    "ze": 0.4, "ie": 0.7, "ae": 0.3, "oe": 0.3, "ue": 0.5,
    # -y endings
    "ly": 1.2, "ry": 0.9, "ty": 0.9, "ny": 0.7, "my": 0.6,
    "by": 0.5, "sy": 0.5, "ky": 0.3, "gy": 0.5, "dy": 0.6,
    "hy": 0.4, "py": 0.3, "wy": 0.2, "vy": 0.3, "cy": 0.4,
    "fy": 0.2, "ty": 0.9,
    # Consonant clusters that end words
    "th": 1.0, "sh": 0.7, "ch": 0.8, "gh": 0.5, "ck": 0.6,
    "ng": 1.0, "nk": 0.6, "nd": 1.0, "nt": 1.0, "st": 1.0,
    "rd": 0.9, "rt": 0.9, "ld": 0.9, "lt": 0.7, "ns": 0.7,
    "ts": 0.6, "ds": 0.6, "ks": 0.5, "ms": 0.5, "rs": 0.8,
    "ls": 0.5, "ps": 0.4, "ft": 0.5, "pt": 0.4, "ct": 0.5,
    "ht": 0.7, "xt": 0.3, "mp": 0.4, "mb": 0.3, "nc": 0.3,
    "lf": 0.3, "rm": 0.4, "rn": 0.5, "rk": 0.3, "rl": 0.3,
    "rp": 0.3, "sp": 0.3, "sk": 0.3, "sm": 0.3, "lk": 0.3,
    # Double letters
    "ss": 0.8, "ll": 1.0, "tt": 0.3, "ff": 0.5, "dd": 0.3,
    "nn": 0.3, "mm": 0.3, "zz": 0.3, "gg": 0.3, "pp": 0.2,
    "rr": 0.2, "bb": 0.2, "cc": 0.2,
    # -io / -o / -a / -i endings (loan words, names)
    "io": 0.5, "ro": 0.4, "to": 0.5, "no": 0.4, "co": 0.3,
    "do": 0.4, "so": 0.5, "go": 0.3, "po": 0.2, "mo": 0.3,
    "lo": 0.4, "ho": 0.3,
    "ia": 0.4, "ea": 0.3, "oa": 0.2, "ua": 0.2,
    "ki": 0.2, "li": 0.3, "ri": 0.3, "ni": 0.3, "mi": 0.3,
    "si": 0.3, "ti": 0.3, "di": 0.2, "bi": 0.2, "ci": 0.2,
}

# Bigrams that are VERY unlikely as word endings — they mark the
# middle of a word, not the end.
_RARE_END_BIGRAMS: dict[str, float] = {
    # Onset clusters (appear at word-START, not word-END)
    "tr": -1.0, "pr": -1.0, "br": -1.0, "cr": -0.9, "dr": -0.9,
    "fr": -1.0, "gr": -0.9, "wr": -0.8,
    "bl": -0.9, "cl": -0.9, "fl": -0.9, "gl": -0.9, "pl": -0.9,
    "sl": -0.8, "sp": -0.5, "sc": -0.6, "sk": -0.3,
    "sm": -0.5, "sn": -0.8, "sw": -0.9, "tw": -0.8,
    "str": -1.0,  # 2-letter lookup only; str isn't in this table anyway
    # Vowel + rare final
    "aj": -0.9, "ej": -0.9, "ij": -0.9, "oj": -0.9, "uj": -0.9,
    "aq": -1.0, "eq": -1.0, "iq": -1.0, "oq": -1.0, "uq": -1.0,
    # Consonant + consonant mid-word patterns (typically want a vowel next)
    "kn": -0.6, "pn": -0.6, "gn": -0.4, "bn": -0.7, "dn": -0.7,
    "fn": -0.8, "hn": -0.7, "mn": -0.5, "tn": -0.7, "vn": -0.8,
    "wn": -0.2,  # -own/-awn is attested, so mild
    "lr": -0.8, "mr": -0.9, "nr": -0.9, "pr": -0.9, "zr": -0.9,
    "lm": -0.3,  # elm/calm — attested, very mild
    # Vowel-pairs that rarely end
    "uo": -0.6, "ui": -0.2, "iu": -0.5, "yu": -0.8, "yi": -0.8,
    "yo": -0.3,
    # Consonant + h that rarely ends (bh/dh/fh/jh/kh/lh/mh/nh/ph/rh/wh/zh)
    "bh": -0.9, "dh": -0.9, "fh": -0.9, "jh": -0.9, "kh": -0.7,
    "lh": -0.9, "mh": -0.9, "nh": -0.9, "ph": -0.4, "rh": -0.6,
    "wh": -0.9, "zh": -0.8,
    # Rare consonant + vowel starters (common at word-start, not end)
    "ju": -0.6, "qu": -1.0, "qa": -1.0, "qe": -1.0, "qi": -1.0, "qo": -1.0,
    "xa": -0.7, "xe": -0.6, "xi": -0.7, "xo": -0.8, "xu": -0.9,
    "za": -0.6, "ze": -0.4, "zi": -0.7, "zo": -0.7, "zu": -0.8,
    # Three-consonant mid-run (just bigrams we'll see)
    "fr": -0.9, "tf": -0.9, "vf": -0.9, "df": -0.9, "bf": -0.9,
    "pf": -0.9, "lf": -0.3,  # self/elf attested
}


def word_end_bigram_bias(
    word_buffer: str,
    letter_run_len: int,
    on_word_trie: bool,
    letters_off_trie: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a bias vector favoring " " if the last-2-letter suffix
    forms a common word-end bigram, or penalizing " " if it forms a
    rare end bigram (the word is clearly mid-growth).

    Active only when:
      - We're outside speaker-label territory (state 0)
      - The buffer has at least 4 letters
      - We're off the word_trie OR have drifted >= 2 off-trie letters
    """
    if speaker_label_state != 0:
        return None
    if letter_run_len < 3:
        return None
    if len(word_buffer) < 2:
        return None
    # Only fire at off-trie drift.
    if letters_off_trie < 1:
        return None

    suffix = word_buffer[-2:].lower()
    # Consider only alphabetic suffixes (skip apostrophe-bearing tails).
    if not (suffix.isalpha() and len(suffix) == 2):
        return None

    common = _COMMON_END_BIGRAMS.get(suffix)
    rare = _RARE_END_BIGRAMS.get(suffix)
    if common is None and rare is None:
        return None

    # Scale by depth into the off-trie drift.
    depth = max(letters_off_trie, min(letter_run_len - 3, 6))
    depth_scale = min(0.2 + 0.25 * depth, 2.0)

    vec = [0.0] * VOCAB_SIZE

    # Only consume the RARE signal — penalize word-terminators when
    # the suffix is clearly mid-word orthography (e.g., "tr", "bl",
    # "qu", "xa"). The COMMON side is risky because legitimate off-
    # trie words (names, archaic forms, compounds) also end in those
    # same bigrams, and boosting space then is training-negative.
    if rare is not None:
        pen = rare * depth_scale  # negative
        if " " in VOCAB_INDEX:
            vec[VOCAB_INDEX[" "]] += pen
        if "," in VOCAB_INDEX:
            vec[VOCAB_INDEX[","]] += pen * 0.5
        if "." in VOCAB_INDEX:
            vec[VOCAB_INDEX["."]] += pen * 0.35

    return vec
