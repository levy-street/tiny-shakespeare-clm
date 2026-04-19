"""Word-onset consonant-cluster legality.

English word onsets follow tight phonotactic rules: at most three
consonants, and only certain clusters. This predictor, at positions
INSIDE the onset (letter_run_len in {1, 2}), suppresses next-letter
choices that would produce an illegal onset cluster given the letters
already emitted.

Coverage:

  letter_run_len == 1  (first letter is a consonant, next is letter 2)
    * After "b": legal continuations are vowels + l/r/y + '
      (e.g., be, by, but, black, brown, blade; "bn"/"bm"/"bk"/"bt"
      etc. are illegal).
    * After "c": vowels + h/l/r/y; illegal "cn"/"cm"/"cd"/"cs".
    * After "d": vowels + r/w/y/' ; illegal "dl"/"dn"/"dm"/"db".
    * After "f": vowels + l/r/y/'; illegal "fn"/"fm"/"fd".
    * After "g": vowels + h/l/n (gnomon)/r/y; illegal "gd"/"gm"/"gb".
    * After "h": vowels + y/' only; everything else illegal.
    * After "k": vowels + n (knight)/y/r/'; illegal "kd"/"km"/"kb".
    * After "l": vowels + y/' only; illegal "lm"/"ln"/"lr".
    * After "m": vowels + n (mnemonic)/y/'; illegal "md"/"mt"/"ml".
    * After "n": vowels + y/'; illegal "nm"/"nl"/"nr".
    * After "p": vowels + h/l/n (pneumonia)/r/s (psalm)/y/'; illegal "pd"/"pb".
    * After "q": u only (queue, queen, quick).
    * After "r": vowels + y/h (rhyme/rheum)/'; illegal "rs"/"rn"/"rm"/"rl".
    * After "s": vowels + c/h/k/l/m (smile)/n (snow)/p/q/t/w/y/'; broad.
    * After "t": vowels + h/r/w/y/'; illegal "tb"/"tn"/"tm"/"td"/"tf".
    * After "v": vowels + y/' only.
    * After "w": vowels + h/r/y/'; illegal "wn"/"wm".
    * After "x": vowels only (rare — xylophone, Xerxes).
    * After "y": vowels + '; illegal cluster afterwards.
    * After "z": vowels only.
    * After "j": vowels only.

  letter_run_len == 2  (we have 2 letters; both consonants → very
    restricted 3rd-letter set)
    * "str-": vowels only (strong, straw, strange).
    * "spl-": vowels only (splash, splinter).
    * "spr-": vowels only.
    * "scr-": vowels only.
    * "thr-": vowels only.
    * "sch-": vowels only (scheme).
    * "shr-": vowels only.
    * "sph-": vowels only.
    * "bl-", "br-", "cl-", "cr-", "dr-", "fl-", "fr-", "gl-", "gr-",
      "pl-", "pr-", "sl-", "sm-", "sn-", "sp-", "st-", "sw-", "tr-",
      "tw-", "th-", "sh-", "ch-", "wh-", "gh-", "ph-", "qu-": vowel
      required next (few exceptions like "chr-", "phr-", "sch-" are
      borderline — we allow r/l/y/h after those).

Gates:
  * word-start territory — `letter_run_len in {1, 2}` and `word_buffer`
    non-empty (ASCII letters).
  * Outside speaker-label territory.

Penalty mode: ILLEGAL next letters receive a strong negative log-bias
(-3.0 to -5.0); permitted continuations get 0.0. No positive boosts
(other layers handle those).

No corpus statistics — all rules come from English phonotactics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

_VOWELS = frozenset("aeiouy")  # includes 'y' as post-consonant vowel

# After a single consonant, what SECOND letters are legal?
# Encoded as the permitted set (everything else penalized).
# Vowels + register-specific consonants.
_POST_CONS_1: dict[str, frozenset[str]] = {
    "b": frozenset("aeiouylr'"),
    "c": frozenset("aeiouyhlr'"),
    "d": frozenset("aeiouyrw'"),
    "f": frozenset("aeiouylr'"),
    "g": frozenset("aeiouyhlnr'"),
    "h": frozenset("aeiouy'"),
    "j": frozenset("aeiouy"),
    "k": frozenset("aeiouynr'"),
    "l": frozenset("aeiouy'"),
    "m": frozenset("aeiouyn'"),
    "n": frozenset("aeiouy'"),
    "p": frozenset("aeiouyhlnrsy'"),
    "q": frozenset("u"),
    "r": frozenset("aeiouyh'"),
    "s": frozenset("aeiouychklmnpqtwy'"),
    "t": frozenset("aeiouyhrwy'"),
    "v": frozenset("aeiouy'"),
    "w": frozenset("aeiouyhry'"),
    "x": frozenset("aeiouy"),
    "y": frozenset("aeiouy'"),
    "z": frozenset("aeiouy"),
}

# Common word-starts where first letter is a VOWEL — second letter is
# already well-covered by startbigram; skip here (return None).

# After a legal 2-consonant prefix, what THIRD letters are legal?
# (Includes the "str-/spr-/scr-/..." triple-cluster continuations.)
_POST_CONS_2: dict[str, frozenset[str]] = {
    "bl": frozenset("aeiouy'"),
    "br": frozenset("aeiouy'"),
    "cl": frozenset("aeiouy'"),
    "cr": frozenset("aeiouy'"),
    "dr": frozenset("aeiouy'"),
    "fl": frozenset("aeiouy'"),
    "fr": frozenset("aeiouy'"),
    "gl": frozenset("aeiouy'"),
    "gr": frozenset("aeiouy'"),
    "pl": frozenset("aeiouy'"),
    "pr": frozenset("aeiouy'"),
    "sl": frozenset("aeiouy'"),
    "sm": frozenset("aeiouy'"),
    "sn": frozenset("aeiouy'"),
    "sp": frozenset("aeiouylrh'"),   # sp, spl, spr, sph
    "st": frozenset("aeiouyrh'"),    # st, str
    "sw": frozenset("aeiouy'"),
    "sc": frozenset("aeiouyhkrly'"), # sc, sch, scr, scl
    "sh": frozenset("aeiouyr'"),     # sh, shr
    "ch": frozenset("aeiouyr'"),     # ch, chr (archaic)
    "wh": frozenset("aeiouy'"),
    "gh": frozenset("aeiouy'"),
    "ph": frozenset("aeiouylr'"),    # ph, phr
    "th": frozenset("aeiouyrw'"),    # th, thr, thw
    "tr": frozenset("aeiouy'"),
    "tw": frozenset("aeiouy'"),
    "qu": frozenset("aeiouy'"),
    "kn": frozenset("aeiouy'"),
    "wr": frozenset("aeiouy'"),
    "pn": frozenset("aeiouy"),
    "ps": frozenset("aeiouy"),
    "gn": frozenset("aeiouy'"),
    "mn": frozenset("aeiouy"),
    "rh": frozenset("aeiouy'"),
}


# Illegal-letter penalty strength.
_PENALTY = 1.0


def _make_vec(permitted: frozenset[str]) -> list[float]:
    """Penalize every lowercase letter NOT in the permitted set.
    Uppercase and non-letter chars untouched (those are handled
    by the word-form state, and shouldn't appear mid-word anyway)."""
    vec = [0.0] * VOCAB_SIZE
    for ch in "abcdefghijklmnopqrstuvwxyz":
        if ch not in permitted:
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] = -_PENALTY
    return vec


_POST_CONS_1_VEC: dict[str, list[float]] = {
    k: _make_vec(v) for k, v in _POST_CONS_1.items()
}
_POST_CONS_2_VEC: dict[str, list[float]] = {
    k: _make_vec(v) for k, v in _POST_CONS_2.items()
}


def onset_cluster_bias(
    word_buffer: str,
    letter_run_len: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if not word_buffer:
        return None

    if letter_run_len == 1:
        # word_buffer has exactly 1 character (the first).
        first = word_buffer[-1].lower()
        if first in _VOWELS:
            return None
        vec = _POST_CONS_1_VEC.get(first)
        return vec

    if letter_run_len == 2:
        # word_buffer's last 2 chars form a consonant onset pair
        # if both are consonants. Otherwise skip.
        pair = word_buffer[-2:].lower()
        if len(pair) < 2:
            return None
        if pair[0] in _VOWELS or pair[1] in _VOWELS:
            return None
        vec = _POST_CONS_2_VEC.get(pair)
        return vec

    return None
