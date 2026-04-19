"""Off-trie departure-position predict consumer.

Reads `state.offtrie_depart_pos` (see pipeline/flow.py) — the
letter_run_len at which the current word FIRST left the word-trie.
This is richer than the existing letters_off_trie axis (drift
distance) or has_seen_complete (did we ever hit a complete form).
It tells us WHERE the word broke from known English.

Three regimes map to three qualitatively different pathologies:

  * Early departure (depart_pos in {1, 2}): the first or second letter
    already put us off all known word-prefixes. These are gibberish
    tokens from the start — like "zq-" or "dg-" or any sequence that
    has no English word starting with it. Strong terminator boost;
    strong gibberish-letter penalty.

  * Mid departure (depart_pos in {3, 4}): we had a solid 3-4 letter
    prefix and stepped off. Could be a real-word inflection that the
    trie doesn't cover (e.g., an unusual morphological form), OR the
    start of gibberish. Moderate terminator boost; moderate penalty
    on high-entropy continuations (j/q/x/z/etc).

  * Late departure (depart_pos >= 5): the trie knew a 5+ letter prefix
    but none of its completions — so we're now extending a real-looking
    prefix into invented morphology. Real Shakespeare has few such
    words. This is the "outflown / arthure / resistan" case — heavy
    word-end bias, heavy gibberish-letter penalty.

Bias is only applied once the word has also accumulated some letters
AFTER the departure (so we don't over-penalize the single letter that
caused the departure — that letter is already sampled). Specifically
we require letters_off_trie >= 1 (we're AT or PAST the departure).

All weights come from prior knowledge of English phonotactics / word
structure — no corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Letters to boost (word-enders and common word-ending letters).
_TERMINATORS: dict[str, float] = {
    " ": 1.0,
    ",": 0.55,
    ".": 0.40,
    ";": 0.30,
    ":": 0.20,
    "!": 0.30,
    "?": 0.30,
    "\n": 0.35,
}

# Common word-ending letters — if we MUST extend, at least pick a
# plausible ending letter so a further word-end char can close it
# next step.
_END_LETTERS: dict[str, float] = {
    "e": 1.0, "s": 0.95, "d": 0.85, "t": 0.80, "n": 0.75,
    "h": 0.65, "r": 0.70, "y": 0.65, "l": 0.55, "g": 0.40,
}

# Rare / gibberish-extending letters. Strongly penalized when we're
# past the word-trie.
_GIBBERISH: dict[str, float] = {
    "j": -1.0, "q": -1.0, "x": -1.0, "z": -1.0,
    "v": -0.40, "w": -0.25, "k": -0.30, "b": -0.30,
    "c": -0.25, "f": -0.30, "p": -0.25, "m": -0.20,
}


def _build_vec(depart_pos: int, off_len: int) -> list[float] | None:
    """depart_pos = where we left the trie; off_len = how far past.
    Returns None if bias is zero."""
    if off_len < 1:
        return None
    if depart_pos <= 0:
        return None

    # Only fire in clearly-pathological regimes. Most off-trie runs are
    # legitimate morphology that the trie doesn't cover (archaic forms,
    # uncommon inflections), so we require BOTH a specific departure
    # pattern AND real drift before nudging.
    #
    # Early (1-2): gibberish from the start. Fire only when drifted 2+.
    # Mid (3-4): ambiguous. Don't fire — trie_recovery handles this.
    # Late (5+): extending a real prefix into invented morphology.
    #            Fire only when drifted 2+, otherwise we'd penalize
    #            legit inflections like reading->readings.
    if depart_pos <= 2:
        if off_len < 2:
            return None
        term_scale = 0.03
        end_scale = 0.08
        gib_scale = 0.25
    elif depart_pos <= 4:
        return None
    else:  # depart_pos >= 5
        if off_len < 2:
            return None
        term_scale = 0.04
        end_scale = 0.12
        gib_scale = 0.30

    # Escalate with off_len.
    if off_len == 2:
        drift = 0.50
    elif off_len == 3:
        drift = 0.80
    elif off_len == 4:
        drift = 1.10
    else:
        drift = 1.40

    term_scale *= drift
    end_scale *= drift
    gib_scale *= drift

    vec = [0.0] * VOCAB_SIZE
    for ch, w in _TERMINATORS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += term_scale * w
    for ch, w in _END_LETTERS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += end_scale * w
    for ch, w in _GIBBERISH.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += gib_scale * w  # w is negative
    return vec


# Precompute: depart_pos in [0..10], off_len in [0..8].
_VECS: list[list[list[float] | None]] = [
    [_build_vec(d, o) for o in range(9)]
    for d in range(11)
]


def offtrie_depart_bias(
    offtrie_depart_pos: int,
    letters_off_trie: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Escalate word-end / gibberish-penalty based on HOW the word
    departed the trie. Fires only outside speaker-label territory.
    Returns None when not applicable."""
    if speaker_label_state != 0:
        return None
    if offtrie_depart_pos <= 0:
        return None
    if letters_off_trie < 1:
        return None
    d = min(offtrie_depart_pos, 10)
    o = min(letters_off_trie, 8)
    return _VECS[d][o]
