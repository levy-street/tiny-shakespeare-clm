"""Monosyllabic-run momentum predict consumer.

Reads `state.monosyllabic_run` (see pipeline/flow.py) and pushes the
current word to end SOONER — biasing space/punctuation over further
letters — when we're deep in a monosyllabic-run cadence.

The field counts consecutive 1-syllable completed words. When the
run is sustained (>= 4), we're inside one of Shakespeare's most
distinctive textural modes — the percussive stacked-monosyllable
cadence of lines like "To be, or not to be, that is the question",
"Words, words, words", "Out, out, brief candle", or "This above
all: to thine own self be true".

In this mode, the current word is strongly likely to ALSO be
monosyllabic — meaning it will end at letter position 2-4 rather
than run out to position 6+. None of the existing layers read
monosyllabic_run, so this is genuinely new information.

The bias only fires at already-long mid-word positions (letter
position >= 4) — i.e., exactly the positions where the word is at
risk of BECOMING polysyllabic. There we give a gentle boost to the
word-ending characters (space / punctuation) so the word closes.

Scale is gentle because word-end already has many competing signals
(word_trie, cadence, line_break). Fires only outside speaker-label
territory and skips when we're firmly on-trie at a known short word.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Characters that END words — spaces, punctuation. These get the
# positive word-close bias when we're deep in a mono run and the
# current word is already at letter position 4+ (at risk of becoming
# polysyllabic).
_WORD_ENDER_WEIGHTS: dict[str, float] = {
    " ": 1.4,
    ",": 0.9,
    ".": 0.5,
    "?": 0.4,
    "!": 0.4,
    ";": 0.5,
    ":": 0.3,
    "\n": 0.4,
}


def _build_vec(run: int, pos: int) -> list[float]:
    """Build bias vector for run length `run` and letter position
    `pos` within the current word. pos is letter_run_len (1-based).
    """
    vec = [0.0] * VOCAB_SIZE
    if run < 4 or pos < 4:
        return vec

    # Position factor: stronger as the word grows longer. At position
    # 4 we're at risk of polysyllable; at position 7+ we very likely
    # already AT a polysyllable, so push harder.
    if pos == 4:
        pos_scale = 0.2
    elif pos == 5:
        pos_scale = 0.4
    elif pos == 6:
        pos_scale = 0.6
    else:
        pos_scale = 0.8

    # Run factor: scales with momentum magnitude. Kept small —
    # word_trie and line_break already carry most of the word-end
    # signal; this is a flow-tier nudge on top.
    if run < 6:
        run_scale = 0.02
    elif run < 9:
        run_scale = 0.04
    else:
        run_scale = 0.06

    scale = pos_scale * run_scale

    for ch, w in _WORD_ENDER_WEIGHTS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += scale * w

    return vec


# Precompute vectors indexed by (run, pos) with run in [0..12], pos in [0..10].
_VECS: list[list[list[float] | None]] = [
    [_build_vec(r, p) if (r >= 4 and p >= 4) else None for p in range(11)]
    for r in range(13)
]


def rhythm_wordend_bias(
    monosyllabic_run: int,
    letter_run_len: int,
    speaker_label_state: int,
    on_word_trie: bool,
) -> list[float] | None:
    """Nudge toward word-end (space/punct) when deep in a monosyllabic
    run. Fires at mid-word letter positions >= 4 to push the current
    word to end before becoming polysyllabic. Skips speaker-label
    territory; skips while still firmly on-trie at short-word positions
    (the trie knows better). Returns None when not applicable.
    """
    if speaker_label_state != 0:
        return None
    if monosyllabic_run < 4:
        return None
    if letter_run_len < 3:
        return None
    r = min(monosyllabic_run, 12)
    p = min(letter_run_len, 10)
    return _VECS[r][p] if p >= 4 else _VECS[r][4]
