"""Coda-plausibility bias — use the syllable-coda FSM signal.

Reads `state.post_vowel_cluster` — the literal lowercase consonant
cluster emitted since the most recent vowel within the current word.

This cluster is the sequence of consonants sitting in the CODA of the
current syllable (or the coda of the prior syllable + onset of a new
one, ambiguously). Its SHAPE tells us whether closing the word here
would land on a legal English word-final coda, or whether we'd leave
the word mid-cluster in phonotactically unreachable territory.

Three cases:

  1. Cluster is empty (just emitted a vowel, or no vowel yet in word):
     No signal — ordinary bias layers govern.

  2. Cluster IS a complete legal English word-final coda (matches the
     trailing consonant cluster of some real word):
        → Boost terminators moderately. The word CAN legitimately
          end here; we shouldn't underbias closure.

  3. Cluster is a strict prefix of some legal coda but NOT itself legal:
        → Neutral. Still extending toward a plausible endpoint.

  4. Cluster is neither legal nor a legal prefix (phonotactically
     impossible as a word-final coda):
        → Strong terminator push and letter penalty. We're extending
          into a dead-end; close now or at most with a vowel.

The legal-coda set is derived from the hand-curated word_trie — the
same vocabulary used by `word_ending_shape`. Each word's trailing
consonant cluster (letters after its last vowel; `y` counted as vowel
unless word-initial) is added to the set. Prefixes of those clusters
form the prefix-legal set.

No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE
from .word_trie import _WORDS as _TRIE_WORDS  # type: ignore[attr-defined]


_VOWELS: frozenset[str] = frozenset("aeiou")


def _extract_coda(word: str) -> str:
    w = word.lower()
    for i in range(len(w) - 1, -1, -1):
        c = w[i]
        if c in _VOWELS:
            return w[i + 1:]
        if c == "y" and i > 0:
            return w[i + 1:]
    # No vowel in word: no coda (all-consonant "word", likely a label).
    return ""


def _build_sets() -> tuple[frozenset[str], frozenset[str]]:
    """Return (legal_codas, legal_coda_prefixes)."""
    codas: set[str] = set()
    for w in _TRIE_WORDS:
        c = _extract_coda(w)
        if c and len(c) <= 6:
            codas.add(c)
    # Filter out codas containing apostrophes or non-letters.
    codas = {c for c in codas if c.isalpha()}

    prefixes: set[str] = set()
    for c in codas:
        for i in range(1, len(c) + 1):
            prefixes.add(c[:i])
    # Always include the full codas.
    prefixes |= codas

    return frozenset(codas), frozenset(prefixes)


_LEGAL_CODAS, _LEGAL_CODA_PREFIXES = _build_sets()


_TERMINATORS: tuple[tuple[str, float], ...] = (
    (" ", 1.0),
    (",", 0.55),
    (".", 0.45),
    (";", 0.30),
    (":", 0.20),
    ("\n", 0.40),
    ("!", 0.28),
    ("?", 0.28),
)

_LOWER = "abcdefghijklmnopqrstuvwxyz"
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def coda_plausibility_bias(
    post_vowel_cluster: str,
    letter_run_len: int,
    letters_off_trie: int,
    on_word_trie: bool,
    speaker_label_state: int,
    word_ending_shape_score: int,
) -> list[float] | None:
    """Emit a bias vector reflecting whether the coda-in-progress is
    legally closable, prefix-legal, or phonotactically impossible.

    Gates:
      * Skip inside speaker-label territory.
      * Skip on-trie (the trie itself encodes legitimate extensions;
        off-trie is where our phonotactic prior helps most).
      * Skip when cluster is empty (no signal).
      * Skip when word_ending_shape_score == 2 (buffer IS a known word
        — stronger signal already firing).
    """
    if speaker_label_state != 0:
        return None
    if on_word_trie:
        return None
    if not post_vowel_cluster:
        return None
    if letter_run_len < 3:
        return None
    if word_ending_shape_score == 2:
        # Already a complete known word — let word_trie layers drive.
        return None

    cluster = post_vowel_cluster
    clen = len(cluster)

    # Classify cluster.
    is_full_legal = cluster in _LEGAL_CODAS
    is_prefix_legal = cluster in _LEGAL_CODA_PREFIXES

    vec = [0.0] * VOCAB_SIZE

    if is_full_legal:
        # Case 2: cluster is itself a legal word-ender. Require some
        # drift before firing — near trie we trust the trie layers.
        drift = max(letters_off_trie, 0)
        if drift < 2:
            return None
        if drift >= 5:
            base = 0.32
        elif drift >= 3:
            base = 0.22
        else:  # drift == 2
            base = 0.14
        # Extra kicker when the coda is a STRONG close (th/ng/nd/nt/st/rd/rk/rt).
        strong_closes = {"th", "ng", "nd", "nt", "st", "rd", "rk", "rt",
                         "nk", "mp", "ck", "ld", "lt", "lk", "lf", "lm",
                         "ss", "ll", "ff", "sh", "ch", "gh", "ct", "pt",
                         "ft"}
        if cluster in strong_closes:
            base *= 1.15
        for t, w in _TERMINATORS:
            idx = VOCAB_INDEX.get(t)
            if idx is not None:
                vec[idx] += base * w
        return vec

    if is_prefix_legal:
        # Case 3: cluster is a prefix of some legal coda. Neutral —
        # we might still be en route to a legal ending. Return None
        # to avoid noise.
        return None

    # Case 4: cluster is NOT any legal coda-prefix. This is
    # phonotactically unreachable as a word ending — closure will
    # have to happen either via a terminator NOW (leaving an
    # un-reconciled cluster — still bad) OR via a vowel (which starts
    # a new syllable, breaking this cluster apart).
    #
    # Strategy:
    #   * Boost terminators moderately (closing ends the pain).
    #   * Boost vowels more strongly (they legally re-segment the
    #     cluster: "mbr" + "e" → "m.bre" where "m" was the old coda
    #     and "bre" starts a new syllable).
    #   * Penalize further consonants (they just extend the dead-end).
    #
    # Depth scaling: shallow clusters are sometimes just false
    # positives (our legal-coda set is incomplete for rare forms).
    if clen <= 1:
        # Single-consonant clusters should almost always be prefix-legal;
        # if not, something is odd but the signal is weak.
        return None

    # Scale pressure by cluster length and off-trie drift.
    depth_scale = min(1.0, 0.4 + 0.25 * (clen - 1))
    drift_scale = 1.0 + 0.30 * min(max(letters_off_trie, 0), 4)
    scale = depth_scale * drift_scale

    # Terminator boost.
    term_mag = 0.50 * scale
    for t, w in _TERMINATORS:
        idx = VOCAB_INDEX.get(t)
        if idx is not None:
            vec[idx] += term_mag * w

    # Vowel boost (breaks the cluster via resegmentation).
    vowel_mag = 2.70 * scale
    for ch in "aeiou":
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += vowel_mag
    # y as vowel — weaker
    idx = VOCAB_INDEX.get("y")
    if idx is not None:
        vec[idx] += vowel_mag * 0.4

    # Consonant penalty (stacking more just extends the dead-end).
    cons_pen = -0.42 * scale
    for ch in "bcdfghjklmnpqrstvwxz":
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += cons_pen
    # Capitals: harder penalty (a cap mid-word is already wrong on top).
    cap_pen = -0.60 * scale
    for ch in _UPPER:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += cap_pen

    return vec
