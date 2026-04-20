"""Absolute word-length prior.

Orthogonal to the trie-drift layers (which only fire off-trie): this
layer applies termination pressure based solely on how long the
current letter run has become, independent of trie status.

Motivation. English — and early-modern English in particular — has a
strongly right-skewed word-length distribution. Words of 1-6 chars
dominate; 7-9 are common; 10+ become rare; 13+ are very rare; 15+ is
essentially nonexistent outside deliberate jokes (Shakespeare's
"honorificabilitudinitatibus" in LLL being the famous exception).
Sampling chains routinely produce invented "words" of 10-18 chars
("etustarted", "rytsapiosen", "inafulnyeer", "Darknesarnent'shen")
because the bigram / trigram / word-trie layers all provide local
letter-level continuation pull that, when integrated, can extend
indefinitely even when the off-trie recovery layers are firing.

This layer adds an absolute length prior:
  * length 8-9: no-op (still a reasonable word length)
  * length 10: mild termination boost
  * length 11-12: moderate termination boost, mild letter penalty
  * length 13-14: strong termination boost, moderate letter penalty
  * length 15+: hard cap — very strong termination, heavy letter penalty

The layer is dampened when `on_word_trie` is True AND `has_seen_complete`
is False: we may be building a legitimate long word ("importunately")
that hasn't yet passed through a complete-word node. The dampener
lowers the letter penalty; terminator boost still applies to keep
pressure on.

Gated on `speaker_label_state == 0` (labels have their own length limit
via the speaker trie).

No corpus statistics — the length thresholds come from prior knowledge
of English word-length distributions.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_LOWER_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_UPPER_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Gibberish-extending letters — always penalized harder than plain
# letters when the length prior fires.
_GIBBERISH = ("j", "q", "x", "z", "v", "J", "Q", "X", "Z", "V")


def word_length_prior_bias(
    letter_run_len: int,
    on_word_trie: bool,
    has_seen_complete: bool,
    letters_off_trie: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if letter_run_len < 12:
        return None

    n = letter_run_len

    # Base schedule: (space_boost, punct_boost, nl_boost, letter_pen, gib_pen)
    # Fires only at 12+ to protect BPC — real Shakespeare words of
    # length 12+ are rare; of length 15+ essentially nonexistent.
    if n == 12:
        sp_b, pn_b, nl_b, l_p, g_p = 0.45, 0.25, 0.18, -0.12, -0.35
    elif n == 13:
        sp_b, pn_b, nl_b, l_p, g_p = 0.85, 0.48, 0.38, -0.28, -0.70
    elif n == 14:
        sp_b, pn_b, nl_b, l_p, g_p = 1.45, 0.80, 0.65, -0.55, -1.15
    else:  # n >= 15 — hard cap regime.
        # Strong push; escalate slightly per extra letter.
        extra = min(n - 15, 6)
        sp_b = 2.20 + 0.35 * extra
        pn_b = 1.35 + 0.22 * extra
        nl_b = 1.10 + 0.18 * extra
        l_p = -1.00 - 0.18 * extra
        g_p = -1.80 - 0.30 * extra

    # Dampen the letter penalty when we may still be building a
    # legitimate on-trie word that hasn't passed a complete-word
    # node yet. Terminator boost is retained (we still want to end
    # soon), but don't fight the trie's pull with a strong letter
    # penalty.
    if on_word_trie and not has_seen_complete and letters_off_trie == 0:
        l_p *= 0.35
        g_p *= 0.55
        # Also dampen the terminator boost slightly — we're not that
        # confident the word is gibberish.
        sp_b *= 0.65
        pn_b *= 0.65
        nl_b *= 0.65

    vec = [0.0] * VOCAB_SIZE

    if " " in VOCAB_INDEX:
        vec[VOCAB_INDEX[" "]] += sp_b
    for ch, w in ((",", 0.60), (".", 0.45), (";", 0.35), (":", 0.20),
                  ("!", 0.30), ("?", 0.28)):
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += pn_b * w
    if "\n" in VOCAB_INDEX:
        vec[VOCAB_INDEX["\n"]] += nl_b

    for ch in _LOWER_LETTERS:
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += l_p
    for ch in _UPPER_LETTERS:
        if ch in VOCAB_INDEX:
            # Upper mid-word is already implausible; same pen level.
            vec[VOCAB_INDEX[ch]] += l_p

    for ch in _GIBBERISH:
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += g_p

    # Apostrophe continues the run (e.g. "ne'er"); if we've already
    # hit length 10+, the apostrophe is almost certainly gibberish-
    # extending ("Darknesarnent'shen"). Penalize it like a letter.
    if "'" in VOCAB_INDEX:
        vec[VOCAB_INDEX["'"]] += l_p * 0.75

    return vec
