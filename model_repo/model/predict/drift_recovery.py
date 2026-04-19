"""Predict layer — scene-drift recovery bias.

Reads `state.drift_streak` (set by pipeline/drift.py). When we've
completed 2+ consecutive off-trie words, the scene is in runaway
letter-ngram gibberish mode. This layer applies an increasingly
aggressive recovery bias at word-start:

  Streak 0-1: no bias (normal operation).
  Streak 2:   mild pull toward common English starters.
  Streak 3+:  strong pull, scaled with streak length.

The recovery targets the 10 most productive first-letter classes in
English and Shakespeare's vocabulary:

  t — the, thou, thy, to, that, this, those, ...
  a — a, an, and, as, at, all, art, ...
  i — I, is, in, it, if, ...
  o — of, or, on, O, our, out, ...
  h — he, him, his, her, have, ...
  w — will, what, where, with, when, who, ...
  b — but, be, by, before, both, ...
  s — shall, so, sir, speak, still, see, ...
  f — for, from, fair, find, ...
  m — my, me, more, must, make, ...

Plus common capital starters at sentence-start-like positions (which
we can approximate by low words_in_sentence).

Suppresses rare/gibberish-leaning starters: x, z, j, q, and also
dampens v/k/b slightly (these open fewer words than the top 10).

No corpus statistics — all weights from prior knowledge of English
word-frequency shape.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Safe starter letters — very common first-letter classes.
_SAFE_STARTERS: dict[str, float] = {
    "t": 0.60,   # the, thou, to, that
    "a": 0.45,   # a, an, and, as, all
    "i": 0.40,   # I, is, in, it
    "o": 0.35,   # of, or, on, our, O
    "h": 0.45,   # he, his, have, heart
    "w": 0.45,   # with, what, will, when
    "b": 0.35,   # but, be, by, before
    "s": 0.50,   # so, sir, shall, see
    "f": 0.30,   # for, from, fair
    "m": 0.35,   # my, more, must, make
    "n": 0.22,   # no, not, now
    "c": 0.22,   # come, can, cannot
    "d": 0.20,   # do, did, death, dear
    "l": 0.20,   # love, lord, like
    "p": 0.15,   # pray, poor, peace
    "r": 0.15,   # right, run, remain
    "e": 0.18,   # ever, every, eye
    "y": 0.18,   # you, your, yet, ye
    "g": 0.15,   # good, give, go, great
}

# Capitals for sentence-start recovery.
_SAFE_CAP_STARTERS: dict[str, float] = {
    "T": 0.50,   # The, Thou, To, That
    "A": 0.35,   # And, A, As, All
    "I": 0.45,   # I, In, Is, If
    "O": 0.30,   # O, Our, Of
    "H": 0.30,   # He, His, Have
    "W": 0.40,   # What, Why, Will, With
    "B": 0.25,   # But, Be, Before
    "S": 0.35,   # So, Sir, Speak, Shall
    "F": 0.22,   # For, From, Fair
    "M": 0.25,   # My, More, Must
    "N": 0.18,   # No, Now, Nay
    "C": 0.18,   # Come, Can
    "D": 0.15,   # Do, Did, Dear
    "L": 0.15,   # Lord, Let
    "P": 0.12,   # Pray, Peace
    "Y": 0.15,   # You, Your, Yet
    "G": 0.12,   # Good, Give, Go
    "K": 0.15,   # King, Know
    "E": 0.12,   # Ever, Eye
    "R": 0.10,   # Rather, Roman
}

# Suppress rare gibberish-starters.
_GIBBERISH_STARTERS: dict[str, float] = {
    "x": -0.60, "X": -0.40,
    "z": -0.45, "Z": -0.30,
    "j": -0.30, "J": -0.20,
    "q": -0.40, "Q": -0.25,
    # Letters that are common mid-word but rarely START words.
    "v": -0.15, "V": -0.05,
}

# Also push space/terminators a bit — if drifting, ending the current
# "word" (which might be a garbage orphan prefix from prior emissions)
# is a win. But this only fires at word-start (letter_run_len == 0),
# where space is typically already very likely from prior biases.
# Avoid double-pressuring; keep this at 0 here.


def drift_recovery_bias(
    drift_streak: int,
    speaker_label_state: int,
    words_in_sentence: int,
    consecutive_newlines: int,
) -> list[float] | None:
    """At word-start, when consecutive off-trie word-completions exceed
    a threshold, apply a recovery bias pulling toward common English
    starters and away from rare letters."""
    if speaker_label_state != 0:
        return None
    if drift_streak < 2:
        return None

    # Scale: 0 at streak<2, linear ramp to full at streak >= 5.
    if drift_streak >= 5:
        scale = 1.0
    else:
        scale = (drift_streak - 1) / 4.0  # streak 2 → 0.25, 3 → 0.50, 4 → 0.75

    # At sentence-start-like positions (fresh line, 0 words in sentence),
    # capital letters are the targets. Elsewhere, lowercase.
    at_sentence_start = words_in_sentence == 0

    vec = [0.0] * VOCAB_SIZE

    if at_sentence_start:
        for ch, w in _SAFE_CAP_STARTERS.items():
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += w * scale
        # Mild lowercase boost too (many sentence-starts are still lc
        # after a comma/semicolon followed by newline).
        for ch, w in _SAFE_STARTERS.items():
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += 0.4 * w * scale
    else:
        for ch, w in _SAFE_STARTERS.items():
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += w * scale

    for ch, w in _GIBBERISH_STARTERS.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale  # w is negative

    return vec


def drift_recovery_midword_bias(
    drift_streak: int,
    letter_run_len: int,
    letters_off_trie: int,
    on_word_trie: bool,
    speaker_label_state: int,
) -> list[float] | None:
    """Mid-word terminator pressure when drift streak is high.

    Complements the word-start recovery: once the CURRENT word is
    also off-trie and has run long enough, push aggressively toward
    space/punctuation to end the gibberish EARLY rather than letting
    it extend to "rseanhalgmiefsem" length.
    """
    if speaker_label_state != 0:
        return None
    if drift_streak < 2:
        return None
    if on_word_trie:
        return None
    if letters_off_trie < 2:
        return None
    if letter_run_len < 4:
        return None

    # Streak scale — ramp to full at streak >= 5.
    if drift_streak >= 5:
        streak_scale = 1.0
    else:
        streak_scale = (drift_streak - 1) / 4.0

    # Length scale — longer current gibberish word → stronger end push.
    if letter_run_len >= 10:
        length_scale = 1.5
    elif letter_run_len >= 7:
        length_scale = 1.0
    else:
        length_scale = 0.5

    scale = streak_scale * length_scale

    vec = [0.0] * VOCAB_SIZE
    # Terminators (same set as offtrie_depart).
    for ch, w in (
        (" ", 1.5), (",", 0.80), (".", 0.60), (";", 0.40),
        (":", 0.30), ("!", 0.40), ("?", 0.40), ("\n", 0.50),
    ):
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
    # Common word-ending letters — at least LAND on a plausible ender.
    for ch, w in (
        ("e", 0.40), ("s", 0.40), ("d", 0.35), ("t", 0.30),
        ("n", 0.30), ("r", 0.25), ("y", 0.25), ("h", 0.20),
    ):
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
    # Extra gibberish penalty on rare letters.
    for ch, w in (
        ("x", -0.50), ("z", -0.40), ("j", -0.30), ("q", -0.35),
    ):
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
    return vec
