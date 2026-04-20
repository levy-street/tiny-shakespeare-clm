"""Post-apostrophe elision bias inside a word.

English (and Shakespeare) use the apostrophe inside a word in a very
small number of patterns:

  's     possessive, contraction of "is" / "has"
  'd     contraction of "would" / "had" / past tense
  't     contraction of "it" ('tis, 'twas) and "not" (is't, on't)
  'll    contraction of "will"
  're    contraction of "are"
  've    contraction of "have"
  'er    elision in "o'er", "ne'er", "where'er"
  'en    archaic ('gen, 'twen) — rare but attested
  'em    short for "them"
  'tis   / 'twas / 'twere (word-initial apostrophe)

Therefore, at position 1 after an apostrophe inside a word, the next
letter is drawn from a tight set: s, d, t, l, r, v, e, n, m. Any other
letter is almost certainly gibberish.

This layer reads `letters_since_apostrophe`:
  * 1 → we just emitted "'", next char is position-1-after-apos.
        Strong bias toward {s, d, t, l, r, v, e, n, m}; strong
        penalty on all other letters.
  * 2 → one letter past the apostrophe. The second letter is also
        constrained: after 'l' comes 'l' ('ll); after 'r' comes 'e'
        ('re); after 'v' comes 'e' ('ve); after 'e' comes 'r' or 'n'
        ('er / 'en); after 's' the word usually ends. Gentle second-
        position bias.
  * 3+ → bias fades; word-trie / letter-ngram take over.

No corpus statistics — these patterns come from English morphology.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_POS1_STRONG = {
    "s": 1.6,
    "d": 1.3,
    "t": 1.4,
    "l": 1.3,   # 'll
    "r": 1.2,   # 're
    "v": 1.1,   # 've
    "e": 1.0,   # 'er, 'em, 'en
    "n": 0.8,   # 'n / 'neath
    "m": 0.9,   # 'em
}


# Second-letter continuations keyed by the letter at position 1 after
# the apostrophe (i.e., the last letter of word_buffer when
# letters_since_apostrophe == 2).
_POS2_BY_PREV: dict[str, dict[str, float]] = {
    "l": {"l": 1.6},          # 'll
    "r": {"e": 1.5},          # 're
    "v": {"e": 1.6},          # 've
    "e": {"r": 0.9, "n": 0.8, "m": 0.6, "s": 0.3},  # 'er / 'en / 'em / 'es
    "t": {"w": 0.8, "h": 0.3, "i": 0.3},  # 'twas / 'thou / 'tis (short)
    # Most 'd / 's / 't / 'n / 'm words are 1-letter elisions → next
    # char is a word-ender (space/punct), so no position-2 letter bias.
}


# Letters to penalize at position 1 (everything NOT in _POS1_STRONG).
_LOWER = "abcdefghijklmnopqrstuvwxyz"
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def apostrophe_elision_bias(
    letters_since_apostrophe: int,
    word_buffer: str,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if letters_since_apostrophe < 1 or letters_since_apostrophe > 2:
        return None
    if not word_buffer:
        return None

    vec = [0.0] * VOCAB_SIZE

    if letters_since_apostrophe == 1:
        # Position 1: strong bias toward elision letters; penalize
        # everything else.
        for ch, w in _POS1_STRONG.items():
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += w
        # Penalize all OTHER lowercase letters moderately — they are
        # not impossible (very rare archaic forms), but improbable.
        for ch in _LOWER:
            if ch in _POS1_STRONG:
                continue
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] -= 1.2
        # Penalize uppercase even harder — an uppercase letter right
        # after an apostrophe inside a word is essentially impossible.
        for ch in _UPPER:
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] -= 2.5
        return vec

    # letters_since_apostrophe == 2: the last letter of word_buffer is
    # the letter that was emitted at position 1. Look up continuation.
    prev = word_buffer[-1].lower()
    continuations = _POS2_BY_PREV.get(prev)
    if continuations is None:
        return None
    any_nonzero = False
    for ch, w in continuations.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w
            any_nonzero = True
    if not any_nonzero:
        return None
    return vec
