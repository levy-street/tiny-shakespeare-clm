"""Line-starter anaphora bias.

Reads `state.recent_line_starters` — up to 3 first-words-of-line
(oldest first). Fires when at least two recent line-starters are
the EXACT SAME word (lowercased) — that's the actual Shakespearean
anaphora pattern: "Now is... / Now are..." or "Blow winds... /
Blow blow...". Pure letter-match is too permissive (T starts
thousands of different words) so we require word-level agreement.

Applied in compose.py at verse-line-start positions outside
speaker labels. Strength depends on agreement strength: exact
2-of-last-2 gives a strong bump; 3-of-last-3 gives the strongest.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_STRONG_2OF2 = 1.2   # last two starters identical (fresh anaphora)
_STRONGEST_3 = 2.4   # all three starters identical (established)
_WEAK_2OF3 = 0.5     # starters[0] == starters[2], middle differs

# Very common English/Shakespeare line-openers whose repetition is
# not necessarily anaphoric — they're just frequent enough to repeat
# by chance. Require content-word-ish starters for the bias.
_NON_ANAPHORIC: frozenset[str] = frozenset({
    "the", "a", "an", "and", "but", "or", "for", "so", "then",
    "i", "he", "she", "we", "they", "you", "it",
    "my", "his", "her", "our", "your", "their",
    "to", "of", "in", "on", "by", "with",
    "is", "are", "was", "were", "be",
    "that", "this", "these", "those",
    "if", "as", "when", "where", "while",
})


def anaphora_start_bias(
    starters: tuple[str, ...],
) -> list[float] | None:
    """Return word-start first-letter bias vector when recent line-
    starters repeat the same word. Returns None otherwise."""
    if not starters:
        return None
    words = [s.lower() for s in starters if s]
    if len(words) < 2:
        return None

    if len(words) == 2:
        if words[0] == words[1] and words[0] not in _NON_ANAPHORIC:
            ch = words[-1][0]
            strength = _STRONG_2OF2
        else:
            return None
    else:
        # len == 3
        if words[0] == words[1] == words[2]:
            if words[0] in _NON_ANAPHORIC:
                return None
            ch = words[-1][0]
            strength = _STRONGEST_3
        elif words[1] == words[2]:
            if words[1] in _NON_ANAPHORIC:
                return None
            ch = words[-1][0]
            strength = _STRONG_2OF2
        elif words[0] == words[2]:
            if words[0] in _NON_ANAPHORIC:
                return None
            ch = words[-1][0]
            strength = _WEAK_2OF3
        else:
            return None

    vec = [0.0] * VOCAB_SIZE
    if ch in VOCAB_INDEX:
        vec[VOCAB_INDEX[ch]] += strength
    up = ch.upper()
    if up != ch and up in VOCAB_INDEX:
        vec[VOCAB_INDEX[up]] += strength
    return vec


def anaphora_repeated_word(
    starters: tuple[str, ...],
) -> str | None:
    """Return the word currently being anaphora-repeated, or None.
    Identical logic to anaphora_start_bias's trigger, returning the
    word rather than a bias vector."""
    if not starters:
        return None
    words = [s.lower() for s in starters if s]
    if len(words) < 2:
        return None
    if len(words) == 2:
        if words[0] == words[1] and words[0] not in _NON_ANAPHORIC:
            return words[-1]
        return None
    if words[0] == words[1] == words[2]:
        if words[0] in _NON_ANAPHORIC:
            return None
        return words[-1]
    if words[1] == words[2]:
        if words[1] in _NON_ANAPHORIC:
            return None
        return words[-1]
    if words[0] == words[2]:
        if words[0] in _NON_ANAPHORIC:
            return None
        return words[-1]
    return None


_MIDWORD_STRENGTH = 1.4


def anaphora_midword_bias(
    starters: tuple[str, ...],
    buffer: str,
    chars_since_newline: int,
) -> list[float] | None:
    """Mid-word continuation bias for anaphora. Only fires when the
    current line is fresh (chars_since_newline small) and the buffer
    is a strict prefix of the anaphora-repeated word.
    """
    if not buffer:
        return None
    # Only during the first word of the new line.
    if chars_since_newline > len(buffer) + 1:
        return None
    repeated = anaphora_repeated_word(starters)
    if not repeated:
        return None
    lb = buffer.lower()
    if len(repeated) <= len(lb):
        return None
    if not repeated.startswith(lb):
        return None
    nxt = repeated[len(lb)]
    vec = [0.0] * VOCAB_SIZE
    if nxt in VOCAB_INDEX:
        vec[VOCAB_INDEX[nxt]] += _MIDWORD_STRENGTH
    return vec
