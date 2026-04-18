"""Formulaic-phrase bias layer.

Reads `state.formula_node` — our current position in the formula trie —
and, at appropriate positions, biases the next character toward the
first letter (or continuation letter) of the expected next word.

Two application points:

  1. At word-start (letter_run_len == 0 AND last was space/newline):
     boost the first letters of words that would advance the current
     formula.

  2. Mid-word (letter_run_len >= 1 with non-empty buffer): if the
     buffer is a strict prefix of an expected next word, boost the
     letter that continues toward that word.

Strongest bias applies at deeper formula nodes — the further into a
formula we are, the more constrained the next word is.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE
from .formula_trie import expected_next_words


# Strength of the first-letter bias. Grows with formula depth (deeper
# nodes are more discriminating).
_START_BASE_BUMP = 1.2
_MIDWORD_BASE_BUMP = 1.7


def formula_start_bias(
    node: int,
) -> list[float] | None:
    """Return a word-start first-letter bias vector for the current
    formula node, or None if the node has no children."""
    if node <= 0:
        return None
    children = expected_next_words(node)
    if not children:
        return None
    vec = [0.0] * VOCAB_SIZE
    # Aggregate bump per starter letter.
    letter_hits: dict[str, float] = {}
    for word in children:
        if not word:
            continue
        first = word[0]
        letter_hits[first] = letter_hits.get(first, 0.0) + 1.0
    if not letter_hits:
        return None
    # Normalize so the total positive mass is bounded; the more
    # discriminating the node (fewer alternatives), the stronger the
    # bump on each alternative.
    # With N alternative letters, each gets BUMP / sqrt(N), so single
    # expected words get full bump, but diffuse nodes still help.
    import math as _math
    n = len(letter_hits)
    per_letter = _START_BASE_BUMP / max(1.0, _math.sqrt(n))
    for ch, _count in letter_hits.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += per_letter
        up = ch.upper()
        if up != ch and up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] += per_letter * 0.5
    return vec


def formula_midword_bias(
    node: int,
    buffer: str,
) -> list[float] | None:
    """Return a mid-word letter bias vector: boost letters that
    continue buffer into any expected next word at this node."""
    if node <= 0 or not buffer:
        return None
    children = expected_next_words(node)
    if not children:
        return None
    # Find all expected words that start with buffer.
    matches = []
    for word in children:
        if len(word) > len(buffer) and word.startswith(buffer):
            matches.append(word)
    if not matches:
        return None
    vec = [0.0] * VOCAB_SIZE
    # For each match, boost the next letter of that word.
    letter_hits: dict[str, float] = {}
    for word in matches:
        nxt = word[len(buffer)]
        letter_hits[nxt] = letter_hits.get(nxt, 0.0) + 1.0
    import math as _math
    n = len(letter_hits)
    per_letter = _MIDWORD_BASE_BUMP / max(1.0, _math.sqrt(n))
    for ch, _count in letter_hits.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += per_letter
    # Also: if buffer IS exactly an expected word (buffer in children),
    # boost space/terminators.
    if buffer in children:
        for term in (" ", "\n"):
            if term in VOCAB_INDEX:
                vec[VOCAB_INDEX[term]] += _MIDWORD_BASE_BUMP * 0.4
    return vec
