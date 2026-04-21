"""Block preposition→preposition sequences.

Reads `state.prep_block_active` (True immediately after a preposition
completes, cleared by any content-word / punctuation boundary). When
active, and the current `word_buffer` is a prefix of a known English
preposition (including Shakespearean archaic prepositions), penalize
the letters that would extend the buffer down the preposition trie.
Also penalize terminators when the buffer IS an exact preposition
(preventing the word from closing as a 2-3 letter preposition after
another preposition).

Catches sample patterns like:
  "the noon of with"    (prep "of" → prep "with")
  "By in me did he"     (prep "By" → prep "in")
  "at the he when"      (prep at → det "the" → pron "he"; this layer
                         fires between "at" and "the" but "the" isn't
                         a prep so no block; then "the he" is an NP
                         issue handled elsewhere)

Mechanism: a hand-coded trie of common English + archaic prepositions.
For each current word_buffer (lowercased) that is a prefix of any
preposition:
  - penalize letters that would extend the buffer along a prep path
  - if the buffer is itself a complete preposition, penalize terminators
    (space, comma, period, etc.) so the model must extend into a
    non-preposition word instead (e.g. "o" → "out", "own", "other"
    rather than closing as "of" or "on").

Gates:
  * prep_block_active == True
  * speaker_label_state == 0
  * letter_run_len >= 1 (word is in progress)

No corpus statistics — the preposition set is a fixed English
grammar fact; weights chosen from prior knowledge.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Narrow set of prepositions that most commonly appear as the second
# prep in a forbidden prep→prep pattern. We deliberately EXCLUDE long
# prepositions like "before", "through", "between", etc. because
# their prefixes collide heavily with common non-prep words
# ("the" shares "th" with "through"; "become"/"behold" share "be"
# with "before"/"behind") — including them causes BPC regression.
# Short, high-traffic prepositions have low collision cost and catch
# the most visible prep→prep offenders.
_PREPS: frozenset[str] = frozenset({
    # 2-letter
    "of", "to", "in", "on", "at", "as", "by", "up",
    # 3-letter
    "for",
    # 4-letter
    "into", "unto", "upon", "with", "from",
})


def _build_trie() -> dict[str, set[str]]:
    """For each prefix (including empty), the set of next-letters that
    keep us on the prep trie."""
    trie: dict[str, set[str]] = {}
    for w in _PREPS:
        for i in range(len(w)):
            pref = w[:i]
            nxt = w[i]
            trie.setdefault(pref, set()).add(nxt)
    return trie


_TRIE: dict[str, set[str]] = _build_trie()


# Penalty for extending buffer along a prep path.
_EXTEND_PEN = -3.5
# Penalty for terminators when buffer == complete preposition.
_CLOSE_PEN = -4.0

_TERMINATORS: tuple[str, ...] = (" ", ",", ".", ";", ":", "!", "?", "\n", "'")


def prep_governor_bias(
    prep_block_active: bool,
    word_buffer: str,
    letter_run_len: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if not prep_block_active:
        return None
    # Gate at letter_run_len >= 2: at single-letter buffers the
    # trie is too ambiguous (e.g. buffer "t" would penalize "o"
    # which would also block "tomb"/"today"; buffer "b" would
    # penalize "y" which would also block "bond"/"body"). At
    # 2-letter buffers the trie becomes much more specific
    # (buffer "wi" → only penalize "t" (with); buffer "in" → only
    # penalize "t"/"s" (into/inside), etc.)
    if letter_run_len < 2:
        return None
    if not word_buffer:
        return None

    pref = word_buffer.lower()
    # Only consider pure-letter prefixes (strip apostrophes for the
    # prep-trie lookup, as "o'er" has an apostrophe at position 1).
    # For simplicity: exact-match lookup on the prefix string.
    nxts = _TRIE.get(pref)
    is_prep = pref in _PREPS

    if nxts is None and not is_prep:
        return None

    vec = [0.0] * VOCAB_SIZE

    # Penalize letters that would extend the buffer along a prep path.
    if nxts:
        for ch in nxts:
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += _EXTEND_PEN
            up = ch.upper()
            if up != ch:
                uidx = VOCAB_INDEX.get(up)
                if uidx is not None:
                    vec[uidx] += _EXTEND_PEN

    # Penalize terminators when buffer is exactly a preposition — the
    # model should extend into a non-preposition word.
    if is_prep:
        for t in _TERMINATORS:
            idx = VOCAB_INDEX.get(t)
            if idx is not None:
                vec[idx] += _CLOSE_PEN

    return vec
