"""Post-object word-trie — mid-word bias toward clause-chain starters.

Third corner of the clause-FSM-aware predict family (after
verb_word_trie and object_word_trie). Fires when
clause_slot == POST_OBJ — the position immediately after an object/
complement, where the natural next tokens are either sentence
punctuation or a word that chains into a new clause or extends the
current one.

Word set (closed-class, hand-curated):
  - Coordinating conjunctions: and, but, or, nor, yet, so, for
  - Subordinating conjunctions & relatives: that, which, who, whom,
    whose, what, where, when, while, since, though, although, if,
    unless, because, till, until, ere, lest, than, as, wherefore,
    whereby, whereof, whereto, whence, thence, hence
  - Chain adverbs: then, thus, therefore, hence, still, now, yet,
    also, too, even, indeed
  - Extending prepositions: with, to, of, in, on, at, for, by, from,
    upon, into, unto, within, without, through, under, over

Design mirrors verb_word_trie / object_word_trie: descend the
trie, boost continuation letters and — when buffer is a complete
entry — gently boost terminators.

This closes the drift gap visible in POST_OBJ samples where a
noun-object is followed by yet another noun rather than a
conjunction, punctuation, or extending-preposition.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


POST_OBJ_WORDS: frozenset[str] = frozenset({
    # Coordinating
    "and", "but", "or", "nor", "yet", "so", "for",
    # Subordinating / relatives
    "that", "which", "who", "whom", "whose", "what", "where",
    "when", "while", "whilst", "since", "though", "although",
    "if", "unless", "because", "till", "until", "ere", "lest",
    "than", "as",
    "wherefore", "whereby", "whereof", "whereto", "whence",
    "thence", "hence",
    # Chain adverbs
    "then", "thus", "therefore", "still", "now", "also", "too",
    "even", "indeed", "howbeit", "nevertheless", "however",
    "moreover", "anon", "presently", "straight", "straightway",
    # Prepositions extending a PP chain
    "with", "to", "of", "in", "on", "at", "by", "from",
    "upon", "into", "unto", "within", "without", "through",
    "under", "over", "against", "before", "after", "beyond",
    "above", "below", "behind", "beneath", "beside", "between",
    "among", "amongst", "about", "around", "across",
    # Closing interjections / tags
    "perchance", "mayhap", "haply", "forsooth", "verily",
})


def _build_trie() -> dict:
    root: dict = {}
    for w in POST_OBJ_WORDS:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node["$"] = True
    return root


_TRIE = _build_trie()


def _descend(buf: str) -> dict | None:
    node = _TRIE
    for ch in buf:
        if ch not in node:
            return None
        node = node[ch]
    return node


def _nearest_end(node: dict, depth: int, limit: int) -> int | None:
    if depth > limit:
        return None
    if node.get("$"):
        return depth
    best: int | None = None
    for ch, child in node.items():
        if ch == "$":
            continue
        if not isinstance(child, dict):
            continue
        d = _nearest_end(child, depth + 1, limit)
        if d is not None and (best is None or d < best):
            best = d
    return best


def post_obj_word_trie_bias(
    word_buffer: str,
    letter_run_len: int,
    clause_slot: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a bias vector pushing toward clause-chain completions."""
    if speaker_label_state != 0:
        return None
    if clause_slot != 3:  # POST_OBJ
        return None
    if letter_run_len < 1:
        return None
    if not word_buffer:
        return None

    buf = word_buffer.lower()
    if not buf.isalpha():
        return None

    node = _descend(buf)
    if node is None:
        return None

    # Moderate scale. POST_OBJ competes with open-ended noun-phrase
    # extensions (many real sentences have object + comma + more
    # content of many shapes), so keep it gentle.
    base = 0.40

    vec = [0.0] * VOCAB_SIZE
    any_bias = False

    for ch, child in node.items():
        if ch == "$":
            continue
        if not isinstance(child, dict):
            continue
        proximity = _nearest_end(child, depth=1, limit=4)
        if proximity is None:
            lean = 0.35
        else:
            lean = 1.0 - 0.17 * (proximity - 1)
            lean = max(lean, 0.35)
        idx_lo = VOCAB_INDEX.get(ch)
        if idx_lo is not None:
            vec[idx_lo] += base * lean
            any_bias = True

    if node.get("$") and letter_run_len >= 2:
        term_scale = 0.40
        for ch, w in ((" ", 1.0), (",", 0.45), (".", 0.30),
                      (";", 0.25), ("\n", 0.30)):
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += term_scale * w
                any_bias = True

    if not any_bias:
        return None
    return vec
