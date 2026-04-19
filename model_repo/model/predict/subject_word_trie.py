"""Subject / clause-opener word-trie — mid-word bias at clause start.

Fourth corner of the clause-FSM-aware predict family. Fires when
clause_slot == FRESH (sentence or clause just began, no subject
yet). Biases mid-word letter choices toward words that typically
OPEN a clause: subject pronouns, determiners opening subject NPs,
wh-words, interjections, and negation.

Word set (closed-class, hand-curated):
  - Subject pronouns (incl. EME): I, thou, you, ye, we, he, she,
    they, it, none, nothing, one, who
  - Determiners opening a subject NP: the, a, an, this, that, these,
    those, my, thy, his, her, our, your, their, no, some, any, every,
    each, all, both, such, yon, yonder
  - Wh-words / interrogatives: who, whose, whom, what, which, when,
    where, why, how, whither, whence
  - Interjections opening a line: O, Alas, Ah, Oh, Nay, Ay, Hark,
    Lo, Pray, Prithee, Marry, Indeed, Tut, Fie, Soft, Peace, Hush,
    Faith, Mercy, Come, Go, Stay, Look, See, Hear, Hold, Now
  - Temporal / conditional clause-openers: now, then, here, there,
    hence, thence, whence, once, ere, till, until, if, though,
    unless, when, while, because
  - Proper-noun seed letters are handled elsewhere (capital bias)
    and are NOT in this trie (too broad).

Design mirrors verb_word_trie / object_word_trie / post_obj_word_trie.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


SUBJECT_WORDS: frozenset[str] = frozenset({
    # Subject pronouns
    "i", "thou", "you", "ye", "we", "he", "she", "they", "it",
    "none", "nothing", "one", "who", "whoever", "whosoever",
    "somebody", "nobody", "anybody", "everybody",
    "someone", "everyone",
    # Determiners
    "the", "a", "an", "this", "that", "these", "those",
    "my", "thy", "his", "her", "its", "our", "your", "their",
    "no", "some", "any", "every", "each", "all", "both", "such",
    "yon", "yonder", "what", "which", "mine", "thine",
    # Wh-words
    "who", "whose", "whom", "what", "which", "when", "where",
    "why", "how", "whither", "whence", "wherefore", "whereby",
    "whereof", "whereto",
    # Interjections / imperatives opening a clause
    "o", "alas", "ah", "oh", "nay", "ay", "aye", "hark", "lo",
    "pray", "prithee", "marry", "indeed", "tut", "fie", "soft",
    "peace", "hush", "faith", "mercy", "come", "go", "stay",
    "look", "see", "hear", "hold", "now", "behold",
    # Temporal / conditional clause-openers
    "then", "here", "there", "hence", "thence", "whence",
    "once", "ere", "till", "until", "if", "though", "although",
    "unless", "while", "whilst", "because", "since",
    # Short negations / confirmations
    "not", "never", "ever", "still", "only", "also",
    "yes", "no",
    # Common frame
    "let", "let's",
})


def _build_trie() -> dict:
    root: dict = {}
    for w in SUBJECT_WORDS:
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


def subject_word_trie_bias(
    word_buffer: str,
    letter_run_len: int,
    clause_slot: int,
    speaker_label_state: int,
    chars_since_sentence_end: int,
) -> list[float] | None:
    """Return a bias vector pushing toward subject / clause-opener
    completions.

    Active at:
      - speaker_label_state == 0
      - clause_slot == 0 (FRESH)
      - letter_run_len >= 1
      - chars_since_sentence_end <= 20 (only at actual clause start;
        beyond this we're probably mid-sentence and FRESH is stale)
      - word_buffer is a prefix of some subject-opener
    """
    if speaker_label_state != 0:
        return None
    if clause_slot != 0:  # FRESH
        return None
    if letter_run_len < 1:
        return None
    if not word_buffer:
        return None
    # FRESH can persist for a while between punctuation marks; we only
    # want to fire shortly after the clause actually began.
    if chars_since_sentence_end > 20:
        return None

    buf = word_buffer.lower()
    if not buf.isalpha():
        return None

    node = _descend(buf)
    if node is None:
        return None

    # Scale: comparable to post_obj, slightly lower (FRESH spans many
    # possible subject shapes and a narrow trie shouldn't dominate).
    base = 0.35

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
            lean = 1.0 - 0.18 * (proximity - 1)
            lean = max(lean, 0.32)
        idx_lo = VOCAB_INDEX.get(ch)
        if idx_lo is not None:
            vec[idx_lo] += base * lean
            any_bias = True

    if node.get("$") and letter_run_len >= 2:
        term_scale = 0.35
        for ch, w in ((" ", 1.0), (",", 0.35), (".", 0.25),
                      (";", 0.20), ("\n", 0.25)):
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += term_scale * w
                any_bias = True

    if not any_bias:
        return None
    return vec
