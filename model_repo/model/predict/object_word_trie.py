"""Object-phrase word-trie — mid-word bias toward object/complement starts.

Parallel structural move to verb_word_trie: once the clause has a
verb (clause_slot == HAS_VERB), syntax tells us the next word is
most likely a determiner, pronoun, preposition, or direct noun.
Determiners / pronouns / prepositions are a small closed class — a
trie restricted to them gives a strong, low-risk mid-word bias.

Examples of the gap this closes: "Imogen is se" / "Charities is" /
"is oil" — after "is", the model is drifting into improbable
continuations. A "se" mid-word at HAS_VERB should pull toward
common follow-ups like "set", "sent"; but more importantly an "i"
at HAS_VERB should pull toward "in" / "it" / "is" (as a repeat
copula is unlikely after an already-placed verb).

Word set (closed-class, ~80 members):
  - determiners: the / a / an / my / thy / his / her / this / that
    / these / those / our / your / their / no / some / any / every
    / each / all / both / such / what / which / yon / yonder
  - pronouns (object): him / her / them / us / me / thee / ye / it
  - demonstratives: this / that / these / those
  - prepositions: to / of / in / on / at / by / for / with / from /
    through / upon / unto / under / over / about / against /
    amongst / among / between / into / within / without / during
  - conjunctions that open a complement: that / as
  - adverbs that often open complement: not / never / ever / only

Design mirrors verb_word_trie: descend the trie by current
word_buffer, boost continuation letters (weighted by proximity to
a word-end), and boost terminators when buffer is a complete
entry.

Gate:
  - speaker_label_state == 0
  - clause_slot == 2 (HAS_VERB)
  - letter_run_len >= 1
  - word_buffer is a prefix of some word in OBJ_WORDS

Scale is smaller than verb_word_trie because HAS_VERB is shorter-
lived (one word resolves it), so over-biasing causes churn.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


OBJ_WORDS: frozenset[str] = frozenset({
    # Determiners
    "the", "a", "an",
    "my", "thy", "his", "her", "its", "our", "your", "their",
    "this", "that", "these", "those",
    "no", "some", "any", "every", "each", "all", "both", "such",
    "what", "which", "yon", "yonder",
    # Object pronouns
    "him", "her", "them", "us", "me", "thee", "ye", "it",
    "myself", "himself", "herself", "themselves", "ourselves",
    "yourself", "yourselves", "itself",
    # Prepositions
    "to", "of", "in", "on", "at", "by", "for", "with", "from",
    "through", "upon", "unto", "under", "over", "about", "against",
    "amongst", "among", "between", "into", "within", "without",
    "during", "after", "before", "across", "beyond", "beside",
    "around", "behind", "beneath", "below", "above", "until", "till",
    # Complement-opening conjunctions / adverbs
    "that", "as", "not", "never", "ever", "only", "still",
    "now", "here", "there", "hence", "thence", "whence",
    # Common short objects (very-high-frequency Shakespearean nouns)
    "lord", "lords", "lady", "king", "queen", "man", "men", "sir",
    "father", "mother", "son", "daughter", "heart", "hand", "head",
    "eye", "eyes", "love", "life", "death", "time", "day", "night",
    "word", "words", "name", "world", "god", "gods", "heaven",
    "soul", "mind", "blood", "sword", "house", "home", "friend",
    "friends", "enemy", "grace", "honour", "honor", "truth",
    "peace", "war", "fate", "hope", "fear", "joy", "grief",
    # Adjectives that very often lead an object NP
    "good", "sweet", "gentle", "fair", "dear", "poor", "noble",
    "brave", "true", "false", "dead", "alive", "young", "old",
    "great", "small", "strong", "weak", "kind", "bad", "mad",
    # Infinitive marker "to" already listed
})


def _build_trie() -> dict:
    root: dict = {}
    for w in OBJ_WORDS:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node["$"] = True
    return root


_OBJ_TRIE = _build_trie()


def _descend(buf: str) -> dict | None:
    node = _OBJ_TRIE
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


def object_word_trie_bias(
    word_buffer: str,
    letter_run_len: int,
    clause_slot: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a bias vector pushing toward object-phrase completions."""
    if speaker_label_state != 0:
        return None
    if clause_slot != 2:  # HAS_VERB
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

    # Modest scale — the trie has more words than verb_word_trie, so
    # each individual prediction carries less weight.
    base = 0.45

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

    # If buf itself is a complete object word, gently favor terminators.
    if node.get("$") and letter_run_len >= 2:
        term_scale = 0.45
        for ch, w in ((" ", 1.0), (",", 0.40), (".", 0.30),
                      (";", 0.20), ("\n", 0.25)):
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += term_scale * w
                any_bias = True

    if not any_bias:
        return None
    return vec
