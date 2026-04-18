"""Formulaic-phrase trie.

Precomputes a trie of common multi-word Shakespeare/Early-Modern-English
formulas (3+ words; 2-word is already covered by phrase_bigram).

Each node has an integer ID. The root is 0. Transitions are by fully
completed lowercased word. At a node, the set of outgoing edges gives
the "expected next words" — useful for biasing the first letter of
the next word.

All formulas come from prior knowledge of Shakespearean idiom. No
corpus statistics.
"""

from __future__ import annotations


# Raw formula list. Each tuple is a sequence of expected completed words.
# Words are lowercased (matching pipeline/pos.py's storage of completed
# words). Chains of 3+ words only — 2-gram is already well-handled.
_FORMULAS: tuple[tuple[str, ...], ...] = (
    # Politeness / request formulas
    ("i", "pray", "thee"),
    ("i", "pray", "you"),
    ("i", "pray", "thee", "tell"),
    ("i", "pray", "you", "sir"),
    ("i", "do", "pray", "thee"),
    ("i", "do", "beseech", "you"),
    ("i", "do", "beseech", "thee"),
    ("i", "beseech", "you"),
    ("i", "beseech", "thee"),
    ("i", "beseech", "your"),
    ("i", "thank", "you"),
    ("i", "thank", "thee"),
    ("i", "thank", "thee", "for"),
    ("god", "save", "you"),
    ("god", "save", "your"),
    ("god", "save", "the"),
    ("god", "give", "you"),
    ("god", "bless", "you"),
    ("god", "bless", "thee"),
    ("heaven", "help", "me"),
    ("heaven", "forgive", "me"),
    # Oaths / assertions
    ("by", "my", "troth"),
    ("by", "my", "faith"),
    ("by", "my", "soul"),
    ("by", "my", "life"),
    ("by", "the", "mass"),
    ("by", "this", "hand"),
    ("by", "heaven"),
    ("upon", "my", "soul"),
    ("upon", "my", "life"),
    ("upon", "my", "honour"),
    ("upon", "my", "word"),
    ("on", "my", "life"),
    ("on", "my", "honour"),
    ("on", "my", "soul"),
    ("as", "i", "am"),
    ("as", "i", "live"),
    ("as", "god", "shall"),
    ("as", "my", "soul"),
    ("for", "my", "part"),
    ("for", "my", "soul"),
    ("for", "my", "life"),
    ("in", "good", "faith"),
    ("in", "good", "sooth"),
    ("in", "sooth"),
    # Vocatives
    ("my", "good", "lord"),
    ("my", "good", "lady"),
    ("my", "good", "friend"),
    ("my", "good", "sir"),
    ("my", "gracious", "lord"),
    ("my", "noble", "lord"),
    ("my", "dear", "lord"),
    ("my", "dear", "friend"),
    ("my", "dear", "lady"),
    ("my", "sweet", "lord"),
    ("my", "sweet", "lady"),
    ("my", "sweet", "love"),
    ("my", "lord", "and"),
    ("good", "my", "lord"),
    ("good", "my", "lady"),
    ("good", "my", "friend"),
    ("sweet", "my", "lord"),
    ("gentle", "my", "lord"),
    ("dear", "my", "lord"),
    ("fair", "my", "lord"),
    # Address tags
    ("o", "my", "lord"),
    ("o", "my", "god"),
    ("o", "my", "soul"),
    ("o", "my", "heart"),
    ("o", "my", "love"),
    ("o", "gentle"),
    ("o", "sweet"),
    # Emotive openers
    ("alas", "poor"),
    ("alas", "the", "day"),
    ("alas", "alas"),
    ("ah", "me"),
    ("woe", "is", "me"),
    ("woe", "unto"),
    ("fie", "upon"),
    ("fie", "on"),
    # Time / place formulas
    ("this", "very", "day"),
    ("this", "very", "hour"),
    ("this", "very", "night"),
    ("even", "now"),
    ("but", "now"),
    ("long", "ago"),
    ("once", "more"),
    ("never", "more"),
    ("ever", "more"),
    ("forth", "with"),
    ("here", "and", "now"),
    # Common verb + object phrases
    ("i", "warrant", "you"),
    ("i", "warrant", "thee"),
    ("i", "warrant", "it"),
    ("i", "dare", "say"),
    ("i", "dare", "swear"),
    ("i", "can", "not"),
    ("i", "do", "know"),
    ("i", "do", "think"),
    ("i", "do", "hope"),
    ("i", "do", "fear"),
    ("i", "do", "swear"),
    ("i", "do", "love"),
    ("i", "do", "remember"),
    ("i", "have", "been"),
    ("i", "have", "seen"),
    ("i", "have", "heard"),
    ("i", "have", "done"),
    ("i", "have", "said"),
    ("i", "must", "go"),
    ("i", "must", "be"),
    ("i", "will", "be"),
    ("i", "will", "not"),
    ("i", "will", "tell"),
    ("i", "shall", "not"),
    ("i", "shall", "be"),
    ("thou", "shalt", "not"),
    ("thou", "shalt", "be"),
    ("thou", "wilt", "not"),
    ("thou", "wilt", "be"),
    ("thou", "art", "a"),
    ("thou", "art", "my"),
    ("thou", "art", "the"),
    ("thou", "hast", "a"),
    ("thou", "hast", "my"),
    ("thou", "hast", "the"),
    ("thou", "hast", "been"),
    ("thou", "hast", "seen"),
    ("he", "hath", "a"),
    ("he", "hath", "the"),
    ("he", "hath", "been"),
    ("she", "hath", "a"),
    ("she", "hath", "the"),
    ("she", "hath", "been"),
    ("it", "is", "a"),
    ("it", "is", "the"),
    ("it", "is", "my"),
    ("it", "is", "not"),
    ("it", "is", "no"),
    ("it", "is", "so"),
    ("it", "is", "done"),
    ("it", "is", "enough"),
    ("it", "is", "well"),
    ("there", "is", "no"),
    ("there", "is", "a"),
    ("there", "is", "the"),
    ("there", "is", "some"),
    ("this", "is", "a"),
    ("this", "is", "the"),
    ("this", "is", "my"),
    ("this", "is", "no"),
    ("that", "is", "a"),
    ("that", "is", "the"),
    ("that", "is", "my"),
    ("that", "is", "no"),
    ("that", "is", "not"),
    ("what", "is", "this"),
    ("what", "is", "that"),
    ("what", "is", "the"),
    ("what", "is", "your"),
    ("what", "say", "you"),
    ("what", "say", "thou"),
    ("what", "think", "you"),
    ("how", "now", "my"),
    ("how", "now", "sir"),
    ("how", "now", "what"),
    ("how", "do", "you"),
    ("how", "fares"),
    ("fare", "thee", "well"),
    ("fare", "you", "well"),
    ("fare", "well"),
    ("come", "come"),
    ("come", "hither"),
    ("come", "away"),
    ("come", "on"),
    ("away", "with"),
    ("out", "upon"),
    ("out", "of"),
    ("hold", "thy"),
    ("hold", "your"),
    ("peace", "peace"),
    # Negation / prohibitions
    ("i", "will", "not", "have"),
    ("i", "cannot", "tell"),
    ("i", "know", "not"),
    ("i", "know", "not", "what"),
    ("i", "know", "not", "why"),
    ("i", "care", "not"),
    ("i", "fear", "not"),
    ("i", "doubt", "it"),
    ("i", "doubt", "not"),
    # Existential formulas
    ("to", "be", "or"),
    ("to", "be", "or", "not"),
    ("not", "to", "be"),
    ("if", "it", "be"),
    ("if", "it", "were"),
    ("if", "i", "were"),
    ("if", "i", "had"),
    ("if", "thou", "wilt"),
    ("if", "thou", "hast"),
    ("if", "thou", "art"),
    # Misc common formulas
    ("let", "me", "see"),
    ("let", "me", "go"),
    ("let", "me", "speak"),
    ("let", "me", "have"),
    ("let", "us", "go"),
    ("let", "us", "hence"),
    ("let", "us", "be"),
    ("make", "haste"),
    ("take", "heed"),
    ("take", "hold"),
    ("give", "me", "your"),
    ("give", "me", "thy"),
    ("give", "me", "leave"),
    ("give", "me", "thine"),
    ("give", "me", "a"),
    ("give", "me", "the"),
    # Relative / definitional
    ("that", "which"),
    ("which", "is"),
    ("such", "as"),
    ("such", "a"),
    ("such", "is"),
    ("he", "that"),
    ("he", "who"),
    ("she", "that"),
    ("she", "who"),
    # Discourse / backchannel
    ("well", "said"),
    ("well", "met"),
    ("well", "done"),
    ("no", "no"),
    ("ay", "ay"),
    ("nay", "nay"),
    ("tut", "tut"),
    ("what", "ho"),
    # Additional time/aspect
    ("in", "the", "morning"),
    ("in", "the", "night"),
    ("in", "the", "mean", "time"),
    ("by", "and", "by"),
    ("ere", "long"),
    ("long", "since"),
    ("at", "this", "hour"),
    ("at", "this", "present"),
    ("at", "hand"),
    ("in", "due", "time"),
    ("at", "once"),
    # Place / direction
    ("to", "the", "king"),
    ("to", "the", "queen"),
    ("to", "the", "court"),
    ("from", "the", "king"),
    ("at", "court"),
    ("to", "and", "fro"),
    # Wishes / optatives
    ("would", "god"),
    ("would", "to", "god"),
    ("would", "to", "heaven"),
    ("i", "would", "i"),
    ("i", "would", "to"),
    ("heaven", "grant"),
    ("god", "grant"),
    ("god", "forbid"),
    ("heaven", "forbid"),
    # Imperative softeners
    ("be", "it", "so"),
    ("so", "be", "it"),
    ("let", "it", "be"),
    ("let", "it", "pass"),
    ("let", "it", "go"),
    # Rhetorical questions
    ("is", "it", "not"),
    ("is", "he", "not"),
    ("is", "she", "not"),
    ("art", "thou", "not"),
    ("art", "thou", "a"),
    ("art", "thou", "the"),
    ("dost", "thou", "not"),
    ("dost", "thou", "love"),
    ("dost", "thou", "know"),
    ("dost", "thou", "think"),
    ("hast", "thou", "not"),
    ("hast", "thou", "seen"),
    ("do", "you", "know"),
    ("do", "you", "think"),
    ("do", "you", "hear"),
    ("do", "you", "see"),
    # Famous openers / closers
    ("to", "thine", "own"),
    ("to", "thine", "own", "self"),
    ("this", "above", "all"),
    ("but", "soft"),
    ("but", "see"),
    ("but", "hark"),
    ("hark", "ye"),
    ("hark", "you"),
    ("lo", "there"),
    ("lo", "here"),
    ("see", "here"),
    ("see", "there"),
    # Exclamations
    ("oh", "me"),
    ("o", "me"),
    ("ah", "ha"),
    ("out", "alas"),
    ("heaven", "help"),
    ("o", "heaven"),
    ("o", "heavens"),
    # Concessions / conditionals
    ("and", "yet"),
    ("yet", "again"),
    ("yet", "more"),
    ("no", "more"),
    ("some", "more"),
    ("one", "more"),
    ("no", "less"),
    ("much", "more"),
    ("how", "much"),
    # Agreements / assent
    ("ay", "sir"),
    ("ay", "marry"),
    ("ay", "my", "lord"),
    ("no", "sir"),
    ("no", "my", "lord"),
    ("yes", "my", "lord"),
    ("yes", "sir"),
    # Address chains
    ("sir", "i"),
    ("madam", "i"),
    ("lord", "i"),
    ("sir", "it"),
    ("sir", "the"),
    ("madam", "the"),
    # Possessive + body/mind
    ("in", "my", "heart"),
    ("in", "my", "soul"),
    ("in", "my", "mind"),
    ("on", "my", "head"),
    ("in", "my", "youth"),
    ("with", "all", "my"),
    ("all", "my", "heart"),
    ("all", "my", "life"),
    # Request / command
    ("pray", "you"),
    ("pray", "thee"),
    ("pray", "sir"),
    ("prithee", "tell"),
    ("prithee", "speak"),
    ("prithee", "come"),
    ("prithee", "go"),
)


# Build trie. Each node is an integer ID. Children[node][word] = next_node.
# All lists are kept immutable-ish at module scope.

_CHILDREN: list[dict[str, int]] = [{}]  # node 0 = root
_EXPECTED_STARTERS: list[frozenset[str]] = []  # per-node set of first letters of expected next words


def _build_trie() -> None:
    """Populate _CHILDREN. Root has id=0."""
    for formula in _FORMULAS:
        node = 0
        for word in formula:
            children = _CHILDREN[node]
            if word in children:
                node = children[word]
            else:
                new_id = len(_CHILDREN)
                children[word] = new_id
                _CHILDREN.append({})
                node = new_id


_build_trie()


# Per-node starter-letter set: letters that begin any child word.
for _children in _CHILDREN:
    starters: set[str] = set()
    for w in _children:
        if w:
            starters.add(w[0])
    _EXPECTED_STARTERS.append(frozenset(starters))


N_NODES = len(_CHILDREN)


def advance_node(node: int, word: str) -> int:
    """Return the next node given current node and the just-completed word.

    Behavior:
      - If `word` is a child of `node`, return that child.
      - Otherwise, if `word` is a child of root (0), return that child
        (fresh formula start).
      - Otherwise, return 0 (reset).
    """
    if node < 0 or node >= N_NODES:
        return 0
    children = _CHILDREN[node]
    nxt = children.get(word)
    if nxt is not None:
        return nxt
    # Try fresh start from root.
    if node != 0:
        return _CHILDREN[0].get(word, 0)
    return 0


def expected_next_words(node: int) -> dict[str, int]:
    """Return the children map at `node` — word -> next_node."""
    if node < 0 or node >= N_NODES:
        return {}
    return _CHILDREN[node]


def expected_starter_letters(node: int) -> frozenset[str]:
    """Return the set of first-letters of expected next words at `node`."""
    if node < 0 or node >= N_NODES:
        return frozenset()
    return _EXPECTED_STARTERS[node]
