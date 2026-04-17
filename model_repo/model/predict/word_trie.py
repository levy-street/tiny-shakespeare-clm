"""Word-trie completion bias layer.

Given `state.word_buffer` (the letters written since the last non-letter),
if the buffer is a prefix of one or more common English/Shakespearean
words, bias the distribution toward the letters (and word-ending
characters) that would complete those words. The longer the prefix, the
more confident the completion bias.

All knowledge here comes from prior knowledge of common Shakespearean
vocabulary. No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# A list of common Shakespearean / Early Modern English words (lowercased).
# Mixed with common modern English frequent words. Apostrophes are
# represented literally.
_WORDS: tuple[str, ...] = (
    # frequent English function words
    "the", "and", "to", "of", "a", "in", "that", "is", "was", "for",
    "on", "with", "as", "at", "by", "this", "it", "be", "but", "or",
    "an", "if", "so", "not", "are", "from", "were", "been", "have",
    "has", "had", "do", "did", "does", "will", "would", "should",
    "could", "may", "might", "shall", "can", "must",
    # pronouns & possessives
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "it", "its", "itself",
    # Shakespeare-flavored archaic pronouns/verbs
    "thou", "thee", "thy", "thine", "thyself",
    "ye", "art", "hath", "doth", "dost", "hast",
    "shalt", "wilt", "wouldst", "shouldst", "couldst",
    "wert", "prithee", "methinks", "mayst",
    # Common verbs
    "go", "come", "know", "think", "say", "see", "make", "take",
    "tell", "give", "find", "leave", "bring", "hold", "speak", "stand",
    "hear", "put", "keep", "let", "set", "run", "sit", "work",
    "play", "move", "live", "love", "like", "want", "need", "feel",
    "look", "seem", "show", "turn", "seek", "mean", "call", "shall",
    "lie", "die", "live", "rise", "fall", "fight", "meet", "grant",
    "follow", "beg", "kneel", "swear", "curse", "bless", "forgive",
    # Shakespeare's common lexicon
    "lord", "lady", "master", "mistress", "sir", "madam", "king", "queen",
    "prince", "duke", "earl", "friar", "nurse", "knight", "knave", "fool",
    "gentle", "noble", "valiant", "fair", "sweet", "dear", "kind",
    "true", "false", "good", "bad", "great", "small", "young", "old",
    "new", "dear", "brave", "cruel", "wicked", "holy", "sacred", "mortal",
    "heart", "soul", "mind", "head", "hand", "eye", "face", "blood",
    "life", "death", "love", "hate", "fear", "hope", "dream", "time",
    "day", "night", "morn", "eve", "hour", "year", "world", "heaven",
    "earth", "hell", "sky", "sun", "moon", "star", "sea", "land",
    "flower", "rose", "thorn", "sword", "crown", "throne", "court",
    "castle", "house", "home", "field", "tomb", "grave", "cross",
    # Adverbs / connectors / sentence starters
    "now", "then", "here", "there", "where", "when", "why", "how",
    "who", "what", "which", "whom", "whose",
    "hence", "thence", "whence", "hither", "thither", "whither",
    "before", "after", "between", "within", "without", "against",
    "through", "upon", "unto", "until", "about", "above", "below",
    "anon", "yet", "still", "ever", "never", "always", "oft", "often",
    "perhaps", "perchance", "surely", "truly", "indeed", "forsooth",
    "verily", "belike", "haply", "marry", "faith", "sooth", "troth",
    "alack", "alas", "ay", "aye", "nay", "yea",
    # Contractions (handled with apostrophe in buffer)
    "'tis", "'twas", "'twere", "'twould", "'gainst",
    "o'er", "e'er", "ne'er", "'em",
    # Greeting / address
    "hello", "farewell", "adieu", "welcome", "pray",
    "friend", "foe", "enemy", "brother", "sister", "father", "mother",
    "son", "daughter", "child", "man", "woman", "men", "women",
    "boy", "girl", "lad", "lass", "fellow", "soldier", "servant",
    # Verse fillers
    "oh", "ah", "o",
    # Common bigram starts filled in
    "am", "is", "are",
    # Numbers as words (less likely but present)
    "one", "two", "three",
)

# Build a trie: prefix -> {next_char: count}
# A "count" is a small integer used to derive bias magnitude.
_TRIE: dict[str, dict[str, int]] = {}


def _add_word(word: str) -> None:
    for i in range(len(word) + 1):
        prefix = word[:i]
        _TRIE.setdefault(prefix, {})
        if i < len(word):
            nxt = word[i]
            _TRIE[prefix][nxt] = _TRIE[prefix].get(nxt, 0) + 1
        else:
            # Word terminator: plausible next chars are space, newline,
            # or punctuation.
            for term, weight in (
                (" ", 3), (",", 2), (".", 2), (";", 1), (":", 1),
                ("!", 1), ("?", 1), ("\n", 2), ("'", 1),
            ):
                _TRIE[prefix][term] = _TRIE[prefix].get(term, 0) + weight


for _w in _WORDS:
    _add_word(_w)


def _bias_for(prefix: str) -> list[float] | None:
    """Return a VOCAB_SIZE-length bias vector (or None if no matching
    prefix) that boosts next chars consistent with completing a known
    word. The bias scale grows with the prefix length.
    """
    if prefix not in _TRIE:
        return None
    nexts = _TRIE[prefix]
    if not nexts:
        return None
    import math
    n = len(prefix)
    # Scale: how strongly we believe the prefix implies a known word.
    # Too strong on short prefixes would overfit to our word list; so we
    # ramp up aggressively with length.
    scale = min(0.4 + 0.8 * n, 3.5)
    total = sum(nexts.values())
    vec = [0.0] * VOCAB_SIZE
    # Also apply a negative bump to *all* letters so that unlisted
    # continuations are gently penalized. This makes the trie act as a
    # soft prior toward our vocabulary.
    negative_bump = -0.5 * min(scale, 2.0)
    for ch in "abcdefghijklmnopqrstuvwxyz":
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = negative_bump
    # Boost listed continuations by log-ratio against uniform.
    for ch, w in nexts.items():
        if ch not in VOCAB_INDEX:
            continue
        frac = w / total
        # log-ratio against a baseline of 0.05 — treats frac 0.05 as
        # neutral, larger fracs as boosts.
        bias = scale * math.log((frac + 0.02) / 0.05)
        vec[VOCAB_INDEX[ch]] = bias
        if ch.isalpha():
            up = ch.upper()
            if up in VOCAB_INDEX:
                vec[VOCAB_INDEX[up]] = bias * 0.3
    return vec


# Precompute bias vectors for every prefix in the trie. Keeps predict
# fast: just a dict lookup.
def _precompute() -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for prefix in _TRIE:
        v = _bias_for(prefix)
        if v is not None:
            out[prefix] = v
    return out


PREFIX_BIAS: dict[str, list[float]] = _precompute()


def word_trie_bias(buffer: str) -> list[float] | None:
    """Return a bias vector for the current partial-word buffer, or None."""
    if not buffer:
        return None
    # Try exact prefix; fall back to trimming a leading apostrophe if the
    # buffer starts with one (for contractions).
    if buffer in PREFIX_BIAS:
        return PREFIX_BIAS[buffer]
    # Try with leading apostrophe preserved
    return None
