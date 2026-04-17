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
    # --- frequent English function words ---
    "the", "and", "to", "of", "a", "in", "that", "is", "was", "for",
    "on", "with", "as", "at", "by", "this", "it", "be", "but", "or",
    "an", "if", "so", "not", "are", "from", "were", "been", "have",
    "has", "had", "do", "did", "does", "will", "would", "should",
    "could", "may", "might", "shall", "can", "must", "than", "though",
    "there", "here", "them", "these", "those", "such", "no", "yes",
    "all", "any", "some", "each", "every", "both", "either", "neither",
    "more", "most", "less", "much", "many", "few",
    # --- pronouns & possessives ---
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "it", "its", "itself", "one", "ones",
    # --- Shakespeare archaic pronouns/verbs ---
    "thou", "thee", "thy", "thine", "thyself",
    "ye", "art", "hath", "doth", "dost", "hast",
    "shalt", "wilt", "wouldst", "shouldst", "couldst",
    "didst", "canst", "mayst", "wert", "prithee", "methinks",
    "yond", "yonder", "hither", "thither", "whither",
    "hence", "thence", "whence",
    # --- Common verbs (base + common inflections) ---
    "go", "goes", "going", "gone", "went",
    "come", "comes", "coming", "came",
    "know", "knows", "knew", "known",
    "think", "thinks", "thinking", "thought",
    "say", "says", "said", "saying",
    "see", "sees", "seen", "saw", "seeing",
    "make", "makes", "made", "making",
    "take", "takes", "took", "taken", "taking",
    "tell", "tells", "told", "telling",
    "give", "gives", "gave", "given", "giving",
    "find", "finds", "found", "finding",
    "leave", "leaves", "left", "leaving",
    "bring", "brings", "brought", "bringing",
    "hold", "holds", "held", "holding",
    "speak", "speaks", "spoke", "spoken", "speaking",
    "stand", "stands", "stood", "standing",
    "hear", "hears", "heard", "hearing",
    "keep", "keeps", "kept", "keeping",
    "let", "lets", "letting",
    "set", "sets", "setting",
    "run", "runs", "ran", "running",
    "sit", "sits", "sat", "sitting",
    "work", "works", "worked", "working",
    "play", "plays", "played", "playing",
    "move", "moves", "moved", "moving",
    "live", "lives", "lived", "living",
    "love", "loves", "loved", "loving", "lovely",
    "like", "likes", "liked", "likely",
    "want", "wants", "wanted",
    "need", "needs", "needed",
    "feel", "feels", "felt", "feeling",
    "look", "looks", "looked", "looking",
    "seem", "seems", "seemed",
    "show", "shows", "showed", "shown", "showing",
    "turn", "turns", "turned", "turning",
    "seek", "seeks", "sought", "seeking",
    "mean", "means", "meant", "meaning",
    "call", "calls", "called", "calling",
    "lie", "lies", "lay", "lying",
    "die", "dies", "died", "dying",
    "rise", "rises", "rose", "risen", "rising",
    "fall", "falls", "fell", "fallen", "falling",
    "fight", "fights", "fought", "fighting",
    "meet", "meets", "met", "meeting",
    "grant", "grants", "granted",
    "follow", "follows", "followed", "following",
    "beg", "begs", "begged", "begging",
    "kneel", "kneels", "knelt",
    "swear", "swears", "swore", "sworn", "swearing",
    "curse", "curses", "cursed", "cursing",
    "bless", "blesses", "blessed", "blessing",
    "forgive", "forgives", "forgave", "forgiven",
    "lose", "loses", "lost", "losing",
    "win", "wins", "won", "winning",
    "kill", "kills", "killed", "killing",
    "slay", "slays", "slew", "slain", "slaying",
    "weep", "weeps", "wept", "weeping",
    "cry", "cries", "cried", "crying",
    "laugh", "laughs", "laughed", "laughing",
    "sigh", "sighs", "sighed", "sighing",
    "sleep", "sleeps", "slept", "sleeping",
    "wake", "wakes", "woke", "waking",
    "dream", "dreams", "dreamed", "dreaming",
    "remember", "remembers", "remembered",
    "forget", "forgets", "forgot", "forgetting",
    "write", "writes", "wrote", "written",
    "read", "reads", "reading",
    "hope", "hopes", "hoped",
    "wish", "wishes", "wished",
    "dare", "dares", "dared",
    "fear", "fears", "feared",
    "deny", "denies", "denied",
    "confess", "confesses", "confessed",
    "marry", "marries", "married",
    "part", "parts", "parted",
    "depart", "departs", "departed",
    "return", "returns", "returned",
    "arrive", "arrives", "arrived",
    "enter", "enters", "entered", "entering",
    "exit", "exits", "exited",
    "pass", "passes", "passed", "passing",
    "let", "let's",
    # --- Shakespeare lexicon ---
    "lord", "lords", "lady", "ladies", "master", "masters",
    "mistress", "sir", "madam", "king", "kings", "queen", "queens",
    "prince", "princes", "princess", "duke", "dukes", "duchess",
    "earl", "earls", "count", "friar", "friars", "nurse", "nurses",
    "knight", "knights", "knave", "knaves", "fool", "fools",
    "villain", "villains", "rogue", "rogues", "wretch", "coward",
    "tyrant", "traitor", "murderer", "ghost", "ghosts", "spirit", "spirits",
    "witch", "witches", "god", "gods", "angel", "angels",
    "devil", "devils",
    "gentle", "gentleman", "gentlemen", "noble", "noblest", "nobler",
    "valiant", "fair", "fairest", "fairer", "sweet", "sweetest",
    "dear", "dearest", "dearer", "kind", "kindest", "kinder",
    "true", "truth", "false", "falsehood", "good", "goodness",
    "bad", "wicked", "great", "greater", "greatest", "small", "smaller",
    "young", "younger", "old", "older", "elder", "eldest",
    "new", "brave", "bravest", "braver", "cruel", "cruelty",
    "holy", "holiest", "sacred", "mortal", "immortal", "divine",
    "hollow", "royal", "humble", "honest", "honour", "honoured",
    "disgrace", "shame", "ashamed", "shameful",
    # --- body/nature ---
    "heart", "hearts", "soul", "souls", "mind", "minds",
    "head", "heads", "hand", "hands", "eye", "eyes",
    "face", "faces", "blood", "tongue", "tongues", "lips",
    "arm", "arms", "foot", "feet", "hair", "breath",
    "tear", "tears", "smile", "smiles", "voice", "voices",
    "life", "lives", "death", "deaths",
    "love", "hate", "fear", "hope", "dream", "dreams",
    "grief", "joy", "sorrow", "sorrows", "peace", "war",
    "time", "times", "day", "days", "night", "nights",
    "morn", "morning", "eve", "evening", "hour", "hours",
    "year", "years", "month", "months", "week", "weeks",
    "world", "worlds", "heaven", "heavens", "earth", "earths",
    "hell", "sky", "skies", "sun", "moon", "star", "stars",
    "sea", "seas", "land", "lands", "fire", "fires", "air",
    "water", "waters", "river", "rivers", "wind", "winds",
    "storm", "storms", "rain", "snow", "light", "lights",
    "dark", "darkness", "shadow", "shadows",
    "flower", "flowers", "rose", "roses", "thorn", "thorns",
    "sword", "swords", "crown", "crowns", "throne", "thrones",
    "court", "courts", "castle", "castles",
    "house", "houses", "home", "homes", "field", "fields",
    "tomb", "tombs", "grave", "graves", "cross", "crosses",
    "gold", "silver", "jewel", "jewels", "treasure",
    # --- adverbs / connectors / sentence starters ---
    "now", "then", "here", "there", "where", "when", "why", "how",
    "who", "what", "which", "whom", "whose",
    "before", "after", "between", "within", "without", "against",
    "through", "throughout", "upon", "unto", "until", "till",
    "about", "above", "below", "beneath", "beyond", "around",
    "among", "amongst", "amidst", "amid",
    "anon", "yet", "still", "ever", "never", "always",
    "oft", "often", "seldom", "rarely", "sometimes",
    "perhaps", "perchance", "surely", "truly", "indeed", "forsooth",
    "verily", "belike", "haply", "marry", "faith", "sooth", "troth",
    "alack", "alas", "ay", "aye", "nay", "yea", "well", "why",
    "fie", "pish", "tut", "hark", "lo", "behold",
    "therefore", "thereby", "thereof", "thereto", "herein", "herein",
    "whereby", "whereof", "wherein", "wheresoever",
    # --- Contractions (handled with apostrophe in buffer) ---
    "'tis", "'twas", "'twere", "'twould", "'gainst", "'tisn't",
    "o'er", "e'er", "ne'er", "'em", "'mongst",
    "i'll", "you'll", "he'll", "she'll", "we'll", "they'll",
    "i've", "you've", "we've", "they've",
    "i'd", "you'd", "he'd", "she'd", "we'd", "they'd",
    "i'm", "you're", "we're", "they're", "he's", "she's", "it's",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't",
    "can't", "cannot", "couldn't", "isn't", "aren't", "wasn't",
    "weren't", "hasn't", "haven't", "hadn't",
    "'s", "'d", "'ll", "'ve", "'re", "'t", "'m",
    # --- Address / exclamation ---
    "hello", "farewell", "adieu", "welcome", "pray", "prithee",
    "friend", "friends", "foe", "foes", "enemy", "enemies",
    "brother", "brothers", "sister", "sisters",
    "father", "fathers", "mother", "mothers",
    "son", "sons", "daughter", "daughters",
    "child", "children", "man", "men", "woman", "women",
    "boy", "boys", "girl", "girls",
    "lad", "lads", "lass", "lasses",
    "fellow", "fellows", "soldier", "soldiers",
    "servant", "servants", "page", "pages",
    "messenger", "messengers", "herald",
    # --- Verse fillers ---
    "oh", "ah", "o",
    # --- Numbers ---
    "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "hundred", "thousand",
    "first", "second", "third", "last",
    # --- Adjectives ---
    "beautiful", "ugly", "tall", "short", "long", "quick", "slow",
    "strong", "weak", "rich", "poor", "happy", "sad", "angry",
    "gentle", "rough", "hard", "soft", "warm", "cold",
    "full", "empty", "deep", "high", "low", "near", "far",
    "wise", "foolish", "mad", "sane", "bold", "timid",
    "pale", "bright", "dull", "keen", "sharp", "blunt",
    # --- Misc useful ---
    "yes", "nothing", "something", "anything", "everything",
    "none", "nobody", "somebody", "anybody", "everybody",
    "am", "be", "been", "being",
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
