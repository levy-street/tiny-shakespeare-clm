"""Topical-coherence bias layer.

Reads `state.content_words` — the rolling tuple of the 4 most-recent
content words — and, at word-start positions, biases the first letter
of the next word toward starter letters of words belonging to a
dominant topical cluster.

Shakespearean scenes have strong lexical coherence: once "blood",
"death", "sword" appear, more dark lexicon follows; once "love",
"sweet", "heart" appear, more tender lexicon follows; once "king",
"lord", "crown" appear, more courtly lexicon follows. Function-word
scaffolding ("the", "of", "that", "my", "I") does NOT break this
coherence — it interleaves it. The content_words tuple lets us see
past the scaffolding.

Scoring: each of the 4 slots contributes a decayed weight to the
cluster its word belongs to (most-recent: 1.0, next: 0.75, 0.5, 0.25).
If the highest cluster score exceeds THRESHOLD, we apply that
cluster's starter-letter bias, scaled by (score - THRESHOLD).

All knowledge here comes from prior knowledge of Shakespearean
register. No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# --- Cluster membership ---
# Each cluster is a frozenset of lowercased word forms (including
# common inflections) that are strongly associated with that register.

_DARK = frozenset({
    "blood", "bloody", "bloods", "bleed", "bleeds", "bleeding", "bled",
    "death", "deaths", "deadly", "die", "dies", "died", "dying", "dead",
    "kill", "kills", "killed", "killing",
    "slay", "slays", "slew", "slain", "slaying",
    "murder", "murders", "murdered", "murderer", "murderous",
    "grave", "graves", "tomb", "tombs", "corpse", "corpses",
    "wound", "wounds", "wounded", "wounding",
    "sword", "swords", "dagger", "daggers", "knife", "knives",
    "war", "wars", "battle", "battles", "fight", "fights", "fighting", "fought",
    "foe", "foes", "enemy", "enemies", "strike", "strikes", "struck",
    "hate", "hates", "hated", "hating", "hatred",
    "grief", "griefs", "grieve", "grieves", "grieved", "grieving", "grievous",
    "sorrow", "sorrows", "sorrowful", "sad", "sadness",
    "tears", "weep", "weeps", "weeping", "wept",
    "mourn", "mourns", "mourning", "mourned", "mourner",
    "pain", "pains", "painful", "agony", "anguish",
    "hell", "devil", "devils", "damn", "damned", "damnation",
    "fire", "flame", "flames", "burn", "burns", "burning", "burned", "burnt",
    "rage", "raging", "raged", "fury", "furious", "wrath",
    "fall", "falls", "fallen", "falling", "fell",
    "fear", "fears", "fearful", "feared", "fearing", "afraid",
    "dread", "dreads", "dreadful", "dreaded",
    "dark", "darkness", "darker", "darkest", "night", "nights",
    "cold", "colder", "coldest", "pale", "paler", "palest",
    "grim", "grimmer", "grim",
    "curse", "curses", "cursed", "cursing",
    "treason", "traitor", "traitors", "traitorous",
    "poison", "poisons", "poisoned", "poisoning",
    "vengeance", "revenge", "avenge", "avenged",
    "tyrant", "tyranny", "tyrants",
    "shadow", "shadows", "shadowy",
    "bone", "bones", "skull", "skulls",
})

_LIGHT = frozenset({
    "love", "loves", "loved", "loving", "lover", "lovers", "lovely",
    "heart", "hearts", "hearted",
    "sweet", "sweets", "sweeter", "sweetest", "sweetly", "sweetness",
    "kiss", "kisses", "kissed", "kissing",
    "fair", "fairer", "fairest", "fairly", "fairness",
    "beauty", "beauties", "beautiful", "beauteous",
    "beloved", "darling", "dear", "dearer", "dearest", "dearly",
    "tender", "tenderly", "tenderness",
    "gentle", "gentler", "gentlest", "gently", "gentleness",
    "joy", "joys", "joyful", "joyous", "enjoy", "enjoys", "enjoyed",
    "bliss", "blissful", "blessed", "blessing", "blessings",
    "grace", "graces", "graceful", "gracious", "graciously",
    "smile", "smiles", "smiled", "smiling",
    "delight", "delights", "delighted", "delightful",
    "bright", "brighter", "brightest", "brightly", "brightness",
    "light", "lights", "lighted", "lighten",
    "peace", "peaceful", "peacefully",
    "hope", "hopes", "hoped", "hoping", "hopeful",
    "warm", "warmth", "warmer", "warmest",
    "mild", "milder", "mildest",
    "soft", "softer", "softest", "softly", "softness",
    "friend", "friends", "friendly", "friendship",
    "happy", "happier", "happiest", "happiness", "happily",
    "pleasure", "pleasures", "pleasant", "pleased", "pleasing", "please",
    "virtue", "virtues", "virtuous",
    "honest", "honesty", "honestly",
    "faithful", "faith", "faithfully",
    "true", "truth", "truly", "truest",
    "good", "goodness", "goodly",
    "kind", "kindness", "kindly", "kinder", "kindest",
    "merry", "merrier", "merriest", "merriment",
    "young", "younger", "youngest", "youth", "youthful",
    "spring", "flower", "flowers", "rose", "roses",
    "sing", "sings", "sang", "sung", "singing", "song", "songs",
    "dance", "dances", "danced", "dancing",
})

_ROYAL = frozenset({
    "king", "kings", "kingly", "kingdom", "kingdoms",
    "queen", "queens", "queenly",
    "lord", "lords", "lordship", "lordships",
    "prince", "princes", "princely", "princess", "princesses",
    "duke", "dukes", "duchess", "duchy",
    "earl", "earls",
    "crown", "crowns", "crowned", "crowning",
    "throne", "thrones",
    "court", "courts", "courtly", "courtier", "courtiers",
    "royal", "royals", "royalty",
    "realm", "realms", "majesty", "majesties",
    "noble", "nobles", "nobly", "nobility", "nobleman", "noblemen",
    "honour", "honours", "honoured", "honour", "honourable",
    "honor", "honors", "honored", "honorable",
    "sir", "sirs", "madam", "madams", "lady", "ladies",
    "knight", "knights", "knighthood",
    "monarch", "monarchs",
    "sceptre", "sceptres", "scepter",
    "castle", "castles", "palace", "palaces", "throne",
    "ruler", "rulers", "rule", "rules", "ruled", "ruling",
    "sovereign", "sovereigns", "sovereignty",
    "subject", "subjects",
    "liege", "liegeman",
    "master", "masters", "mastery",
    "lieutenant", "captain", "captains",
    "duke", "duchess",
})

_WORD_TO_CLUSTER: dict[str, int] = {}
for w in _DARK:
    _WORD_TO_CLUSTER[w] = 0
for w in _LIGHT:
    _WORD_TO_CLUSTER[w] = 1
for w in _ROYAL:
    _WORD_TO_CLUSTER[w] = 2

_CLUSTERS: tuple[frozenset[str], ...] = (_DARK, _LIGHT, _ROYAL)

N_CLUSTERS = 3

# --- Per-cluster starter-letter biases ---
# Letters most likely to begin words within each cluster. Values are
# small; they get scaled by (cluster_score - threshold) at runtime.
_CLUSTER_STARTERS: list[dict[str, float]] = [
    # DARK: d, b, s, w, f, g, k, m, h, c, p, r, t
    {
        "d": 1.0,  # death, die, dead, dark, dread, damn, devil, dagger, despair
        "b": 0.9,  # blood, battle, bleed, burn, bone, bane
        "s": 0.9,  # sword, slay, slain, sorrow, sad, shadow, struck
        "w": 0.85, # wound, war, weep, wept, wrath
        "f": 0.85, # foe, fight, fear, fire, fall, fury, furious, fell
        "g": 0.75, # grief, grave, grim, grieve
        "k": 0.6,  # kill, knife
        "m": 0.65, # murder, mourn, misery
        "h": 0.55, # hate, hell, hatred, horror
        "c": 0.55, # corpse, curse, cold, corpse
        "p": 0.55, # pain, pale, poison, plague
        "r": 0.55, # rage, revenge, ruin
        "t": 0.55, # tears, tomb, treason, traitor, tyranny
        "a": 0.4,  # agony, anguish, afraid
    },
    # LIGHT: l, h, s, k, f, b, d, t, g, j, p, w, m, y
    {
        "l": 1.0,  # love, lovely, light
        "h": 0.85, # heart, hope, happy, honest
        "s": 0.85, # sweet, smile, soft, song, sing
        "k": 0.6,  # kiss, kind, kindness
        "f": 0.85, # fair, faithful, friend, flower
        "b": 0.8,  # beauty, beloved, bright, bliss, blessed
        "d": 0.7,  # dear, darling, delight, dance
        "t": 0.7,  # tender, true, truth
        "g": 0.85, # gentle, grace, gracious, good, goodness, glad
        "j": 0.6,  # joy, joyful
        "p": 0.65, # peace, pleasure, please
        "w": 0.55, # warm, warmth
        "m": 0.55, # mild, merry, merriment
        "y": 0.45, # young, youth, youthful
        "v": 0.55, # virtue, virtuous
    },
    # ROYAL: k, q, l, p, t, c, r, d, e, n, m, h, s
    {
        "k": 1.0,  # king, kingdom, knight
        "q": 0.95, # queen
        "l": 0.95, # lord, lordship, lady, liege, lieutenant
        "p": 0.85, # prince, princess, palace
        "t": 0.75, # throne
        "c": 0.85, # crown, court, castle, courtly, courtier, captain
        "r": 0.8,  # royal, realm, ruler, rule
        "d": 0.7,  # duke, duchess, duchy
        "e": 0.55, # earl
        "n": 0.75, # noble, nobility
        "m": 0.8,  # majesty, monarch, master, madam
        "h": 0.7,  # honour, honoured, honourable
        "s": 0.75, # sir, sceptre, sovereign, subject
    },
]


def _build_cluster_vectors() -> list[list[float]]:
    vectors: list[list[float]] = []
    for starters in _CLUSTER_STARTERS:
        vec = [0.0] * VOCAB_SIZE
        for ch, w in starters.items():
            lo = ch
            up = ch.upper()
            if lo in VOCAB_INDEX:
                vec[VOCAB_INDEX[lo]] = w
            if up in VOCAB_INDEX:
                # Capital starter at sentence-start also gets the bias
                # (proportionally weaker to reflect capitals being rarer).
                vec[VOCAB_INDEX[up]] = w * 0.6
        # Apply a small negative to other letters so the cluster actively
        # tilts. Scale so the net sum is ~0 (informational, not mass-shift).
        n_pos = sum(1 for v in vec if v > 0.0)
        if n_pos > 0:
            pos_mass = sum(v for v in vec if v > 0.0)
            neg_per_letter = -pos_mass / 26.0 * 0.15
            for ch in "abcdefghijklmnopqrstuvwxyz":
                if ch in VOCAB_INDEX and vec[VOCAB_INDEX[ch]] == 0.0:
                    vec[VOCAB_INDEX[ch]] = neg_per_letter
        vectors.append(vec)
    return vectors


_CLUSTER_VECS: list[list[float]] = _build_cluster_vectors()


# Decay weights for content_words[0..7] (most-recent first).
_SLOT_WEIGHTS: tuple[float, ...] = (
    1.00, 0.75, 0.55, 0.40, 0.28, 0.18, 0.10, 0.06,
)

# If top cluster score < THRESHOLD: no bias.
# Above THRESHOLD, scale = (score - THRESHOLD) * GAIN, capped at MAX.
_THRESHOLD = 0.55
_GAIN = 0.45
_MAX_SCALE = 0.75


def _dominant_cluster(
    content_words: tuple[str, ...]
) -> tuple[int, float] | None:
    """Return (cluster_id, score) if a topical cluster dominates the
    recent content-words window above _THRESHOLD; else None.
    """
    if not content_words:
        return None
    scores = [0.0] * N_CLUSTERS
    for i, w in enumerate(content_words[:8]):
        cid = _WORD_TO_CLUSTER.get(w)
        if cid is None:
            continue
        scores[cid] += _SLOT_WEIGHTS[i]
    top = 0
    top_score = scores[0]
    for cid in range(1, N_CLUSTERS):
        if scores[cid] > top_score:
            top = cid
            top_score = scores[cid]
    if top_score < _THRESHOLD:
        return None
    return top, top_score


def topic_bias(content_words: tuple[str, ...]) -> list[float] | None:
    """Compute a first-letter bias vector from recent content words.

    Returns None if no clear topical signal is present.
    """
    dom = _dominant_cluster(content_words)
    if dom is None:
        return None
    top, top_score = dom
    scale = min((top_score - _THRESHOLD) * _GAIN, _MAX_SCALE)
    if scale <= 0.0:
        return None
    vec = _CLUSTER_VECS[top]
    return [scale * v for v in vec]


# --- Midword consumer ---
# For each (cluster, buffer-prefix) we precompute a tiny map of
# "next letters that continue buffer into a cluster word" with raw
# weights. Weights are small; final bias scales by activation score.
#
# Precomputing by buffer avoids iterating the cluster on every token.

_MIDWORD_MAX_PREFIX = 8  # cap precomputed prefix length

def _build_midword_maps() -> list[dict[str, dict[str, float]]]:
    maps: list[dict[str, dict[str, float]]] = []
    for cluster in _CLUSTERS:
        prefix_map: dict[str, dict[str, float]] = {}
        for w in cluster:
            # Skip very short words (no midword context) and very long.
            if len(w) < 3:
                continue
            for i in range(1, min(len(w), _MIDWORD_MAX_PREFIX + 1)):
                pref = w[:i]
                nxt = w[i] if i < len(w) else " "
                if nxt.isalpha():
                    nxt = nxt.lower()
                prefix_map.setdefault(pref, {})
                # Accumulate weight — prefixes shared by multiple
                # cluster words get stronger next-letter signal on
                # their shared continuation.
                prefix_map[pref][nxt] = prefix_map[pref].get(nxt, 0.0) + 1.0
        # Normalize per-prefix so large clusters don't dominate
        # (each prefix's total next-letter weight is 1.0).
        for pref, nmap in prefix_map.items():
            total = sum(nmap.values())
            if total > 0:
                for k in nmap:
                    nmap[k] = nmap[k] / total
        maps.append(prefix_map)
    return maps


_MIDWORD_MAPS: list[dict[str, dict[str, float]]] = _build_midword_maps()


_MIDWORD_THRESHOLD = 0.40  # slightly softer than word-start
_MIDWORD_GAIN = 0.35
_MIDWORD_MAX_SCALE = 0.60


def content_repeat_bias(
    buffer: str, content_words: tuple[str, ...]
) -> list[float] | None:
    """At mid-word position, if the current buffer is a strict prefix of
    a word in the recent content_words window, boost the letter that
    would continue buffer toward that word. Captures Shakespeare's
    motif-repetition texture ("Never, never, never, never, never";
    "Blood will have blood"; lexical echoes within a scene).

    Applies equally to all 4 slots with decaying weight — the most-
    recent slot contributes most. Returns None if no match.
    """
    if not buffer or not content_words:
        return None
    if len(buffer) < 2:
        return None
    vec = [0.0] * VOCAB_SIZE
    hit = False
    # Longer buffer = more specific match = stronger boost
    # (2 chars = 1.00x, 3 = 1.25x, 4 = 1.45x, 5+ = 1.6x).
    blen = len(buffer)
    if blen <= 2:
        len_scale = 1.00
    elif blen == 3:
        len_scale = 1.90
    elif blen == 4:
        len_scale = 2.70
    elif blen == 5:
        len_scale = 3.30
    else:
        len_scale = 3.70
    # Slot 0 is the MOST-recent completed content word — it's the one
    # we literally just said. Echoing it back mid-word at high weight
    # creates "insatiate, insatiate" / "there there there" stuck
    # loops. We want motif resonance (slots 1-3) not just-said repeat.
    slot_weights = (0.60, 1.05, 0.55, 0.25, 0.12, 0.06, 0.03, 0.02)
    for i, w in enumerate(content_words[:8]):
        if (
            len(w) > len(buffer)
            and w.startswith(buffer)
        ):
            nxt = w[len(buffer)]
            if nxt in VOCAB_INDEX:
                weight = slot_weights[i] * len_scale
                vec[VOCAB_INDEX[nxt]] += weight
                hit = True
    if not hit:
        return None
    return vec


def topic_midword_bias(
    buffer: str, content_words: tuple[str, ...]
) -> list[float] | None:
    """At mid-word position, if a topical cluster dominates, bias the
    next letter toward letters that continue buffer into one of the
    cluster's words. Returns None if no signal applies.
    """
    if not buffer or not content_words:
        return None
    if len(buffer) > _MIDWORD_MAX_PREFIX:
        return None
    dom = _dominant_cluster(content_words)
    if dom is None:
        return None
    top, top_score = dom
    if top_score < _MIDWORD_THRESHOLD:
        return None
    nmap = _MIDWORD_MAPS[top].get(buffer)
    if not nmap:
        return None
    scale = min(
        (top_score - _MIDWORD_THRESHOLD) * _MIDWORD_GAIN,
        _MIDWORD_MAX_SCALE,
    )
    if scale <= 0.0:
        return None
    vec = [0.0] * VOCAB_SIZE
    for nxt, w in nmap.items():
        if nxt in VOCAB_INDEX:
            vec[VOCAB_INDEX[nxt]] += scale * w * 2.5
    return vec
