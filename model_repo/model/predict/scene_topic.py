"""Scene-topic predict layer.

Reads `state.scene_topics` (an 8-element activation vector over WAR,
LOVE, DEATH, ROYALTY, NATURE, BODY, FAITH, FORTUNE) and returns a
word-start first-letter bias vector pulling toward the dominant
cluster's characteristic starter letters.

Gating:
  - fires only at word-start outside a speaker label
  - requires the top cluster's activation to be strictly greater than
    a minimum threshold (0.60) AND strictly greater than the second-
    place cluster by at least a margin (0.30) — to avoid biasing when
    topics are ambiguous or freshly decayed.

The weights capture characteristic starter-letter distributions per
cluster. Weight scale is modulated by the top cluster's activation.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE
from ..pipeline.topic_tracker import (
    TOPIC_BODY,
    TOPIC_DEATH,
    TOPIC_FAITH,
    TOPIC_FORTUNE,
    TOPIC_LOVE,
    TOPIC_NATURE,
    TOPIC_ROYALTY,
    TOPIC_WAR,
)


# For each topic cluster, the characteristic starter-letter distribution
# of words that belong to that cluster. Hand-crafted from knowledge.
# Weights sum to ~1.0 per cluster (before scaling).
_TOPIC_STARTERS: dict[int, dict[str, float]] = {
    TOPIC_WAR: {
        # sword, slain, soldier, spear, siege, strike, steel, shield
        "s": 0.26,
        # war, wound, weapon
        "w": 0.14,
        # foe, fight, field, fought
        "f": 0.14,
        # battle, blood, banner, bow, blade
        "b": 0.16,
        # arms, army, arrow
        "a": 0.10,
        # conquer, captain, cannon
        "c": 0.10,
        # drum, dagger, defeat
        "d": 0.08,
        # helm
        "h": 0.04,
    },
    TOPIC_LOVE: {
        # love, lover, lip
        "l": 0.20,
        # heart, honey
        "h": 0.14,
        # sweet, sigh
        "s": 0.14,
        # kiss
        "k": 0.10,
        # fair
        "f": 0.12,
        # dear, darling
        "d": 0.12,
        # rose, romance
        "r": 0.10,
        # beauty, bride, beloved
        "b": 0.12,
        # tender
        "t": 0.06,
    },
    TOPIC_DEATH: {
        # death, die, dead, dust, doom, dying
        "d": 0.28,
        # grave, ghost
        "g": 0.14,
        # tomb
        "t": 0.10,
        # mourn, murder, mortal
        "m": 0.14,
        # soul (overlap FAITH)
        "s": 0.10,
        # pale, perish
        "p": 0.10,
        # kill
        "k": 0.08,
        # corpse, coffin
        "c": 0.08,
        # bury
        "b": 0.06,
    },
    TOPIC_ROYALTY: {
        # king, kingdom, knight
        "k": 0.16,
        # queen
        "q": 0.06,
        # crown, court
        "c": 0.14,
        # throne
        "t": 0.08,
        # prince, princess
        "p": 0.08,
        # duke, duchess
        "d": 0.06,
        # royal, realm, reign
        "r": 0.12,
        # lord, lady
        "l": 0.14,
        # noble
        "n": 0.08,
        # majesty, monarch
        "m": 0.08,
        # sovereign, sceptre, subject
        "s": 0.10,
        # empress, emperor
        "e": 0.06,
        # grace
        "g": 0.06,
    },
    TOPIC_NATURE: {
        # sun, stars, sea, sky, shore, snow, storm, spring, summer
        "s": 0.22,
        # moon, morn, mountain
        "m": 0.10,
        # wind, wave, winter, wood
        "w": 0.14,
        # rain, river
        "r": 0.10,
        # flower, field, forest, fire
        "f": 0.14,
        # tree, thunder
        "t": 0.08,
        # night
        "n": 0.06,
        # day
        "d": 0.06,
        # bird
        "b": 0.06,
        # earth
        "e": 0.06,
        # cloud
        "c": 0.06,
        # leaf, lightning
        "l": 0.06,
        # garden
        "g": 0.04,
    },
    TOPIC_BODY: {
        # hand, head, hair, heart
        "h": 0.22,
        # eye, ear
        "e": 0.12,
        # face, foot, flesh
        "f": 0.10,
        # lip
        "l": 0.08,
        # tongue, tears, throat
        "t": 0.14,
        # arm
        "a": 0.06,
        # breast, bone, brow, back
        "b": 0.14,
        # neck
        "n": 0.04,
        # skin
        "s": 0.06,
        # cheek
        "c": 0.06,
        # bosom (already b)
    },
    TOPIC_FAITH: {
        # god, grace
        "g": 0.10,
        # heaven, hell, holy
        "h": 0.18,
        # pray, prayer, priest, prophet
        "p": 0.14,
        # sin, soul, sacred, spirit, saint
        "s": 0.20,
        # mercy, mass
        "m": 0.08,
        # angel
        "a": 0.08,
        # devil, damn
        "d": 0.10,
        # faith
        "f": 0.08,
        # curse, church
        "c": 0.08,
        # bless, blessed
        "b": 0.06,
    },
    TOPIC_FORTUNE: {
        # fate, fortune, future
        "f": 0.24,
        # destiny, doom
        "d": 0.14,
        # hap, haply, hour
        "h": 0.14,
        # time
        "t": 0.16,
        # chance
        "c": 0.10,
        # luck
        "l": 0.06,
        # providence
        "p": 0.08,
        # wheel
        "w": 0.06,
        # star, stars
        "s": 0.10,
    },
}

# Scale of the starter bias applied to the dominant cluster. The top
# cluster's activation (clipped to [0, 4]) is normalized by 4.0 and
# then multiplied by this scale to determine final weight magnitude.
_TOP_SCALE = 0.30

# Minimum activation in top cluster to fire.
_MIN_TOP = 0.80
# Minimum margin (top - second) to fire.
_MIN_MARGIN = 0.35


# Mid-word letter biases per topic. These are common letter
# continuations in words of each cluster. Applied at mid-word positions
# only when the buffer is off-trie (on-trie the word_trie itself
# handles continuation) and when the top cluster is clearly dominant.
# Weights are gentler than the start-bias.
_TOPIC_MIDWORD: dict[int, dict[str, float]] = {
    # war — sword, slain, battle, blood: heavy on -rd, -dd, -tt, -ain
    TOPIC_WAR: {"o": 0.10, "r": 0.08, "l": 0.06, "d": 0.06, "a": 0.06},
    # love — kiss, sweet, heart, dear, fair: heavy on vowels, -rt, -ee
    TOPIC_LOVE: {"e": 0.10, "a": 0.08, "r": 0.06, "t": 0.06, "i": 0.04},
    # death — dying, grave, tomb, dead, corpse: heavy on -d, -ave
    TOPIC_DEATH: {"e": 0.08, "a": 0.06, "d": 0.06, "i": 0.04, "r": 0.04},
    # royalty — king, crown, throne: heavy on -ng, -wn, -ne
    TOPIC_ROYALTY: {"i": 0.08, "o": 0.08, "n": 0.06, "r": 0.04, "a": 0.04},
    # nature — stars, storm, sea, sky, sun: varied vowels + -n endings
    TOPIC_NATURE: {"o": 0.08, "a": 0.08, "r": 0.06, "n": 0.04, "e": 0.06},
    # body — hand, head, heart, eye: short words
    TOPIC_BODY: {"a": 0.08, "e": 0.08, "n": 0.04, "r": 0.04, "d": 0.04},
    # faith — soul, heaven, sacred, pray, spirit: -l, -en, -ay
    TOPIC_FAITH: {"a": 0.08, "e": 0.08, "i": 0.04, "r": 0.04, "n": 0.04},
    # fortune — fate, doom, chance, time, hour: -e/-te/-oo-
    TOPIC_FORTUNE: {"a": 0.08, "e": 0.06, "o": 0.06, "u": 0.04, "t": 0.04},
}


def scene_topic_midword_bias(
    scene_topics: tuple[float, ...],
    speaker_label_state: int,
    letter_run_len: int,
    letters_off_trie: int,
    on_word_trie: bool,
) -> list[float] | None:
    """Return a mid-word letter bias when a topic is dominant and we
    are off the word_trie (so the trie itself isn't guiding us).
    """
    if speaker_label_state != 0:
        return None
    if letter_run_len < 2:
        return None
    if on_word_trie:
        # Trust the trie when on-trie.
        return None
    if letters_off_trie < 1:
        return None
    if not scene_topics:
        return None

    top_val = -1.0
    top_idx = -1
    second_val = -1.0
    for i, v in enumerate(scene_topics):
        if v > top_val:
            second_val = top_val
            top_val = v
            top_idx = i
        elif v > second_val:
            second_val = v

    # Mid-word uses a looser threshold than start bias — even a modestly
    # active topic should subtly shape off-trie continuations.
    if top_idx < 0 or top_val < 0.40:
        return None
    if (top_val - second_val) < 0.20:
        return None

    weights = _TOPIC_MIDWORD.get(top_idx)
    if weights is None:
        return None
    norm = min(top_val, 4.0) / 4.0
    # Escalate with letters_off_trie (the deeper we are off-trie, the
    # more we want to pull toward topic-plausible continuations).
    escalate = min(letters_off_trie / 3.0, 1.0) * 0.5 + 0.5
    scale = norm * escalate * 0.35

    vec = [0.0] * VOCAB_SIZE
    for ch, w in weights.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
    return vec


def scene_topic_start_bias(
    scene_topics: tuple[float, ...],
    speaker_label_state: int,
) -> list[float] | None:
    """Return a word-start first-letter bias for the dominant topic.

    Returns None (no-op) when no cluster is clearly dominant, or when
    we're in a speaker-label.
    """
    if speaker_label_state != 0:
        return None
    if not scene_topics:
        return None

    # Find top and second-place.
    top_val = -1.0
    top_idx = -1
    second_val = -1.0
    for i, v in enumerate(scene_topics):
        if v > top_val:
            second_val = top_val
            top_val = v
            top_idx = i
        elif v > second_val:
            second_val = v

    if top_idx < 0:
        return None
    if top_val < _MIN_TOP:
        return None
    if (top_val - second_val) < _MIN_MARGIN:
        return None

    weights = _TOPIC_STARTERS.get(top_idx)
    if weights is None:
        return None

    # Clamp top_val to [0, 4] and normalize.
    norm = min(top_val, 4.0) / 4.0
    scale = norm * _TOP_SCALE

    vec = [0.0] * VOCAB_SIZE
    for ch, w in weights.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
        # Also bias the capital form at smaller weight (useful at
        # sentence-start positions).
        up = ch.upper()
        uidx = VOCAB_INDEX.get(up)
        if uidx is not None:
            vec[uidx] += w * scale * 0.4
    return vec
