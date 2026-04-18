"""Scene-topic tracker pipeline stage.

Maintains a rolling 8-dimensional non-negative activation vector
over semantic clusters. Updated at word completion by looking the
completed word up in WORD_TO_TOPICS (a hand-crafted dictionary from
prior knowledge of Shakespeare's lexicon — no corpus statistics).

Clusters:
  0 WAR, 1 LOVE, 2 DEATH, 3 ROYALTY,
  4 NATURE, 5 BODY, 6 FAITH, 7 FORTUNE

Update rule at each completed word:
  1. Decay all activations by 0.90 (multiplicative).
  2. If the word maps to one or more clusters in WORD_TO_TOPICS,
     add +1.0 to each matched cluster (a word can belong to multiple;
     e.g. "soul" fires both DEATH and FAITH).
  3. Cap each cluster at 4.0.

Speaker-turn boundary (consecutive_newlines >= 2 and last char == "\n"):
  Multiply all activations by 0.35 — the new speaker may carry residual
  topical mood but mostly starts fresh.

Sentence-end punctuation: no special handling (topics persist across
sentences within a turn — that's the whole point).

Runs after update_pos (so last_word_pos and just_finished_word are fresh).
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


NUM_TOPICS = 8

TOPIC_WAR = 0
TOPIC_LOVE = 1
TOPIC_DEATH = 2
TOPIC_ROYALTY = 3
TOPIC_NATURE = 4
TOPIC_BODY = 5
TOPIC_FAITH = 6
TOPIC_FORTUNE = 7


# Word → list of topic indices. Hand-crafted from knowledge of
# Shakespearean lexicon. Lowercased lookup. A word may belong to
# multiple clusters (e.g. "soul" ∈ {DEATH, FAITH}, "heart" ∈ {LOVE,
# BODY}, "blood" ∈ {WAR, BODY}, "star" ∈ {NATURE, FORTUNE}).
WORD_TO_TOPICS: dict[str, tuple[int, ...]] = {
    # --- WAR ---
    "sword": (TOPIC_WAR,),
    "swords": (TOPIC_WAR,),
    "battle": (TOPIC_WAR,),
    "battles": (TOPIC_WAR,),
    "war": (TOPIC_WAR,),
    "wars": (TOPIC_WAR,),
    "foe": (TOPIC_WAR,),
    "foes": (TOPIC_WAR,),
    "enemy": (TOPIC_WAR,),
    "enemies": (TOPIC_WAR,),
    "arms": (TOPIC_WAR,),
    "armour": (TOPIC_WAR,),
    "steel": (TOPIC_WAR,),
    "slain": (TOPIC_WAR, TOPIC_DEATH),
    "slay": (TOPIC_WAR, TOPIC_DEATH),
    "fight": (TOPIC_WAR,),
    "fought": (TOPIC_WAR,),
    "siege": (TOPIC_WAR,),
    "wound": (TOPIC_WAR, TOPIC_BODY),
    "wounds": (TOPIC_WAR, TOPIC_BODY),
    "wounded": (TOPIC_WAR, TOPIC_BODY),
    "strike": (TOPIC_WAR,),
    "strikes": (TOPIC_WAR,),
    "struck": (TOPIC_WAR,),
    "field": (TOPIC_WAR, TOPIC_NATURE),
    "spear": (TOPIC_WAR,),
    "arrow": (TOPIC_WAR,),
    "arrows": (TOPIC_WAR,),
    "soldier": (TOPIC_WAR,),
    "soldiers": (TOPIC_WAR,),
    "captain": (TOPIC_WAR,),
    "victory": (TOPIC_WAR,),
    "defeat": (TOPIC_WAR,),
    "banner": (TOPIC_WAR,),
    "trumpet": (TOPIC_WAR,),
    "drum": (TOPIC_WAR,),
    "cannon": (TOPIC_WAR,),
    "shield": (TOPIC_WAR,),
    "helm": (TOPIC_WAR,),
    "bow": (TOPIC_WAR,),
    "dagger": (TOPIC_WAR,),
    "blade": (TOPIC_WAR,),
    "conquer": (TOPIC_WAR,),
    "conqueror": (TOPIC_WAR,),
    "army": (TOPIC_WAR,),
    "armies": (TOPIC_WAR,),
    "revenge": (TOPIC_WAR,),

    # --- LOVE ---
    "love": (TOPIC_LOVE,),
    "loves": (TOPIC_LOVE,),
    "loved": (TOPIC_LOVE,),
    "loving": (TOPIC_LOVE,),
    "heart": (TOPIC_LOVE, TOPIC_BODY),
    "hearts": (TOPIC_LOVE, TOPIC_BODY),
    "dear": (TOPIC_LOVE,),
    "kiss": (TOPIC_LOVE, TOPIC_BODY),
    "kisses": (TOPIC_LOVE, TOPIC_BODY),
    "sweet": (TOPIC_LOVE,),
    "sweetest": (TOPIC_LOVE,),
    "rose": (TOPIC_LOVE, TOPIC_NATURE),
    "roses": (TOPIC_LOVE, TOPIC_NATURE),
    "charm": (TOPIC_LOVE,),
    "beauty": (TOPIC_LOVE,),
    "beauties": (TOPIC_LOVE,),
    "beautiful": (TOPIC_LOVE,),
    "cheek": (TOPIC_LOVE, TOPIC_BODY),
    "cheeks": (TOPIC_LOVE, TOPIC_BODY),
    "mistress": (TOPIC_LOVE,),
    "bride": (TOPIC_LOVE,),
    "wedding": (TOPIC_LOVE,),
    "wed": (TOPIC_LOVE,),
    "marry": (TOPIC_LOVE,),
    "married": (TOPIC_LOVE,),
    "lover": (TOPIC_LOVE,),
    "lovers": (TOPIC_LOVE,),
    "tender": (TOPIC_LOVE,),
    "gentle": (TOPIC_LOVE,),
    "fair": (TOPIC_LOVE,),
    "darling": (TOPIC_LOVE,),
    "passion": (TOPIC_LOVE,),
    "beloved": (TOPIC_LOVE,),
    "woo": (TOPIC_LOVE,),
    "wooed": (TOPIC_LOVE,),

    # --- DEATH ---
    "death": (TOPIC_DEATH,),
    "deaths": (TOPIC_DEATH,),
    "die": (TOPIC_DEATH,),
    "dies": (TOPIC_DEATH,),
    "died": (TOPIC_DEATH,),
    "dying": (TOPIC_DEATH,),
    "dead": (TOPIC_DEATH,),
    "grave": (TOPIC_DEATH,),
    "graves": (TOPIC_DEATH,),
    "tomb": (TOPIC_DEATH,),
    "tombs": (TOPIC_DEATH,),
    "corpse": (TOPIC_DEATH, TOPIC_BODY),
    "bury": (TOPIC_DEATH,),
    "buried": (TOPIC_DEATH,),
    "soul": (TOPIC_DEATH, TOPIC_FAITH),
    "souls": (TOPIC_DEATH, TOPIC_FAITH),
    "ghost": (TOPIC_DEATH,),
    "ghosts": (TOPIC_DEATH,),
    "dust": (TOPIC_DEATH,),
    "rot": (TOPIC_DEATH,),
    "mourn": (TOPIC_DEATH,),
    "mourning": (TOPIC_DEATH,),
    "funeral": (TOPIC_DEATH,),
    "pale": (TOPIC_DEATH,),
    "murder": (TOPIC_DEATH, TOPIC_WAR),
    "murdered": (TOPIC_DEATH, TOPIC_WAR),
    "kill": (TOPIC_DEATH, TOPIC_WAR),
    "killed": (TOPIC_DEATH, TOPIC_WAR),
    "killing": (TOPIC_DEATH, TOPIC_WAR),
    "perish": (TOPIC_DEATH,),
    "perished": (TOPIC_DEATH,),
    "mortal": (TOPIC_DEATH,),
    "mortals": (TOPIC_DEATH,),
    "coffin": (TOPIC_DEATH,),
    "shroud": (TOPIC_DEATH,),
    "ashes": (TOPIC_DEATH,),

    # --- ROYALTY ---
    "king": (TOPIC_ROYALTY,),
    "kings": (TOPIC_ROYALTY,),
    "queen": (TOPIC_ROYALTY,),
    "queens": (TOPIC_ROYALTY,),
    "crown": (TOPIC_ROYALTY,),
    "crowns": (TOPIC_ROYALTY,),
    "crowned": (TOPIC_ROYALTY,),
    "throne": (TOPIC_ROYALTY,),
    "thrones": (TOPIC_ROYALTY,),
    "prince": (TOPIC_ROYALTY,),
    "princes": (TOPIC_ROYALTY,),
    "princess": (TOPIC_ROYALTY,),
    "duke": (TOPIC_ROYALTY,),
    "duchess": (TOPIC_ROYALTY,),
    "royal": (TOPIC_ROYALTY,),
    "sceptre": (TOPIC_ROYALTY,),
    "scepter": (TOPIC_ROYALTY,),
    "lord": (TOPIC_ROYALTY,),
    "lords": (TOPIC_ROYALTY,),
    "lady": (TOPIC_ROYALTY,),
    "ladies": (TOPIC_ROYALTY,),
    "noble": (TOPIC_ROYALTY,),
    "nobles": (TOPIC_ROYALTY,),
    "court": (TOPIC_ROYALTY,),
    "majesty": (TOPIC_ROYALTY,),
    "sovereign": (TOPIC_ROYALTY,),
    "reign": (TOPIC_ROYALTY,),
    "realm": (TOPIC_ROYALTY,),
    "subject": (TOPIC_ROYALTY,),
    "subjects": (TOPIC_ROYALTY,),
    "monarch": (TOPIC_ROYALTY,),
    "emperor": (TOPIC_ROYALTY,),
    "empress": (TOPIC_ROYALTY,),
    "kingdom": (TOPIC_ROYALTY,),
    "grace": (TOPIC_ROYALTY, TOPIC_FAITH),

    # --- NATURE ---
    "sun": (TOPIC_NATURE,),
    "moon": (TOPIC_NATURE,),
    "stars": (TOPIC_NATURE, TOPIC_FORTUNE),
    "star": (TOPIC_NATURE, TOPIC_FORTUNE),
    "wind": (TOPIC_NATURE,),
    "winds": (TOPIC_NATURE,),
    "rain": (TOPIC_NATURE,),
    "sea": (TOPIC_NATURE,),
    "seas": (TOPIC_NATURE,),
    "sky": (TOPIC_NATURE,),
    "skies": (TOPIC_NATURE,),
    "flower": (TOPIC_NATURE,),
    "flowers": (TOPIC_NATURE,),
    "tree": (TOPIC_NATURE,),
    "trees": (TOPIC_NATURE,),
    "bird": (TOPIC_NATURE,),
    "birds": (TOPIC_NATURE,),
    "storm": (TOPIC_NATURE,),
    "storms": (TOPIC_NATURE,),
    "morn": (TOPIC_NATURE,),
    "morning": (TOPIC_NATURE,),
    "night": (TOPIC_NATURE,),
    "day": (TOPIC_NATURE,),
    "shore": (TOPIC_NATURE,),
    "leaf": (TOPIC_NATURE,),
    "leaves": (TOPIC_NATURE,),
    "fire": (TOPIC_NATURE,),
    "fires": (TOPIC_NATURE,),
    "earth": (TOPIC_NATURE,),
    "ocean": (TOPIC_NATURE,),
    "wave": (TOPIC_NATURE,),
    "waves": (TOPIC_NATURE,),
    "river": (TOPIC_NATURE,),
    "mountain": (TOPIC_NATURE,),
    "forest": (TOPIC_NATURE,),
    "wood": (TOPIC_NATURE,),
    "garden": (TOPIC_NATURE,),
    "summer": (TOPIC_NATURE,),
    "winter": (TOPIC_NATURE,),
    "spring": (TOPIC_NATURE,),
    "autumn": (TOPIC_NATURE,),
    "snow": (TOPIC_NATURE,),
    "cloud": (TOPIC_NATURE,),
    "clouds": (TOPIC_NATURE,),
    "thunder": (TOPIC_NATURE,),
    "lightning": (TOPIC_NATURE,),

    # --- BODY ---
    "hand": (TOPIC_BODY,),
    "hands": (TOPIC_BODY,),
    "eye": (TOPIC_BODY,),
    "eyes": (TOPIC_BODY,),
    "face": (TOPIC_BODY,),
    "faces": (TOPIC_BODY,),
    "lip": (TOPIC_BODY,),
    "lips": (TOPIC_BODY,),
    "tongue": (TOPIC_BODY,),
    "tongues": (TOPIC_BODY,),
    "arm": (TOPIC_BODY,),
    "breast": (TOPIC_BODY,),
    "breasts": (TOPIC_BODY,),
    "head": (TOPIC_BODY,),
    "heads": (TOPIC_BODY,),
    "foot": (TOPIC_BODY,),
    "feet": (TOPIC_BODY,),
    "tears": (TOPIC_BODY,),
    "blood": (TOPIC_BODY, TOPIC_WAR),
    "flesh": (TOPIC_BODY,),
    "bone": (TOPIC_BODY,),
    "bones": (TOPIC_BODY,),
    "hair": (TOPIC_BODY,),
    "brow": (TOPIC_BODY,),
    "skin": (TOPIC_BODY,),
    "throat": (TOPIC_BODY,),
    "neck": (TOPIC_BODY,),
    "back": (TOPIC_BODY,),
    "bosom": (TOPIC_BODY,),

    # --- FAITH ---
    "god": (TOPIC_FAITH,),
    "gods": (TOPIC_FAITH,),
    "heaven": (TOPIC_FAITH,),
    "heavens": (TOPIC_FAITH,),
    "hell": (TOPIC_FAITH,),
    "prayer": (TOPIC_FAITH,),
    "pray": (TOPIC_FAITH,),
    "prayed": (TOPIC_FAITH,),
    "sin": (TOPIC_FAITH,),
    "sins": (TOPIC_FAITH,),
    "holy": (TOPIC_FAITH,),
    "faith": (TOPIC_FAITH,),
    "mercy": (TOPIC_FAITH,),
    "angel": (TOPIC_FAITH,),
    "angels": (TOPIC_FAITH,),
    "devil": (TOPIC_FAITH,),
    "devils": (TOPIC_FAITH,),
    "sacred": (TOPIC_FAITH,),
    "spirit": (TOPIC_FAITH,),
    "spirits": (TOPIC_FAITH,),
    "church": (TOPIC_FAITH,),
    "blessed": (TOPIC_FAITH,),
    "bless": (TOPIC_FAITH,),
    "curse": (TOPIC_FAITH,),
    "cursed": (TOPIC_FAITH,),
    "damn": (TOPIC_FAITH,),
    "damned": (TOPIC_FAITH,),
    "saint": (TOPIC_FAITH,),
    "saints": (TOPIC_FAITH,),
    "priest": (TOPIC_FAITH,),
    "prophet": (TOPIC_FAITH,),
    "sacrament": (TOPIC_FAITH,),
    "mass": (TOPIC_FAITH,),

    # --- FORTUNE ---
    "fate": (TOPIC_FORTUNE,),
    "fates": (TOPIC_FORTUNE,),
    "chance": (TOPIC_FORTUNE,),
    "luck": (TOPIC_FORTUNE,),
    "fortune": (TOPIC_FORTUNE,),
    "fortunes": (TOPIC_FORTUNE,),
    "doom": (TOPIC_FORTUNE, TOPIC_DEATH),
    "doomed": (TOPIC_FORTUNE, TOPIC_DEATH),
    "destiny": (TOPIC_FORTUNE,),
    "providence": (TOPIC_FORTUNE,),
    "hap": (TOPIC_FORTUNE,),
    "haply": (TOPIC_FORTUNE,),
    "wheel": (TOPIC_FORTUNE,),
    "time": (TOPIC_FORTUNE,),
    "times": (TOPIC_FORTUNE,),
    "hour": (TOPIC_FORTUNE,),
    "hours": (TOPIC_FORTUNE,),
    "future": (TOPIC_FORTUNE,),
}


_DECAY = 0.90
_BUMP = 1.0
_CAP = 4.0
_TURN_MULT = 0.35


def update_topic_tracker(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-turn boundary dampen (applied on the turn-closing \n
    # rather than on word completions within turn).
    if state.consecutive_newlines >= 2 and ch == "\n":
        dampened = tuple(v * _TURN_MULT for v in state.scene_topics)
        return state.model_copy(update={"scene_topics": dampened})

    if not state.just_finished_word:
        return state
    if state.speaker_label_state != 0:
        return state

    word = state.last_completed_word
    if not word:
        return state

    # Strip leading apostrophe for lookup ('tis, 'gainst, etc.).
    lookup = word.lstrip("'")

    # Decay everything first.
    topics = [v * _DECAY for v in state.scene_topics]

    # Bump matched clusters.
    hits = WORD_TO_TOPICS.get(lookup)
    if hits is not None:
        for k in hits:
            topics[k] = min(_CAP, topics[k] + _BUMP)

    return state.model_copy(update={"scene_topics": tuple(topics)})
