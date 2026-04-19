"""Sensory-charge flow update.

Runs at word completion. Tilts `sensory_charge` toward +3 when recent
completed words are corporeal / sensory (blood, sword, heart, fire,
night, tears, storm, flesh, ...) and toward -3 when they are abstract
/ discursive (cause, matter, reason, truth, justice, virtue, fault).
All words (including function words) apply a small decay so the
signal reflects the *recent* window, not the whole passage.

Resets to 0 on speaker-turn boundary (double-newline), since a new
speaker re-opens register space.
"""

from __future__ import annotations

from ..state import ModelState


# Strongly corporeal / sensory vocabulary. Body parts, elements,
# weather, weapons, physical objects, light/dark, blood, death,
# concrete natural features. Prior knowledge — no corpus counting.
_SENSORY: frozenset[str] = frozenset(
    [
        # Body
        "blood", "bloody", "heart", "hearts", "tears", "tear", "breath",
        "eye", "eyes", "hand", "hands", "face", "faces", "lips", "lip",
        "tongue", "tongues", "flesh", "bone", "bones", "skin", "cheek",
        "cheeks", "brow", "brows", "ear", "ears", "head", "heads",
        "arm", "arms", "foot", "feet", "knee", "knees", "breast",
        "breasts", "bosom", "throat", "shoulder", "body", "bodies",
        "limb", "limbs", "pulse", "neck", "back",
        # Weapons / violence
        "sword", "swords", "blade", "dagger", "spear", "lance", "axe",
        "bow", "arrow", "arrows", "shield", "helm", "armour", "cannon",
        "wound", "wounds", "stroke", "blow", "blows", "stab", "slash",
        "gash", "scar", "scars", "steel", "iron", "war", "wars",
        "battle", "battles", "sword's", "blade's",
        # Fire / light / dark
        "fire", "fires", "flame", "flames", "spark", "sparks",
        "light", "lights", "dark", "darkness", "night", "nights",
        "shadow", "shadows", "dawn", "dusk", "sun", "sunset", "moon",
        "stars", "star", "sky", "heavens", "heaven", "lightning",
        "thunder", "storm", "storms", "tempest", "gale", "wind",
        "winds", "rain", "snow", "mist", "cloud", "clouds", "dew",
        "frost", "ice",
        # Nature / elements
        "sea", "seas", "ocean", "wave", "waves", "shore", "tide",
        "river", "stream", "water", "waters", "earth", "ground",
        "dust", "ashes", "smoke", "stone", "stones", "rock", "rocks",
        "sand", "mountain", "mountains", "hill", "hills", "valley",
        "wood", "woods", "forest", "tree", "trees", "leaf", "leaves",
        "flower", "flowers", "rose", "roses", "thorn", "thorns",
        "root", "roots", "branch", "branches", "grass", "meadow",
        "field", "fields", "cave", "cliff",
        # Creatures
        "bird", "birds", "dove", "raven", "eagle", "owl", "serpent",
        "snake", "dragon", "lion", "wolf", "wolves", "bear", "bull",
        "horse", "horses", "hound", "dog", "cat", "lamb", "deer",
        "stag", "hart", "beast", "beasts", "crow", "worm", "worms",
        # Death / grave
        "death", "deaths", "grave", "graves", "tomb", "tombs", "corpse",
        "bones", "shroud", "coffin", "funeral", "burial",
        # Liquid / substance
        "wine", "milk", "honey", "bread", "feast", "meat",
        # Light / dark sensory abstract
        "gold", "silver", "silk", "velvet", "crown", "ring", "cup",
        "jewel", "pearl", "gem",
        # Weather of feeling (hybrid)
        "plague", "poison", "venom", "fever",
        # More battle/hunt
        "prey", "arrow",
    ]
)


# Strongly abstract / discursive vocabulary. Reasoning, deliberation,
# moral-political abstractions, process/discourse words.
_ABSTRACT: frozenset[str] = frozenset(
    [
        "cause", "causes", "reason", "reasons", "purpose", "purposes",
        "matter", "matters", "question", "questions", "answer", "answers",
        "truth", "truths", "justice", "virtue", "virtues", "honour",
        "honours", "duty", "duties", "right", "rights", "wrong", "wrongs",
        "fault", "faults", "blame", "proof", "proofs", "sense",
        "senses", "thought", "thoughts", "mind", "minds", "wit",
        "wits", "reason", "service", "business", "suit", "suits",
        "end", "ends", "means", "part", "parts", "case", "cases",
        "course", "courses", "consideration", "respect", "regard",
        "circumstance", "circumstances", "decree", "office", "offices",
        "charge", "order", "orders", "counsel", "opinion", "opinions",
        "intent", "intents", "intention", "motive", "motives",
        "doubt", "doubts", "conscience", "honour", "nature",
        # Process / state / discourse
        "state", "states", "government", "policy", "law", "laws",
        "rule", "rules", "custom", "customs", "practice",
        "occasion", "occasions", "event", "events", "time", "times",
        "term", "terms", "manner", "fashion", "way", "ways",
        "report", "reports", "news", "tidings", "promise", "promises",
        "word", "words", "sentence", "sentences", "meaning",
        "prayer", "prayers", "warning", "warnings",
        # Ethical-moral abstract
        "mercy", "pity", "shame", "guilt", "sin", "sins", "sinner",
        "grace", "faith", "trust", "truth", "hope", "hopes",
        "despair",
    ]
)


def update_sensory_charge(state: ModelState, token_id: int) -> ModelState:
    # Reset on speaker-turn boundary (double newline).
    if state.consecutive_newlines >= 2:
        if state.sensory_charge != 0.0:
            return state.model_copy(update={"sensory_charge": 0.0})
        return state

    # Only update at word-completion events.
    if not state.just_finished_word:
        return state

    word = state.last_completed_word
    if not word:
        return state

    w = word.lower()
    # Strip trailing apostrophe-suffix (blood's, sword's, heart's).
    if "'" in w:
        head = w.split("'", 1)[0]
    else:
        head = w

    # Decay toward zero every word completion.
    charge = state.sensory_charge * 0.93

    if head in _SENSORY:
        charge += 0.6
    elif head in _ABSTRACT:
        charge -= 0.5

    # Clamp.
    if charge > 3.0:
        charge = 3.0
    elif charge < -3.0:
        charge = -3.0

    if charge == state.sensory_charge:
        return state
    return state.model_copy(update={"sensory_charge": charge})
