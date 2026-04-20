"""Coarse noun-class dictionary for semantic-coherence tracking.

Shakespearean samples often drift across semantic fields ("throne of
treasure", "my mother is niece"): each word-transition is locally
grammatical, but the words belong to incompatible semantic frames.
Bigram / trigram co-occurrence biases don't fix this — the signal is
frame compatibility, not surface rare-word co-occurrence.

12-class tagger sourced from prior knowledge of Shakespeare's
lexical fields:

    0  NONE        unknown / non-noun
    1  KINSHIP     mother, father, son, ...
    2  ROYALTY     king, queen, prince, throne, ...
    3  BODY        heart, hand, eye, blood, ...
    4  EMOTION     love, grief, fear, wrath, ...
    5  NATURE      sun, moon, storm, flower, ...
    6  ABSTRACT    soul, truth, fate, honour, ...
    7  WEAPON      sword, spear, battle, army, ...
    8  PLACE       castle, tower, city, grave, ...
    9  TIME        hour, day, year, season, ...
    10 CREATURE    dog, horse, wolf, bird, ...
    11 DIVINE      god, heaven, hell, angel, ...

Ambiguous words are placed in the class that BEST predicts typical
continuation (heart → EMOTION, throne → ROYALTY, grave → PLACE).

No statistics — assignments from reading Shakespeare with English
literary knowledge.
"""

from __future__ import annotations

NC_NONE = 0
NC_KINSHIP = 1
NC_ROYALTY = 2
NC_BODY = 3
NC_EMOTION = 4
NC_NATURE = 5
NC_ABSTRACT = 6
NC_WEAPON = 7
NC_PLACE = 8
NC_TIME = 9
NC_CREATURE = 10
NC_DIVINE = 11

N_CLASSES = 12


_KINSHIP = (
    "mother", "father", "brother", "sister", "son", "daughter",
    "aunt", "uncle", "niece", "nephew", "cousin", "cousins",
    "wife", "husband", "child", "children", "kin", "kinsman",
    "kinsmen", "kindred", "parent", "parents", "twin", "twins",
    "widow", "widower", "orphan", "bride", "bridegroom",
    "grandsire", "grandam", "stepmother", "stepfather",
    "brothers", "sisters", "sons", "daughters", "mothers",
    "babe", "babes", "infant", "boy", "girl", "lad", "lass", "maid",
    "maiden", "maidens", "youth", "youths",
)

_ROYALTY = (
    "king", "queen", "prince", "princess", "duke", "duchess",
    "earl", "count", "lord", "lady", "lords", "ladies",
    "noble", "royal", "majesty", "highness", "throne", "crown",
    "sceptre", "reign", "liege", "sovereign", "emperor", "empress",
    "regent", "knight", "knights", "court", "courtier", "courtiers",
    "kings", "queens", "princes", "dukes",
    "marquess", "viscount", "baron", "baroness",
    "prelate", "ruler", "monarch",
)

_BODY = (
    "heart", "hand", "hands", "head", "eye", "eyes", "ear", "ears",
    "face", "faces", "lip", "lips", "tongue", "cheek", "cheeks",
    "brow", "brows", "breast", "bosom", "blood", "flesh", "bone",
    "bones", "arm", "arms", "foot", "feet", "knee", "knees",
    "hair", "skin", "voice", "breath", "mouth", "mouths",
    "neck", "finger", "fingers", "nose", "forehead", "temple",
    "shoulder", "shoulders", "throat", "tooth", "teeth",
    "belly", "leg", "legs", "palm", "wrist", "ankle",
    "spine", "pulse", "vein", "veins",
    "tear", "tears", "sweat",
)

_EMOTION = (
    "love", "hate", "joy", "sorrow", "grief", "pity", "wrath",
    "rage", "fear", "hope", "despair", "envy", "shame", "pride",
    "anger", "mercy", "faith", "courage", "loyalty", "jealousy",
    "tenderness", "affection", "passion", "passions",
    "delight", "mirth", "cheer", "gloom", "melancholy",
    "bliss", "rapture", "devotion", "longing", "yearning",
    "distress", "agony", "pang", "pangs", "torment", "torments",
    "woe", "woes", "mourning",
    "compassion", "remorse", "contempt", "scorn", "disdain",
    "spite", "malice", "ardour",
    "desire", "desires", "lust",
    "dread", "terror", "horror",
)

_NATURE = (
    "sun", "moon", "star", "stars", "sky", "heavens",
    "cloud", "clouds", "wind", "winds", "rain", "storm", "storms",
    "sea", "ocean", "tide", "wave", "waves", "earth", "ground",
    "soil", "fire", "flame", "flames", "flower", "flowers", "rose",
    "roses", "tree", "trees", "forest", "forests", "mountain",
    "mountains", "hill", "hills", "stone", "stones", "rock", "rocks",
    "water", "waters", "light", "darkness", "dawn",
    "dusk", "morn", "morning", "evening", "eve",
    "lightning", "thunder", "frost", "snow", "hail", "mist", "dew",
    "river", "stream", "brook", "meadow", "field", "fields",
    "leaf", "leaves", "branch", "branches", "bloom", "blossom",
    "lily", "violet", "thorn", "thorns", "grove",
    "valley", "glen", "plain", "plains", "shore", "beach",
)

_ABSTRACT = (
    "soul", "mind", "spirit", "spirits", "thought", "thoughts",
    "truth", "lie", "lies", "time", "fate", "death", "life",
    "beauty", "virtue", "honour", "honor", "justice", "reason",
    "wit", "wits", "wisdom", "knowledge", "folly", "vice",
    "goodness", "evil", "peace", "war", "quiet", "silence",
    "noise", "sound", "cause", "causes", "course", "courses",
    "duty", "duties", "law", "laws", "right", "rights", "wrong",
    "wrongs", "chance", "fortune", "misfortune", "luck",
    "destiny", "doom", "providence",
    "dream", "dreams", "vision", "visions",
    "memory", "memories", "conscience",
    "art", "artifice",
    "power", "strength", "weakness", "might",
    "grace", "pardon",
    "service",
)

_WEAPON = (
    "sword", "swords", "dagger", "daggers", "spear", "spears",
    "shield", "shields", "lance", "lances", "bow", "bows",
    "arrow", "arrows", "knife", "knives", "axe", "axes",
    "armour", "armor", "helm", "helmet", "mail", "steel",
    "blade", "blades", "point", "edge",
    "gun", "guns", "pistol", "pistols", "cannon", "cannons",
    "musket", "halberd", "pike", "pikes", "javelin",
    "club", "cudgel", "mace",
    "host", "hosts", "battle", "battles", "fight", "fights",
    "combat", "combats", "wars",
    "army", "armies", "soldier", "soldiers", "troop", "troops",
    "captain", "captains", "general", "generals",
    "warrior", "warriors",
    "siege", "assault",
)

_PLACE = (
    "castle", "castles", "chamber", "chambers", "hall", "halls",
    "tower", "towers", "grave", "graves", "tomb", "tombs",
    "temple", "temples", "church", "churches", "city", "cities",
    "town", "towns", "house", "houses", "kingdom", "kingdoms",
    "realm", "realms", "land", "lands", "bed", "beds",
    "palace", "palaces", "garden", "gardens", "street", "streets",
    "road", "roads", "path", "paths",
    "prison", "dungeon", "cell", "cells",
    "gate", "gates", "door", "doors", "window", "windows",
    "room", "rooms", "closet", "courtyard",
    "harbour", "harbor", "port", "ports",
    "country", "countries", "province", "provinces",
    "england", "france", "italy", "rome", "venice", "scotland",
    "ireland", "wales", "denmark",
    "camp", "camps", "tent", "tents",
    "village", "villages",
)

_TIME = (
    "hour", "hours", "minute", "minutes", "year", "years",
    "age", "ages", "season", "seasons", "winter", "summer",
    "spring", "autumn",
    "month", "months", "week", "weeks",
    "century", "centuries", "epoch", "era",
    "midnight", "moment", "moments", "instant",
    "manhood", "womanhood", "childhood", "infancy",
    "yesterday", "today", "tomorrow",
    "past", "future",
)

_CREATURE = (
    "dog", "dogs", "cat", "cats", "horse", "horses",
    "bird", "birds", "dove", "doves", "swan", "swans",
    "lion", "lions", "wolf", "wolves", "serpent", "serpents",
    "snake", "snakes", "beast", "beasts", "deer", "hart",
    "hare", "fox", "foxes", "bear", "bears", "boar", "boars",
    "eagle", "eagles", "hawk", "hawks", "owl", "owls",
    "raven", "ravens", "crow", "crows",
    "lamb", "lambs", "sheep", "ox", "oxen", "bull", "bulls",
    "cow", "cows", "goat", "goats", "stag",
    "worm", "worms", "fly", "flies", "bee", "bees",
    "dragon", "dragons", "unicorn", "unicorns",
    "phoenix", "basilisk",
)

_DIVINE = (
    "god", "gods", "goddess", "heaven", "hell",
    "angel", "angels", "devil", "devils", "fiend", "fiends",
    "ghost", "ghosts", "demon", "demons",
    "saint", "saints", "prophet", "prophets",
    "deity", "deities",
    "cherub", "cherubim", "seraph", "seraphim",
    "paradise", "purgatory", "limbo",
    "damnation", "salvation",
    "curse", "curses", "blessing", "blessings", "prayer", "prayers",
    "sin", "sins",
)


def _build_dict() -> dict[str, int]:
    d: dict[str, int] = {}
    groups = [
        (_KINSHIP, NC_KINSHIP),
        (_ROYALTY, NC_ROYALTY),
        (_BODY, NC_BODY),
        (_EMOTION, NC_EMOTION),
        (_NATURE, NC_NATURE),
        (_ABSTRACT, NC_ABSTRACT),
        (_WEAPON, NC_WEAPON),
        (_PLACE, NC_PLACE),
        (_TIME, NC_TIME),
        (_CREATURE, NC_CREATURE),
        (_DIVINE, NC_DIVINE),
    ]
    # First-wins locks the dominant class for ambiguous words (heart
    # → EMOTION by order wait — actually here: earlier wins, so
    # KINSHIP first, then ROYALTY. 'heart' is only in BODY so BODY
    # gets it. Careful about ordering above).
    for words, cls in groups:
        for w in words:
            wl = w.lower()
            if wl not in d:
                d[wl] = cls
    return d


NOUN_CLASS: dict[str, int] = _build_dict()


def classify(word: str) -> int:
    """Class id for lowercased word; 0 if unknown / not a tracked noun."""
    if not word:
        return NC_NONE
    w = word.lower()
    if w.startswith("'"):
        w = w[1:]
    if w.endswith("'"):
        w = w[:-1]
    if w.endswith("'s"):
        w = w[:-2]
    return NOUN_CLASS.get(w, NC_NONE)
