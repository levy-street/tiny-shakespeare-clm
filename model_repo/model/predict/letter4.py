"""Letter-4gram (4-letter prefix → next letter) bias layer.

For very common 4-letter prefixes, bias the next letter strongly.
Complements the 3-letter layer for the highest-confidence cases —
things like "ough" → t, "ight" → space, "tion" → space/s, "ness" →
space, "ment" → space, "ship" → space.

All hand-specified from English orthography.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

_L4: dict[str, dict[str, float]] = {
    # -ough → t (thought, fought, nought, ought, bought, sought)
    "ough": {"t": 2.5, " ": 1.5, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.3},
    # -ight → space (might, night, sight, light, right, bright)
    "ight": {" ": 2.5, ",": 0.8, ".": 0.6, "\n": 0.5, "s": 0.6, "'": 0.3,
             "e": 0.4, "l": 0.3},
    # -aigh → t (straight)
    "aigh": {"t": 2.0, " ": 0.3},
    # -eigh → t (eight, weight, neighbour)
    "eigh": {"t": 2.0, "b": 0.5, " ": 0.4},
    # -tion → space/s (nation, action, motion)
    "tion": {" ": 2.5, "s": 1.5, ",": 0.7, ".": 0.5, "\n": 0.4, "a": 0.3},
    # -sion → space/s (vision, passion, confusion)
    "sion": {" ": 2.5, "s": 1.5, ",": 0.7, ".": 0.5, "\n": 0.4},
    # -ness → space (darkness, sweetness, wickedness)
    "ness": {" ": 2.8, ",": 0.8, ".": 0.6, "\n": 0.5, "!": 0.3, ";": 0.4,
             "e": 0.4},
    # -ment → space (moment, comment, ornament)
    "ment": {" ": 2.5, ",": 0.8, ".": 0.5, "\n": 0.4, "s": 0.7, "a": 0.3},
    # -ship → space (friendship, fellowship, lordship)
    "ship": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.6},
    # -ways -ward -wise
    "ward": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.8, "l": 0.3},
    "wise": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "r": 0.3},
    # -full -fulled
    "full": {"y": 1.5, " ": 1.3, ",": 0.5, "n": 0.3},
    # -less (useless, helpless, hopeless)
    "less": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "l": 0.5, "n": 0.3,
             "o": 0.3, "e": 0.3},
    # -able (table, able, stable, capable)
    "able": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.4, ";": 0.3},
    # -ible (possible, terrible, horrible)
    "ible": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.3, ";": 0.3},
    # -ally (usually, really, totally) -edly (hurriedly)
    "ally": {" ": 2.0, ",": 0.5, ".": 0.4, "\n": 0.3, ";": 0.3},
    # -edly (decidedly)
    "edly": {" ": 1.5, ",": 0.4},
    # -ing after common stems — very strong word-end
    "ling": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.5},
    "ting": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.5},
    "ming": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.5},
    "ning": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.5},
    "ring": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.6},
    "ping": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.5},
    "ding": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.5},
    "king": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.7,
             "d": 0.5, "l": 0.4},  # kingdom, kingly
    "ying": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.5},
    "wing": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.5},
    "uing": {" ": 2.0, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.4},
    # -ance (distance, chance, ignorance)
    "ance": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.6, ";": 0.3},
    # -ence (silence, patience, difference)
    "ence": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.6, ";": 0.3},
    # -hood (childhood, manhood, brotherhood)
    "hood": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.4},
    # -some (handsome, lonesome, tiresome)
    "some": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.3},
    # -ture (nature, picture, creature)
    "ture": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.5, "d": 0.4,
             "r": 0.3, "l": 0.3},
    # -sure (sure, measure, pleasure, treasure)
    "sure": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.4, "d": 0.3,
             "l": 0.3},
    # -here/-there/-where + ?
    "here": {" ": 2.0, ",": 0.7, ".": 0.5, "\n": 0.4, "i": 0.4, "o": 0.3,
             "u": 0.3, "'": 0.3, ";": 0.3, "a": 0.3, "b": 0.3},
    "ther": {"e": 2.0, "i": 1.0, " ": 0.3, ",": 0.2, "l": 0.3, "s": 0.3},
    "uest": {" ": 1.8, ",": 0.5, "i": 0.4, "s": 0.3, "e": 0.3},
    # -hold (behold, household, uphold)
    "hold": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.5, "e": 0.3,
             "i": 0.3},
    # -down, -town
    "down": {" ": 1.5, ",": 0.4, ".": 0.3},
    "town": {" ": 1.5, ",": 0.4, ".": 0.3, "s": 0.4},
    # -thing (nothing, something, anything)
    "hing": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.5},
    # -world
    "worl": {"d": 2.5, "y": 0.3},
    # -ought already covered by -ough
    # -ever, -every (ever, never, forever, every)
    "ever": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "y": 1.2, "s": 0.3,
             "!": 0.3, ";": 0.3, "m": 0.3, "l": 0.3},
    "very": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "!": 0.3, "o": 0.3,
             ";": 0.3, "t": 0.3},
    # -olve (solve, resolve, involve)
    "olve": {" ": 1.5, "d": 0.5, "s": 0.5, "r": 0.3},
    # -use/-used
    "used": {" ": 1.5, ",": 0.4, ".": 0.3, "\n": 0.3},
    # -pped, -tted (doubled past)
    "pped": {" ": 1.8, ",": 0.4, ".": 0.3, "\n": 0.3},
    "tted": {" ": 1.8, ",": 0.4, ".": 0.3, "\n": 0.3},
    # -ning, -lling, -mmed, -ssed
    "lled": {" ": 1.8, ",": 0.4, ".": 0.3, "\n": 0.3},
    "rred": {" ": 1.6, ",": 0.4, ".": 0.3},
    "ssed": {" ": 1.8, ",": 0.4, ".": 0.3, "\n": 0.3},
    # -aken, -oken, -iken (taken, spoken, broken)
    "aken": {" ": 1.5, ",": 0.4, ".": 0.3, "\n": 0.3, "s": 0.3},
    "oken": {" ": 1.5, ",": 0.4, ".": 0.3, "\n": 0.3, "s": 0.3},
    # -ight covered; -ought covered
    # -ount (count, mount, amount)
    "ount": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.5, "e": 0.3,
             "a": 0.3, "r": 0.3, "i": 0.3},
    # -ound (round, sound, found, wound, ground)
    "ound": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.6, "e": 0.3,
             "l": 0.3},
    # -alth (health, wealth, stealth)
    "alth": {" ": 1.6, ",": 0.4, ".": 0.3, "\n": 0.3, "s": 0.3, "i": 0.3,
             "y": 0.3},
    # -ilth, -olph
    # -orth (north, forth, worth)
    "orth": {" ": 1.6, ",": 0.4, ".": 0.3, "\n": 0.3, "s": 0.3, "w": 0.4,
             "l": 0.3, "y": 0.3, "i": 0.3, "e": 0.3},
    # -isth, -outh (south, mouth, youth, truth)
    "outh": {" ": 1.5, ",": 0.4, ".": 0.3, "\n": 0.3, "s": 0.3},
    # -aith (faith)
    "aith": {" ": 1.7, ",": 0.4, ".": 0.3, "\n": 0.3, "f": 0.4, "s": 0.3},
    # -ings (kings, things, bringings)
    "ings": {" ": 2.2, ",": 0.6, ".": 0.4, "\n": 0.3, ";": 0.3},
    # -ands, -inds, -ends (lands, minds, friends, ends)
    "ands": {" ": 2.2, ",": 0.6, ".": 0.4, "\n": 0.3, ";": 0.3},
    "ends": {" ": 2.2, ",": 0.6, ".": 0.4, "\n": 0.3, ";": 0.3},
    "inds": {" ": 2.2, ",": 0.6, ".": 0.4, "\n": 0.3, ";": 0.3},
    "olds": {" ": 2.0, ",": 0.5, ".": 0.4, "\n": 0.3},
    # -orts, -arts
    "orts": {" ": 2.0, ",": 0.5, ".": 0.4, "\n": 0.3},
    "arts": {" ": 2.0, ",": 0.5, ".": 0.4, "\n": 0.3},
    # -iest, -eest
    "iest": {" ": 2.2, ",": 0.6, ".": 0.4, "\n": 0.3, ";": 0.3},
    # -dest (modest, saddest, oldest)
    "dest": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3},
    # -lest (paltriest, blessedness-est) best, blest
    "lest": {" ": 1.5, ",": 0.4, ".": 0.3},
    # -gest (largest, biggest, dangest)
    "gest": {" ": 1.6, ",": 0.4, ".": 0.3, "\n": 0.3, "u": 0.3},
    # -mest
    "mest": {" ": 1.5, ",": 0.4, ".": 0.3},
    # -rest (rest, forest, dearest, interest)
    "rest": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.4, ";": 0.3,
             "l": 0.3, "i": 0.3},
    # -eath (death, breath, heath, wreath)
    "eath": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "s": 0.4, "e": 0.3,
             "l": 0.3},
    # -ower (power, tower, lower, flower)
    "ower": {" ": 1.6, ",": 0.5, ".": 0.3, "\n": 0.3, "s": 0.5, "e": 0.3,
             "f": 0.3, "l": 0.3},
    # -over (over, cover, lover, rover)
    "over": {" ": 2.0, ",": 0.6, ".": 0.4, "\n": 0.3, "s": 0.4, "e": 0.3,
             "l": 0.3, "t": 0.3, "n": 0.3, "h": 0.3},
    # -arly (early, nearly, dearly, clearly)
    "arly": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3},
    # -ious (various, curious, precious, serious, glorious)
    "ious": {" ": 2.2, ",": 0.6, ".": 0.4, "\n": 0.3, ";": 0.3},
    # -eous (piteous, hideous, righteous, courteous)
    "eous": {" ": 2.0, ",": 0.5, ".": 0.4, "\n": 0.3, ";": 0.3},
    # -uous (continuous, virtuous, strenuous)
    "uous": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3},
    # -each (reach, each, beach, teach, peach)
    "each": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "e": 0.4, "i": 0.3},
    # -ouch (touch, couch, pouch)
    "ouch": {" ": 1.6, ",": 0.4, ".": 0.3, "\n": 0.3, "e": 0.3, "i": 0.3},
    # -urch (church)
    "urch": {" ": 1.5, ",": 0.4, ".": 0.3},
    # -unto, -upon as word endings
    "unto": {" ": 2.0, ",": 0.5, ".": 0.3, "\n": 0.3},
    "upon": {" ": 2.2, ",": 0.6, ".": 0.4, "\n": 0.3, ";": 0.3},
    # -inst (against)
    "inst": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3},
    # -ndst (against, amongst, midst, wilst)
    "ongs": {"t": 2.0, ",": 0.3, ".": 0.3, " ": 0.3, "'": 0.3},
    # -hilst (whilst)
    "ilst": {" ": 1.8, ",": 0.5, ".": 0.3, "\n": 0.3},
    # -adst (hadst)
    "adst": {" ": 1.8, ",": 0.5, ".": 0.4},
    # -ouldst (wouldst, shouldst, couldst)
    "ldst": {" ": 1.8, ",": 0.5, ".": 0.4},
    # -ayst, -eest (sayst, seest)
    "ayst": {" ": 1.5, ",": 0.4, ".": 0.3},
    "eest": {" ": 1.5, ",": 0.4, ".": 0.3},
    # -orld (world)
    "orld": {" ": 2.5, ",": 0.5, "s": 0.5, "l": 0.3, ".": 0.3, "\n": 0.3},
    # -ance, -ence already listed; -ince (since, prince, convince)
    "ince": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, "l": 0.3, "s": 0.3,
             "e": 0.3},
    # -ould (would, should, could)
    "ould": {" ": 2.0, ",": 0.3, "'": 0.5, "s": 0.4, "n": 0.3, ";": 0.3,
             ".": 0.3},
    # -urse (curse, course, nurse, verse, horse)
    # Note: technically -orse; this is just -urse
    "urse": {" ": 1.6, ",": 0.4, ".": 0.3, "\n": 0.3, "s": 0.3, "d": 0.3},
    # -orse
    "orse": {" ": 1.6, ",": 0.4, ".": 0.3, "\n": 0.3, "s": 0.3, "r": 0.3,
             "l": 0.3},
    # -erse (verse, diverse, universe, adverse)
    "erse": {" ": 1.6, ",": 0.4, ".": 0.3, "\n": 0.3, "s": 0.3, "l": 0.3,
             "d": 0.3},
    # -aven, -even, -iven (heaven, seven, given, driven)
    "aven": {" ": 1.5, ",": 0.4, ".": 0.3, "\n": 0.3, "s": 0.5, "'": 0.3,
             "l": 0.3},
    "even": {" ": 1.5, ",": 0.4, ".": 0.3, "\n": 0.3, "i": 0.3, "s": 0.3,
             "e": 0.3},
    "iven": {" ": 1.5, ",": 0.4, ".": 0.3, "\n": 0.3, "s": 0.3},
    "oven": {" ": 1.4, ",": 0.4, ".": 0.3, "\n": 0.3},
    # -urns, -ourn
    "urns": {" ": 1.8, ",": 0.5, ".": 0.4, "\n": 0.3, ";": 0.3},
    "ourn": {"s": 0.8, "e": 0.8, " ": 0.4, ",": 0.3},
    # -ways (always, sideways, anyways)
    "ways": {" ": 2.0, ",": 0.5, ".": 0.4, "\n": 0.3},
    # -rink/-rank/-rink (drink, think, bank, rank, frank)
    "rink": {" ": 1.5, ",": 0.4, "s": 0.4, ".": 0.3, "l": 0.3},
    "rank": {" ": 1.5, ",": 0.4, "s": 0.4, ".": 0.3},
    # -eant (meant, leant)
    "eant": {" ": 1.6, ",": 0.4, ".": 0.3, "\n": 0.3, "s": 0.3},
    # -ream (dream, cream, stream)
    "ream": {" ": 1.6, ",": 0.4, ".": 0.3, "\n": 0.3, "s": 0.4, "t": 0.3,
             "i": 0.3, "e": 0.3},
    # -eave (leave, weave, cleave, behave, brave)
    "eave": {" ": 1.6, ",": 0.4, ".": 0.3, "\n": 0.3, "s": 0.4, "n": 0.3,
             "d": 0.3, "r": 0.3},
}


_GLOBAL_SCALE = 0.15
# Default penalty for letters not listed
_NEG_DEFAULT = 0.0


def _build_bias_vectors() -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    lowers = "abcdefghijklmnopqrstuvwxyz"
    for prefix, entries in _L4.items():
        vec = [0.0] * VOCAB_SIZE
        for target in lowers:
            if target not in entries:
                vec[VOCAB_INDEX[target]] = _NEG_DEFAULT * _GLOBAL_SCALE
        for nxt, bias in entries.items():
            if nxt in VOCAB_INDEX:
                scaled = bias * _GLOBAL_SCALE
                vec[VOCAB_INDEX[nxt]] = scaled
                if nxt.isalpha() and nxt.lower() == nxt:
                    up = nxt.upper()
                    if up in VOCAB_INDEX:
                        vec[VOCAB_INDEX[up]] = scaled * 0.3
        out[prefix] = vec
    return out


LETTER4_BIAS_VECTORS: dict[str, list[float]] = _build_bias_vectors()


def letter4_bias(word_buffer: str) -> list[float] | None:
    """Bias vector keyed on last 4 letters of word_buffer (lowercased,
    ignoring apostrophes). None if fewer than 4 letters or prefix not
    listed."""
    if len(word_buffer) < 4:
        return None
    letters = [c for c in word_buffer if c != "'"]
    if len(letters) < 4:
        return None
    key = "".join(letters[-4:]).lower()
    return LETTER4_BIAS_VECTORS.get(key)
