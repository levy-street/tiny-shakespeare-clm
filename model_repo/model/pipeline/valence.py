"""Tier 3 FLOW — emotional valence texture tracker.

Tracks a signed scalar in [-1.0, +1.0] representing the positive ↔
negative polarity of recent content-word diction. Complements the
existing intensity-only registers (fury/lament/tenderness/doubt) and
the corporeal/abstract sensory_charge axis — neither of which measures
POS↔NEG polarity.

Update rule (on every completed content word):
  * Polarized word pulls valence toward its sign by 0.25 of the
    remaining distance to ±1. (e.g. current 0.0, positive word →
    0.25. Current 0.4, positive word → 0.55.)
  * Neutral content word: decay valence toward 0 by factor 0.96.
  * Function words and short articles: no-op (they don't carry
    valence information).

Reset on speaker-turn boundary and on speaker-label entry (new
character may have opposite register).

Word lists are hand-authored from prior knowledge of the moral /
affective register of Early Modern English vocabulary. No corpus
statistics.
"""

from __future__ import annotations

from ..state import ModelState


# Positive-valence word stems / words. Lowercased; matches use
# startswith on the lowercased last_completed_word so that e.g.
# "lovely", "lovest", "loveth" all match "love".
_POSITIVE_STEMS: frozenset[str] = frozenset({
    "love", "lov",       # love, loves, loved, loving, lovely, lovest
    "fair",               # fair, fairer, fairest, fairly
    "sweet",              # sweet, sweeter, sweetly, sweetness, sweetest
    "dear",               # dear, dearer, dearest, dearly
    "gentle", "gently",   # gentle, gentler, gentlest
    "kind",               # kind, kindness, kindly (also kin)
    "noble", "nobly",     # noble, nobler, noblest, nobility
    "honour", "honor",    # honour, honourable, honoured
    "honest",             # honest, honesty, honestly
    "grace", "gracious",  # grace, graceful, graces, gracious
    "blessed", "bless",   # bless, blessed, blessing
    "bliss",              # bliss, blissful
    "joy", "joyful", "joyous",
    "merry", "mirth",
    "pure", "pur",        # pure, purely, purity, purest
    "saint",              # saint, sainted, saintly
    "virtue", "virtu",    # virtue, virtuous, virtues
    "valiant", "valour", "valor",
    "holy", "holi",       # holy, holily, holiness
    "heaven",             # heaven, heavenly, heavens
    "truth", "true", "truly",
    "trust",
    "peace",
    "mercy", "merci",
    "hope", "hopeful",
    "faith", "faithful",
    "mild", "meek",
    "smile", "smil",      # smile, smiles, smiled, smiling
    "bright", "brightly",
    "brave", "bravely",
    "glad", "gladly",
    "bless",
    "good",
    "glory", "glorio",    # glory, glorious, gloriously
    "triumph",
    "sweet",
    "tender",
    "rose",               # rose, roses, rosy, rosier (as praise diction)
    "praise",
    "beautif", "beauty",
    "graceful",
    "courage", "courageous",
    "divine",
    "happy", "happi",
    "content", "contentm",
    "loyal", "loyalty",
    "faithful",
})

_NEGATIVE_STEMS: frozenset[str] = frozenset({
    "hate", "hat",        # hate, hates, hated, hateful — NB "hat" also matches "hatred"
    "foul", "foully", "foulness",
    "false", "falsely", "falsehood",
    "vile", "vilely", "vilest",
    "base", "basely", "baseness",  # includes "basely", "baseness"
    "sin", "sinful", "sinner",
    "shame", "shameful", "shamed",
    "dark", "darker", "darkest", "darkness", "darkly",
    "hell", "hellish",
    "damn", "damned",
    "curse", "cursed", "curst",
    "cruel", "cruelly", "cruelty",
    "harsh", "harshly",
    "rank",               # rank, ranker (as in "rank corruption")
    "rude", "rudely", "rudeness",
    "villain", "villainy", "villainous",
    "wrong", "wrongly",
    "wretch", "wretched", "wretchedness",
    "woe", "woeful",
    "grief", "grievous", "grievo",
    "weep", "weepin",     # weep, weeping, weepest
    "ugly",
    "dread", "dreadful", "dreading",
    "frown", "frowning",
    "scorn", "scornful",
    "spite", "spiteful",
    "rage", "raging",
    "angry", "anger", "angrier",
    "grim", "grimly",
    "filth", "filthy",
    "rotten", "rot",
    "corrupt", "corruption",
    "poison", "poisoned", "poisono",
    "venom", "venomo",
    "evil", "evilly",
    "treacher",           # treacherous, treachery
    "lie", "liar",        # CAUTION: "lie" is also "to lie down"
    "murder", "murderer", "murderous",
    "tyrant", "tyranny", "tyrannous",
    "slay", "slew", "slain",
    "blood",              # blood often carries negative valence
    "deadly",
    "betray", "betrayal", "betrayed",
    "monster", "monstrous",
    "beast",              # frequently pejorative in Shakespeare
    "fiend", "fiendish",
    "devil",
    "pain", "painful",
    "bane", "baneful",
    "horrid", "horrible", "horror",
    "fear",
    "terror", "terribl",
    "hideous",
})


# Words that LOOK polarized but are ambiguous — exclude to avoid
# miscounting (e.g., "lie" can mean "recline"; "fair" can mean
# "equitable" or "just enough"). Keep the lists pure.
_IGNORE_POLARIZATION: frozenset[str] = frozenset({
    "fair",   # has enough non-praise senses to be noisy
    "lie",    # recline, deception — ambiguous
    "lies",
    "rose",   # flower vs praise diction — ambiguous
    "hat",    # literal hat, not "hate"
    "rank",   # rank-order, military rank
    "rot",    # onomatopoeia
    "bless",  # verb "to bless" — keep
    "bless",  # keep positive
})


def _word_polarity(word: str) -> int:
    """Return +1 if positive, -1 if negative, 0 if neutral.

    Uses stem/prefix matching: any stem in the set that the word
    starts with produces a match. Words in _IGNORE_POLARIZATION are
    forced to 0 regardless.
    """
    if not word:
        return 0
    w = word.lower()
    if w in _IGNORE_POLARIZATION:
        return 0
    # Positive match
    for stem in _POSITIVE_STEMS:
        if w.startswith(stem):
            return 1
    # Negative match
    for stem in _NEGATIVE_STEMS:
        if w.startswith(stem):
            return -1
    return 0


def update_valence(state: ModelState, token_id: int) -> ModelState:
    # Reset inside speaker-label territory. A new character may bring
    # opposite moral register, so don't carry across.
    if state.speaker_label_state != 0:
        if abs(state.emotional_valence) > 1e-4:
            return state.model_copy(update={"emotional_valence": 0.0})
        return state

    if not state.just_finished_word:
        return state
    word = state.last_completed_word
    if not word:
        return state

    # Skip very short words (articles, pronouns, conjunctions) as
    # function-word noise.
    if len(word) <= 2:
        return state

    pol = _word_polarity(word)

    cur = state.emotional_valence

    if pol == 0:
        # Slow decay toward 0.
        new = cur * 0.96
        if abs(new) < 1e-4:
            new = 0.0
    elif pol > 0:
        # Pull up toward +1 by 0.25 of remaining distance.
        new = cur + 0.25 * (1.0 - cur)
    else:
        # Pull down toward -1 by 0.25 of remaining distance.
        new = cur + 0.25 * (-1.0 - cur)

    # Clip just to be safe against floating-point drift.
    if new > 1.0:
        new = 1.0
    elif new < -1.0:
        new = -1.0

    if abs(new - cur) < 1e-4:
        return state
    return state.model_copy(update={"emotional_valence": new})
