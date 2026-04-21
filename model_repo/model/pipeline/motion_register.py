"""Tier 3 FLOW — kinetic register (motion ↔ stasis) tracker.

Tracks `motion_register` in [-1.0 stasis, +1.0 motion]. Captures
whether the recent diction is driving the scene toward MOTION
(chase, march, flee, leap, ride, sail, fall, rise, haste, rush,
hither, forth, away, on) or STASIS (stand, stay, sit, dwell, lie,
abide, remain, pause, linger, here, there, still, yonder, long).

Distinct axis from:
  * martial_register (battlefield vocab)
  * sensory_charge (corporeal ↔ abstract)
  * emotional_valence (moral polarity)
  * mirth / lament / fury / gravitas (mood magnitudes)

Two words like "march" and "stand" share martial/sensory/valence
yet diverge sharply on kinetic mode. Shakespeare's scenes
self-reinforce kinetic mode: once a scene turns kinetic, verbs and
adverbs continue in that mode.

Update rule (on every completed content word):
  * Motion-stem word: pull +0.22 of remaining distance to +1.
  * Stasis-stem word: pull -0.22 of remaining distance to -1.
  * Neutral content word: slow decay toward 0 (×0.965 per word).
  * Function / 1-2-letter words: no-op.

Reset to 0 on speaker-turn boundary (\\n\\n) / in speaker-label
territory — a new character may set a different kinetic mode.

All stems from prior knowledge of Early Modern English kinetic
vocabulary. No corpus statistics.
"""

from __future__ import annotations

from ..state import ModelState


# Motion stems — match with startswith (captures inflections).
_MOTION_STEMS: frozenset[str] = frozenset({
    # Verbs of going / coming / traveling
    "come", "go", "goes", "goe",     # come, comes, coming, cometh
    "went", "gone", "going",
    "run", "runs", "running", "ran",
    "fly", "flies", "flew", "flown", "flying",
    "flee", "fled", "fleeing",
    "march", "marching", "marched",
    "ride", "rides", "riding", "rode", "ridden",
    "sail", "sailing", "sailed",
    "haste", "hasten", "hasted", "hastily",
    "rush", "rushing", "rushed",
    "storm", "storming", "stormed",
    "speed", "speeds", "speeding", "sped",
    "chase", "chasing", "chased",
    "charge", "charging", "charged",
    "strike", "strikes", "striking", "struck",
    "leap", "leaps", "leaping", "leapt", "leaped",
    "soar", "soars", "soaring", "soared",
    "depart", "departing", "departed",
    "fall", "falls", "falling", "fell",
    "rise", "rises", "rising", "rose",   # caution — "rose" also flower
    "arise", "arises", "arising",
    "arose", "arisen",
    "mount", "mounts", "mounting", "mounted",
    "advance", "advancing",
    "retreat", "retreating",
    "pursue", "pursuing", "pursued",
    "seek", "seeks", "seeking", "sought",
    "bring", "bringing", "brought",
    "fetch", "fetching", "fetched",
    "send", "sending", "sent",
    "follow", "following", "followed",
    "ascend", "ascending",
    "descend", "descending",
    "journey", "journeying",
    "travel", "travell", "travelling",
    "voyage",
    "hurry", "hurried", "hurrying",
    "fly'st", "go'st", "com'st",   # contracted thou-forms

    # Direction adverbs
    "hither", "thither", "whither",
    "hence", "thence", "whence",
    "forth", "forward",
    "away", "aback", "aloft", "alofti",
    "onward", "onwards",
    "outward", "outwards",
    "upward", "upwards",
    "downward", "downwards",
    "hitherto", "thitherward",
})

_STASIS_STEMS: frozenset[str] = frozenset({
    # Verbs of dwelling / remaining
    "stand", "stands", "standing", "stood",
    "stay", "stays", "staying", "stayed",
    "sit", "sits", "sitting", "sat",
    "lie", "lies", "lying", "lay", "lain",
    "dwell", "dwells", "dwelling", "dwelt",
    "bide", "bides", "biding", "bided",
    "abide", "abides", "abiding", "abode",
    "rest", "rests", "resting", "rested",
    "wait", "waits", "waiting", "waited",
    "remain", "remains", "remaining", "remained",
    "pause", "pauses", "pausing", "paused",
    "linger", "lingers", "lingering", "lingered",
    "keep", "keeps", "keeping", "kept",
    "hold", "holds", "holding", "held",
    "tarry", "tarries", "tarrying",
    "sojourn",
    "slumber", "slumbering",
    "sleep", "sleeping", "slept",

    # Location / position adverbs
    "here", "there", "yonder",
    "still",               # careful: also modal adverb "still alive"
    "within", "without",
    "long", "ever", "forever", "always",
    "between",
})


# Ambiguous stems to exclude so the register doesn't drift on
# non-kinetic senses.
_IGNORE: frozenset[str] = frozenset({
    "fall",   # autumn noun usage
    "rose",   # flower
    "still",  # "still alive" — adverb sense, not stasis
    "long",   # "long for", "how long" — not stasis
    "ever",   # adverb
    "rest",   # "the rest of them" — noun sense
    "keep",   # noun (castle keep)
    "seek",   # pursuit — not strong motion
    "bring",  # transitive — weak
    "send",   # transitive — weak
    "follow", # many senses
    "mount",  # noun sense (Mount Olivet)
    "wait",   # expectation, not just stasis
})


def _word_kinetic(word: str) -> int:
    """+1 motion, -1 stasis, 0 neutral/ignored."""
    if not word:
        return 0
    w = word.lower()
    if w in _IGNORE:
        return 0
    if w in _MOTION_STEMS:
        return 1
    if w in _STASIS_STEMS:
        return -1
    # Fallback: stem-prefix match for inflections we didn't enumerate.
    # We only use this for longer forms (>= 4 chars) to avoid spurious
    # matches on "go/do" etc.
    if len(w) >= 5:
        for stem in _MOTION_STEMS:
            if len(stem) >= 4 and w.startswith(stem):
                return 1
        for stem in _STASIS_STEMS:
            if len(stem) >= 4 and w.startswith(stem):
                return -1
    return 0


def update_motion_register(state: ModelState, token_id: int) -> ModelState:
    # Reset in speaker-label territory.
    if state.speaker_label_state != 0:
        if abs(state.motion_register) > 1e-4:
            return state.model_copy(update={"motion_register": 0.0})
        return state

    if not state.just_finished_word:
        return state
    word = state.last_completed_word
    if not word:
        return state
    if len(word) <= 2:
        return state

    pol = _word_kinetic(word)
    cur = state.motion_register

    if pol == 0:
        new = cur * 0.965
        if abs(new) < 1e-4:
            new = 0.0
    elif pol > 0:
        new = cur + 0.22 * (1.0 - cur)
    else:
        new = cur + 0.22 * (-1.0 - cur)

    if new > 1.0:
        new = 1.0
    elif new < -1.0:
        new = -1.0
    if abs(new - cur) < 1e-4:
        return state
    return state.model_copy(update={"motion_register": new})
