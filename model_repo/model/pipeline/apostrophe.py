"""Tier-3 apostrophe-mode tracker.

Maintains `apostrophe_mode` (0..3), `apostrophe_words_since_cue`, and
`apostrophe_target`. See state/schema.py for the full contract.

Apostrophe is the Shakespearean rhetorical figure of addressing an
absent/abstract/inanimate entity: "O Fortune!", "Ye gods!", "O night!".
It is a distinct lyric register, orthogonal to mood (grief, fury) and
to literal-character addressee (vocative).

Updates happen at word completion (`just_finished_word`) so we can act
on `last_completed_word`. The stage is small and self-contained: reads
only a handful of state fields and writes three.

Runs AFTER update_basic_counters (for `just_finished_word` and
`last_completed_word`) and AFTER update_linguistic (for sentence /
line context). Runs BEFORE the predict-facing mood and flow stages so
downstream predict layers can condition on the apostrophe mode.
"""

from __future__ import annotations

from ..state import ModelState

# Interjection particles that open an apostrophe. Shakespeare uses
# all of these to summon / invoke:
#   "O Fortune, I will not trust thee"
#   "Oh heaven, be witness"
#   "Ah, gentle night!"
#   "Alas, poor Yorick!"
#   "Ye gods, look down!"
#   "Hark, the lark!"
#   "Lo, where he comes!"
#   "Fie, how my bones ache!"
#   "Alack, what blood is this?"
_APOSTROPHE_CUES: frozenset[str] = frozenset({
    "o", "oh", "ah", "alas", "alack", "ye", "yea",
    "hark", "lo", "fie",
})

# Abstract / figurative / inanimate targets that a speaker can
# apostrophize. Organised loosely (see state docstring for family
# hints). Case-folded.
_APOSTROPHE_TARGETS: frozenset[str] = frozenset({
    # Celestial / cosmic
    "heaven", "heavens", "sky", "skies", "star", "stars", "moon",
    "sun", "night", "day", "morn", "morrow", "dawn", "dusk",
    # Mortality / fate
    "death", "grave", "fate", "fates", "fortune", "fortunes",
    "time", "hour", "doom", "destiny",
    # Affective / moral
    "love", "heart", "soul", "mind", "spirit", "hope", "faith",
    "honour", "honor", "truth", "peace", "pity", "mercy",
    "beauty", "virtue", "shame", "pride", "conscience",
    # Emotion-as-addressee
    "grief", "sorrow", "joy", "despair", "rage", "wrath", "fear",
    # Natural / elemental
    "earth", "sea", "wind", "fire", "nature", "world",
    # Divine / supernatural
    "god", "gods", "jove", "heavens", "hell", "angels", "saints",
    # Abstract faculties / muses
    "muse", "fancy", "reason", "thought",
    # Body-as-apostrophe (common in soliloquy)
    "hand", "eyes", "tongue",
})

# Imperative verbs commonly directed at an apostrophized target.
# "Come, night!", "Speak, heaven!", "Hide, love!", "Stand, friends!".
_APOSTROPHE_IMPERATIVES: frozenset[str] = frozenset({
    "come", "speak", "hide", "show", "stand", "hold", "hear",
    "mark", "behold", "look", "see", "shine", "rise", "fall",
    "weep", "rest", "sleep", "wake", "bear", "take", "give",
    "bring", "send", "grant", "shield", "guard", "save", "help",
    "pity", "pardon", "forgive", "curse", "smite", "strike",
})

# Words that, when referring to a CONCRETE interlocutor (character in
# scene), typically signal we are NOT in apostrophe mode. Used as a
# soft demoter: if the current sentence centres on "my lord" / "sir"
# / a known speaker's name, we clear apostrophe_mode.
_CONCRETE_ADDRESS_HINTS: frozenset[str] = frozenset({
    "sir", "madam", "lord", "lords", "lady", "ladies",
    "master", "mistress", "friend", "friends", "gentlemen",
    "knave", "villain", "sirrah",
})

_DECAY_WINDOW = 3  # words after cue 1 before decay to 0 if no target


def update_apostrophe(state: ModelState, token_id: int) -> ModelState:
    # Cheap fast-exit: no word completion => no mode-state change
    # (apostrophe transitions all happen at word boundaries).
    # EXCEPT turn-boundary reset, which fires on the \n\n.
    mode = state.apostrophe_mode

    # Turn-boundary reset.
    if state.consecutive_newlines >= 2 and (
        mode != 0
        or state.apostrophe_words_since_cue != 0
        or state.apostrophe_target
    ):
        return state.model_copy(
            update={
                "apostrophe_mode": 0,
                "apostrophe_words_since_cue": 0,
                "apostrophe_target": "",
            }
        )

    if not state.just_finished_word:
        return state
    lcw = (state.last_completed_word or "").lower().strip("'")
    if not lcw:
        return state

    words_since = state.apostrophe_words_since_cue
    target = state.apostrophe_target

    # Update rules, evaluated in order:
    # 1. If current word is a concrete-address hint AND it follows a
    #    possessive ("my lord", "thy sir"), demote mode.
    # 2. If current word is an apostrophe cue interjection AND we're at
    #    sentence-start context, ENTER mode 1 (or reinforce to mode 3
    #    if we were already 2).
    # 3. If current word is an apostrophe target AND we were in mode 1,
    #    promote to mode 2 and record the target.
    # 4. If current word is an apostrophe imperative AND we were in
    #    mode 2 with a target locked, promote to mode 3.
    # 5. Otherwise, increment words_since_cue. If mode == 1 and the
    #    window expires (_DECAY_WINDOW), decay to 0 (interjection
    #    was filler, not apostrophe).

    # Rule 1: concrete-address demoter.
    if lcw in _CONCRETE_ADDRESS_HINTS and mode >= 1:
        # Only demote if preceded by a possessive or a comma (concrete
        # address shape: "my lord," / "good sir"). Weak signal; use
        # previous word if available.
        prev = (state.prev_completed_word or "").lower().strip("'")
        if prev in ("my", "thy", "your", "our", "good", "fair", "dear",
                    "noble", "sweet", "gentle"):
            return state.model_copy(
                update={
                    "apostrophe_mode": 0,
                    "apostrophe_words_since_cue": 0,
                    "apostrophe_target": "",
                }
            )

    # Rule 2: apostrophe cue particle.
    if lcw in _APOSTROPHE_CUES:
        # Sentence-start context: few words into sentence. Use
        # words_in_sentence from counters if available; otherwise
        # approximate with chars_since_sentence_end threshold.
        at_sentence_start = (
            getattr(state, "words_in_sentence", 999) <= 1
            or state.chars_since_sentence_end <= 4
            or state.consecutive_newlines >= 1
        )
        if at_sentence_start:
            # Reinforce if already active; else enter mode 1.
            if mode >= 2:
                new_mode = 3
            else:
                new_mode = max(mode, 1)
            return state.model_copy(
                update={
                    "apostrophe_mode": new_mode,
                    "apostrophe_words_since_cue": 0,
                    # keep existing target if we had one locked.
                    "apostrophe_target": target,
                }
            )

    # Rule 3: target noun after a cue — promote 1 -> 2.
    if mode >= 1 and lcw in _APOSTROPHE_TARGETS:
        # Don't regress if already at 3.
        new_mode = max(mode, 2)
        return state.model_copy(
            update={
                "apostrophe_mode": new_mode,
                "apostrophe_words_since_cue": 0,
                "apostrophe_target": lcw,
            }
        )

    # Rule 4: imperative verb while active with target locked — lock.
    if mode == 2 and target and lcw in _APOSTROPHE_IMPERATIVES:
        return state.model_copy(
            update={
                "apostrophe_mode": 3,
                "apostrophe_words_since_cue": 0,
                "apostrophe_target": target,
            }
        )

    # Rule 5: decay / idle.
    if mode == 0 and words_since == 0 and not target:
        return state
    new_words_since = words_since + 1
    # Window-based decay when priming didn't resolve to a target.
    if mode == 1 and new_words_since >= _DECAY_WINDOW:
        return state.model_copy(
            update={
                "apostrophe_mode": 0,
                "apostrophe_words_since_cue": 0,
                "apostrophe_target": "",
            }
        )
    # Mild sustained-decay for modes 2/3: if we go 6+ words without any
    # reinforcement, drop one level.
    if mode >= 2 and new_words_since >= 6:
        return state.model_copy(
            update={
                "apostrophe_mode": mode - 1,
                "apostrophe_words_since_cue": 0,
                "apostrophe_target": target if mode - 1 >= 2 else "",
            }
        )
    if new_words_since != words_since:
        return state.model_copy(
            update={"apostrophe_words_since_cue": new_words_since}
        )
    return state
