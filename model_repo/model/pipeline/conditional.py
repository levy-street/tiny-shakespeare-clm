"""Tier 2 — conditional / concessive discourse FSM.

Shakespeare sentences frequently use a PROTASIS→APODOSIS structure:

  If thou lovest me, [then] say so.
  Though he be honest, he is rash.
  When I am gone, weep not.
  Since you ask, I will answer.

The APODOSIS (main clause) reliably begins with:
  - a subject pronoun (I, thou, he, she, we, they, ye, you, it)
  - a modal / imperative verb (shall, will, must, can; go, come, hear,
    speak, tell, stay, leave, look, mark, think)
  - the adverbs "then" or "so"
  - "there" / "here" existential

No existing state signals this. subord_depth tracks nested-clause
depth but doesn't fire a specific "apodosis pending" signal when the
protasis closes (typically by comma). This stage closes that gap.

Fires on EVERY token (not just word completion) because the protasis→
apodosis transition is driven by a comma, which is a single char.

Reset on sentence-end (. ? !) and on turn boundary (consecutive
newlines >= 2).

No corpus statistics — the subordinator inventory is hand-curated
English / Early Modern English grammar.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


# Mode enum.
MODE_NONE = 0
MODE_PROTASIS = 1
MODE_APODOSIS = 2
MODE_RESOLVED = 3


# Opener dictionary — key = word (lowercased, no punctuation), value =
# opener ID. 0 reserved for "none".
_OPENERS: dict[str, int] = {
    "if": 1,
    "though": 2,
    "when": 3,
    "since": 4,
    "unless": 5,
    "lest": 6,
    "whereas": 7,
    "albeit": 8,
    "although": 9,
    "while": 10,
    # "whenever": use same slot as when
    "whenever": 3,
    "whensoever": 3,
    # "an" sometimes "an if" in EME — treat weakly
}


def update_conditional(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    mode = state.conditional_mode
    opener = state.conditional_opener
    age = state.conditional_age

    # Sentence-end fully resets.
    if ch in ".?!":
        if mode != 0 or opener != 0 or age != 0:
            return state.model_copy(update={
                "conditional_mode": 0,
                "conditional_opener": 0,
                "conditional_age": 0,
            })
        return state

    # Turn boundary also resets. consecutive_newlines is pre-update
    # but good enough as a signal.
    if ch == "\n" and state.consecutive_newlines >= 1:
        # This will be consecutive_newlines 2+ after the update. Reset.
        if mode != 0 or opener != 0 or age != 0:
            return state.model_copy(update={
                "conditional_mode": 0,
                "conditional_opener": 0,
                "conditional_age": 0,
            })
        return state

    updates: dict = {}

    # Word-completion events: potentially open the protasis, or age
    # the apodosis counter.
    if state.just_finished_word and state.last_completed_word:
        word = state.last_completed_word.lower()

        # Opener detection — only if we're in MODE_NONE and this is
        # sentence-opening-ish (words_in_sentence <= 1, i.e., this is
        # the first or second word of the sentence — accommodates
        # light fillers like "And if", "But though").
        if (
            mode == MODE_NONE
            and state.speaker_label_state == 0
            and state.words_in_sentence <= 2
            and word in _OPENERS
        ):
            mode = MODE_PROTASIS
            opener = _OPENERS[word]
            age = 0

        # Aging: in PROTASIS or APODOSIS or RESOLVED, count words.
        elif mode in (MODE_PROTASIS, MODE_APODOSIS, MODE_RESOLVED):
            # In APODOSIS, a substantive word transitions to RESOLVED.
            # Substantive = anything that isn't a comma/conjunction/
            # opener filler. Since we're at word-completion, we have
            # a word: always transition after the first word post-comma.
            if mode == MODE_APODOSIS and age >= 0:
                # First word of apodosis has now been completed → RESOLVED.
                mode = MODE_RESOLVED
                age = 0
            else:
                age = min(age + 1, 20)

    # Char-level event: comma closes the protasis → enter APODOSIS.
    # Require at least 2 words since opener to avoid false closes on
    # "If, as I said, ..." (nested commas within protasis). Actually
    # EME uses parentheticals heavily so require 3 words since opener.
    if ch == "," and mode == MODE_PROTASIS and age >= 3:
        mode = MODE_APODOSIS
        age = 0

    # ; also closes protasis into apodosis (strong clause-break).
    if ch == ";" and mode == MODE_PROTASIS and age >= 2:
        mode = MODE_APODOSIS
        age = 0

    # Write back if anything changed.
    if (
        mode != state.conditional_mode
        or opener != state.conditional_opener
        or age != state.conditional_age
    ):
        updates["conditional_mode"] = mode
        updates["conditional_opener"] = opener
        updates["conditional_age"] = age
        return state.model_copy(update=updates)
    return state
