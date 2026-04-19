"""Antithesis / rhetorical-contrast state tracker.

Shakespeare's signature rhetorical device is the two-part contrast:

  "To be, or not to be"
  "Not that I loved Caesar less, but that I loved Rome more"
  "Neither a borrower nor a lender be"
  "More in sorrow than in anger"
  "The fault, dear Brutus, is not in our stars / But in ourselves"
  "To err is human, to forgive divine"
  "What's done is done"  (not — but the paired antithesis pattern)

The machinery has three signals:
  1. an OPENER word — "not", "nor", "neither", "either", "rather",
     "more", "less", "though"  — arms the contrast
  2. a PIVOT word — "but", "or", "nor", "than", "yet", "else" — flips
     into the complement half
  3. distance from each — used to decay state when the contrast never
     materializes and to time sentence closure in the complement half.

State transitions (applied at word completion):
  NONE + opener       -> OPENER_SEEN (words_since_opener = 0)
  OPENER_SEEN + pivot -> PIVOTED (words_since_pivot = 0; clear opener counter)
  OPENER_SEEN + non-pivot word -> stay OPENER_SEEN; increment counter.
      If counter reaches 7 without a pivot, return to NONE.
  PIVOTED + any word -> stay PIVOTED; increment counter. After 6 words
      in the complement without a new pivot or sentence-end, decay
      back to NONE.
  Any state + sentence-end punctuation (. ? !) -> NONE.
  Any state + speaker-turn boundary (blank line) -> NONE.

The opener "not" is intentionally broad — it fires many times without
a proper "but" follow-up. That's fine: the decay brings us back to
NONE, and the predict layer only weights the signal gently.
"""

from __future__ import annotations

from ..state import ModelState


# Pure opener words — when seen, arm the contrast-pending state.
# "not" and "nor" are also pivot words (nor is pivot if we're already
# in OPENER_SEEN); they're handled specially in the transition logic.
_ANT_OPENERS: frozenset[str] = frozenset({
    "not", "neither", "either", "rather", "more", "less", "though",
    "although", "whether",
})

# Pivot words — when seen in OPENER_SEEN state, flip to PIVOTED.
_ANT_PIVOTS: frozenset[str] = frozenset({
    "but", "or", "nor", "than", "yet", "else",
})

ANT_NONE = 0
ANT_OPENER = 1
ANT_PIVOTED = 2

_OPENER_DECAY_AFTER = 7  # words without a pivot
_PIVOT_DECAY_AFTER = 6   # words after pivot


def update_antithesis(state: ModelState, token_id: int) -> ModelState:
    # Sentence-end: reset.
    if state.last_char in (".", "?", "!"):
        if (
            state.antithesis_state != ANT_NONE
            or state.antithesis_words_since_opener != 0
            or state.antithesis_words_since_pivot != 0
        ):
            return state.model_copy(
                update={
                    "antithesis_state": ANT_NONE,
                    "antithesis_words_since_opener": 0,
                    "antithesis_words_since_pivot": 0,
                }
            )
        return state

    # Speaker-turn boundary: reset.
    if state.consecutive_newlines >= 2 and state.last_char == "\n":
        if (
            state.antithesis_state != ANT_NONE
            or state.antithesis_words_since_opener != 0
            or state.antithesis_words_since_pivot != 0
        ):
            return state.model_copy(
                update={
                    "antithesis_state": ANT_NONE,
                    "antithesis_words_since_opener": 0,
                    "antithesis_words_since_pivot": 0,
                }
            )
        return state

    # Only transition on word completion.
    if not state.just_finished_word:
        return state

    w = state.last_completed_word.lower() if state.last_completed_word else ""
    if not w:
        return state

    st = state.antithesis_state
    so = state.antithesis_words_since_opener
    sp = state.antithesis_words_since_pivot

    if st == ANT_PIVOTED:
        sp += 1
        if sp >= _PIVOT_DECAY_AFTER:
            # Complement half is played out — return to neutral.
            st = ANT_NONE
            sp = 0
        # A second pivot while already pivoted keeps state but resets
        # the counter (new contrast within the complement).
        if w in _ANT_PIVOTS:
            sp = 0
        # A fresh opener inside the complement re-arms a nested contrast.
        if w in _ANT_OPENERS:
            st = ANT_OPENER
            so = 0
            sp = 0
    elif st == ANT_OPENER:
        if w in _ANT_PIVOTS:
            st = ANT_PIVOTED
            sp = 0
            so = 0
        else:
            so += 1
            if w in _ANT_OPENERS:
                # Fresh opener replaces the old one, resetting the clock.
                so = 0
            if so >= _OPENER_DECAY_AFTER:
                st = ANT_NONE
                so = 0
                sp = 0
    else:  # ANT_NONE
        if w in _ANT_OPENERS:
            st = ANT_OPENER
            so = 0
            sp = 0

    if (
        st == state.antithesis_state
        and so == state.antithesis_words_since_opener
        and sp == state.antithesis_words_since_pivot
    ):
        return state
    return state.model_copy(
        update={
            "antithesis_state": st,
            "antithesis_words_since_opener": so,
            "antithesis_words_since_pivot": sp,
        }
    )
