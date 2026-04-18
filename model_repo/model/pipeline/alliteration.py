"""Tier 2/3 — within-line alliteration tracking.

Shakespeare uses alliteration heavily as a rhetorical device:
"primrose path", "full fathom five", "sound and fury", "shuffle off
this mortal coil", "feather of lead, bright smoke, cold fire, sick
health", "blow, blow, thou winter wind". Existing state tracks
word-level anaphora across lines (recent_line_starters) but nothing
within a single line remembers that the last two content words both
started with 'f'.

This stage maintains two fields:

  line_alliteration_letter  — the lowercase first letter currently
                              alliterated (or "" if no run active)
  line_alliteration_run     — count of consecutive matching content
                              words on this line (>= 2 = active)

Function words (articles, possessives, prepositions, conjunctions,
aux verbs, modals, pronouns, wh, negations, interjections) are
TRANSPARENT: they do not advance the run, nor do they break it. This
matches how alliteration is actually heard — "blow, thou winter wind"
alliterates on 'b-w-w' ignoring "thou".

Reset on newline (fresh line, fresh alliteration scope).

Consumed by predict/alliteration.py at word-start positions to boost
the alliteration letter when the run is active.
"""

from __future__ import annotations

from ..state import ModelState
from .pos import (
    POS_ARTICLE,
    POS_AUX_VERB,
    POS_CONJUNCTION,
    POS_MODAL,
    POS_NEGATION,
    POS_POSSESSIVE,
    POS_PREPOSITION,
    POS_PRONOUN,
    POS_WH,
)


# POS classes that are TRANSPARENT to alliteration (don't count,
# don't break). These are the "glue" words a listener skips when
# perceiving alliteration.
_TRANSPARENT_POS: frozenset[int] = frozenset({
    POS_ARTICLE,
    POS_POSSESSIVE,
    POS_PREPOSITION,
    POS_CONJUNCTION,
    POS_AUX_VERB,
    POS_MODAL,
    POS_PRONOUN,
    POS_NEGATION,
    POS_WH,
})


def update_alliteration(state: ModelState, token_id: int) -> ModelState:
    # Reset on newline — alliteration is bounded to a single line.
    if state.last_char == "\n":
        if state.line_alliteration_letter != "" or state.line_alliteration_run != 0:
            return state.model_copy(update={
                "line_alliteration_letter": "",
                "line_alliteration_run": 0,
            })
        return state

    # Only act when a word has just completed.
    if not state.just_finished_word:
        return state

    w = state.last_completed_word
    if not w:
        return state

    # Skip if the first character isn't a letter (shouldn't happen
    # for completed words, but defensive).
    first = w[0].lower()
    if not ("a" <= first <= "z"):
        return state

    # Transparent POS — preserve the current run as-is.
    if state.last_word_pos in _TRANSPARENT_POS:
        return state

    # Content word. Decide: advance run, or reset it.
    if state.line_alliteration_letter == first:
        new_run = min(state.line_alliteration_run + 1, 8)
        return state.model_copy(update={
            "line_alliteration_run": new_run,
        })
    else:
        return state.model_copy(update={
            "line_alliteration_letter": first,
            "line_alliteration_run": 1,
        })
