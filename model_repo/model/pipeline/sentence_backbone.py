"""Sentence backbone tracker.

Maintains two simple booleans on the sentence currently in progress:
  * sentence_has_subject — seen a subject-like completed word since
    the last terminal punctuation / turn boundary.
  * sentence_has_verb    — seen a finite-verb-like completed word
    likewise.

A well-formed English sentence normally contains BOTH. If we reach a
word-end without a verb after 3+ words, terminal punctuation is almost
certainly premature. If both are present and we've written 4+ words,
a terminal punctuation becomes plausible.

Heuristics (reading last_word_pos from pipeline/pos.py):
  Subject candidates:
    POS_PRONOUN, POS_POSSESSIVE, POS_PROPER_NOUN, POS_NOUN, POS_WH,
    POS_ARTICLE (satisfies subject as "the" introducing a subject NP)
  Finite verb candidates:
    POS_AUX_VERB, POS_MODAL, POS_VERB, POS_VERB_ED
  (POS_VERB_ING alone is a participle, NOT a finite verb.)

Resets on PUNCT_END and on speaker-turn boundary. Must run AFTER
pipeline/pos.py so last_word_pos is fresh.
"""

from __future__ import annotations

from ..state import ModelState
from .linguistic import PUNCT_END
from .pos import (
    POS_ARTICLE,
    POS_AUX_VERB,
    POS_MODAL,
    POS_NOUN,
    POS_POSSESSIVE,
    POS_PRONOUN,
    POS_PROPER_NOUN,
    POS_VERB,
    POS_VERB_ED,
    POS_WH,
)

# POS classes that count as a "subject" candidate.
_SUBJECT_POS: frozenset[int] = frozenset({
    POS_PRONOUN,
    POS_POSSESSIVE,
    POS_PROPER_NOUN,
    POS_NOUN,
    POS_WH,
    POS_ARTICLE,
})

# POS classes that count as a "finite verb" candidate.
_VERB_POS: frozenset[int] = frozenset({
    POS_AUX_VERB,
    POS_MODAL,
    POS_VERB,
    POS_VERB_ED,
})


def update_sentence_backbone(state: ModelState, token_id: int) -> ModelState:
    ch = state.last_char
    cls = state.last_char_class

    # Sentence-end: reset both backbone flags.
    if cls == PUNCT_END:
        if state.sentence_has_subject or state.sentence_has_verb:
            return state.model_copy(
                update={
                    "sentence_has_subject": False,
                    "sentence_has_verb": False,
                }
            )
        return state

    # Speaker-turn boundary: reset.
    if state.consecutive_newlines >= 2 and ch == "\n":
        if state.sentence_has_subject or state.sentence_has_verb:
            return state.model_copy(
                update={
                    "sentence_has_subject": False,
                    "sentence_has_verb": False,
                }
            )
        return state

    # Only update at word completion.
    if not state.just_finished_word:
        return state
    # Inside a speaker label — that's a label, not a sentence word.
    if state.speaker_label_state != 0:
        return state

    pos = state.last_word_pos
    new_subj = state.sentence_has_subject
    new_verb = state.sentence_has_verb

    if not new_subj and pos in _SUBJECT_POS:
        new_subj = True
    if not new_verb and pos in _VERB_POS:
        new_verb = True

    if new_subj == state.sentence_has_subject and new_verb == state.sentence_has_verb:
        return state
    return state.model_copy(
        update={
            "sentence_has_subject": new_subj,
            "sentence_has_verb": new_verb,
        }
    )
