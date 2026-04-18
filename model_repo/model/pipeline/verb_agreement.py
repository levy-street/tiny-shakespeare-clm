"""Subject–verb agreement expectation pipeline stage.

Maintains `state.verb_agreement` — a small integer FSM that captures,
once a subject has been identified in the current clause, what
morphology the upcoming main verb is expected to carry in Early
Modern English.

Agreement classes (see schema for semantics):
  VA_NONE=0, VA_THOU=1, VA_THIRD_SG=2, VA_FIRST_SG=3, VA_PLURAL=4,
  VA_IMPERATIVE=5

Transitions:

  - Reset to VA_NONE on sentence-end punctuation (. ! ?) or at a
    CONJUNCTION that also resets the clause slot.

  - When a word just completed AND the previous clause_slot was
    FRESH AND the new clause_slot is HAS_SUBJ, read the last
    completed word and classify it into an agreement class. This is
    the "subject is now identified" transition.

  - When a word just completed AND clause_slot transitioned to
    HAS_VERB (verb role filled), leave the agreement class alone —
    but if it was VA_NONE, promote to VA_IMPERATIVE (bare verb with
    no preceding subject).

  - Clausal break (, ; :) resets to VA_NONE so a new sub-clause
    gets a fresh subject-agreement expectation.

This stage MUST run after `update_clause_slot` so it can observe
slot transitions.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB

VA_NONE = 0
VA_THOU = 1
VA_THIRD_SG = 2
VA_FIRST_SG = 3
VA_PLURAL = 4
VA_IMPERATIVE = 5


# Direct subject pronoun → agreement class.
_PRONOUN_AGREEMENT: dict[str, int] = {
    "thou": VA_THOU,
    # "thee" appears as a subject in some dialect/inverted forms
    # ("prithee thee"); very rare as true subject — map to THOU
    # for safety when it IS a subject.
    "thee": VA_THOU,
    "i": VA_FIRST_SG,
    "he": VA_THIRD_SG,
    "she": VA_THIRD_SG,
    "it": VA_THIRD_SG,
    "this": VA_THIRD_SG,
    "that": VA_THIRD_SG,
    "one": VA_THIRD_SG,
    "who": VA_THIRD_SG,
    "whoso": VA_THIRD_SG,
    "whoever": VA_THIRD_SG,
    "which": VA_THIRD_SG,
    # Plural / 2nd-person-formal (historically plural, synced to plural verb):
    "we": VA_PLURAL,
    "they": VA_PLURAL,
    "you": VA_PLURAL,
    "ye": VA_PLURAL,
    "these": VA_PLURAL,
    "those": VA_PLURAL,
    "both": VA_PLURAL,
}


# Interjections that often precede an imperative: "O come!", "Alas,
# hear me!". We treat a fresh clause opening with one of these as
# still VA_NONE; the next verb will promote to VA_IMPERATIVE.
# (No special handling needed here — the HAS_VERB transition
# handles it.)

# Short noun heuristic: if the subject is a completed word that is
# neither a pronoun in the table above nor ends in a plural marker,
# assume 3rd-singular.
def _noun_agreement(word: str) -> int:
    if not word:
        return VA_NONE
    w = word.lower()
    # Obvious plural markers: "-s" tail where not a known singular
    # word. Early Modern Shakespeare has many "-s"-final singular
    # forms ("princess", "witness", "cause", "was") but as a prior,
    # most "-s" nouns in subject position are plural in Shakespeare's
    # prose. We apply only when the word is 4+ chars and ends in
    # "s" but not "ss", "us", "is", "as" (a very small disambiguator).
    if len(w) >= 4 and w.endswith("s") and not (
        w.endswith("ss") or w.endswith("us") or w.endswith("is") or w.endswith("as")
    ):
        return VA_PLURAL
    return VA_THIRD_SG


def update_verb_agreement(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]
    va = state.verb_agreement

    # Sentence-end or clausal break: reset.
    if ch in ".?!":
        if va != VA_NONE:
            return state.model_copy(update={"verb_agreement": VA_NONE})
        return state

    if ch in ",;:" and state.speaker_label_state == 0:
        if va != VA_NONE:
            return state.model_copy(update={"verb_agreement": VA_NONE})
        return state

    # Only transition on word completion.
    if not (state.just_finished_word and state.last_completed_word):
        return state

    # Detect "subject just landed": clause_slot moved FRESH → HAS_SUBJ.
    # Note: update_clause_slot ran before us, so state.clause_slot is
    # the POST-transition value. We can't easily observe the FROM
    # state here (it's been overwritten), but we CAN check: if
    # clause_slot == HAS_SUBJ AND va == VA_NONE, we're at the
    # exact word that just filled the subject role.
    if state.clause_slot == 1 and va == VA_NONE:
        w = state.last_completed_word.lower()
        cls = _PRONOUN_AGREEMENT.get(w)
        if cls is None:
            cls = _noun_agreement(w)
        if cls != VA_NONE:
            return state.model_copy(update={"verb_agreement": cls})
        return state

    # Detect imperative: clause_slot == HAS_VERB AND va == VA_NONE
    # (verb appeared with no subject ever having been filled).
    if state.clause_slot == 2 and va == VA_NONE:
        return state.model_copy(update={"verb_agreement": VA_IMPERATIVE})

    return state
