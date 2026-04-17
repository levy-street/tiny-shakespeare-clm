"""Tier 2 — clause-structure tracking.

Runs after `update_pos` (so last_completed_word and prev_completed_word
are current) and before `update_prosody` / `update_flow`. Maintains:

  - clauses_in_sentence: number of clausal breaks (, ; :) since the
    last sentence-end punctuation. Resets to 0 on . ? !.
  - in_dependent_clause: True iff the current clause was opened by a
    subordinating conjunction. Toggled on clause breaks.
  - subject_pronoun: the most recent subject pronoun in the current
    sentence. Used by downstream layers for verb-agreement-aware bias.
    Reset on sentence end.

These fields let later stages (both predict and flow) condition on
syntactic position: after multiple clauses, sentence-end is overdue;
inside a dependent clause starting with "thou", the expected verb form
is "hast" / "wilt" rather than "have" / "will".
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB

# Subject pronouns we track.
_SUBJ_PRONOUNS: frozenset[str] = frozenset({
    "i", "thou", "he", "she", "it", "we", "ye", "you", "they",
})

# Subordinating conjunctions that open dependent clauses.
_SUBORDINATORS: frozenset[str] = frozenset({
    "that", "which", "who", "whom", "whose", "when", "where", "while",
    "whilst", "if", "though", "although", "because", "since", "unless",
    "until", "till", "ere", "lest", "as", "whenever", "wherever",
})


def update_clause(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    clauses = state.clauses_in_sentence
    in_dep = state.in_dependent_clause
    subj = state.subject_pronoun

    # Sentence-end punctuation resets clause state.
    if ch in ".?!":
        clauses = 0
        in_dep = False
        subj = ""
    elif ch in ",;:" and state.speaker_label_state == 0:
        # Clausal break. Increment and reset dependent-clause flag
        # (next clause starts fresh).
        clauses = clauses + 1
        in_dep = False

    # When a word just completed, update subject_pronoun and
    # in_dependent_clause based on the newly completed word.
    if state.just_finished_word and state.last_completed_word:
        w = state.last_completed_word
        # Set/update subject pronoun if we see one that could be a
        # sentence subject. Prefer the FIRST one we see in a sentence
        # (subjects usually come first); don't overwrite once set,
        # except when a fresh clause begins.
        if w in _SUBJ_PRONOUNS and not subj:
            subj = w
        # Detect clause-opening by subordinator. Only tag if it appears
        # at a plausible position (after a comma or at sentence start).
        if w in _SUBORDINATORS:
            # Only flip if the previous char before this word was a
            # clausal break or newline (a fresh clause opening).
            pc = state.prev_char_class
            if pc in (1, 0) and state.chars_since_comma <= 6:
                # After a ", " (comma + space then word) or at clause
                # boundary. Marks a dependent clause opening.
                in_dep = True

    return state.model_copy(
        update={
            "clauses_in_sentence": clauses,
            "in_dependent_clause": in_dep,
            "subject_pronoun": subj,
        }
    )
