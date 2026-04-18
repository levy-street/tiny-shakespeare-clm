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
    depth = state.clause_depth
    wis = state.words_in_subordinate

    # Sentence-end punctuation resets clause state.
    if ch in ".?!":
        clauses = 0
        in_dep = False
        subj = ""
        depth = 0
        wis = 0
    elif ch in ",;:" and state.speaker_label_state == 0:
        # Clausal break. Increment and reset dependent-clause flag
        # (next clause starts fresh).
        clauses = clauses + 1
        in_dep = False

    # Speaker-turn boundary: reset subordination state.
    if state.consecutive_newlines >= 2 and ch == "\n":
        depth = 0
        wis = 0

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
            # A subordinator opens a dependent clause when it appears
            # right after a clause boundary. We detect this by checking
            # that chars_since_comma is close to len(w)+2 (word length
            # plus ", " separator) OR chars_since_sentence_end is small
            # (sentence start). Being lenient: most subordinators ARE
            # opening clauses in practice.
            wl = len(w)
            near_comma = state.chars_since_comma <= wl + 3
            near_sent = state.chars_since_sentence_end <= wl + 3
            if near_comma or near_sent:
                in_dep = True
                # Increment subordinator depth, capped at 3.
                if depth < 3:
                    depth = depth + 1
                wis = 0  # fresh subordinate clause — reset word count
        else:
            # Any other word completion bumps the in-subordinate count
            # if we're inside one.
            if depth > 0:
                wis = wis + 1

    return state.model_copy(
        update={
            "clauses_in_sentence": clauses,
            "in_dependent_clause": in_dep,
            "subject_pronoun": subj,
            "clause_depth": depth,
            "words_in_subordinate": wis,
        }
    )
