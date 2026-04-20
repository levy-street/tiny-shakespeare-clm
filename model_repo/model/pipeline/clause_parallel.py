"""Tier 2 — intra-sentence clause-parallelism tracker.

Records the first letter of each clause within the current sentence.
When two consecutive clauses (separated by comma or semicolon) open
with the same first word or letter, the pattern tends to continue:

  "I came, I saw, I conquered."
  "She is fair, she is wise, she is true."
  "Speak soft, speak low, speak truly."

This is a TURN-INTERNAL echo pattern distinct from:
  - anaphora (line-starter tracking — newline-boundary anchored)
  - antithesis (opener/pivot contrast structure)
  - list_structure (coarser list FSM, items are separated by
                   commas but no first-letter echo pressure)

Fields maintained:
  clause_opener_letter       — first letter of current clause's opener
  prev_clause_opener_letter  — first letter of prior clause's opener
                               within the same sentence
  clauses_in_sentence_index  — 0 at sentence start, +1 per clause break

Transitions:
  - On "," or ";" (clause break): rotate opener letters. The CURRENT
    clause's letter becomes PREV; CURRENT is cleared to "". Index++.
  - On first word completed in a clause where clause_opener_letter
    is empty: record its first letter.
  - On "." "?" "!" (sentence end): reset all three to defaults.
  - On turn boundary (\\n\\n): reset.

No corpus statistics — this is purely a structural rhetoric tracker.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


_CLAUSE_BREAKS: frozenset[str] = frozenset({",", ";"})
_SENT_END: frozenset[str] = frozenset({".", "?", "!"})


def update_clause_parallel(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    col = state.clause_opener_letter
    pcol = state.prev_clause_opener_letter
    idx = state.clauses_in_sentence_index

    # Sentence end resets.
    if ch in _SENT_END:
        if col or pcol or idx != 0:
            return state.model_copy(update={
                "clause_opener_letter": "",
                "prev_clause_opener_letter": "",
                "clauses_in_sentence_index": 0,
            })
        return state

    # Turn boundary resets (blank line between turns).
    if ch == "\n" and state.consecutive_newlines >= 1:
        if col or pcol or idx != 0:
            return state.model_copy(update={
                "clause_opener_letter": "",
                "prev_clause_opener_letter": "",
                "clauses_in_sentence_index": 0,
            })
        return state

    # Clause break: rotate.
    if ch in _CLAUSE_BREAKS:
        # If the current clause has no opener recorded (e.g., double
        # comma pathological case), keep pcol as-is; otherwise roll
        # col → pcol.
        if col:
            new_pcol = col
        else:
            new_pcol = pcol
        if new_pcol != pcol or col != "" or idx == state.clauses_in_sentence_index and (col or pcol):
            return state.model_copy(update={
                "clause_opener_letter": "",
                "prev_clause_opener_letter": new_pcol,
                "clauses_in_sentence_index": min(idx + 1, 10),
            })
        return state.model_copy(update={
            "clauses_in_sentence_index": min(idx + 1, 10),
        })

    # Speaker-label: don't mess with.
    if state.speaker_label_state != 0:
        if col or pcol or idx != 0:
            return state.model_copy(update={
                "clause_opener_letter": "",
                "prev_clause_opener_letter": "",
                "clauses_in_sentence_index": 0,
            })
        return state

    # Word completion: if clause opener letter is unset, record it.
    if state.just_finished_word and state.last_completed_word and not col:
        first_letter = state.last_completed_word[0].lower()
        # Only record if it's a letter (skip apostrophe-starters or
        # other weird buffer tails).
        if first_letter.isalpha():
            return state.model_copy(update={
                "clause_opener_letter": first_letter,
            })
        return state

    return state
