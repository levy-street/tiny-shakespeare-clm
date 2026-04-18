"""List-parallelism pipeline stage.

Shakespeare uses comma-separated list structures heavily:

  - "X, Y, and Z"
  - "by heaven, by earth, by all that's holy"
  - "nor A, nor B, nor C"
  - "to die, to sleep, to dream"

This stage tracks list-progression state so predict layers can bias
toward parallel continuations (alliterative starts, parallel POS,
and the closing conjunction "and" / "or" / "nor").

State updated:
  - commas_since_sent_end: running count of comma-class punctuation
    since the last sentence-end punctuation (. ? !). Also reset on
    speaker-turn boundary.
  - list_item_pending: True immediately after a comma until the
    following word starts (captures the "first word after comma"
    position).
  - list_last_item_first_letter: first letter (lowercased) of the
    most-recent post-comma word.
  - list_parallel_run: count of consecutive items (including the
    most-recent) that shared the same first letter as the prior
    item. >=2 signals an alliterative parallel list.
  - list_first_item_pos: POS tag of the first post-comma word of
    the current sentence.

Design:
  - On PUNCT_MID (, ; :): bump comma counter; arm list_item_pending.
  - On PUNCT_END (. ? !): reset everything.
  - On speaker-turn boundary (consecutive_newlines >= 2): reset.
  - On first letter of a word while list_item_pending: record
    the first letter; compare against prior list_last_item_first_letter
    to update list_parallel_run.
  - On word completion (just_finished_word): if list_item_pending,
    finalize — mark pending=False, update first_item_pos if empty.

Runs after update_pos (needs last_word_pos) and after update_linguistic
(needs last_char_class and just_finished_word).
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB
from .linguistic import (
    LOWER_CONS,
    LOWER_VOWEL,
    NEWLINE,
    PUNCT_END,
    PUNCT_MID,
    SPACE,
    UPPER,
)


def update_list_structure(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]
    cls = state.last_char_class  # already updated by linguistic

    # Speaker-turn boundary reset.
    turn_reset = (
        state.consecutive_newlines >= 2 and ch == "\n"
    ) or state.speaker_label_state != 0

    # Sentence-end reset.
    sent_end_reset = cls == PUNCT_END

    if turn_reset or sent_end_reset:
        return state.model_copy(
            update={
                "commas_since_sent_end": 0,
                "list_item_pending": False,
                "list_last_item_first_letter": "",
                "list_parallel_run": 0,
                "list_first_item_pos": 0,
            }
        )

    commas = state.commas_since_sent_end
    item_pending = state.list_item_pending
    last_first = state.list_last_item_first_letter
    parallel_run = state.list_parallel_run
    first_item_pos = state.list_first_item_pos

    # On comma/semicolon/colon, bump counter and arm a pending list item.
    # Skip ":" when inside a speaker label (handled elsewhere by FSM).
    if cls == PUNCT_MID:
        commas += 1
        item_pending = True
        return state.model_copy(
            update={
                "commas_since_sent_end": min(commas, 15),
                "list_item_pending": item_pending,
                "list_last_item_first_letter": last_first,
                "list_parallel_run": parallel_run,
                "list_first_item_pos": first_item_pos,
            }
        )

    # When list_item_pending and we see the first letter of a word,
    # capture it. A "first letter" is any letter that increments
    # letter_run_len to exactly 1.
    is_letter = cls in (UPPER, LOWER_VOWEL, LOWER_CONS)
    if item_pending and is_letter and state.letter_run_len == 1:
        first_letter = ch.lower()
        if last_first and first_letter == last_first:
            parallel_run = min(parallel_run + 1, 6)
        else:
            parallel_run = 1  # fresh item sets run=1
        last_first = first_letter
        # Don't clear pending yet — wait for word completion.

    # When word completes and list_item_pending was armed, finalize.
    if state.just_finished_word and item_pending:
        # Record POS of the first list item in this sentence if we
        # haven't already.
        if first_item_pos == 0:
            first_item_pos = state.last_word_pos
        item_pending = False

    return state.model_copy(
        update={
            "commas_since_sent_end": commas,
            "list_item_pending": item_pending,
            "list_last_item_first_letter": last_first,
            "list_parallel_run": parallel_run,
            "list_first_item_pos": first_item_pos,
        }
    )
