"""Within-turn line-word-count cadence tracker.

Maintains two fields:

  line_word_count: int
    Completed-word count in the current body line. Incremented at
    just_finished_word when we're not inside a speaker label and
    not mid-newline. Reset to 0 at a body newline (the \n that
    triggers consecutive_newlines == 1) and at the turn boundary
    (consecutive_newlines >= 2).

  recent_line_word_counts: tuple[int, ...]  (cap 3, most-recent first)
    Rolling history of completed line word counts WITHIN the current
    turn. Pushed at the body-newline moment — before turn_progress
    resets in-line state, we capture the count the current line just
    completed.

Together these let a downstream predict layer know whether the
in-progress line has reached the cadence established by earlier
lines in this same speech, so it can nudge the newline terminator
at the right word count — Shakespeare's verse lines cluster tightly
around a shared word-count within a single speech.

Design notes:
  - Only captures BODY newlines (consecutive_newlines == 1). The
    turn-terminator blank line is not a "completed line" in this
    sense — it's the close of the whole speech, handled by
    turn_shape.
  - Not guarded by had_content: an empty body line (two \n with no
    text between) is very rare in Shakespeare prose and when it
    does occur the 0-entry in the tuple is simply ignored by the
    predict consumer (which thresholds on target >= 4 anyway).
  - Runs near the end of the pipeline so it sees the completed
    consecutive_newlines and just_finished_word values from
    update_basic_counters and update_linguistic.

Placement: after update_turn_progress (so turn counters are already
current) but the reset semantics need to be correct — we base
state.line_word_count on pre-reset values of word-boundary triggers.
Just like turn_shape runs before turn_progress to see lines_in_turn
before it resets, this stage runs AFTER turn_progress because our
line_word_count field is independent — turn_progress's reset of
lines_in_turn doesn't affect us.

No corpus statistics — just bookkeeping and a bounded rolling tuple.
"""

from __future__ import annotations

from ..state import ModelState

_CAP = 3


def update_line_word_cadence(state: ModelState, token_id: int) -> ModelState:
    # Turn boundary: reset everything. Fires on the second \n of a
    # \n\n run, same canonical turn boundary as other stages.
    if state.consecutive_newlines >= 2:
        if state.line_word_count != 0 or state.recent_line_word_counts:
            return state.model_copy(
                update={
                    "line_word_count": 0,
                    "recent_line_word_counts": (),
                }
            )
        return state

    # Body newline (consecutive_newlines just became 1 from 0): push
    # the line we just closed onto the rolling tuple. Must skip when
    # we're inside a speaker label — a \n ending a speaker label
    # (NAME:\n) isn't a body line.
    if (
        state.consecutive_newlines == 1
        and state.speaker_label_state == 0
        and state.last_char == "\n"
    ):
        lwc = state.line_word_count
        # Even a 0-word line gets recorded; downstream consumer
        # thresholds. We avoid pushing if the current value is
        # already identical to keep updates minimal.
        new_tuple = (lwc,) + state.recent_line_word_counts
        if len(new_tuple) > _CAP:
            new_tuple = new_tuple[:_CAP]
        if new_tuple != state.recent_line_word_counts or lwc != 0:
            return state.model_copy(
                update={
                    "recent_line_word_counts": new_tuple,
                    "line_word_count": 0,
                }
            )
        return state

    # Otherwise, at a completed word inside the body, increment
    # line_word_count.
    if (
        state.just_finished_word
        and state.speaker_label_state == 0
        and state.consecutive_newlines == 0
    ):
        return state.model_copy(
            update={"line_word_count": state.line_word_count + 1}
        )

    return state
