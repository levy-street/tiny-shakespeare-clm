"""Rhyme-position / line-tail memory.

Maintains a rolling 3-letter tail of the current line, and at the
moment a newline closes a verse-plausible line, shifts it into
`prev_line_tail` (and `prev_prev_line_tail`) so downstream predict
layers can bias toward end-of-line letters that rhyme with the
previous couplet.

Reset on speaker-turn change (consecutive_newlines >= 2). Lines that
are speaker labels (end in ":") don't participate — they can't rhyme.

Runs after `update_linguistic` so `state.last_char`, `prev_char`,
`prev_line_length`, and `consecutive_newlines` reflect the current
token.
"""

from __future__ import annotations

from ..state import ModelState


def _is_letter(c: str) -> bool:
    return len(c) == 1 and c.isalpha()


def update_rhyme(state: ModelState, token_id: int) -> ModelState:
    ch = state.last_char
    prev = state.prev_char

    # Speaker-turn change — wipe.
    if state.consecutive_newlines >= 2:
        if (
            state.prev_line_tail == ""
            and state.prev_prev_line_tail == ""
            and state.line_tail_buffer == ""
            and state.verse_line_run == 0
        ):
            return state
        return state.model_copy(
            update={
                "prev_line_tail": "",
                "prev_prev_line_tail": "",
                "line_tail_buffer": "",
                "verse_line_run": 0,
            }
        )

    # A newline that just closed a non-empty, non-label line:
    #   - ch == "\n"
    #   - prev was NOT "\n" (the prior char was on the same line)
    #   - prev was NOT ":" (not a speaker label closing)
    #   - state.prev_line_length (set by linguistic) > 0
    if (
        ch == "\n"
        and prev != "\n"
        and prev != ":"
        and state.prev_line_length > 0
        and state.line_tail_buffer != ""
    ):
        line_len = state.prev_line_length
        is_verse_plausible = 15 <= line_len <= 55
        new_prev_prev = state.prev_line_tail
        new_prev = state.line_tail_buffer
        new_buf = ""
        if is_verse_plausible:
            new_run = state.verse_line_run + 1
        else:
            new_run = 0
        if (
            new_prev == state.prev_line_tail
            and new_prev_prev == state.prev_prev_line_tail
            and new_buf == state.line_tail_buffer
            and new_run == state.verse_line_run
        ):
            return state
        return state.model_copy(
            update={
                "prev_line_tail": new_prev,
                "prev_prev_line_tail": new_prev_prev,
                "line_tail_buffer": new_buf,
                "verse_line_run": new_run,
            }
        )

    # Speaker-label line closed — reset buffer but don't shift.
    if ch == "\n" and prev == ":":
        if state.line_tail_buffer == "":
            return state
        return state.model_copy(update={"line_tail_buffer": ""})

    # Any other newline (blank line separator, empty line) — reset
    # current-line buffer but don't shift.
    if ch == "\n":
        if state.line_tail_buffer == "":
            return state
        return state.model_copy(update={"line_tail_buffer": ""})

    # Letter on current line — update rolling tail.
    if _is_letter(ch):
        new_buf = (state.line_tail_buffer + ch.lower())[-3:]
        if new_buf == state.line_tail_buffer:
            return state
        return state.model_copy(update={"line_tail_buffer": new_buf})

    return state
