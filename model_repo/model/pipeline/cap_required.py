"""Capital-required-at-next-word-start gate.

Sets `state.cap_required_mode`, a single explicit categorical signal
naming the structural reason (if any) that the next emitted letter
must be uppercase. Modes:

  0 NONE            — no structural capital requirement.
  1 SENTENCE_START  — just past ". ", "? ", or "! " (or single-\\n with
                      sentence_end within 2 chars). Fresh sentence opens
                      with a capital.
  2 VERSE_LINE      — single-\\n terminated a verse-plausible line
                      (len 1-80, non-empty) that was NOT enjambed.
                      Every new verse line opens capital.
  3 POST_LABEL      — just past "\\n" following a short ":"-terminated
                      speaker label. Dialogue opens capital.

Only meaningful at a word-start position (letter_run_len == 0 AND
last_char is not a letter). Conditions match exactly the previously-
inline logic in compose.py so BPC is preserved on training-text
continuations (enjambed lines, prose-wrap lines, etc.), while giving
downstream layers and future tuning a single explicit state field to
condition on instead of re-deriving the condition at every call site.

Runs late in the pipeline so prev_line_length / prev_line_final_class /
prev_char_class / sentence_start_pending are already current. Does NOT
read any corpus statistic — the rules are standard Shakespeare
orthographic conventions.
"""

from __future__ import annotations

from ..state import ModelState
from .linguistic import LOWER_VOWEL, NEWLINE, PUNCT_END, PUNCT_MID, SPACE


# Mode enumeration (kept in sync with schema docstring).
NONE = 0
SENTENCE_START = 1
VERSE_LINE = 2
POST_LABEL = 3


def _compute_mode(state: ModelState) -> int:
    # Only at word-start outside speaker-label FSM state 2.
    if state.speaker_label_state == 2:
        return NONE
    if state.letter_run_len != 0:
        return NONE

    last_cls = state.last_char_class

    # Must be at a position where a fresh word could start: last char
    # is a space, or single newline. (This matches the outer block
    # guard in compose.py so downstream bias only fires in context.)
    if not (last_cls == SPACE or (last_cls == NEWLINE and state.consecutive_newlines == 1)):
        return NONE

    # SENTENCE_START (matches inline `is_sentence_start` exactly).
    is_sentence_start = (
        state.prev_char_class == PUNCT_END
        and last_cls == SPACE
    ) or (
        last_cls == NEWLINE and state.consecutive_newlines == 1
        and state.chars_since_sentence_end <= 2
    )
    if is_sentence_start:
        return SENTENCE_START

    # VERSE_LINE (matches inline `on_verse_line_start and not is_enjambed`).
    on_verse_line_start = (
        last_cls == NEWLINE
        and state.consecutive_newlines == 1
        and 1 <= state.prev_line_length <= 80
    )
    if on_verse_line_start:
        # Enjambment exactly as inline: prev line ended on a lowercase
        # vowel AND was prose-length. (Using the narrow vowel-only
        # class 3 match, not any letter class — this matches the
        # pre-existing behavior.)
        is_enjambed = (
            state.prev_line_final_class == LOWER_VOWEL
            and state.prev_line_length >= 50
        )
        if not is_enjambed:
            return VERSE_LINE
        # Note: enjambed case is handled separately in compose.py as
        # a NEGATIVE cap bias; we leave mode NONE here so that layer
        # still fires with its original weights.
        return NONE

    # POST_LABEL (matches inline `on_post_label_start`).
    on_post_label_start = (
        last_cls == NEWLINE
        and state.consecutive_newlines == 1
        and not is_sentence_start
        and not on_verse_line_start
        and 1 < state.prev_line_length < 15
        and state.prev_char_class == PUNCT_MID  # the ":" of the label
    )
    if on_post_label_start:
        return POST_LABEL

    return NONE


def update_cap_required(state: ModelState, token_id: int) -> ModelState:
    mode = _compute_mode(state)
    if mode == state.cap_required_mode:
        return state
    return state.model_copy(update={"cap_required_mode": mode})
