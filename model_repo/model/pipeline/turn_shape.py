"""Cross-turn rhythm / stichomythia mode tracker.

Maintains `recent_turn_line_counts` (rolling tuple, cap 4, most-recent
first) and derives `stichomythia_mode` (categorical: 0=UNKNOWN,
1=RAPID, 2=SUSTAINED).

Fires at the same trigger moment as update_dialogue_adjacency — the
consecutive_newlines 1→2 transition — and MUST run immediately after
that stage so the snapshot of lines_in_turn is still correct (before
update_turn_progress resets it).

Derivation rules for stichomythia_mode:
  - If tuple has 0 entries → UNKNOWN.
  - If len(tuple) >= 2 AND all first two entries are <= 2 lines
    → RAPID (rapid exchange / stichomythia).
  - Else if most-recent entry is >= 6 lines → SUSTAINED (declamatory).
  - Else UNKNOWN.

Reset behavior: never clears the rolling tuple mid-session; the tuple
naturally rotates out old entries. (Scene-change boundaries aren't
independently detected here; a new scene's first turn simply adds a
fresh entry to the rolling tuple.)

No corpus statistics — the mode categories and thresholds are from
prior knowledge of dialogue rhythm (stichomythia is the classical
term for line-by-line rapid exchange in Greek drama and Shakespeare).
"""

from __future__ import annotations

from ..state import ModelState

_MAX = 4
UNKNOWN = 0
RAPID = 1
SUSTAINED = 2


def _derive_mode(recent: tuple[int, ...]) -> int:
    if len(recent) == 0:
        return UNKNOWN
    if len(recent) >= 2 and recent[0] <= 2 and recent[1] <= 2:
        return RAPID
    if recent[0] >= 6:
        return SUSTAINED
    return UNKNOWN


def update_turn_shape(state: ModelState, token_id: int) -> ModelState:
    # Fire only at the first blank-newline that closes a turn.
    if state.consecutive_newlines != 2:
        return state

    # Only snapshot turns that actually had content (mirror the
    # had_content rule in dialogue_adjacency — otherwise blank-line
    # runs between scene-change stage directions would pollute the
    # rhythm tuple with zero-line "turns").
    had_content = (
        state.words_in_turn > 0
        or state.sentences_in_turn > 0
        or state.lines_in_turn > 0
        or state.current_turn_final_char != ""
    )
    if not had_content:
        return state

    # dialogue_adjacency runs just before us and resets
    # current_turn_final_char to "", but it has NOT yet reset the
    # in-turn counters (that's update_turn_progress's job later).
    # We can safely read lines_in_turn here.
    lt = state.lines_in_turn
    new_tuple = (lt,) + state.recent_turn_line_counts
    if len(new_tuple) > _MAX:
        new_tuple = new_tuple[:_MAX]
    new_mode = _derive_mode(new_tuple)

    updates: dict = {}
    if new_tuple != state.recent_turn_line_counts:
        updates["recent_turn_line_counts"] = new_tuple
    if new_mode != state.stichomythia_mode:
        updates["stichomythia_mode"] = new_mode
    if not updates:
        return state
    return state.model_copy(update=updates)
