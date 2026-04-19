"""Tier 2/3 — dialogue-turn progress tracking.

Maintains three counters that describe where we are within the current
speaker's turn:

  - words_in_turn      — completed words since the turn began
  - sentences_in_turn  — sentence-end punctuation since the turn began
  - lines_in_turn      — body newlines since the turn began

Reset trigger:
  consecutive_newlines >= 2  (the blank line between turns)

Increment rules:
  - words_in_turn:      whenever just_finished_word is True AND
                        speaker_label_state == 0
  - sentences_in_turn:  whenever last_char is . ? ! AND
                        speaker_label_state == 0
  - lines_in_turn:      whenever last_char is \n AND
                        consecutive_newlines == 1 AND
                        speaker_label_state == 0

Why this captures something the model can't already see:
  `words_in_sentence` resets per sentence, so it can't tell you
  whether this sentence is the first of the turn or the fifth.
  `last_speaker_label` identifies WHO but not HOW FAR.

Downstream consumers (predict layer):
  - Turn-opener interjection / vocative biases when
    sentences_in_turn == 0 and words_in_turn == 0.
  - Wind-down cadence signals when lines_in_turn grows large
    (suggests the turn is approaching a closure point).
  - Turn-position-conditioned capital and punctuation tuning.
"""

from __future__ import annotations

from ..state import ModelState


def update_turn_progress(state: ModelState, token_id: int) -> ModelState:
    # Reset on turn boundary. A double newline (>=2 consecutive)
    # marks the blank line separating speaker turns.
    if state.consecutive_newlines >= 2:
        if (
            state.words_in_turn
            or state.sentences_in_turn
            or state.lines_in_turn
            or state.turn_exclam_count
            or state.turn_question_count
        ):
            return state.model_copy(update={
                "words_in_turn": 0,
                "sentences_in_turn": 0,
                "lines_in_turn": 0,
                "turn_exclam_count": 0,
                "turn_question_count": 0,
            })
        return state

    # Don't increment while inside a speaker label itself; the turn
    # body begins after the label's closing ":" and post-label newline.
    if state.speaker_label_state != 0:
        return state

    updates: dict = {}

    if state.just_finished_word:
        updates["words_in_turn"] = state.words_in_turn + 1

    lc = state.last_char
    if lc in (".", "?", "!"):
        updates["sentences_in_turn"] = state.sentences_in_turn + 1
        if lc == "!":
            updates["turn_exclam_count"] = state.turn_exclam_count + 1
        elif lc == "?":
            updates["turn_question_count"] = state.turn_question_count + 1

    # Body newline: a single \n that doesn't mark the turn boundary.
    if lc == "\n" and state.consecutive_newlines == 1:
        updates["lines_in_turn"] = state.lines_in_turn + 1

    if updates:
        return state.model_copy(update=updates)
    return state
