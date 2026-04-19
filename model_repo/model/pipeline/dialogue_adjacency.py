"""Tier 2/3 — dialogue-adjacency memory.

Snapshots the PREVIOUS turn's shape at the moment it closes, so that
the CURRENT turn's opening can react to it. Distinct from
update_turn_progress, which tracks within-turn counters and then
resets them at the blank-line turn boundary — those resets happen
immediately AFTER this stage runs, so we see the pre-reset values.

Maintains:
  * current_turn_final_char — the last non-whitespace content char
    emitted inside the still-open turn. Keeps track of what
    punctuation / letter the turn body is currently ending on.
  * prev_turn_final_punct — snapshot at turn close.
  * prev_turn_* counters — carry-over of words_in_turn,
    sentences_in_turn, lines_in_turn, turn_exclam_count,
    turn_question_count at the moment of the turn close.
  * prev_turn_speaker_label — last_speaker_label at turn close.
  * turns_closed — monotonic count.

Trigger points:
  * At every token inside a turn body (speaker_label_state == 0 and
    last_char is neither space nor newline): update
    current_turn_final_char = last_char.
  * At the token that makes consecutive_newlines == 2 (the first
    blank newline that closes the turn): snapshot prev_turn_*
    from the about-to-be-reset in-turn state; reset
    current_turn_final_char to "".

Because update_turn_progress (in PIPELINE order) performs the resets
on the SAME token, this stage MUST run immediately before it.

Downstream consumer: predict/dialogue_opener.py which at the start
of a new turn's first word boosts appropriate opener letters based
on prev_turn_final_punct and prev_turn_word_count.
"""

from __future__ import annotations

from ..state import ModelState


def update_dialogue_adjacency(state: ModelState, token_id: int) -> ModelState:
    lc = state.last_char  # already the incoming token thanks to update_linguistic
    sp = state.speaker_label_state
    cn = state.consecutive_newlines

    updates: dict = {}

    # Turn-close detection: consecutive_newlines has just reached 2
    # for the first time. At cn >= 3 the same snapshot is already
    # written; don't overwrite with new data.
    if cn == 2:
        # Snapshot the prior turn's in-turn counters (still non-zero
        # because update_turn_progress hasn't reset them yet).
        #
        # current_turn_final_char carries the terminator of the prior
        # turn's body. If the turn ended with "," (continuation) or
        # ";" / ":" we record that literal; if it ended with . ? !
        # we record that; if the turn was empty (e.g., the very first
        # block is a stage direction), fall back to "".
        prev_punct = state.current_turn_final_char
        # Keep only punctuation chars as a categorical-ish slot;
        # letters and other chars collapse to "" so the downstream
        # bias treats them as "uninformative turn close".
        if prev_punct in (".", "?", "!", ",", ";", ":", "-"):
            final_punct = prev_punct
        else:
            final_punct = ""

        # Only count as a real closed turn if the closed body
        # actually had content.
        had_content = (
            state.words_in_turn > 0
            or state.sentences_in_turn > 0
            or state.lines_in_turn > 0
            or state.current_turn_final_char != ""
        )

        updates["prev_turn_final_punct"] = final_punct
        updates["prev_turn_word_count"] = state.words_in_turn
        updates["prev_turn_sentence_count"] = state.sentences_in_turn
        updates["prev_turn_line_count"] = state.lines_in_turn
        updates["prev_turn_exclam_count"] = state.turn_exclam_count
        updates["prev_turn_question_count"] = state.turn_question_count
        updates["prev_turn_speaker_label"] = state.last_speaker_label
        # Cross-turn content echo: snapshot the closing turn's content
        # cache before update_turn_content resets it a few stages later.
        # Cap at 6 entries — enough to capture the thematic spine of
        # the prior turn without drowning the start of the new one.
        updates["prev_turn_content_tail"] = state.turn_content_cache[:6]
        if had_content:
            updates["turns_closed"] = state.turns_closed + 1
        # Reset running final-char for the next turn.
        updates["current_turn_final_char"] = ""
        return state.model_copy(update=updates)

    # Inside a turn body: update running final-content-char.
    # - Skip speaker-label territory (the ":" closing the label isn't
    #   "turn-final content", it's structural).
    # - Skip whitespace (space, newline) — we want the last SUBSTANTIVE
    #   char.
    if sp == 0 and lc and lc != " " and lc != "\n":
        if state.current_turn_final_char != lc:
            updates["current_turn_final_char"] = lc

    if updates:
        return state.model_copy(update=updates)
    return state
