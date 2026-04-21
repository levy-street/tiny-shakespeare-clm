"""Tier 2 — coordinator-parallelism echo (POS + first-letter + case).

Shakespeare strongly favors parallel structure across a coordinating
conjunction ("and", "or", "nor"):
  - "fair and foul"       adjective + adjective, alliterative f/f
  - "night and day"       noun + noun
  - "Romeo and Juliet"    proper noun + proper noun (both capital)
  - "thou and I"          pronoun + pronoun
  - "kith and kin"        alliterative k/k
  - "beck and call"       alliterative b/c (shared initial sound)
  - "tooth and nail"      noun + noun

When a coordinator word completes we record three echo signals about
the word that sat immediately BEFORE the coordinator:
  1. Its POS tag (when known) — echo typical first-letter starters
     of that POS class.
  2. Its first letter (lowercased) — alliterative echo, covering
     the many content-word pairs the POS tagger leaves UNKNOWN.
  3. Whether it was a MID-SENTENCE capital (proper-noun-like) — echo
     case for pairs like "Romeo and Juliet", "Cassio and Iago".

State maintained:
  coord_echo_pos                  — POS tag to echo (0 = inactive)
  coord_echo_pending              — True between coord-word's
                                     trailing space and the first
                                     letter of the next word
  coord_echo_first_letter         — lowercase first letter of
                                     pre-coord word
  coord_echo_was_capital          — True iff pre-coord word was a
                                     mid-sentence cap (proper-noun-
                                     like)
  coord_prev_word_started_cap     — helper: snapshot of
                                     current_word_started_cap taken
                                     at the previous word-completion
                                     (provides the 1-word lag we
                                     need so the coord can read the
                                     PRE-coord word's cap status)

Lifecycle:
  * On just_finished_word:
      a) IF just-completed word is a coordinator, ARM the echo using
         prev_completed_word + coord_prev_word_started_cap + POS
         from prev_word_pos.
      b) ELSE consume any pending echo.
      c) In either case, snapshot current_word_started_cap →
         coord_prev_word_started_cap for the NEXT word-completion's
         benefit.
  * On sentence-end (. ? !) or speaker-turn boundary: clear echo.
  * In speaker-label territory: clear echo.

Runs BEFORE update_proper_noun_memory so current_word_started_cap
still reflects the just-completed word's cap status (not yet reset).

No corpus statistics — coord parallelism is a universal English
syntactic regularity; the lexical first-letter echo exploits a
well-known Shakespearean rhetorical habit.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


_COORDINATORS: frozenset[str] = frozenset({"and", "or", "nor"})


def _maybe_snapshot(state: ModelState, updates: dict) -> None:
    """At a word-completion tick, snapshot current_word_started_cap
    into coord_prev_word_started_cap so the NEXT word-completion
    (which may be a coord) can read it."""
    if not state.just_finished_word:
        return
    # state.current_word_started_cap on this tick still reflects the
    # just-completed word's cap status because proper_noun_memory
    # (which resets it) runs AFTER this stage.
    snap = state.current_word_started_cap
    if snap != state.coord_prev_word_started_cap:
        updates["coord_prev_word_started_cap"] = snap


def _clear_echo_into(state: ModelState, updates: dict) -> None:
    if (
        state.coord_echo_pos != 0
        or state.coord_echo_pending
        or state.coord_echo_first_letter
        or state.coord_echo_was_capital
    ):
        updates["coord_echo_pos"] = 0
        updates["coord_echo_pending"] = False
        updates["coord_echo_first_letter"] = ""
        updates["coord_echo_was_capital"] = False


def update_coord(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]
    updates: dict = {}

    # Speaker-label territory: keep echo cleared. (Still snapshot.)
    if state.speaker_label_state != 0:
        _clear_echo_into(state, updates)
        _maybe_snapshot(state, updates)
        if not updates:
            return state
        return state.model_copy(update=updates)

    # Speaker-turn boundary.
    if ch == "\n" and state.consecutive_newlines >= 2:
        _clear_echo_into(state, updates)
        _maybe_snapshot(state, updates)
        if not updates:
            return state
        return state.model_copy(update=updates)

    # Sentence-end punctuation.
    if ch in ".?!":
        _clear_echo_into(state, updates)
        _maybe_snapshot(state, updates)
        if not updates:
            return state
        return state.model_copy(update=updates)

    # Word completion.
    if state.just_finished_word and state.last_completed_word:
        w = state.last_completed_word
        if w in _COORDINATORS:
            # Arm echo.
            target_pos = state.prev_word_pos
            pcw = state.prev_completed_word
            first_letter = ""
            if pcw and pcw[0].isalpha():
                first_letter = pcw[0].lower()
            # was_cap uses coord_prev_word_started_cap (lagged
            # snapshot from when the pre-coord word completed).
            was_cap = bool(state.coord_prev_word_started_cap)
            if target_pos == 0 and not first_letter:
                # No useful echo signal.
                _clear_echo_into(state, updates)
            else:
                if state.coord_echo_pos != target_pos:
                    updates["coord_echo_pos"] = target_pos
                if not state.coord_echo_pending:
                    updates["coord_echo_pending"] = True
                if state.coord_echo_first_letter != first_letter:
                    updates["coord_echo_first_letter"] = first_letter
                if state.coord_echo_was_capital != was_cap:
                    updates["coord_echo_was_capital"] = was_cap
        else:
            # Non-coord word: consume any pending echo.
            _clear_echo_into(state, updates)
        _maybe_snapshot(state, updates)
        if not updates:
            return state
        return state.model_copy(update=updates)

    # Non-boundary tick: nothing to do.
    return state
