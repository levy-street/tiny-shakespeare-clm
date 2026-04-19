"""Subordinate / relative clause depth tracker.

New axis: track nesting of subordinate clauses inside the current
sentence. Increments when a subordinator word completes in a
subordinator-opening position; decrements on clause closure.

Subordinators (EME-inclusive):
  RELATIVE:  that, which, who, whom, whose, where, when
  TEMPORAL:  while, whilst, till, until, ere, since, once, when
  CONDITIONAL: if, unless, though, although, albeit, lest
  CAUSAL:    because, for, as (ambiguous, not always subord)
  CONCESSIVE: though, although, yet (as subord)

Open-policy:
  - On completed word = subordinator
  - AND previous context is either:
    (a) HAS_SUBJ / HAS_VERB / POST_OBJ (not FRESH), meaning we're
        inside a clause that the subordinator is embedding into
    (b) OR the word is one of {"if", "when", "though", "while",
        "because", "unless", "since", "until", "till", "whilst",
        "ere", "lest", "albeit"} which can open a dependent clause
        even at clause-start (clause_slot == FRESH).
  - AND speaker_label_state == 0
  - AND subord_depth < 3 (hard cap)

Close-policy:
  - Sentence-end punctuation (. ? !) — reset depth to 0.
  - Speaker-turn boundary (handled by update_sentence / counters
    resetting via its own sentence-break logic).
  - When subord_words_since_open >= 8 and we see a comma or
    coordinating conjunction, decrement by 1 (the subordinate
    clause has run its course).

This stage runs AFTER update_pos and update_clause_slot so it sees
their decisions.

All thresholds and word lists are hand-chosen from prior knowledge
of English syntax — no corpus counting.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


# Words that when completed open a subordinate clause. Matched against
# last_completed_word (lowercase).
_SUBORDINATORS_STRONG: frozenset[str] = frozenset({
    "that", "which",
    "who", "whom", "whose", "where", "when",
    "while", "whilst", "till", "until", "ere",
    "if", "unless", "though", "although", "albeit", "lest",
    "because", "whereas", "whereby", "wherein", "whereof",
    "wherefore", "whereupon", "whither", "whence",
})

# Words that can open a subordinate clause even at clause_slot == FRESH
# (they can appear at sentence start as subordinators, e.g.,
# "If thou know'st the road").
_SUBORDINATORS_AT_FRESH: frozenset[str] = frozenset({
    "if", "when", "whenever", "whensoever", "though", "although",
    "while", "whilst", "because", "unless", "since", "until",
    "till", "ere", "lest", "albeit", "though", "whereas",
    "whensoever", "whereas",
})

# Cap
_MAX_DEPTH: int = 3
_SLOT_MASK: int = 0b111  # 3 bits per level
_BITS_PER_LEVEL: int = 3


def _push_slot(stack: int, slot: int, depth_after_push: int) -> int:
    """Push a 3-bit slot value onto the stack at the correct level.

    depth_after_push is the new depth AFTER push (1-indexed).
    Level 1 occupies bits 0-2; level 2 occupies bits 3-5; etc.
    """
    if depth_after_push < 1 or depth_after_push > _MAX_DEPTH:
        return stack
    shift = (depth_after_push - 1) * _BITS_PER_LEVEL
    # Clear slot bits at that level and write new.
    cleared = stack & ~(_SLOT_MASK << shift)
    return cleared | ((slot & _SLOT_MASK) << shift)


def _pop_slot(stack: int, depth_before_pop: int) -> int:
    """Pop the level at depth_before_pop (1-indexed), returning
    the stack with that level cleared."""
    if depth_before_pop < 1 or depth_before_pop > _MAX_DEPTH:
        return stack
    shift = (depth_before_pop - 1) * _BITS_PER_LEVEL
    return stack & ~(_SLOT_MASK << shift)


def update_subord(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Sentence-end resets.
    if ch in ".?!":
        if state.subord_depth == 0 and state.subord_words_since_open == 0 and state.subord_slot_stack == 0:
            return state
        return state.model_copy(update={
            "subord_depth": 0,
            "subord_words_since_open": 0,
            "subord_slot_stack": 0,
        })

    updates: dict = {}
    depth = state.subord_depth
    stack = state.subord_slot_stack
    wsince = state.subord_words_since_open

    # When a word just completed, check for subordinator opening and
    # increment word-since counter.
    if state.just_finished_word and state.last_completed_word:
        word = state.last_completed_word.lower()
        # Advance counter first (counts words inside subord, but also
        # matures pre-open tracking; cap at 15).
        if depth > 0:
            wsince = min(wsince + 1, 15)

        # Try to open.
        if (
            state.speaker_label_state == 0
            and depth < _MAX_DEPTH
            and word in _SUBORDINATORS_STRONG
        ):
            slot = state.clause_slot
            # Require either inside-clause context OR FRESH-allowed
            # subordinator.
            allow = (slot != 0) or (word in _SUBORDINATORS_AT_FRESH)
            if allow:
                new_depth = depth + 1
                stack = _push_slot(stack, slot, new_depth)
                depth = new_depth
                wsince = 0

    # Comma-close policy: a long-running subordinate clause can close
    # on comma. Only fires if depth > 0 AND wsince >= 4.
    # (Also: coordinating conjunctions and / but / or / nor could
    # close it, but those come as completed words; we'd need to
    # inspect last_completed_word again. For simplicity we only
    # close on comma here.)
    if ch == "," and depth > 0 and wsince >= 4:
        depth -= 1
        stack = _pop_slot(stack, depth + 1)
        wsince = 0

    if (
        depth != state.subord_depth
        or stack != state.subord_slot_stack
        or wsince != state.subord_words_since_open
    ):
        updates["subord_depth"] = depth
        updates["subord_slot_stack"] = stack
        updates["subord_words_since_open"] = wsince
        return state.model_copy(update=updates)
    return state
