"""Line-break propriety tracker.

Computes `state.line_break_propriety` from already-maintained syntactic
fields — no new semantics, just a combined read that downstream predict
layers can cheaply consume.

Reads:
  - letter_run_len, word_buffer, on_word_trie, just_finished_word
  - np_open, np_wait_words
  - clause_slot (FRESH=0, HAS_SUBJ=1, HAS_VERB=2, POST_OBJ=3)
  - chars_since_comma, chars_since_sentence_end
  - last_char_class (apostrophe / comma / space / ...)

Derives an integer 0..3:

  3 BREAK_CLAUSE_END      — the last char was a clause-break punctuation
                            (, ; :) recently (chars_since_comma <= 2),
                            OR sentence-end was just hit — a natural
                            line-end position.
  2 BREAK_PHRASE_END      — NP is resolved, clause has a verb, we're
                            at a complete word (letter_run_len == 0 or
                            buffer matches a complete word). Natural
                            phrase close.
  1 BREAK_WEAK            — mid-clause but not clearly bad: clause has
                            a verb but no object, or buffer is at a
                            complete word but np is still open.
  0 BREAK_DEEP_MID_PHRASE — mid-word (word_buffer non-empty and not
                            complete), OR np_open with no verb seen,
                            OR clause_slot == FRESH (no subject).

Resets on speaker-turn (consecutive_newlines >= 2) — freshly-entering
a turn starts at propriety 0 by default.

Runs after `update_np_head` and `update_clause_slot` so their fields
are current.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


# Complete-word check is done via the word_trie module used in predict;
# we duplicate the minimal import pattern here to keep the pipeline
# stage independent of predict layers.
from ..predict.word_trie import COMPLETE_WORDS


BREAK_DEEP = 0
BREAK_WEAK = 1
BREAK_PHRASE = 2
BREAK_CLAUSE = 3


def update_line_break(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-turn boundary: reset to 0. A fresh turn has no carried
    # syntactic state and should not be treated as a valid break point
    # immediately.
    if state.consecutive_newlines >= 2 and ch == "\n":
        if state.line_break_propriety != BREAK_DEEP:
            return state.model_copy(update={"line_break_propriety": BREAK_DEEP})
        return state

    # Speaker label: propriety irrelevant (no verse lines inside a
    # label); treat as DEEP.
    if state.speaker_label_state != 0:
        if state.line_break_propriety != BREAK_DEEP:
            return state.model_copy(update={"line_break_propriety": BREAK_DEEP})
        return state

    # Highest priority: just past a clause-break punctuation (chars_since_comma
    # is small after ", ; :"). The linguistic stage zeros chars_since_comma
    # on any PUNCT_END or PUNCT_MID; if that value is <= 2 AND the current
    # char isn't still a letter mid-word, we're at a clause-break close.
    # A sentence-end punctuation also counts (chars_since_sentence_end <= 2).
    at_clause_break = (
        state.chars_since_comma <= 2
        and state.letter_run_len == 0
    )
    at_sentence_end = (
        state.chars_since_sentence_end <= 2
        and state.letter_run_len == 0
    )
    if at_clause_break or at_sentence_end:
        new = BREAK_CLAUSE
        if state.line_break_propriety != new:
            return state.model_copy(update={"line_break_propriety": new})
        return state

    # Mid-word check: a letter is in progress and buffer isn't at a
    # complete form. Can't break here.
    mid_word = (
        state.letter_run_len > 0
        and state.word_buffer not in COMPLETE_WORDS
    )
    if mid_word:
        new = BREAK_DEEP
        if state.line_break_propriety != new:
            return state.model_copy(update={"line_break_propriety": new})
        return state

    # Otherwise, combine np_open and clause_slot to tier the position.
    # clause_slot: 0 FRESH (no subject), 1 HAS_SUBJ, 2 HAS_VERB, 3 POST_OBJ.
    slot = state.clause_slot
    np_open = state.np_open
    np_wait = state.np_wait_words

    if slot == 0 or (slot == 1 and np_open and np_wait <= 1):
        # No verb yet, or subject opened an NP that's still pending.
        new = BREAK_DEEP
    elif np_open and np_wait <= 2:
        # NP is open with limited pre-head modifiers — mid-phrase.
        new = BREAK_DEEP
    elif slot == 1:
        # Has subject, no verb yet — weak.
        new = BREAK_WEAK
    elif slot == 2 and np_open:
        # Has verb but object NP is open — weak.
        new = BREAK_WEAK
    elif slot == 2:
        # Has verb, no open NP — phrase-end is plausible.
        new = BREAK_PHRASE
    elif slot == 3:
        # Post-object — strong phrase-end.
        new = BREAK_PHRASE
    else:
        new = BREAK_WEAK

    if new != state.line_break_propriety:
        return state.model_copy(update={"line_break_propriety": new})
    return state
