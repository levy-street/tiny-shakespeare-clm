"""Dependent-clause closing bias.

Reads `state.in_dependent_clause` (maintained by pipeline/clause.py —
True iff the current clause was opened by a subordinator like "if",
"though", "when", "while", "because", "since", "unless", "until",
"that", "which") and `state.words_in_subordinate` (the word-count
within the current subordinate clause so far).

A dependent clause in Shakespeare normally runs 3-8 content words
before closing with a comma, then the main clause follows. While
inside a dependent clause:

  * Comma at word-end is the natural closer once the dep clause is
    complete (has subject + verb): push "," up.
  * Sentence-enders ". ! ?" close the WHOLE sentence — but the main
    clause hasn't been uttered yet if we're still inside the dep
    clause. So penalize ". ! ?" inside an active dep clause that
    has at least 2 words.

Gating:
  * speaker_label_state == 0
  * in_dependent_clause is True
  * words_in_subordinate >= 2 (dep clause has some content)
  * at word-end on-trie complete-word position (letter_run_len >= 2,
    on_word_trie, word_buffer in complete_words)
  * chars_since_sentence_end >= 15 (not right at sentence start)

Bias magnitudes are modest so they layer cleanly with existing
clause_rhythm/context biases.

All weights from prior knowledge of English / Shakespeare dep-clause
structure — no corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def dependent_clause_bias(
    in_dependent_clause: bool,
    words_in_subordinate: int,
    clause_slot: int,
    chars_since_sentence_end: int,
    chars_since_comma: int,
    word_buffer: str,
    on_word_trie: bool,
    letter_run_len: int,
    speaker_label_state: int,
    complete_words: frozenset[str],
) -> list[float] | None:
    """Bias toward comma (close dep clause) and against sentence-end
    (main clause still pending) while inside an active dep clause."""
    if speaker_label_state != 0:
        return None
    if not in_dependent_clause:
        return None
    if words_in_subordinate < 2:
        return None
    if letter_run_len < 2:
        return None
    if not on_word_trie:
        return None
    if not word_buffer:
        return None
    if word_buffer not in complete_words:
        return None
    if chars_since_sentence_end < 15:
        return None
    # If a comma just fired recently, don't double-comma — skip if
    # chars_since_comma is small.
    if chars_since_comma < 6:
        return None

    vec = [0.0] * VOCAB_SIZE

    # Comma push — scales with dep-clause maturity (words_in_subordinate)
    # and clause_slot (once the dep clause has its own subj+verb, the
    # comma is ready).
    if words_in_subordinate >= 6:
        comma_push = 0.38
    elif words_in_subordinate >= 4:
        comma_push = 0.25
    elif words_in_subordinate >= 3:
        comma_push = 0.15
    else:  # 2 words — early, gentle
        comma_push = 0.08

    # Slot modulation: only add the full comma push when the dep
    # clause has a verb (slot >= 2 = HAS_VERB or POST_OBJ).
    if clause_slot >= 3:       # POST_OBJ — ready to close
        slot_mul = 1.15
    elif clause_slot >= 2:     # HAS_VERB — can close
        slot_mul = 1.00
    else:
        slot_mul = 0.55

    comma_push *= slot_mul

    if "," in VOCAB_INDEX:
        vec[VOCAB_INDEX[","]] += comma_push
    if ";" in VOCAB_INDEX:
        vec[VOCAB_INDEX[";"]] += comma_push * 0.35

    # Sentence-end penalty: main clause is still coming, so ". ! ?"
    # is premature. Scale with dep-clause maturity — the deeper we
    # are, the more suspicious an early sentence-end.
    if words_in_subordinate >= 3:
        sent_end_pen = -0.30
    else:
        sent_end_pen = -0.15

    if "." in VOCAB_INDEX:
        vec[VOCAB_INDEX["."]] += sent_end_pen
    if "!" in VOCAB_INDEX:
        vec[VOCAB_INDEX["!"]] += sent_end_pen * 0.9
    if "?" in VOCAB_INDEX:
        vec[VOCAB_INDEX["?"]] += sent_end_pen * 0.9

    return vec
