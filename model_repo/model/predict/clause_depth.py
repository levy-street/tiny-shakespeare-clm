"""Clause-depth close-pressure bias.

Reads `state.clause_depth` (subordinator nesting depth, 0-3) and
`state.words_in_subordinate` (words since the most recent subordinator
opened a nested clause). At word-end on-trie, as depth * words grows,
we escalate sentence-end / clausal-break bias to pull the text back
toward the main clause.

This is distinct from the overdue-sentence-end bias that fires on
`chars_since_sentence_end`: that one reacts to *time spent* without
closing; this one reacts to *syntactic complexity* — we're deep in
nested subordinates and need to come back up.

No corpus statistics — thresholds from well-known English subordinate
clause length norms (most subordinates are 3-8 words).
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def clause_depth_close_bias(
    clause_depth: int,
    words_in_subordinate: int,
    letter_run_len: int,
    on_word_trie: bool,
    word_buffer: str,
    speaker_label_state: int,
    complete_words: frozenset[str] | None = None,
) -> list[float] | None:
    """Return a bias vector pushing toward clause-close when we're
    lingering deep in a subordinate.

    Gating:
      - speaker_label_state == 0 (not in a speaker label)
      - clause_depth >= 1 (we're in a subordinate)
      - words_in_subordinate >= 3 (subordinate has had some content)
      - word-end position (letter_run_len >= 1 AND word is complete)
    """
    if speaker_label_state != 0:
        return None
    if clause_depth < 1:
        return None
    if words_in_subordinate < 3:
        return None
    # Require word-end at a complete known word.
    if letter_run_len < 1:
        return None
    if complete_words is None or word_buffer not in complete_words:
        return None

    # Escalation: depth 1 at 4+ words → soft; depth 2 at 3+ → medium;
    # depth 3 at 3+ → strong.
    if clause_depth == 1:
        if words_in_subordinate < 6:
            return None
        comma_b = 0.06
        sent_b = 0.0
    elif clause_depth == 2:
        if words_in_subordinate < 5:
            return None
        comma_b = 0.12 if words_in_subordinate < 7 else 0.22
        sent_b = 0.05
    else:  # depth >= 3
        comma_b = 0.20 if words_in_subordinate < 6 else 0.35
        sent_b = 0.12

    vec = [0.0] * VOCAB_SIZE
    if "," in VOCAB_INDEX:
        vec[VOCAB_INDEX[","]] += comma_b
    if ";" in VOCAB_INDEX:
        vec[VOCAB_INDEX[";"]] += comma_b * 0.55
    if sent_b > 0.0:
        if "." in VOCAB_INDEX:
            vec[VOCAB_INDEX["."]] += sent_b
        if "?" in VOCAB_INDEX:
            vec[VOCAB_INDEX["?"]] += sent_b * 0.3
        if "!" in VOCAB_INDEX:
            vec[VOCAB_INDEX["!"]] += sent_b * 0.3
    return vec
