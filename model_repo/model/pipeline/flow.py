"""Tier 3 — flow / mood / register state updates.

Reads Tier 1 (base counters) and Tier 2 (linguistic) fields and the
incoming token, and updates *flow*-level fields that capture the
texture of the emerging text — whether the current word has gone
off the known-vocabulary trie, how long the current line has run, how
overdue the next sentence-ending mark is, and whether the last word
was a short closed-class function word.

These flow fields are later consumed by the predict layer, where they
modulate biases toward line-ending, sentence-ending, or content-word
continuations.
"""

from __future__ import annotations

from ..state import ModelState
from ..predict.word_trie import is_on_trie

# Short closed-class words whose follow-up is almost always a
# content word (noun / verb / adjective), not another function word.
_FUNCTION_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "my", "thy", "his", "her", "our", "your", "their",
    "this", "that", "these", "those",
    "of", "to", "in", "on", "with", "for", "by", "at", "as", "from",
    "into", "unto", "upon", "o'er", "'gainst",
    "and", "but", "or", "nor", "so", "yet", "for",
    "i", "thou", "he", "she", "we", "ye", "they", "you", "me", "thee",
    "him", "us", "them",
    "is", "are", "was", "were", "be", "been", "am", "art", "hath", "doth",
    "hast", "dost", "shall", "will", "would", "should", "could", "may",
    "might", "must", "do", "did", "does", "have", "has", "had",
    "not", "no", "nay", "yea",
    "if", "when", "where", "while", "though", "than", "then", "now",
    "here", "there",
})


def _line_length_bucket(chars_since_newline: int) -> int:
    if chars_since_newline < 20:
        return 0
    if chars_since_newline < 35:
        return 1
    if chars_since_newline < 50:
        return 2
    return 3


def _sent_distance_bucket(chars_since_sentence_end: int) -> int:
    if chars_since_sentence_end < 40:
        return 0
    if chars_since_sentence_end < 80:
        return 1
    return 2


def update_flow(state: ModelState, token_id: int) -> ModelState:
    # Linguistic updates have already run; use the post-update state.
    wb = state.word_buffer
    if not wb:
        # No partial word in progress.
        on_trie = True
    else:
        on_trie = is_on_trie(wb)

    line_length_bucket = _line_length_bucket(state.chars_since_newline)
    sent_distance_bucket = _sent_distance_bucket(state.chars_since_sentence_end)

    # Did we just complete a function word?
    after_function_word = (
        state.just_finished_word
        and state.last_completed_word in _FUNCTION_WORDS
    )

    # Heuristic: we're in a prose line if we've seen enough chars since
    # the last newline without hitting a colon-newline (speaker label)
    # boundary recently. A simple proxy: chars_since_newline > 55 suggests
    # prose (most verse lines are shorter). This is soft.
    in_prose_line = state.chars_since_newline > 55

    return state.model_copy(
        update={
            "on_word_trie": on_trie,
            "line_length_bucket": line_length_bucket,
            "sent_distance_bucket": sent_distance_bucket,
            "after_function_word": after_function_word,
            "in_prose_line": in_prose_line,
        }
    )
