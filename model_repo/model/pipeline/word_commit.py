"""Word-identity commitment.

Closes the "letter-by-letter drift produces gibberish" failure: when
the formula trie has committed us to a deep, unambiguous next-word
identity, hold onto that identity as a string in state and let the
predict layer bias each subsequent letter toward the target's letters,
instead of re-deciding letter-by-letter via independent n-gram signals.

State fields owned:
  committed_word     — the lowercase target ("" if no commit)
  committed_word_pos — how many of its letters have been emitted

Lifecycle:
  - On sentence-end / turn-end / formula reset → clear.
  - On just_finished_word → clear (a new word-start decision follows).
  - At word-start (letter_run_len == 0, last_char_class ∈ {SPACE, NEWLINE}):
      Consult the formula trie at state.formula_node. If exactly ONE
      expected next word AND it is at least 3 letters long AND all-lower
      (formula trie is lowercase anyway), commit to it with pos = 0.
      Otherwise clear.
  - Mid-word (letter_run_len >= 1, just_finished_word is False):
      If the last emitted letter (word_buffer[-1].lower()) matches
      committed_word[committed_word_pos], advance pos. On mismatch or
      overflow, clear — we don't want a stale commit to poison the next
      letter.

Runs after update_formula (needs formula_node up-to-date) and after
update_basic_counters / update_linguistic (needs letter_run_len,
word_buffer, last_char_class). Cleared eagerly on any boundary so
downstream predict reads a clean signal.

No corpus statistics — the formula trie encodes prior-knowledge
Shakespeare idioms.
"""

from __future__ import annotations

from ..predict.formula_trie import expected_next_words
from ..state import ModelState

# Match linguistic.py class codes.
SPACE = 5
NEWLINE = 6


# Hand-curated bigram → dominant-next-word commitments. Keyed by
# (prev_word, last_word) — the two most recently completed lowercase
# words. Values are the robust-default next word in Early-Modern-English
# dialogue contexts. These are NOT from corpus statistics; they reflect
# the strong default continuation in Shakespearean idiom where the
# formula trie is too diffuse or doesn't fire.
#
# Only include high-confidence cases: places where the likeliest next
# word is dominant enough that a +4.0 letter boost will be on-target
# a large fraction of the time, and where the predict-layer's mismatch-
# clearing behavior (committed_word clears on any letter mismatch)
# protects us when the default is wrong.
_BIGRAM_COMMITS: dict[tuple[str, str], str] = {
    # Strong vocative patterns
    ("good", "my"): "lord",
    ("sweet", "my"): "lord",
    ("gentle", "my"): "lord",
    ("dear", "my"): "lord",
    ("fair", "my"): "lord",
    # Pray-thee register
    ("i", "prithee"): "tell",  # often tell/speak/come/go
    # Wish/grant patterns
    ("heaven", "grant"): "thee",
    ("god", "grant"): "thee",
    # Classic Shakespearean quote-completion hooks
    ("not", "to"): "be",       # "or not to be"
    ("to", "be"): "or",        # "to be or not to be"
    # Reported-speech + addressee
    ("fare", "thee"): "well",
    ("fare", "you"): "well",
    # Dialog tags
    ("by", "my"): "troth",     # coin flip with faith/soul — troth is idiomatic
}


def _compute_commit(state: ModelState) -> tuple[str, int]:
    """Compute the (committed_word, committed_word_pos) for the NEXT state,
    given the just-advanced state."""
    # Reset on sentence-end punctuation / turn-end.
    if state.last_char in (".", "?", "!", ";", ":"):
        return ("", 0)
    if state.consecutive_newlines >= 2:
        return ("", 0)

    # A word just completed → clear; the word-start branch below will
    # re-commit if applicable at the first letter-run-len-0 post-space step.
    if state.just_finished_word:
        # Note: at this moment letter_run_len is 0 and last_char is the
        # terminator char (space/punct/newline). Fall through to the
        # word-start branch below for a fresh commit attempt.
        pass

    # Mid-word continuation: validate against committed target.
    if state.letter_run_len >= 1 and state.committed_word:
        # word_buffer[-1] is the last emitted letter (case-folded).
        if not state.word_buffer:
            return ("", 0)
        last_letter = state.word_buffer[-1].lower()
        pos = state.committed_word_pos
        target = state.committed_word
        if pos >= len(target):
            # Already fully emitted — word should have ended; clear.
            return ("", 0)
        if target[pos] == last_letter:
            new_pos = pos + 1
            return (target, new_pos)
        # Mismatch — clear the commit.
        return ("", 0)

    # Word-start attempt: only at letter_run_len == 0 post space/newline.
    last_cls = state.last_char_class
    if state.letter_run_len != 0 or last_cls not in (SPACE, NEWLINE):
        # Not at a word-start position; preserve existing commit only if
        # nothing else; usually this is first-token or punctuation-mid.
        if state.committed_word:
            return ("", 0)
        return (state.committed_word, state.committed_word_pos)

    # Don't commit inside speaker-label state.
    if state.speaker_label_state == 2:
        return ("", 0)

    # Hand-curated bigram trigger first (overrides diffuse formula nodes).
    lw = state.last_completed_word
    pw = state.prev_completed_word
    if lw and pw:
        key = (pw, lw)
        if key in _BIGRAM_COMMITS:
            target = _BIGRAM_COMMITS[key]
            if len(target) >= 3 and all("a" <= c <= "z" for c in target):
                return (target, 0)

    children = expected_next_words(state.formula_node)
    if not children:
        return ("", 0)
    words = [w for w in children.keys() if w and all("a" <= c <= "z" for c in w)]
    if not words:
        return ("", 0)
    if len(words) == 1:
        # Single expected word — require >= 3 letters to commit.
        target = words[0]
        if len(target) < 3:
            return ("", 0)
        return (target, 0)
    # Multiple expected words — commit to their common prefix if >= 3
    # letters (gives confident n-char rails before the alternation point).
    prefix = words[0]
    for w in words[1:]:
        # Trim prefix to longest common prefix with w.
        i = 0
        m = min(len(prefix), len(w))
        while i < m and prefix[i] == w[i]:
            i += 1
        prefix = prefix[:i]
        if len(prefix) < 3:
            return ("", 0)
    if len(prefix) < 3:
        return ("", 0)
    return (prefix, 0)


def update_word_commit(state: ModelState, token_id: int) -> ModelState:
    new_word, new_pos = _compute_commit(state)
    if new_word == state.committed_word and new_pos == state.committed_word_pos:
        return state
    return state.model_copy(
        update={"committed_word": new_word, "committed_word_pos": new_pos}
    )
