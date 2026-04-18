"""Anti-repetition bias at word-start.

Reads `state.recent_clause_words` and penalizes the first letter of any
word that has already appeared this clause — escalating with count.
The goal is to break echo-loop pathology ("there there there",
"hear hear hear") without damaging legitimate refrain or anaphora
(which resets on speaker turn / sentence end).

The penalty is gentle for 1 repeat, moderate for 2, heavy for 3+.
Returns None if no word has appeared 1+ times in the clause.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Words that are allowed to repeat freely — they legitimately occur in
# anaphora, refrain, and natural close-range repetition in Shakespeare.
# Function words and short closed-class items dominate this list.
_EXEMPT: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on",
    "at", "by", "for", "with", "as", "that", "this", "if", "so",
    "no", "not", "i", "me", "my", "we", "us", "our", "he", "him",
    "his", "she", "her", "it", "its", "they", "them", "their",
    "thou", "thee", "thy", "thine", "you", "your", "ye",
    "be", "am", "is", "are", "was", "were", "been", "being",
    "do", "does", "did", "done", "have", "has", "had",
    "will", "shall", "would", "should", "may", "might", "can", "could",
    "must", "o", "oh", "ah", "ay", "well", "now", "then", "yet",
    "what", "who", "where", "when", "why", "how",
    "from", "up", "out", "than",
    "'tis", "'twas",
})


def repetition_start_bias(
    recent_clause_words: tuple[str, ...],
) -> list[float] | None:
    """Penalize first letters of content-words repeated in this clause.

    Counts occurrences of each word in recent_clause_words. For each
    non-exempt word with count >= 2, apply a penalty to that word's
    first letter proportional to repeat count.
    """
    if not recent_clause_words:
        return None

    # Count occurrences.
    counts: dict[str, int] = {}
    for w in recent_clause_words:
        if w and w not in _EXEMPT:
            counts[w] = counts.get(w, 0) + 1

    if not counts:
        return None

    vec = [0.0] * VOCAB_SIZE
    any_hit = False
    # Penalize the first letter of any non-exempt word seen this clause.
    # Even one recent non-exempt word gets a small nudge away, to break
    # symmetry; further repeats escalate sharply.
    for w, c in counts.items():
        if c < 1:
            continue
        first = w[0]
        # Empirically Shakespeare DOES repeat content words within
        # a clause (anaphora, parallelism), but rarely THREE times
        # in a row. Tuned to catch runaway echo without killing
        # legit 2-peats.
        if c < 2:
            continue
        if c == 2:
            pen = -0.80
        elif c == 3:
            pen = -1.60
        else:
            pen = -2.50
        lo = first
        up = first.upper()
        if lo in VOCAB_INDEX:
            vec[VOCAB_INDEX[lo]] += pen
            any_hit = True
        if up != lo and up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] += pen * 0.5
            any_hit = True

    if not any_hit:
        return None
    return vec
