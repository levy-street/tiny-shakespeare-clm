"""Multi-letter line-start (enjambed) trie bias.

Verse-line enjambment in Shakespeare — a single line break within an
ongoing sentence — almost always opens the next line with a small
set of continuation words:

  - Subject pronouns: I, Thou, He, She, They, We, You, Ye, It
  - Relative / subordinate markers: That, Which, Who, Whom, Whose,
    When, Where, While, Whilst, Though, Since, If
  - Possessive / vocative openers: My, Thy, Our, Your, His, Her
  - Conjunctions: And, But, Yet, Nor, Or, For, So
  - Articles / demonstratives: The, A, An, This, That, These, Those
  - Prepositions: Of, To, With, By, In, On, From, Upon, Unto
  - Auxiliaries / modals: Is, Are, Was, Were, Be, Hath, Have, Will,
    Would, Should, Shall, Can, May, Must, Do, Doth
  - Content verbs (imperatives / continuations): Come, Go, Speak,
    Hear, See, Know, Tell, Look, Make, Take, Give, Let
  - Adjectives (continuation): Fair, Good, Sweet, Dear, Brave,
    Noble, Poor, True, Kind

This is a counterpart to sentence_opener_trie that fires specifically
for ENJAMBED lines — lines that continue a sentence started on the
previous line. In those positions, the first word is grammatically
continuing the prior clause, so the opener distribution is narrow
but different from sentence-start or turn-start.

Gates:
  * speaker_label_state == 0
  * letter_run_len == chars_since_newline (word started at line start)
  * letter_run_len in [1, 5]
  * words_in_sentence >= 1 (ENJAMBED: sentence in progress)
  * words_in_turn >= 1 (not a turn-start)
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_OPENERS: tuple[str, ...] = (
    # Subject pronouns
    "i", "thou", "we", "you", "he", "she", "they", "it", "ye",
    # Relative / subordinate
    "that", "which", "who", "whom", "whose",
    "when", "where", "while", "whilst", "though", "since", "if",
    # Possessive / vocative
    "my", "thy", "our", "your", "his", "her", "their", "thine",
    # Conjunctions
    "and", "but", "yet", "nor", "or", "for", "so",
    # Articles / demonstratives
    "the", "a", "an", "this", "these", "those", "such",
    # Prepositions
    "of", "to", "with", "by", "in", "on", "from", "at", "as",
    "upon", "unto", "into", "against", "among", "between",
    "before", "after", "through", "without", "within",
    # Auxiliaries / modals
    "is", "are", "am", "was", "were", "be", "been", "being",
    "hath", "has", "had", "have", "having",
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
    "do", "does", "did", "doth",
    # Content verbs (imperatives / continuations)
    "come", "go", "speak", "hear", "see", "know", "tell",
    "look", "make", "take", "give", "let", "keep", "hold",
    "say", "think", "love", "die", "live", "rest",
    "stand", "sit", "rise", "fall", "weep", "swear",
    "bring", "send", "mark", "leave",
    # Adjectives (continuation)
    "fair", "good", "sweet", "dear", "brave", "noble", "poor",
    "true", "kind", "false", "proud", "gentle", "fond",
    # Negation
    "not", "never", "none", "no", "nothing",
    # Temporal / adverbial
    "now", "then", "thus", "here", "there", "once", "ever",
    "still", "indeed", "well",
    # Exclamations (rare at enjambment but possible)
    "o", "oh", "ah",
)


_PREFIX_INDEX: dict[str, set[str]] = {}
_COMPLETE_SET: set[str] = set()
for _w in _OPENERS:
    _COMPLETE_SET.add(_w)
    for _i in range(1, len(_w)):
        _pref = _w[:_i]
        _nxt = _w[_i]
        _PREFIX_INDEX.setdefault(_pref, set()).add(_nxt)


_LOWER_ALPHA = "abcdefghijklmnopqrstuvwxyz"


def line_start_trie_bias(
    word_buffer: str,
    letter_run_len: int,
    chars_since_newline: int,
    speaker_label_state: int,
    words_in_sentence: int,
    words_in_turn: int,
) -> list[float] | None:
    """Bias letter-by-letter completion of the first word of an
    enjambed verse line (single line break mid-sentence)."""
    if speaker_label_state != 0:
        return None
    if letter_run_len < 1 or letter_run_len > 5:
        return None
    if words_in_sentence < 1:
        return None
    if words_in_turn < 1:
        return None
    # The word must have started immediately after \n (no leading
    # whitespace/indent between the newline and the first letter).
    if chars_since_newline != letter_run_len:
        return None
    if not word_buffer:
        return None

    buf_lower = word_buffer.lower()
    if len(buf_lower) < letter_run_len:
        return None
    pref = buf_lower[-letter_run_len:]
    for ch in pref:
        if ch not in _LOWER_ALPHA:
            return None

    nxt = _PREFIX_INDEX.get(pref)
    is_complete = pref in _COMPLETE_SET
    if nxt is None and not is_complete:
        return None

    if letter_run_len == 1:
        base = 1.95
    elif letter_run_len == 2:
        base = 2.28
    elif letter_run_len == 3:
        base = 2.08
    elif letter_run_len == 4:
        base = 1.43
    else:  # 5
        base = 1.04

    vec = [0.0] * VOCAB_SIZE

    if nxt:
        per_letter = base / (0.6 + 0.4 * len(nxt))
        for ch in nxt:
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += per_letter
        break_penalty = -base * 0.10
        for ch in _LOWER_ALPHA:
            if ch in nxt:
                continue
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += break_penalty

    if is_complete:
        term_boost = base * (0.50 if not nxt else 0.18)
        for ch in (" ", ",", ";", ":", ".", "!", "?", "\n"):
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += term_boost * (
                    1.0 if ch == " " else 0.50
                )

    return vec
