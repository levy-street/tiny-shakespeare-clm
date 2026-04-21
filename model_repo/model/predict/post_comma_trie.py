"""Multi-letter post-comma word-opener trie bias.

After a mid-clause comma, semicolon, or colon + space, the next word
in Shakespeare falls in a tight cluster:

  - Conjunctions: and, but, or, nor, yet, so, for
  - Adverbs / hinges: indeed, then, now, thus, still, ever, withal
  - Vocatives: sir, madam, friend, lord, lady, boy, master, mistress,
    father, mother, brother, sister, son, dear, sweet, good, gentle,
    noble, fair, brave, my (+ possessive+noun), your (+ noun)
  - Subordinate markers: that, which, who, whom, whose, as, if, since,
    though, when, where, while
  - Enumerative/continuation: to, of, with, from, by, in, on, at,
    unto, upon, the, a, an, this, that
  - Exclamations: O, Alas, Fie, Nay

The model's generic startword biases under-predict these specific
post-comma continuations because the context-class bucket is too
coarse. This layer does a letter-by-letter trie match constrained
to the hand-listed post-comma openers.

Gates:
  * speaker_label_state == 0
  * words_in_sentence >= 1 (we're continuing an existing sentence,
    not starting a new one after `.`, `!`, `?`)
  * chars_since_comma == letter_run_len + 1 — confirms this word
    started immediately after ", " (or "; ", ": ")
  * letter_run_len in [1, 5]

Scale moderate — the post-comma window is large (many possible
continuations), so the trie bias is sharper than first-letter but
softer than turn_opener_trie.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_OPENERS: tuple[str, ...] = (
    # Conjunctions
    "and", "but", "or", "nor", "yet", "so", "for",
    # Adverbs / discourse markers
    "indeed", "then", "now", "thus", "still", "ever", "withal",
    "else", "hence", "here", "there", "once",
    # Vocatives
    "sir", "madam", "friend", "lord", "lady", "boy", "girl",
    "master", "mistress", "father", "mother", "brother", "sister",
    "son", "daughter", "cousin", "husband", "wife",
    "king", "queen", "prince", "princess", "duke",
    "my", "thy", "our", "your", "his", "her",
    # Vocative-modifiers (often precede a vocative noun)
    "dear", "sweet", "good", "gentle", "noble", "fair", "brave",
    "poor", "true", "kind", "mine",
    # Subordinate markers
    "that", "which", "who", "whom", "whose", "as", "if", "since",
    "though", "when", "where", "while", "whilst", "unless", "until",
    "ere",
    # Enumerative / continuation
    "to", "of", "with", "from", "by", "in", "on", "at",
    "unto", "upon", "the", "a", "an", "this", "these",
    "those", "such", "one",
    # Exclamations
    "o", "oh", "ah", "alas", "fie", "nay", "ay", "tis",
    # Subject pronouns (common mid-sentence continuation)
    "i", "thou", "we", "he", "she", "they", "you", "it", "ye",
    # Auxiliaries / modals
    "is", "are", "was", "were", "be", "been",
    "hath", "has", "had", "have",
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
    "do", "does", "did", "doth",
    # Negation
    "not", "never", "none", "no",
    # Other common continuations
    "say", "speak", "hear", "come", "go", "take", "give",
    "look", "see", "let", "make",
    "more", "most", "much", "many", "each", "every", "all",
    "some", "any", "other", "both",
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


def post_comma_trie_bias(
    word_buffer: str,
    letter_run_len: int,
    speaker_label_state: int,
    words_in_sentence: int,
    chars_since_comma: int,
    last_char: str,
) -> list[float] | None:
    """Bias letter-by-letter completion of a word that started
    immediately after a mid-clause comma / semicolon / colon.
    """
    if speaker_label_state != 0:
        return None
    if words_in_sentence < 1:
        return None
    if letter_run_len < 1 or letter_run_len > 5:
        return None
    if not word_buffer:
        return None
    # The word must have started right after ", " (or "; ", ": ").
    # chars_since_comma is reset at any PUNCT_END / PUNCT_MID and
    # incremented per char. After ", A" chars_since_comma == 2 and
    # letter_run_len == 1, so chars_since_comma == letter_run_len + 1.
    if chars_since_comma != letter_run_len + 1:
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
        base = 4.50
    elif letter_run_len == 2:
        base = 5.25
    elif letter_run_len == 3:
        base = 4.75
    elif letter_run_len == 4:
        base = 3.25
    else:  # 5
        base = 2.25

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
