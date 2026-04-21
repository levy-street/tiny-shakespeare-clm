"""Multi-letter sentence-opener trie bias.

Counterpart to `turn_opener_trie.py` for the FIRST word of a fresh
sentence in the middle of a turn (words_in_sentence == 0 after a
sentence-ending punctuation, but words_in_turn > 0 — the turn is in
progress). Shakespeare's within-turn sentences open with a tight
cluster of words:

  - Continuation conjunctions: And, But, Yet, Nor, For, So
  - Temporal hinges: Now, Then, Thus, Here, There
  - Subject pronouns: I, Thou, He, She, They, We, You
  - Possessive+noun openings: My, Thy, Our, Your, His, Her
  - Imperatives: Come, Go, Hear, Speak, Look, Stay, Hold
  - Articles / demonstratives: The, A, An, This, That, These, Those
  - Exclamations: O, Alas, Fie, Ah, Nay
  - Conditional/temporal: If, When, Where, Though, Since, Whilst
  - Interrogatives: What, Why, Who, How
  - Modals / auxiliaries: Is, Are, Was, Were, Be, Hath, Have, Will,
    Shall, Would, Should, Can, May, Must, Doth, Do

The layer biases letter-by-letter completion of the first word
toward one of these openers. Fires in a narrower window than
turn_opener_trie (letter positions 1-5) and at a gentler scale —
mid-turn sentences tend to be more varied than first-sentence
openers.

No corpus statistics; word list is hand-authored.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Common sentence-openers that appear mid-turn. Lowercase; case is
# handled at bias time (first position biases uppercase variant; later
# positions bias lowercase).
_OPENERS: tuple[str, ...] = (
    # Continuation conjunctions (very common mid-turn sentence openers)
    "and", "but", "yet", "nor", "for", "so", "or",
    # Temporal / logical hinges
    "now", "then", "thus", "hence", "thereof",
    "here", "there", "still", "ever",
    # Subject pronouns
    "i", "thou", "we", "you", "he", "she", "they", "it", "ye",
    # Possessive + noun openings (the common vocative/address structure)
    "my", "thy", "our", "your", "his", "her", "their", "thine",
    # Imperative verbs
    "come", "go", "hear", "speak", "tell", "look", "see",
    "hold", "stay", "stand", "yield", "rise", "sit", "mark",
    "behold", "attend", "take", "give", "bring",
    "weep", "kneel", "swear", "think",
    # Articles / demonstratives
    "the", "a", "an", "this", "that", "these", "those", "such",
    # Exclamations and attention-getters
    "o", "oh", "ah", "ay", "alas", "alack", "fie", "nay", "no",
    "hark", "peace", "soft", "tis",
    # Conditional / temporal / concessive
    "if", "when", "where", "whilst", "though", "since", "unless", "ere",
    # Interrogatives
    "what", "why", "who", "whom", "how", "whither", "whence",
    # Modals / auxiliaries
    "is", "are", "am", "was", "were", "be", "been",
    "hath", "has", "had", "have",
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
    "do", "does", "did", "doth",
    # Negation / assertion adverbs
    "not", "never", "neither", "none",
    "yea", "indeed", "verily", "truly",
    # Prepositions (less common opener, but valid)
    "by", "of", "upon", "unto", "to", "with", "from",
    "in", "on", "at", "as",
    # Polite / oath openers
    "pray", "prithee", "beseech", "good", "god", "faith", "marry",
    "well",
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


def sentence_opener_trie_bias(
    word_buffer: str,
    letter_run_len: int,
    speaker_label_state: int,
    words_in_sentence: int,
    words_in_turn: int,
    sentences_in_turn: int,
) -> list[float] | None:
    """Bias letter-by-letter completion of the first word of a new
    within-turn sentence toward known sentence-openers.

    Returns None when inactive.
    """
    if speaker_label_state != 0:
        return None
    # We want mid-turn new sentences: words_in_sentence == 0 AND
    # we're past the turn's opening sentence (sentences_in_turn >= 1
    # OR words_in_turn > 0). Skip the first sentence of the turn —
    # turn_opener_trie handles that with a tighter list/scale.
    if words_in_sentence != 0:
        return None
    # If both counters are zero, we're at the turn-start: defer to
    # turn_opener_trie.
    if words_in_turn == 0 and sentences_in_turn == 0:
        return None
    if letter_run_len < 1 or letter_run_len > 5:
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

    # Scale schedule — narrower/softer than turn_opener_trie. Mid-turn
    # sentences are more varied than turn-openers.
    if letter_run_len == 1:
        base = 2.25
    elif letter_run_len == 2:
        base = 2.55
    elif letter_run_len == 3:
        base = 2.25
    elif letter_run_len == 4:
        base = 1.65
    else:  # 5
        base = 1.20

    vec = [0.0] * VOCAB_SIZE

    if nxt:
        per_letter = base / (0.6 + 0.4 * len(nxt))
        for ch in nxt:
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += per_letter
            up = ch.upper()
            if letter_run_len == 1 and up in VOCAB_INDEX:
                vec[VOCAB_INDEX[up]] += per_letter * 0.20
        break_penalty = -base * 0.10
        for ch in _LOWER_ALPHA:
            if ch in nxt:
                continue
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += break_penalty

    if is_complete:
        term_boost = base * (0.45 if not nxt else 0.15)
        for ch in (" ", ",", ";", ":", ".", "!", "?", "\n"):
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += term_boost * (
                    1.0 if ch == " " else 0.50
                )

    return vec
