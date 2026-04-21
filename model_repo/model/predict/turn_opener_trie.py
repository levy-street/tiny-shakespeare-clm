"""Multi-letter turn-opener trie bias.

`turn_opener.py` provides only a FIRST-LETTER bias for the first word of
a new speaker turn. Once the first letter is chosen, the model drifts
through generic letter-ngram continuations, often producing non-opener
content words like "Heath", "Clot", "Ale", "Aohshs" to open a turn —
whereas real Shakespeare turns open with a tight cluster of fixed
words (O, Alas, My lord, Come, Nay, What, Thou, Good my lord, etc.).

This layer is a multi-letter TRIE of common turn-openers. At each
letter position inside the FIRST word of a fresh turn, it asks:
  "which next letters would extend the current buffer to complete one
   of the known opener words?"
and boosts those letters. Letters that would break every opener
trajectory get a mild penalty (pushing the sampler back onto a real
opener path).

The list is drawn from prior knowledge of Early Modern English
speech-act openers. No corpus statistics.

Gates (hard):
  * speaker_label_state == 0
  * words_in_turn == 0 AND sentences_in_turn == 0
  * letter_run_len in [1, 6] — once we're past 6 letters, the buffer
    is committing to a real, possibly unrelated word; don't over-steer
  * lines_in_turn <= 1 — don't fire on a mid-turn new line

All weights are LOG-BIASES added to the logit vector. Scale is
moderate: opener words already benefit from startword + next_word
biases; this layer sharpens WITHIN those candidates by demanding
letter-by-letter continuation of an opener.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Turn-opener words in lowercase. Each is a word that disproportionately
# begins a Shakespearean speaker turn — interjections, vocatives, oaths,
# demonstratives, pronouns-as-subject, modal/auxiliary starters,
# imperative verbs, and common conjunctive/conditional openers.
#
# Capitalized versions are implicit: these are matched letter-by-letter
# against a case-folded buffer; the bias is distributed to both upper
# and lower case next-letter tokens with the upper-case variant
# slightly higher (turn openers are typically sentence-starts → caps).
_OPENERS: tuple[str, ...] = (
    # Primary exclamations / attention-getters
    "o", "oh", "ah", "ay", "aye", "alas", "alack", "fie",
    "hark", "hist", "hush", "peace", "silence", "soft",
    "zounds", "marry", "indeed", "amen", "behold",
    # Oaths and address openers
    "god", "good", "faith", "prithee", "pray", "beseech",
    # Possessive / vocative address
    "my", "our", "your", "thy", "thine", "his", "her",
    "sir", "madam", "mistress", "master",
    # Imperative verbs (most common turn-open imperatives)
    "come", "go", "hear", "speak", "tell", "look", "see",
    "hold", "stay", "stand", "yield", "rise", "sit", "mark",
    "attend", "await", "begone", "hence", "away", "forth",
    "give", "take", "bring", "send", "lend", "let",
    "weep", "mourn", "kneel", "swear", "think",
    # Negations / assents
    "no", "nay", "nor", "not", "none", "never",
    "yes", "yea", "yet", "ye",
    # Subject pronouns
    "i", "we", "thou", "thee", "you", "he", "she", "they", "it",
    # Demonstratives / articles
    "this", "that", "these", "those", "such", "the", "a", "an",
    # WH-words (for questions and discourse-opens)
    "what", "who", "whom", "whose", "why", "when", "where",
    "whence", "whither", "how",
    # Conditional / temporal / concessive
    "if", "though", "since", "unless", "until", "ere",
    # Auxiliaries / modals — often start a response
    "is", "are", "am", "was", "were", "be", "been",
    "hath", "has", "had", "have",
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
    "do", "does", "did", "doth",
    # Temporal adverbs / hinges
    "now", "then", "here", "there", "once", "tis", "nay",
    "still", "ever",
    # Prepositions that open discourse
    "by", "for", "of", "with", "upon", "unto", "to", "from",
    "in", "on", "at", "as",
    # Adjective openers (for "Fair lord", "Good my lord", etc.)
    "fair", "sweet", "gentle", "fond", "dear", "brave", "noble",
    "false", "true", "poor", "great", "kind",
    # Noun address
    "father", "mother", "brother", "sister", "son", "daughter",
    "cousin", "friend", "husband", "wife",
    "king", "queen", "prince", "princess", "duke", "lord", "lady",
    # Conjunctive openers
    "but", "and", "or", "so", "well",
)


# Build a prefix → set-of-next-letters index.
# _PREFIX_NEXT[(prefix, is_prefix_complete_opener)] = tuple of next letters.
# We'll instead just scan the word list each call — keeping the layer
# stateless and simple; the list is ~140 entries so it's cheap.
# Actually, precompute by prefix length for speed.
_PREFIX_INDEX: dict[str, set[str]] = {}
_COMPLETE_SET: set[str] = set()
for _w in _OPENERS:
    _COMPLETE_SET.add(_w)
    for _i in range(1, len(_w)):
        _pref = _w[:_i]
        _nxt = _w[_i]
        _PREFIX_INDEX.setdefault(_pref, set()).add(_nxt)


_LOWER_ALPHA = "abcdefghijklmnopqrstuvwxyz"


def turn_opener_trie_bias(
    word_buffer: str,
    letter_run_len: int,
    speaker_label_state: int,
    words_in_turn: int,
    sentences_in_turn: int,
    lines_in_turn: int,
) -> list[float] | None:
    """Multi-letter opener trie bias. Returns None when inactive."""
    # Gates.
    if speaker_label_state != 0:
        return None
    if words_in_turn != 0 or sentences_in_turn != 0:
        return None
    if lines_in_turn > 1:
        return None
    if letter_run_len < 1 or letter_run_len > 6:
        return None
    if not word_buffer:
        return None

    # Case-fold buffer — our opener list is lowercase.
    buf_lower = word_buffer.lower()
    # word_buffer may include a leading apostrophe or have case variants.
    # Filter to the pure-letter suffix matching letter_run_len letters.
    if len(buf_lower) < letter_run_len:
        return None
    # Take the last `letter_run_len` chars — these are the current word
    # prefix in letters.
    pref = buf_lower[-letter_run_len:]
    # Restrict to alphabetic only.
    for ch in pref:
        if ch not in _LOWER_ALPHA:
            return None

    nxt = _PREFIX_INDEX.get(pref)
    is_complete = pref in _COMPLETE_SET

    # If this prefix matches no known opener AND is not itself a complete
    # opener, we have nothing to steer toward — do nothing. (Don't penalize
    # everything: the first word may legitimately be a rare word.)
    if nxt is None and not is_complete:
        return None

    vec = [0.0] * VOCAB_SIZE

    # Position-dependent base magnitude. Stronger at position 1-3 where
    # the opener identity is still being decided; softer at 4-6 where
    # we're mostly just confirming a long opener.
    if letter_run_len == 1:
        base = 3.40
    elif letter_run_len == 2:
        base = 3.60
    elif letter_run_len == 3:
        base = 3.20
    elif letter_run_len == 4:
        base = 2.40
    elif letter_run_len == 5:
        base = 1.80
    else:  # 6
        base = 1.20

    if nxt:
        # Boost each next-letter that would continue a known opener.
        # Spread the mass: if many openers share this prefix, each
        # continuation is less discriminating.
        # Use a per-letter boost, amplified when there are FEW options
        # (sharper prediction).
        per_letter = base / (0.6 + 0.4 * len(nxt))
        for ch in nxt:
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += per_letter
            up = ch.upper()
            # Uppercase variants also boosted (turn-start is typically
            # sentence-start, so first letter is cap; subsequent are
            # lower). For letter_run_len==1, mainly cap; for 2+, lower.
            if letter_run_len == 1 and up in VOCAB_INDEX:
                vec[VOCAB_INDEX[up]] += per_letter * 0.25
        # Mild penalty on letters that would break every opener
        # trajectory — pushes the sampler back onto an opener path.
        break_penalty = -base * 0.15
        for ch in _LOWER_ALPHA:
            if ch in nxt:
                continue
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += break_penalty

    if is_complete:
        # If buffer is itself a complete opener word, mildly boost
        # terminators so the model commits the opener and starts the
        # next word rather than extending into nonsense (e.g. "com"+
        # →"come", "come"+" " rather than "come"+"s"+"t"+"e"...).
        # Only when there's NO pending continuation — if nxt exists,
        # the opener could be mid-extension (e.g. "go" vs "good") so
        # be gentler.
        term_boost = base * (0.55 if not nxt else 0.20)
        for ch in (" ", ",", ";", ":", ".", "!", "?", "\n"):
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += term_boost * (
                    1.0 if ch == " " else 0.50
                )

    return vec
