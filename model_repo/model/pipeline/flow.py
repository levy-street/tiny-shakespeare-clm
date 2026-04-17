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


_VOWELS_SET = frozenset("aeiouAEIOU")


# Words whose appearance marks an archaic / early-modern register.
# Not a frequency list — a hand-picked set of lexical markers that
# unambiguously signal "this scene is in archaic mode". Each bumps
# archaic_density by _ARCHAIC_BUMP (or _STRONG_BUMP for strong markers).
_ARCHAIC_STRONG: frozenset[str] = frozenset({
    "thou", "thee", "thy", "thine", "hath", "doth", "hast", "dost",
    "wilt", "shalt", "art", "wert", "canst", "didst", "wouldst",
    "couldst", "shouldst", "mayst", "mightst",
    "prithee", "methinks", "forsooth", "wherefore", "whence",
    "hither", "thither", "whither", "anon", "alack", "ere",
    "marry", "sirrah", "zounds", "quoth", "mayhap",
    "'tis", "'twas", "'twere", "'gainst",
})
_ARCHAIC_MILD: frozenset[str] = frozenset({
    "nay", "yea", "ay", "fie", "oft", "mine",
    "o'er", "ne'er", "e'er", "e'en",
    "unto", "upon",
})
_ARCHAIC_STRONG_BUMP = 0.28
_ARCHAIC_MILD_BUMP = 0.10
_ARCHAIC_DECAY = 0.985  # per completed word
# Modern-only markers that gently pull density down (explicitly
# *not* archaic — we saw a modern form that suggests the register
# is drifting toward modern). Small effect.
_MODERN_MARKERS: frozenset[str] = frozenset({
    "okay", "really",  # won't appear in Shakespeare; kept empty-ish
})


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

    # --- Phonotactic tracking inside the current word ---
    # letters_off_trie: letters written since the buffer first went
    # off-trie. 0 while on-trie or when no word is in progress.
    # consonants_since_vowel: consecutive consonants since the last
    # vowel in the current word (resets at word end, at vowel).
    # vowels_in_word: number of vowels seen in the current word.
    if not wb:
        letters_off_trie = 0
        consonants_since_vowel = 0
        vowels_in_word = 0
        vowels_since_consonant = 0
    else:
        last_ch = state.last_char
        is_letter = len(last_ch) == 1 and (
            ("a" <= last_ch <= "z") or ("A" <= last_ch <= "Z")
        )
        if on_trie:
            letters_off_trie = 0
        elif is_letter:
            letters_off_trie = state.letters_off_trie + 1
        else:
            letters_off_trie = state.letters_off_trie
        if is_letter:
            if last_ch in _VOWELS_SET:
                consonants_since_vowel = 0
                vowels_in_word = state.vowels_in_word + 1
                vowels_since_consonant = state.vowels_since_consonant + 1
            else:
                consonants_since_vowel = state.consonants_since_vowel + 1
                vowels_in_word = state.vowels_in_word
                vowels_since_consonant = 0
        else:
            consonants_since_vowel = state.consonants_since_vowel
            vowels_in_word = state.vowels_in_word
            vowels_since_consonant = state.vowels_since_consonant

    # Verse-mode rolling score. Update only when a line has just
    # completed (newline emitted that terminated a non-blank line).
    # We read from the updated `prev_line_length` (the just-finished
    # line's length). The score decays toward 0 on blank lines so that
    # speaker-label blanks don't force any particular mode.
    verse_score = state.verse_score
    if state.last_char == "\n":
        ln = state.prev_line_length
        if 1 < ln < 60:  # verse-shaped line
            delta = 0.7 if 20 <= ln <= 52 else 0.3
            verse_score = min(3.0, verse_score + delta)
        elif ln >= 70:  # prose-shaped line
            verse_score = max(-3.0, verse_score - 0.9)
        elif ln >= 60:
            verse_score = max(-3.0, verse_score - 0.4)
        else:
            # blank or very short: mild decay toward 0
            verse_score *= 0.9

    # Archaic register density: a rolling [0, 1] float.
    # On each completed word, decay + bump based on the word.
    archaic_density = state.archaic_density
    if state.just_finished_word:
        archaic_density *= _ARCHAIC_DECAY
        w = state.last_completed_word
        if w:
            if w in _ARCHAIC_STRONG:
                archaic_density = min(1.0, archaic_density + _ARCHAIC_STRONG_BUMP)
            elif w in _ARCHAIC_MILD:
                archaic_density = min(1.0, archaic_density + _ARCHAIC_MILD_BUMP)
    # Reset to 0 at start of a new speaker's dialogue (post-label
    # newline + double-newline would give us a fresh scene context).
    # Concretely, reset when we just emitted a blank line after a
    # label (consecutive_newlines == 2). This lets each speaker's
    # register develop fresh but preserves continuity within a speech.
    if state.consecutive_newlines >= 2 and state.last_char == "\n":
        # Preserve a fraction so scene-wide register isn't fully lost.
        archaic_density *= 0.6

    return state.model_copy(
        update={
            "on_word_trie": on_trie,
            "line_length_bucket": line_length_bucket,
            "sent_distance_bucket": sent_distance_bucket,
            "after_function_word": after_function_word,
            "in_prose_line": in_prose_line,
            "letters_off_trie": letters_off_trie,
            "consonants_since_vowel": consonants_since_vowel,
            "vowels_in_word": vowels_in_word,
            "vowels_since_consonant": vowels_since_consonant,
            "verse_score": verse_score,
            "archaic_density": archaic_density,
        }
    )
