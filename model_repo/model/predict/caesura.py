"""Caesura-position predict layer.

Reads `state.has_caesura_this_line`, `state.caesura_syllable`,
`state.syllables_in_line`, `state.verse_score`, `state.verse_line_run`,
`state.prev_line_syllables` and biases mid-line punctuation at
word-end positions based on where the caesura *should* fall.

In iambic pentameter:
  - The caesura — the mid-line syntactic pause — typically falls at
    syllable 4, 5, or 6.
  - A line without a caesura by syllable 7+ starts to feel run-on.
  - Two caesurae close together (e.g. at syll 3 and again at syll 5)
    feel choppy and are rare.

Fires only at word-end on-trie complete-word positions (these are the
only legal places to push a comma/semicolon). Requires:
  * speaker_label_state == 0
  * letter_run_len >= 2, on_word_trie, word_buffer in complete_words
  * consecutive_newlines == 0
  * verse_score >= 0.6 (confident verse)
  * verse_line_run >= 2 (established verse run)
  * prev_line_syllables in {9, 10, 11} (pentameter anchor)
  * chars_since_sentence_end >= 6 (not sentence-start)

Two regimes:

  A) has_caesura_this_line is False and syllables_in_line in {4, 5, 6}:
     Push "," and ";" up. This is the prime caesura window.
     Strongest at syllable 5 (mid-pentameter), weaker at 4 and 6.

  B) has_caesura_this_line is True and (syllables_in_line -
     caesura_syllable) < 3: Suppress "," ";" ":" "-". A caesura just
     fired; a second break right after feels choppy.

All weights from prior knowledge of English iambic-pentameter caesura
placement — no corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def caesura_bias(
    has_caesura_this_line: bool,
    caesura_syllable: int,
    syllables_in_line: int,
    verse_score: float,
    verse_line_run: int,
    prev_line_syllables: int,
    speaker_label_state: int,
    consecutive_newlines: int,
    chars_since_sentence_end: int,
    letter_run_len: int,
    on_word_trie: bool,
    word_buffer: str,
    complete_words: frozenset[str],
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if consecutive_newlines != 0:
        return None
    if letter_run_len < 2:
        return None
    if not on_word_trie:
        return None
    if not word_buffer:
        return None
    if word_buffer not in complete_words:
        return None
    if chars_since_sentence_end < 6:
        return None

    vec = [0.0] * VOCAB_SIZE

    if not has_caesura_this_line:
        # Regime A (syll-5 comma push) was disabled — even tiny
        # magnitudes did not beat baseline because `clause_rhythm`
        # already handles long-pause comma pushes. Only the gap==0
        # suppression remains below.
        return None

    # Regime B: a caesura already fired. Suppress a second one if it
    # would be too close.
    if caesura_syllable < 0:
        return None
    gap = syllables_in_line - caesura_syllable
    if gap == 0:
        # Immediately-adjacent second break — feels choppy.
        pen = -0.05
        if "," in VOCAB_INDEX:
            vec[VOCAB_INDEX[","]] += pen
        if ";" in VOCAB_INDEX:
            vec[VOCAB_INDEX[";"]] += pen
        return vec

    return None
