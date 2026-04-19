"""Urgency / tempo predict layer.

Reads `state.urgency_tempo` (0..1 float maintained by pipeline/flow.py)
and biases the distribution toward the *feel* of frantic vs. languid
text:

  * High urgency (>= 0.55): Shakespeare at full frantic tilt —
    commands, exclamations, short bursts. Bias toward:
      - "!"  at word-end (urgent sentences close loud)
      - space at word-end inside a long word (cut the word short,
        favor brevity)
      - NOT toward "," / ";" / ":" (urgent speech doesn't branch
        into subclauses — it fires and closes)
    Penalize long content-word-continuation letters (the middle of
    a polysyllable in urgent mode is unlikely).

  * Mid urgency (0.30 .. 0.55): mild bump to "!" at sentence-end.

  * Low urgency (< 0.1) and very long sentence-open (words_in_sentence
    >= 10): slight nudge toward continuation — reflective speech
    breathes longer.

Fires at three distinct positions:

  A) AT WORD-END (letter_run_len >= 2, on_word_trie, word_buffer
     in complete_words, speaker_label_state == 0, consecutive_newlines
     == 0, chars_since_sentence_end >= 10):
       - urgency >= 0.55: push "!" and ".".
       - urgency in [0.3, 0.55): gentle "!" push.

  B) MID-WORD inside a word that's grown long (letter_run_len >= 5,
     word_buffer in complete_words, on_word_trie):
       - urgency >= 0.55: push space (close the word).

  C) SENTENCE-END TRIGGER position (just after word-end at sentence
     punct), not currently used — reserved for future consumers.

All weights from prior knowledge of English performative speech —
no corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def urgency_word_end_bias(
    urgency_tempo: float,
    letter_run_len: int,
    on_word_trie: bool,
    word_buffer: str,
    complete_words: frozenset[str],
    speaker_label_state: int,
    consecutive_newlines: int,
    chars_since_sentence_end: int,
    words_in_sentence: int,
) -> list[float] | None:
    """Bias at word-end positions based on urgency_tempo."""
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
    if chars_since_sentence_end < 10:
        return None
    if words_in_sentence < 2:
        return None

    vec = [0.0] * VOCAB_SIZE

    if urgency_tempo >= 0.55:
        # Frantic mode. Push "!" and "."; suppress mid-clause punct.
        # Magnitude scales with how deep into the urgency range.
        strong = 0.10 + 0.20 * (urgency_tempo - 0.55) / 0.45
        strong = min(0.30, strong)
        if "!" in VOCAB_INDEX:
            vec[VOCAB_INDEX["!"]] += strong
        if "." in VOCAB_INDEX:
            vec[VOCAB_INDEX["."]] += strong * 0.55
        # Suppress comma/semicolon/colon — urgent speech doesn't nest.
        if "," in VOCAB_INDEX:
            vec[VOCAB_INDEX[","]] -= strong * 0.30
        if ";" in VOCAB_INDEX:
            vec[VOCAB_INDEX[";"]] -= strong * 0.30
        return vec

    if urgency_tempo >= 0.30:
        # Mid-urgency: gentle "!" push, no other change.
        mild = 0.06
        if "!" in VOCAB_INDEX:
            vec[VOCAB_INDEX["!"]] += mild
        return vec

    if urgency_tempo < 0.08 and words_in_sentence >= 10:
        # Very languid + long sentence: gentle continuation bias
        # (space — keep writing, don't close).
        if " " in VOCAB_INDEX:
            vec[VOCAB_INDEX[" "]] += 0.03
        # Suppress sentence-end (too early to stop a reflective line).
        if "." in VOCAB_INDEX:
            vec[VOCAB_INDEX["."]] -= 0.04
        return vec

    return None


def urgency_long_word_bias(
    urgency_tempo: float,
    letter_run_len: int,
    on_word_trie: bool,
    word_buffer: str,
    complete_words: frozenset[str],
    speaker_label_state: int,
) -> list[float] | None:
    """At high urgency inside a 5+-letter complete-word position,
    gently push toward space (close the word, keep the tempo tight)."""
    if speaker_label_state != 0:
        return None
    if urgency_tempo < 0.55:
        return None
    if letter_run_len < 5:
        return None
    if not on_word_trie:
        return None
    if not word_buffer:
        return None
    if word_buffer not in complete_words:
        return None
    vec = [0.0] * VOCAB_SIZE
    if " " in VOCAB_INDEX:
        vec[VOCAB_INDEX[" "]] += 0.06
    return vec
