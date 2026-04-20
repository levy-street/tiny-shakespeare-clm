"""Sentence-syllable parallelism bias.

Reads `state.syllables_in_sentence`, `state.prev_sentence_syllables`,
and `state.prev_prev_sentence_syllables`. When we're at a word-end
with a completed word-buffer, and the current sentence's syllable
count has caught up to the rolling average of recent sentences, boost
terminal punctuation ( . ? ! ) over continuation. When the current
sentence overshoots, strengthen the push. When significantly SHORT,
slightly suppress terminators (avoid premature closure).

Gate: speaker_label_state==0, letter_run_len >= 2, word_buffer_is_complete
(only at a plausible word boundary where a terminator could occur),
and we require at least ONE prior sentence's length to have a rhythmic
peer. Very short prior sentences (<=3 syllables — likely "Ay." or
"No.") do not anchor a rhythmic budget and get ignored.

Magnitudes are modest: this is a rhythmic cadence signal that should
nudge, not force. It stacks with turn_punct_texture, sentence_backbone,
sentence_length_prior.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Letter indices precomputed once.
_TERM_CHARS = (".", "?", "!")
_CONT_CHARS = (",", ";")


def sentence_syllable_parallel_bias(
    syllables_in_sentence: int,
    prev_sentence_syllables: int,
    prev_prev_sentence_syllables: int,
    letter_run_len: int,
    on_word_trie: bool,
    word_buffer_is_complete: bool,
    speaker_label_state: int,
    words_in_sentence: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if letter_run_len < 2:
        return None
    if not on_word_trie:
        return None
    if not word_buffer_is_complete:
        return None
    # Need at least 4 words in sentence before a terminator becomes
    # rhythmically plausible regardless of syllables — very short
    # openers (<=3 words) often continue with commas and more
    # clause material.
    if words_in_sentence < 4:
        return None

    # Build rolling peer. Prefer average of last two non-trivial peers.
    peers: list[int] = []
    if prev_sentence_syllables >= 4:
        peers.append(prev_sentence_syllables)
    if prev_prev_sentence_syllables >= 4:
        peers.append(prev_prev_sentence_syllables)
    if not peers:
        return None
    peer_avg = sum(peers) / len(peers)

    # Clamp peer_avg to plausible Shakespeare range (avoid wild values
    # from mis-classified sentences).
    if peer_avg < 5.0:
        peer_avg = 5.0
    elif peer_avg > 28.0:
        peer_avg = 28.0

    cur = syllables_in_sentence
    ratio = cur / peer_avg

    # Asymmetric schedule:
    #   ratio < 0.45: too short — small negative on terminators
    #   0.45 <= ratio < 0.80: still short — very gentle negative
    #   0.80 <= ratio < 1.05: in the sweet spot — moderate terminator boost
    #   1.05 <= ratio < 1.35: slightly over — stronger terminator boost
    #   1.35 <= ratio:        well over — strong terminator push
    # Suppress terminators when the current sentence is drastically
    # shorter than its rolling peer (only fires when ratio < 0.35 —
    # this catches truly premature closure attempts while staying out
    # of the noisier mid-ratio region where other layers provide
    # better signal). No-op in the neutral-to-excess range.
    if ratio < 0.35:
        term_boost = -0.08
        cont_boost = 0.01
    else:
        return None

    # More confidence when we have TWO peers agreeing.
    if len(peers) == 2 and peers[0] > 0 and peers[1] > 0:
        # Agreement: peers within 25% of each other.
        lo, hi = (peers[0], peers[1]) if peers[0] <= peers[1] else (peers[1], peers[0])
        if hi / max(lo, 1) <= 1.35:
            term_boost *= 1.20
            cont_boost *= 1.20

    if term_boost == 0.0 and cont_boost == 0.0:
        return None

    vec = [0.0] * VOCAB_SIZE
    for ch in _TERM_CHARS:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += term_boost
    for ch in _CONT_CHARS:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += cont_boost
    return vec
