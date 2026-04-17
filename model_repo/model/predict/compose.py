"""Compose predict layers into the final distribution.

Order:
  1. Start from unigram log-probs (prior).
  2. Add context-class biases (from context.py).
  3. If last char is a letter, add letter-bigram biases (from bigram.py).
  4. If last char is a space or single newline (word start), add
     startword biases (from startword.py).
  5. Apply speaker-label FSM specific boosts.
  6. Log-softmax renormalize.
"""

from __future__ import annotations

import math

from ..pipeline.linguistic import (
    APOSTROPHE,
    LOWER_CONS,
    LOWER_VOWEL,
    NEWLINE,
    SPACE,
    UPPER,
)
from ..state import ModelState
from ..vocab import VOCAB, VOCAB_INDEX, VOCAB_SIZE
from .bigram import bigram_bias
from .context import CTX_BIAS_VECTORS, context_key
from .letter3 import letter3_bias
from .next_word import next_word_bias
from .speaker_trie import speaker_trie_bias
from .startword import START_BIAS
from .trigram import trigram_bias
from .unigram import UNIGRAM_LOGPROBS
from .word_trie import FORCE_END_BIAS, is_on_trie, word_trie_bias


def _log_softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    logz = m + math.log(z)
    return [x - logz for x in logits]


def predict(state: ModelState) -> list[float]:
    # Layer 1: unigram.
    logits = list(UNIGRAM_LOGPROBS)

    # Layer 2: context-class biases.
    ctx = context_key(state)
    ctx_bias = CTX_BIAS_VECTORS[ctx]
    for i in range(VOCAB_SIZE):
        logits[i] += ctx_bias[i]

    last_cls = state.last_char_class

    # Layer 3: letter-bigram biases (only inside letter runs).
    if state.last_char and last_cls in (UPPER, LOWER_VOWEL, LOWER_CONS):
        bi = bigram_bias(state.last_char)
        if bi is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += bi[i]

    # Layer 3b: trigram digraph biases (last two letters).
    if state.last_char and state.prev_char:
        tg = trigram_bias(state.prev_char, state.last_char)
        if tg is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += tg[i]

    # Layer 3b2: letter-trigram bias (last 3 letters → next).
    if state.word_buffer:
        l3 = letter3_bias(state.word_buffer)
        if l3 is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += l3[i]

    # Layer 3c: word-trie completion bias.
    if state.word_buffer:
        wt = word_trie_bias(state.word_buffer)
        if wt is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += wt[i]

    # Layer 3d: speaker-label trie bias.
    if state.speaker_buffer:
        st = speaker_trie_bias(state.speaker_buffer)
        if st is not None:
            for i in range(VOCAB_SIZE):
                logits[i] += st[i]

    # Layer 4: start-of-word bias (after space or single newline).
    if last_cls == SPACE or (
        last_cls == NEWLINE and state.consecutive_newlines == 1
    ):
        for i in range(VOCAB_SIZE):
            logits[i] += START_BIAS[i]

        # Layer 4b: next-word (word-bigram) first-letter bias.
        if state.last_completed_word:
            nw = next_word_bias(state.last_completed_word)
            if nw is not None:
                for i in range(VOCAB_SIZE):
                    logits[i] += nw[i]

        # Layer 4c: at a sentence start (post ". ", post "? ", post "! "
        # or post a double-newline blank line), strongly boost capital
        # letters relative to lowercase. The training corpus always
        # starts new sentences with a capital (outside of mid-sentence
        # continuations).
        is_sentence_start = (
            state.prev_char_class == 6  # PUNCT_END — . ? !
            and last_cls == SPACE
        ) or (
            last_cls == NEWLINE and state.consecutive_newlines == 1
            and state.chars_since_sentence_end <= 2
        )
        if is_sentence_start:
            for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] += 1.2
            for ch in "abcdefghijklmnopqrstuvwxyz":
                if ch in VOCAB_INDEX:
                    logits[VOCAB_INDEX[ch]] -= 0.5

    # Layer 5: speaker-label-specific boosts.
    if state.speaker_label_state == 3:
        # After ":" closing a label: strongly expect \n.
        logits[VOCAB_INDEX["\n"]] += 3.5
    elif state.speaker_label_state == 2 and state.upper_run_len >= 3:
        # Name is long enough to plausibly end; boost ":"
        logits[VOCAB_INDEX[":"]] += 2.0

    # After apostrophe, specifically boost common contraction letters.
    if last_cls == APOSTROPHE:
        for ch, boost in (("s", 2.0), ("d", 1.5), ("t", 1.5), ("l", 1.0),
                          ("r", 0.8), ("v", 0.8)):
            logits[VOCAB_INDEX[ch]] += boost

    # Layer 6: line-position / flow-aware modulations.
    # Only apply when we're outside speaker-label territory.
    if state.speaker_label_state == 0:
        llb = state.line_length_bucket
        sdb = state.sent_distance_bucket
        # At end-of-word position (letter_run >= 2 AND on_word_trie)
        # on progressively longer lines, newline becomes more likely as
        # the word's terminator. Training verse wraps ~30-50 chars;
        # prose ~60-80.
        if state.letter_run_len >= 2 and state.on_word_trie:
            if llb == 1:
                logits[VOCAB_INDEX["\n"]] += 1.8
            elif llb == 2:
                logits[VOCAB_INDEX["\n"]] += 3.5
            elif llb == 3:
                logits[VOCAB_INDEX["\n"]] += 5.0
        # Overdue sentence end: at word-end on-trie, boost sentence-end
        # punctuation so the model actually closes sentences.
        if state.letter_run_len >= 2 and state.on_word_trie and sdb >= 1:
            bump = 3.8 if sdb == 1 else 6.0
            logits[VOCAB_INDEX["."]] += bump
            if "?" in VOCAB_INDEX:
                logits[VOCAB_INDEX["?"]] += bump * 0.3
            if "!" in VOCAB_INDEX:
                logits[VOCAB_INDEX["!"]] += bump * 0.3
            if "," in VOCAB_INDEX:
                logits[VOCAB_INDEX[","]] += bump * 0.6
            if ";" in VOCAB_INDEX:
                logits[VOCAB_INDEX[";"]] += bump * 0.45
        # Even in short sentences, a comma becomes plausible after a
        # few completed words — Shakespeare is comma-heavy.
        if (
            state.letter_run_len >= 2
            and state.on_word_trie
            and state.chars_since_sentence_end >= 10
            and state.chars_since_sentence_end < 40
        ):
            if "," in VOCAB_INDEX:
                logits[VOCAB_INDEX[","]] += 4.8
            if ";" in VOCAB_INDEX:
                logits[VOCAB_INDEX[";"]] += 2.0

    return _log_softmax(logits)
