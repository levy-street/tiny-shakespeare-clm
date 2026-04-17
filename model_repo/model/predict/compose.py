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

    return _log_softmax(logits)
