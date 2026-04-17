"""The `predict` half of the two-function model API.

`predict(state)` returns a list of natural-log-probabilities over VOCAB.
Exponentiated, the vector must be a valid probability distribution.

Strategy
--------

1. Start from a unigram base distribution (computed once at import).
2. Derive a "context key" from the linguistic state — coarse classes
   describing what has just happened (last char class, letter-run length,
   speaker-label FSM state, etc.).
3. For each context key, a hand-coded table of per-target-class log-biases
   is applied to the unigram, then renormalized.

The biases are hand-authored from prior knowledge of English + Shakespeare
char statistics: what tends to follow a newline, a space, a comma, a
period, a letter inside a word, the start of a speaker label, and so on.
No corpus statistics — all biases are prior knowledge.
"""

from __future__ import annotations

import math
from pathlib import Path

from .pipeline.linguistic import (
    APOSTROPHE,
    CLASS_OF_TOKEN,
    DASH,
    LOWER_CONS,
    LOWER_VOWEL,
    NEWLINE,
    OTHER,
    PUNCT_END,
    PUNCT_MID,
    SPACE,
    UPPER,
)
from .state import ModelState
from .vocab import VOCAB, VOCAB_INDEX, VOCAB_SIZE

_TRAIN = Path(__file__).resolve().parents[2] / "corpus" / "train.txt"


def _unigram_logprobs() -> list[float]:
    text = _TRAIN.read_text(encoding="utf-8")
    counts = [0] * VOCAB_SIZE
    for ch in text:
        counts[VOCAB_INDEX[ch]] += 1
    total = sum(counts)
    return [math.log(c / total) for c in counts]


_UNIGRAM_LOGPROBS: list[float] = _unigram_logprobs()


# ---------------------------------------------------------------------------
# Context-conditional class-level log-biases.
#
# Each "context" is a coarse situation after `advance`. Each maps to a
# vector of 10 class log-biases (added to the unigram log-prob of every
# token of that class, then renormalized).
# ---------------------------------------------------------------------------


# Context identifiers (int).
CTX_START = 0
CTX_AFTER_DOUBLE_NL = 1  # consecutive_newlines >= 2 (awaiting speaker/blank)
CTX_AFTER_SINGLE_NL = 2
CTX_AFTER_SPACE_SENT_START = 3  # start of new sentence mid-line
CTX_AFTER_SPACE = 4
CTX_AFTER_PUNCT_END = 5  # . ? !
CTX_AFTER_PUNCT_MID = 6  # , ; :
CTX_AFTER_APOS = 7
CTX_AFTER_DASH = 8
CTX_IN_SPEAKER_LABEL = 9  # inside upper-case name run
CTX_AFTER_COLON_LABEL = 10  # right after ":" closing a speaker label
CTX_IN_WORD_SHORT = 11  # inside a lowercase word, position 1–3
CTX_IN_WORD_MID = 12  # position 4–6
CTX_IN_WORD_LONG = 13  # position 7+
CTX_AFTER_UPPER_START = 14  # first upper-case of a word (not speaker label)
CTX_OTHER = 15

N_CTX = 16

# Per-class log biases indexed by [ctx][class].
# Classes:      NL   SP   UP   LV   LC   AP   PE   PM   DA   OT
# Defaults are small bumps; numbers are additive on top of unigram log-probs.
#
# Think of these as "how much extra mass to spread across this class in
# this context" compared to the unigram prior.

_BIAS: list[list[float]] = [[0.0] * 10 for _ in range(N_CTX)]


def _set(ctx: int, values: dict[int, float]) -> None:
    for k, v in values.items():
        _BIAS[ctx][k] = v


# Start of text: Shakespeare corpus starts with a speaker label ("First
# Citizen:"). So lean heavily on uppercase + letter.
_set(
    CTX_START,
    {
        UPPER: 6.0,
        LOWER_CONS: 1.0,
        LOWER_VOWEL: 0.5,
        NEWLINE: -4.0,
        SPACE: -4.0,
        PUNCT_END: -6.0,
        PUNCT_MID: -6.0,
        APOSTROPHE: -4.0,
        DASH: -4.0,
        OTHER: -6.0,
    },
)

# Two newlines in a row: strongly expect uppercase speaker label or blank.
_set(
    CTX_AFTER_DOUBLE_NL,
    {
        UPPER: 5.0,
        LOWER_CONS: -2.5,
        LOWER_VOWEL: -2.5,
        NEWLINE: -1.0,
        SPACE: -3.0,
        PUNCT_END: -6.0,
        PUNCT_MID: -6.0,
        APOSTROPHE: -4.0,
        DASH: -4.0,
        OTHER: -6.0,
    },
)

# Single newline mid-document: usually the start of the next verse line
# or continuation. Expect letters (often lowercase in prose, upper in
# verse) and double-newline (blank line between scenes/paras).
_set(
    CTX_AFTER_SINGLE_NL,
    {
        UPPER: 1.6,
        LOWER_CONS: 0.6,
        LOWER_VOWEL: 0.5,
        NEWLINE: 0.4,
        SPACE: -3.0,
        PUNCT_END: -5.0,
        PUNCT_MID: -5.0,
        APOSTROPHE: -2.5,
        DASH: -2.0,
        OTHER: -5.0,
    },
)

# Sentence start mid-line (after ". " etc.): expect uppercase letter.
_set(
    CTX_AFTER_SPACE_SENT_START,
    {
        UPPER: 3.5,
        LOWER_CONS: 0.6,
        LOWER_VOWEL: 0.3,
        NEWLINE: -5.0,
        SPACE: -6.0,
        PUNCT_END: -6.0,
        PUNCT_MID: -6.0,
        APOSTROPHE: 0.2,
        DASH: -4.0,
        OTHER: -6.0,
    },
)

# After a mid-line space (mid-sentence word boundary).
_set(
    CTX_AFTER_SPACE,
    {
        UPPER: -0.5,
        LOWER_CONS: 1.5,
        LOWER_VOWEL: 1.3,
        NEWLINE: -5.0,
        SPACE: -6.0,
        PUNCT_END: -5.0,
        PUNCT_MID: -5.0,
        APOSTROPHE: -0.5,
        DASH: -3.0,
        OTHER: -5.0,
    },
)

# After a sentence-ending punctuation (. ? !): expect space or newline.
_set(
    CTX_AFTER_PUNCT_END,
    {
        SPACE: 3.0,
        NEWLINE: 2.0,
        UPPER: -3.0,
        LOWER_CONS: -4.0,
        LOWER_VOWEL: -4.0,
        PUNCT_END: -4.0,
        PUNCT_MID: -4.0,
        APOSTROPHE: -2.0,
        DASH: -3.0,
        OTHER: -4.0,
    },
)

# After , ; : almost certainly space (or newline).
_set(
    CTX_AFTER_PUNCT_MID,
    {
        SPACE: 3.5,
        NEWLINE: 1.5,
        UPPER: -4.0,
        LOWER_CONS: -4.5,
        LOWER_VOWEL: -4.5,
        PUNCT_END: -5.0,
        PUNCT_MID: -5.0,
        APOSTROPHE: -3.0,
        DASH: -3.0,
        OTHER: -5.0,
    },
)

# After apostrophe: contraction letter follows ('s, 'd, 'll, 't, 're, 've).
_set(
    CTX_AFTER_APOS,
    {
        LOWER_CONS: 3.0,
        LOWER_VOWEL: 1.5,
        UPPER: -4.0,
        SPACE: -2.5,
        NEWLINE: -4.0,
        PUNCT_END: -4.0,
        PUNCT_MID: -4.0,
        APOSTROPHE: -4.0,
        DASH: -4.0,
        OTHER: -5.0,
    },
)

# After dash: usually space or letter.
_set(
    CTX_AFTER_DASH,
    {
        SPACE: 2.0,
        LOWER_CONS: 1.0,
        LOWER_VOWEL: 0.6,
        UPPER: -1.0,
        NEWLINE: -2.0,
        PUNCT_END: -3.0,
        PUNCT_MID: -3.0,
        APOSTROPHE: -3.0,
        DASH: -2.0,
        OTHER: -4.0,
    },
)

# Inside a speaker label (upper run after \n\n): continue upper or
# space (multi-word names like "KING HENRY") or close with ":".
_set(
    CTX_IN_SPEAKER_LABEL,
    {
        UPPER: 4.0,
        SPACE: 1.5,
        PUNCT_MID: 2.5,  # the ":" that closes the label
        LOWER_CONS: -2.0,
        LOWER_VOWEL: -2.0,
        NEWLINE: -4.0,
        PUNCT_END: -5.0,
        APOSTROPHE: -4.0,
        DASH: -4.0,
        OTHER: -5.0,
    },
)

# Right after the ":" that closed a speaker label — expect newline.
_set(
    CTX_AFTER_COLON_LABEL,
    {
        NEWLINE: 5.0,
        SPACE: -1.0,
        UPPER: -4.0,
        LOWER_CONS: -4.0,
        LOWER_VOWEL: -4.0,
        PUNCT_END: -5.0,
        PUNCT_MID: -5.0,
        APOSTROPHE: -4.0,
        DASH: -4.0,
        OTHER: -5.0,
    },
)

# Inside word, short (pos 1–3): more letters, slight apostrophe, a bit of
# space.
_set(
    CTX_IN_WORD_SHORT,
    {
        LOWER_CONS: 2.0,
        LOWER_VOWEL: 2.0,
        UPPER: -3.0,
        SPACE: 0.3,
        NEWLINE: -2.0,
        PUNCT_END: -2.5,
        PUNCT_MID: -2.5,
        APOSTROPHE: 0.2,
        DASH: -3.0,
        OTHER: -5.0,
    },
)

# Inside word, medium (pos 4–6): word endings rising.
_set(
    CTX_IN_WORD_MID,
    {
        LOWER_CONS: 1.5,
        LOWER_VOWEL: 1.3,
        UPPER: -3.5,
        SPACE: 1.4,
        NEWLINE: -1.0,
        PUNCT_END: -1.5,
        PUNCT_MID: -1.5,
        APOSTROPHE: 0.2,
        DASH: -3.0,
        OTHER: -5.0,
    },
)

# Inside word, long (pos 7+): most words end by here; favor space/punct.
_set(
    CTX_IN_WORD_LONG,
    {
        LOWER_CONS: 0.5,
        LOWER_VOWEL: 0.2,
        UPPER: -4.0,
        SPACE: 2.6,
        NEWLINE: 0.2,
        PUNCT_END: -0.5,
        PUNCT_MID: 0.0,
        APOSTROPHE: 0.0,
        DASH: -2.0,
        OTHER: -5.0,
    },
)

# After a starting uppercase letter (not in speaker label, e.g. first
# letter of a sentence): lowercase letter strongly expected.
_set(
    CTX_AFTER_UPPER_START,
    {
        LOWER_CONS: 2.2,
        LOWER_VOWEL: 2.2,
        UPPER: -2.0,
        SPACE: -0.5,
        NEWLINE: -3.0,
        PUNCT_END: -3.0,
        PUNCT_MID: -3.0,
        APOSTROPHE: -0.5,
        DASH: -3.0,
        OTHER: -5.0,
    },
)

# Fallback.
_set(CTX_OTHER, {})


# ---------------------------------------------------------------------------


def _context_key(state: ModelState) -> int:
    if state.tokens_seen == 0:
        return CTX_START
    cls = state.last_char_class
    if cls == NEWLINE:
        if state.consecutive_newlines >= 2:
            return CTX_AFTER_DOUBLE_NL
        return CTX_AFTER_SINGLE_NL
    if cls == SPACE:
        # Is this the start of a new sentence (prev char was . ? !)?
        if state.prev_char_class == PUNCT_END:
            return CTX_AFTER_SPACE_SENT_START
        return CTX_AFTER_SPACE
    if cls == PUNCT_END:
        return CTX_AFTER_PUNCT_END
    if cls == PUNCT_MID:
        # Are we closing a speaker label ("NAME:")?
        if state.speaker_label_state == 3:
            return CTX_AFTER_COLON_LABEL
        return CTX_AFTER_PUNCT_MID
    if cls == APOSTROPHE:
        return CTX_AFTER_APOS
    if cls == DASH:
        return CTX_AFTER_DASH
    if cls == UPPER:
        # Speaker label (after \n\n, in upper run)?
        if state.speaker_label_state == 2:
            return CTX_IN_SPEAKER_LABEL
        # Otherwise a capital at sentence/word start.
        if state.upper_run_len == 1:
            return CTX_AFTER_UPPER_START
        # Multi-upper outside speaker label — rare; treat as label-ish.
        return CTX_IN_SPEAKER_LABEL
    # Lowercase letter — inside a word.
    pos = state.letter_run_len
    if pos <= 3:
        return CTX_IN_WORD_SHORT
    if pos <= 6:
        return CTX_IN_WORD_MID
    return CTX_IN_WORD_LONG


# Precompute contextual logprob vectors for each context: unigram +
# class-bias, then log-softmax.
def _precompute_ctx_logprobs() -> list[list[float]]:
    out: list[list[float]] = []
    for ctx in range(N_CTX):
        biases = _BIAS[ctx]
        raw = [
            _UNIGRAM_LOGPROBS[i] + biases[CLASS_OF_TOKEN[i]]
            for i in range(VOCAB_SIZE)
        ]
        m = max(raw)
        exps = [math.exp(x - m) for x in raw]
        z = sum(exps)
        logz = m + math.log(z)
        out.append([x - logz for x in raw])
    return out


_CTX_LOGPROBS: list[list[float]] = _precompute_ctx_logprobs()


def predict(state: ModelState) -> list[float]:
    ctx = _context_key(state)
    return list(_CTX_LOGPROBS[ctx])
