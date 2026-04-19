"""Predict layer: proper-noun scene rolodex.

Consumes `state.proper_nouns_seen` — up to 10 recently-observed
capitalized content words (lowercased form) — and biases the
next-token distribution at two positions:

  Word-start (letter_run_len == 0, after space or single newline):
    For each word in the rolodex, boost its initial character (both
    upper and lower variants, with a stronger boost on the upper
    since proper nouns are capitalized). More recent entries get
    larger boosts. Fires only when proper_noun_slot allows (STRONG,
    QUOTE, or MILD) OR when sentence_start_pending is True (the
    sentence-initial word is often a proper-noun subject: "Rome shall
    see...", "Coriolanus comes."). Decays with distance from head.

  Mid-word (letter_run_len >= 1 and current_word_started_cap == True):
    If the buffer so far is a prefix of one of the rolodex words,
    boost the next letter of that rolodex word. This is a targeted
    continuation bias: we just emitted "Ro" and "Rome" is in the
    rolodex, so strongly boost "m".

No corpus statistics — the rolodex is populated only from the state
stream during advance(); predict just reads it.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Log-bias scale for word-start first-letter push, keyed on rolodex
# position (0 = most recent).
_START_BOOST_BY_POS: tuple[float, ...] = (
    0.95, 0.80, 0.65, 0.55, 0.45, 0.35, 0.28, 0.22, 0.18, 0.15,
)

# Log-bias scale for mid-word continuation push.
_MID_BOOST_BY_POS: tuple[float, ...] = (
    2.20, 1.80, 1.45, 1.20, 1.00, 0.85, 0.70, 0.60, 0.50, 0.40,
)


def proper_noun_memory_start_bias(
    proper_nouns_seen: tuple[str, ...],
    speaker_label_state: int,
    proper_noun_slot: int,
    sentence_start_pending: bool,
    letter_run_len: int,
    word_buffer: str,
) -> list[float] | None:
    """At word-start, bias first letters toward recently-seen proper nouns."""
    if speaker_label_state != 0:
        return None
    if not proper_nouns_seen:
        return None
    if letter_run_len != 0 or word_buffer != "":
        return None
    # Gate: we want the proper noun signal active EITHER when the
    # proper-noun-slot machinery is already favoring a cap, OR at
    # sentence start, OR at the start of a line (which often opens
    # with a subject-proper-noun in Shakespeare).
    # proper_noun_slot enums: 0 NONE, 1 MILD, 2 STRONG, 3 QUOTE.
    if proper_noun_slot == 0 and not sentence_start_pending:
        # Mid-sentence, no PN expectation — a fresh proper-noun start
        # is uncommon here. Use a SMALL fallback boost, because names
        # do still recur ("he loved Rome"); strong boosts risk phantom
        # capitals.
        scale_override = 0.30
    elif proper_noun_slot == 2:  # STRONG
        scale_override = 1.00
    elif proper_noun_slot == 3:  # QUOTE
        scale_override = 0.90
    elif proper_noun_slot == 1:  # MILD
        scale_override = 0.70
    else:
        # sentence_start_pending
        scale_override = 0.65

    vec = [0.0] * VOCAB_SIZE
    for i, word in enumerate(proper_nouns_seen[:len(_START_BOOST_BY_POS)]):
        if not word:
            continue
        first = word[0]
        base = _START_BOOST_BY_POS[i] * scale_override
        # Boost lowercase variant lightly (a proper noun occasionally
        # appears lowercased or inside a contraction), uppercase more
        # strongly (the dominant case for a proper noun).
        if first in VOCAB_INDEX:
            vec[VOCAB_INDEX[first]] += base * 0.35
        up = first.upper()
        if up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] += base
    return vec


def proper_noun_memory_mid_bias(
    proper_nouns_seen: tuple[str, ...],
    speaker_label_state: int,
    current_word_started_cap: bool,
    word_buffer: str,
    letter_run_len: int,
) -> list[float] | None:
    """Mid-word: if the capitalized buffer is a prefix of some rolodex
    word, boost that word's continuation."""
    if speaker_label_state != 0:
        return None
    if not proper_nouns_seen:
        return None
    if not current_word_started_cap:
        return None
    if letter_run_len < 1:
        return None
    if not word_buffer:
        return None
    buf_lower = word_buffer.lower()
    vec: list[float] | None = None
    # Accumulate boosts — multiple rolodex words can share a prefix
    # (e.g., "Coriolanus" and "Cordelia" both start "Cor").
    for i, word in enumerate(proper_nouns_seen[:len(_MID_BOOST_BY_POS)]):
        if not word:
            continue
        if len(word) <= len(buf_lower):
            continue
        if not word.startswith(buf_lower):
            continue
        # Next letter to predict:
        nxt = word[len(buf_lower)]
        if nxt not in VOCAB_INDEX:
            continue
        if vec is None:
            vec = [0.0] * VOCAB_SIZE
        boost = _MID_BOOST_BY_POS[i]
        # Scale down if buffer is very short (prefix is too ambiguous).
        if len(buf_lower) == 1:
            boost *= 0.55
        elif len(buf_lower) == 2:
            boost *= 0.85
        vec[VOCAB_INDEX[nxt]] += boost
        # Also boost the uppercase variant very slightly (should
        # almost never be right mid-word, but not impossible for
        # two-word capitalized names like "Northumberland").
        up = nxt.upper()
        if up in VOCAB_INDEX and up != nxt:
            vec[VOCAB_INDEX[up]] += boost * 0.05
    return vec
