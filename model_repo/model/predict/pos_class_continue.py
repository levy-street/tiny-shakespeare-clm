"""POS-class filtered within-word continuation bias.

Structural move targeting the failure mode
    "... of the let kiss ..."
where — after a determiner — the sampler picks a word whose first
letters alone are plausibly NP-y but whose completion lands on a verb
(e.g. "l" -> "e" -> "t" = "let"). The existing `phrase_slot_bias`
layer operates only at the first character of a new word; it cannot
steer away from a verb completion of an already-started word.

This layer runs at every *within-word* step (`letter_run_len >= 1`)
and at word-terminators. It fires only when the phrase-slot FSM is in
POST_DET (1) or POST_ADJ (2) — contexts where the next content word
MUST be a NOUN or ADJECTIVE. Behavior:

1. **Next-letter continuation bias.** For each candidate letter, look
   up what POS classes of words start with `word_buffer + ch`. If the
   continuation leads ONLY to VERB / ADVERB / FUNCTION-WORD completions
   and not to any NOUN / ADJECTIVE completion, apply a penalty. If it
   leads to NOUN / ADJ completions (and NOT only to disallowed classes)
   apply a mild reward.

2. **Wrong-class terminator block.** If `word_buffer` is itself a
   *complete* word in the VERB / ADVERB / FUNCTION-WORD classes but
   NOT a complete word in NOUN / ADJ classes, penalize word-terminators
   (space, newline, comma, period, etc.). This forces the sampler to
   extend the word rather than commit to a wrong-POS complete form —
   e.g. at "the let", block the terminator so the word grows into
   "letter" or "letters".

Both rules use only prior-knowledge POS lists (`_NOUNS`,
`_ADJECTIVES`, `_VERBS`, `_ADVERBS`, `_PRONOUNS`, `_MODALS`,
`_AUX_VERBS`, `_ARTICLES`, `_POSSESSIVES`, `_PREPOSITIONS`,
`_CONJUNCTIONS`, `_NEGATIONS`, `_WH`) from `pipeline/pos.py`. No
corpus statistics.
"""

from __future__ import annotations

from ..pipeline.pos import (
    _ADJECTIVES,
    _ADVERBS,
    _ARTICLES,
    _AUX_VERBS,
    _CONJUNCTIONS,
    _MODALS,
    _NEGATIONS,
    _NOUNS,
    _POSSESSIVES,
    _PREPOSITIONS,
    _PRONOUNS,
    _VERBS,
    _WH,
)
from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# Words we want to appear after a determiner / inside an NP.
_NOUN_ADJ_WORDS: frozenset[str] = frozenset(_NOUNS | _ADJECTIVES)

# Words that are DISALLOWED as the head of an NP after a determiner.
# Verbs, adverbs, and most function-word classes can't legitimately
# occupy a POST_DET / POST_ADJ slot (barring very rare idiomatic uses
# which we sacrifice for overall grammar).
_DISALLOWED_WORDS: frozenset[str] = frozenset(
    _VERBS | _ADVERBS | _AUX_VERBS | _MODALS | _PRONOUNS | _POSSESSIVES
    | _ARTICLES | _PREPOSITIONS | _CONJUNCTIONS | _NEGATIONS | _WH
)
# Remove anything that is ALSO a noun/adj (shared forms like "love",
# "hate") from the disallowed set — these are ambiguous and should not
# be suppressed.
_DISALLOWED_WORDS = frozenset(_DISALLOWED_WORDS - _NOUN_ADJ_WORDS)


def _build_prefix_next_chars(words: frozenset[str]) -> dict[str, frozenset[str]]:
    """For each non-empty proper prefix of each word in `words`, record
    the set of next letters that continue into some word in the set."""
    out: dict[str, set[str]] = {}
    for w in words:
        for i in range(1, len(w)):
            prefix = w[:i]
            nxt = w[i]
            out.setdefault(prefix, set()).add(nxt)
    return {k: frozenset(v) for k, v in out.items()}


_NOUN_ADJ_PREFIX_NEXTS: dict[str, frozenset[str]] = _build_prefix_next_chars(_NOUN_ADJ_WORDS)
_DISALLOWED_PREFIX_NEXTS: dict[str, frozenset[str]] = _build_prefix_next_chars(_DISALLOWED_WORDS)


# Terminator chars — word-ending characters where the sampler commits
# to the word as-is. We block these when the committed word would be
# wrong-POS.
_TERMINATOR_CHARS: tuple[str, ...] = (" ", "\n", ",", ".", ";", ":", "!", "?")


def pos_class_continue_bias(
    word_buffer: str,
    letter_run_len: int,
    phrase_slot: int,
    speaker_label_state: int,
) -> list[float] | None:
    # Gated on being inside a word (letter_run_len >= 1), in a
    # phrase-slot that requires a noun/adj head, and outside a
    # speaker-label.
    if speaker_label_state != 0:
        return None
    if letter_run_len < 1:
        return None
    if phrase_slot not in (1, 2):
        return None
    if not word_buffer:
        return None
    # The word_buffer as-known for the lookups (lowercase).
    buf_low = word_buffer.lower()

    na_nexts = _NOUN_ADJ_PREFIX_NEXTS.get(buf_low)
    dis_nexts = _DISALLOWED_PREFIX_NEXTS.get(buf_low)
    is_complete_na = buf_low in _NOUN_ADJ_WORDS
    is_complete_dis = buf_low in _DISALLOWED_WORDS

    # If the prefix participates in NOTHING (neither labeled class),
    # skip — we have no information.
    if (
        na_nexts is None
        and dis_nexts is None
        and not is_complete_na
        and not is_complete_dis
    ):
        return None

    # Escalate with phrase_slot pressure.
    #   POST_DET (1): moderate
    #   POST_ADJ (2): slightly stronger (we're further into the NP
    #                  with no head yet)
    if phrase_slot == 1:
        pen_wrong_cont = -0.85
        rew_right_cont = 0.28
        pen_wrong_terminator = -1.40
    else:  # phrase_slot == 2
        pen_wrong_cont = -1.05
        rew_right_cont = 0.36
        pen_wrong_terminator = -1.70

    vec = [0.0] * VOCAB_SIZE

    # --- Next-letter continuation bias -------------------------------
    # For each lowercase letter, decide whether it continues into a
    # noun/adj, a disallowed class, both, or neither.
    if na_nexts is not None or dis_nexts is not None:
        na_set = na_nexts or frozenset()
        dis_set = dis_nexts or frozenset()
        for ch in "abcdefghijklmnopqrstuvwxyz":
            if ch not in VOCAB_INDEX:
                continue
            in_na = ch in na_set
            in_dis = ch in dis_set
            if in_dis and not in_na:
                vec[VOCAB_INDEX[ch]] += pen_wrong_cont
                # Uppercase variant (mid-word should be rare anyway, but
                # apply the same pressure so it can't slip through).
                up = ch.upper()
                if up in VOCAB_INDEX:
                    vec[VOCAB_INDEX[up]] += pen_wrong_cont * 0.5
            elif in_na and not in_dis:
                vec[VOCAB_INDEX[ch]] += rew_right_cont

    # --- Wrong-class terminator block --------------------------------
    # If the current word_buffer is itself a complete DISALLOWED word
    # (verb / adverb / function-word) but NOT a complete NOUN/ADJ,
    # penalize terminators so the word keeps extending toward a valid
    # noun form (e.g. "let" -> "letter", "bad" is noun/adj-ish so N/A).
    # We also apply this block when the buffer is a complete disallowed
    # AND there IS some noun-adj extension reachable (i.e. extending
    # won't paint us into a corner).
    if is_complete_dis and not is_complete_na:
        extensible_to_na = na_nexts is not None and len(na_nexts) > 0
        # Scale: if we CAN extend to a noun/adj, apply full penalty.
        # If we cannot, apply a softer penalty (still prefer to try
        # rather than commit to a wrong-class form, but don't over-
        # push into uncharted territory).
        scale = 1.0 if extensible_to_na else 0.50
        for ch in _TERMINATOR_CHARS:
            if ch in VOCAB_INDEX:
                vec[VOCAB_INDEX[ch]] += pen_wrong_terminator * scale

    return vec
