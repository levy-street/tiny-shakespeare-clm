"""Sentence-level anaphora bias — match the previous sentence's opener.

Consumes:
  - prev_sentence_first_word: first word (lowercased) of the sentence
    that just closed
  - sentence_anaphora_run: count of consecutive sentences starting with
    the same first word

When we're at a sentence-start position and prev_sentence_first_word
is a plausible anaphora trigger (short closed-class or declamatory
opener), nudge the first letter of the new sentence toward that word's
starting letter. When a run is already established (>= 1), the boost
grows — we lean into the rhetorical chain.

Short list of "chain-worthy" openers where anaphora is common in
Shakespeare. Random content words don't count (we don't want to bias
"Gentleman" → another "Gentleman"-opening sentence).
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Words that are canonically repeated as sentence openers in
# Shakespearean rhetoric. Hand-curated. All lowercased.
_CHAIN_OPENERS: frozenset[str] = frozenset({
    # Coordinators and sentence-leaders that genuinely repeat
    "and", "but", "yet", "or", "nor", "so",
    # Temporal / conditional chains
    "when", "where", "if", "though", "while", "since", "till", "until",
    # Declamatory / invocation chains (Sonnets, soliloquies)
    "o", "oh", "ah", "alas",
    "let", "here", "there", "now", "then",
    # First-person rhetorical chains
    "i", "my", "mine",
    # Second-person apostrophe chains
    "thou", "thy", "thine", "you", "your",
    # "To X. To Y. To Z." infinitive chains (Hamlet-style)
    "to",
    # "Why X? Why Y?" rhetorical question chains
    "why", "what", "how",
    # "No X. No Y." denial chains
    "no", "not", "never",
    # "All X. All Y." universal chains
    "all", "every", "each", "none",
    # "Come X. Come Y." / "Go X. Go Y."
    "come", "go",
    # "Is X? Is Y?" inversion-question chains
    "is", "art",
    # "Tis X. Tis Y." (archaic contraction)
    "tis", "twas",
})


def sentence_anaphora_start_bias(
    prev_sentence_first_word: str,
    sentence_anaphora_run: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a first-letter bias for a new sentence when anaphora
    is plausible. Returns None for speaker-label positions or when
    the prior opener isn't a chain-worthy word.

    The caller must gate on `is_sentence_start` (post-PUNCT_END space
    or verse-line-start newline after PUNCT_END).
    """
    if speaker_label_state != 0:
        return None
    prev = prev_sentence_first_word
    if not prev:
        return None
    if prev not in _CHAIN_OPENERS:
        return None

    # Base scale — gentle when no run established, stronger when the
    # chain is visibly active. The scale grows with run length.
    if sentence_anaphora_run == 0:
        scale = 0.22
    elif sentence_anaphora_run == 1:
        scale = 0.55
    elif sentence_anaphora_run == 2:
        scale = 0.85
    else:
        scale = 1.10

    vec = [0.0] * VOCAB_SIZE
    first_letter = prev[0]
    # Boost the capital form (sentence-start position) primarily and
    # the lowercase form lightly.
    up = first_letter.upper()
    lo = first_letter.lower()
    u_idx = VOCAB_INDEX.get(up)
    l_idx = VOCAB_INDEX.get(lo)
    if u_idx is not None:
        vec[u_idx] += scale
    if l_idx is not None:
        vec[l_idx] += scale * 0.25
    return vec
