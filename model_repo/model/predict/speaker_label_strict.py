"""Speaker-label strict-legality bias.

Complements `speaker_trie_bias` by penalizing non-letter characters
(space, colon, apostrophe, newline, digits, most punctuation) and
opposite-case letters when those tokens aren't legal next-characters
at the current `speaker_buffer` prefix in the canonical speaker trie.

The existing speaker_trie_bias penalizes only same-case letters
outside the trie-legal next-set. At a prefix like "MOUNT" (only "J"
legal, for MOUNTJOY), " " (space) and ":" (colon) get 0 bias — the
model can freely emit them, producing malformed labels like
"MOUNT tssayl:" that slip through as FSM-valid.

This layer adds hard penalties on those tokens when they aren't in
the trie-legal next-set, closing the FSM slack.

State flags consumed (populated by pipeline/speaker_strict.py):
  speaker_trie_on_trie     — buffer is a prefix of some known name
  speaker_trie_space_valid — " " is a legal next char
  speaker_trie_colon_valid — ":" is a legal next char

Gates:
  * speaker_label_state in {1, 2} — we're inside a speaker label
  * buffer is non-empty (at buffer == "" all letters are equally
    legal, trie handles the entry)
  * Outside speaker label mode → return None

No corpus statistics — this is a grammar rule derived from the
speaker-label trie.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_LOWER = "abcdefghijklmnopqrstuvwxyz"
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Characters that are NEVER legal inside a speaker label and should
# always be hard-penalized when state is 1 or 2. (Colons and spaces
# are sometimes legal and handled via the trie flags.)
_ALWAYS_FORBIDDEN = (
    "'", "\"", ",", ".", ";", "!", "?",
    "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "(", ")", "[", "]",
)

# Newline is technically legal at the "\n\n" boundary before state
# enters 1, but inside state 1 or 2 mid-label a newline is broken
# (no Shakespeare label contains an embedded newline). Penalize.
_NEWLINE = "\n"


def speaker_label_strict_bias(
    speaker_label_state: int,
    speaker_buffer: str,
    speaker_label_saw_lower: bool,
    speaker_trie_on_trie: bool,
    speaker_trie_space_valid: bool,
    speaker_trie_colon_valid: bool,
) -> list[float] | None:
    if speaker_label_state not in (1, 2):
        return None
    if not speaker_buffer:
        # Entry point — let the trie & compose handle the first letter.
        return None

    vec = [0.0] * VOCAB_SIZE

    # --- Hard penalty on always-forbidden characters. ---
    # Inside a label, apostrophes, commas, quote marks, digits, etc.
    # are never legal — no canonical Shakespeare speaker label contains
    # any of them.
    forbid_strong = -4.0
    for ch in _ALWAYS_FORBIDDEN:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += forbid_strong

    # Newline mid-label is broken; penalize but slightly less harshly
    # (at least it can cleanly reset state downstream).
    nl_idx = VOCAB_INDEX.get(_NEWLINE)
    if nl_idx is not None:
        vec[nl_idx] += -3.0

    # --- Space / colon legality from trie flags. ---
    if speaker_trie_on_trie:
        if not speaker_trie_space_valid:
            sp_idx = VOCAB_INDEX.get(" ")
            if sp_idx is not None:
                vec[sp_idx] += -3.5
        if not speaker_trie_colon_valid:
            co_idx = VOCAB_INDEX.get(":")
            if co_idx is not None:
                vec[co_idx] += -3.5
    else:
        # Off-trie buffer: no known speaker label reaches here. Push
        # hard toward termination so we exit the bad-label state
        # quickly. But don't encourage mid-word space/colon — those
        # would produce malformed compound labels.
        sp_idx = VOCAB_INDEX.get(" ")
        if sp_idx is not None:
            vec[sp_idx] += -2.8
        co_idx = VOCAB_INDEX.get(":")
        if co_idx is not None:
            vec[co_idx] += -3.0

    # --- Opposite-case letter penalty. ---
    # Once we've committed to a case regime via the first letter,
    # the opposite case is rare. Shakespeare's labels are either
    # ALL-CAPS (HAMLET, KING HENRY) or Title-Case (First Citizen,
    # Lady Macbeth) — never mixed within a word after the first
    # letter. speaker_label_saw_lower distinguishes the two modes.
    if speaker_label_saw_lower:
        # We're in Title-Case mode — uppercase mid-label is rare
        # (only acceptable at the start of a NEW word after a space,
        # e.g. "First " → "C" for Citizen. We don't have a per-word
        # flag, so penalize uppercase mid-label gently. At a fresh
        # post-space position `speaker_trie_space_valid` would be
        # False (the space was just emitted) and uppercase may be
        # legitimate — so penalize lightly.
        # Check if the last char in speaker_buffer is a space — if so
        # uppercase is legal for the fresh word-start in Title Case.
        if speaker_buffer and speaker_buffer.endswith(" "):
            pen = 0.0  # fresh word-start in Title Case — uppercase OK
        else:
            pen = -1.2
        for ch in _UPPER:
            idx = VOCAB_INDEX.get(ch)
            if idx is not None:
                vec[idx] += pen
    else:
        # All-caps mode — lowercase mid-label is invalid (a Title-Case
        # label would have emitted lowercase at position 2).
        # At buffer length 1 we haven't committed yet, so skip.
        if len(speaker_buffer) >= 2:
            pen_low = -3.0
            for ch in _LOWER:
                idx = VOCAB_INDEX.get(ch)
                if idx is not None:
                    vec[idx] += pen_low

    return vec
