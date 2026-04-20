"""Post-punctuation whitespace lock.

In Shakespeare (and all English), the character immediately following
any clause-break punctuation (`,`, `.`, `;`, `:`, `!`, `?`) is a space
or newline. Never a letter. Never another punctuation except the
same-char run (`...`, `!!`, `!?`).

Samples regularly produce artifacts like "Enter,eact" where a letter
follows a comma directly — a surface-level violation no real corpus
contains. The existing context biases penalize letters after punct
(-4.5 in the CTX_AFTER_PUNCT_MID class), but those are soft biases
that can be overridden by strong downstream votes.

This layer returns a forbid_mask list over the vocab: tokens which
should get the hard-forbid floor (1e-6) rather than the normal floor
(1.2e-4). All letters (a-z, A-Z) and most symbols are masked; only
space, newline, and the punct-continuation chars are allowed.

Applies only in the main dialogue region (speaker_label_state == 0).
Inside speaker labels the layer returns None — label colons are
followed by newlines anyway, and the label FSM handles its own
transitions.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_PUNCT_TRIGGERS = frozenset(",.;:!?")
# Chars that are LEGAL immediately after a clause-break punctuation:
# space, newline, and the punctuation marks that can chain together
# (ellipses, double punct, quoted aside). Apostrophe legal in rare
# cases like ", 'tis" (but a preceding space is more standard — we
# still allow it for safety). Dash legal for inserted asides.
_LEGAL_AFTER_PUNCT = frozenset(" \n'-,.;:!?")


def post_punct_forbid_letters(
    last_char: str,
    speaker_label_state: int,
) -> list[bool] | None:
    """Return a per-token forbid mask or None if not applicable."""
    if speaker_label_state != 0:
        return None
    if not last_char or last_char not in _PUNCT_TRIGGERS:
        return None
    mask = [False] * VOCAB_SIZE
    for ch, idx in VOCAB_INDEX.items():
        if ch not in _LEGAL_AFTER_PUNCT:
            mask[idx] = True
    return mask
