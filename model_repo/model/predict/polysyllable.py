"""Predict consumer — polysyllable-density biasing at mid-word.

Reads `state.polysyllable_density` (see pipeline/polysyllable.py). Fires
only in a narrow mid-word zone (letter_run_len in [3, 6], on-word-trie,
outside speaker-label territory) where the word's length is still being
decided by the next sampled char (another letter extends; a space closes).

Signal:
  * High density (recent words polysyllabic, e.g. > 0.6) ⇒ register is
    elaborate / Latinate; the current word is likely also polysyllabic.
    Push slightly AGAINST space (keep extending).
  * Low density (recent words monosyllabic, e.g. < 0.4) ⇒ register is
    plain-speech; the current word is likely short. Push slightly TOWARD
    space (close it).

The amplitude is small — this is a rhythm modulator, not a structural
constraint. The existing CTX_IN_WORD_* biases, gibberish_hardcap, and
word_trie layers remain authoritative.

No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_AMP: float = 0.30  # max log-unit swing on space bias from density extremes


def polysyllable_midword_bias(
    polysyllable_density: float,
    letter_run_len: int,
    on_word_trie: bool,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a log-bias vector or None when the layer shouldn't fire."""
    if speaker_label_state != 0:
        return None
    if not on_word_trie:
        return None
    # Only fire at "deciding length" positions — too-short or too-long
    # buffers already have strong signals elsewhere.
    if not (3 <= letter_run_len <= 6):
        return None

    # Center around 0.5 so neutral density produces no bias.
    signed = (polysyllable_density - 0.5) * 2.0  # in [-1, +1]
    # Slightly stronger bias at shorter positions (the register signal
    # matters most when we haven't committed to a long word yet).
    # Linear decay from letter_run_len 3 (1.0) to 6 (0.4).
    position_gain = 1.0 - (letter_run_len - 3) * 0.20  # 1.0, 0.8, 0.6, 0.4
    amp = _AMP * position_gain

    # High density → discourage space (keep extending). Negative signed
    # → encourage space.
    space_bias = -signed * amp

    vec = [0.0] * VOCAB_SIZE
    if " " in VOCAB_INDEX:
        vec[VOCAB_INDEX[" "]] += space_bias
    return vec
