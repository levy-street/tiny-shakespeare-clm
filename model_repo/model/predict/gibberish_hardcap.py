"""Predict layer — hard cap on off-trie word length.

Real Shakespeare rarely produces words longer than 12-13 characters.
Off-trie long words (15+) are almost always gibberish extensions of
real prefixes, fed by letter-ngram momentum. The existing
`offtrie_depart_bias` caps its drift-scaling at letter_run_len ~= 8,
so there's no pressure escalation past that point.

This layer imposes an EXPONENTIALLY growing space/punctuation bias
once `letter_run_len` crosses a threshold and the word is off-trie.
Unlike the additive biases elsewhere, this one scales sharply with
each additional letter to act as a structural hard cap:

  letter_run_len = 10: modest nudge
  letter_run_len = 12: strong nudge
  letter_run_len = 14: dominant push
  letter_run_len = 16+: overwhelming forced-close

Gates:
  * Must be off the word-trie (`on_word_trie == False`)
  * Must have drifted (`letters_off_trie >= 2`) — don't punish
    legitimate long on-trie derivations
  * Outside speaker-label territory

No corpus statistics — purely a structural length constraint.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Growth schedule — log-bias for " " (and scaled for , . ; \n).
# Indexed by letter_run_len.
# Below 10, return None (no bias).
def _schedule(letter_run_len: int) -> float:
    if letter_run_len < 13:
        return 0.0
    if letter_run_len == 13:
        return 0.40
    if letter_run_len == 14:
        return 1.00
    if letter_run_len == 15:
        return 2.00
    if letter_run_len == 16:
        return 3.50
    # 17+
    return 5.00


def gibberish_hardcap_bias(
    letter_run_len: int,
    on_word_trie: bool,
    letters_off_trie: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if on_word_trie:
        return None
    if letters_off_trie < 4:
        return None
    if letter_run_len < 13:
        return None

    space_boost = _schedule(letter_run_len)
    if space_boost <= 0.0:
        return None

    vec = [0.0] * VOCAB_SIZE
    # Primary terminator — strongest.
    if " " in VOCAB_INDEX:
        vec[VOCAB_INDEX[" "]] += space_boost
    # Sentence/clause terminators — slightly smaller.
    if "," in VOCAB_INDEX:
        vec[VOCAB_INDEX[","]] += space_boost * 0.50
    if "." in VOCAB_INDEX:
        vec[VOCAB_INDEX["."]] += space_boost * 0.40
    if ";" in VOCAB_INDEX:
        vec[VOCAB_INDEX[";"]] += space_boost * 0.30
    if "!" in VOCAB_INDEX:
        vec[VOCAB_INDEX["!"]] += space_boost * 0.30
    if "?" in VOCAB_INDEX:
        vec[VOCAB_INDEX["?"]] += space_boost * 0.25
    if "\n" in VOCAB_INDEX:
        vec[VOCAB_INDEX["\n"]] += space_boost * 0.35

    # Common word-ending letters — if we MUST extend (another letter),
    # prefer an English-word-ending letter so next step can close.
    end_boost = space_boost * 0.18
    for ch in ("e", "s", "d", "t", "n", "h", "r", "y"):
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += end_boost

    # Suppress rare letters that would extend gibberish.
    rare_pen = -space_boost * 0.25
    for ch in ("x", "z", "j", "q", "k", "v"):
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += rare_pen

    return vec
