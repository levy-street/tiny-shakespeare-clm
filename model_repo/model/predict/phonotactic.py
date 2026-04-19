"""Predict layer — phonotactic illegal-bigram close-out.

Reads `state.bad_bigram_count` (count of phonotactically illegal
letter-pair bigrams encountered inside the current word). Even ONE
illegal bigram inside a real English word is extraordinarily rare;
two is essentially diagnostic of gibberish.

This complements the existing `gibberish_hardcap_bias` which only
fires at letter_run_len >= 13. Most observed gibberish (etvsudqted,
fnvamonsese, iolmead, Iaehohde, claitagmrt) is length 7-12 and
contains illegal bigrams ("tv", "dq", "nvm", "jvs", "tgbl"). A
bigram-triggered close-out catches these at the exact moment the
phonotactic failure is recognized, instead of waiting for the
hardcap.

Mechanism: when bad_bigram_count >= 1 and letter_run_len >= 3, push
a strong bias toward word-terminators (space, punctuation, newline)
and away from additional letters. The bias scales with the count.

Gates:
  * speaker_label_state == 0 (names have looser phonotactics)
  * letter_run_len >= 3 (don't fire on 2-letter words that happen
    to contain one exotic pair — e.g., "vs" if it were a word)
  * bad_bigram_count >= 1
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


def phonotactic_close_bias(
    bad_bigram_count: int,
    letter_run_len: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if bad_bigram_count < 1:
        return None
    if letter_run_len < 5:
        return None

    # Escalate with count. One bad bigram → moderate; two → hard;
    # three+ → overwhelming.
    if bad_bigram_count == 1:
        sc = 1.2
    elif bad_bigram_count == 2:
        sc = 2.8
    else:
        sc = 4.5

    vec = [0.0] * VOCAB_SIZE
    # Primary terminator — strongest.
    if " " in VOCAB_INDEX:
        vec[VOCAB_INDEX[" "]] += sc
    if "," in VOCAB_INDEX:
        vec[VOCAB_INDEX[","]] += sc * 0.55
    if "." in VOCAB_INDEX:
        vec[VOCAB_INDEX["."]] += sc * 0.45
    if ";" in VOCAB_INDEX:
        vec[VOCAB_INDEX[";"]] += sc * 0.30
    if ":" in VOCAB_INDEX:
        vec[VOCAB_INDEX[":"]] += sc * 0.20
    if "!" in VOCAB_INDEX:
        vec[VOCAB_INDEX["!"]] += sc * 0.30
    if "?" in VOCAB_INDEX:
        vec[VOCAB_INDEX["?"]] += sc * 0.25
    if "\n" in VOCAB_INDEX:
        vec[VOCAB_INDEX["\n"]] += sc * 0.35

    # Prefer word-ending letters if we MUST extend.
    end_boost = sc * 0.15
    for ch in ("e", "s", "d", "t", "n", "h", "r", "y"):
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += end_boost

    # Suppress further rare / extension letters that would pile
    # more gibberish.
    rare_pen = -sc * 0.35
    for ch in ("x", "z", "j", "q", "k", "v", "w"):
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += rare_pen

    # Gentle penalty on ALL letters to tilt mass toward terminators.
    light_pen = -sc * 0.08
    for ch in "abcdefghijklmnopqrstuvwxyz":
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += light_pen

    return vec
