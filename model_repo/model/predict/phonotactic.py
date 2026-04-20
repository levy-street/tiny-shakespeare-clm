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
    bad_trigram_count: int,
    letter_run_len: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    # Combine the two violation counts. Trigrams fire often (my
    # legal-CCC list is hand-compiled and inevitably misses some real
    # English clusters like "ngt" in "strength"), so a SINGLE trigram
    # violation is weak evidence of gibberish — only count trigrams
    # beyond the first. Bigram violations are stricter and count
    # from 1.
    effective_trigrams = max(0, bad_trigram_count - 1)
    effective = bad_bigram_count + effective_trigrams
    if effective < 1:
        return None
    # Fire from letter_run_len == 3. A single illegal bigram at
    # positions 2-3 of a fresh word (e.g., "tvs", "dqr") is already
    # strong evidence of gibberish and we want to close BEFORE more
    # nonsense letters accumulate.
    if letter_run_len < 3:
        return None

    # Escalate with count. One violation → moderate; two → hard;
    # three+ → overwhelming.
    if effective == 1:
        sc = 2.2
    elif effective == 2:
        sc = 3.6
    else:
        sc = 5.2

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
