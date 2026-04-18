"""Subject–verb agreement consumer.

Reads `state.verb_agreement` (see pipeline/verb_agreement.py) and
biases the letter stream when the current word is plausibly the
main verb of a clause whose subject has already been identified.

Firing conditions:
  - outside speaker-label territory (speaker_label_state == 0)
  - clause_slot is HAS_SUBJ (slot == 1) — the verb role is still
    unfilled
  - we are in the middle or end of a word (letter_run_len >= 2)
  - verb_agreement is VA_THOU or VA_THIRD_SG (the two classes with
    distinctive morphological suffixes: -st, -eth)

When VA_THOU (thou-register):
  The upcoming verb typically ends in -st or -est:
    "thou art"         — 3-char word ending in "t"
    "thou hast"        — ends in "st"
    "thou didst"       — ends in "dst"
    "thou knowest"     — ends in "est"
    "thou lovest"      — ends in "est"
    "thou speakest"    — ends in "est"
  Bias:
    - If letter_run_len >= 3 and buffer ends in a vowel: boost "s"
      (building toward "-est") and penalize space.
    - If letter_run_len >= 4 and buffer ends in "s": boost "t"
      (completing "-st") and penalize space.
    - Else (interior of the verb): no bias.

When VA_THIRD_SG (he/she/it/noun-singular):
  The upcoming verb often ends in -s or -eth:
    "he is"            — 2-char
    "he hath"          — ends in "th" (archaic 3sg)
    "she doth"         — ends in "th"
    "the king speaks"  — ends in "s"
    "it comes"         — ends in "s"
  Bias (gentler than VA_THOU because this class is broader):
    - At letter_run_len >= 3 and buffer ends in a vowel: small
      boost to "s" (terminating -s form) and "t" (opening "-th").
    - At letter_run_len >= 4 and buffer ends in "t" with buffer
      looking like "-a"/"-o"/"-e": small boost to "h" (building
      "-ath"/"-oth"/"-eth") and to space (it may already end).

The buffer-letter conditions are checked against the actual
`word_buffer` end to avoid firing in the wrong morphological
context.

No corpus statistics — the -st / -eth / -s endings and their
co-occurrence with subject pronouns come from prior knowledge of
Early Modern English morphology.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

VA_NONE = 0
VA_THOU = 1
VA_THIRD_SG = 2
VA_FIRST_SG = 3
VA_PLURAL = 4
VA_IMPERATIVE = 5

_VOWELS = frozenset("aeiouy")


_THOU_VERB_STARTERS: dict[str, float] = {
    "a": 0.18,  # art
    "h": 0.22,  # hast, hadst, hearst
    "d": 0.18,  # didst, doest, dost, durst
    "c": 0.12,  # canst, couldst
    "w": 0.18,  # wast, wert, wilt, wouldst
    "s": 0.15,  # shalt, shouldst, seest, speakest, sayest
    "m": 0.12,  # mayest, must
    "k": 0.10,  # knowest
    "l": 0.08,  # lovest, liest
    "t": 0.08,  # thinkest
    "g": 0.06,  # givest
    "b": 0.05,  # boughtest, bidst
}


def verb_agreement_start_bias(
    verb_agreement: int,
    clause_slot: int,
    speaker_label_state: int,
    letter_run_len: int,
    last_cls: int,
) -> list[float] | None:
    """Return a first-letter bias when the upcoming word is likely
    a verb matching verb_agreement. Fires at word-start only."""
    if speaker_label_state != 0:
        return None
    if clause_slot != 1:  # need HAS_SUBJ
        return None
    if letter_run_len != 0:
        return None
    if verb_agreement != VA_THOU:
        return None
    vec = [0.0] * VOCAB_SIZE
    for ch, w in _THOU_VERB_STARTERS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += w
    return vec


def verb_agreement_bias(
    verb_agreement: int,
    clause_slot: int,
    speaker_label_state: int,
    word_buffer: str,
    letter_run_len: int,
) -> list[float] | None:
    """Return a VOCAB-sized bias when firing conditions are met."""
    if speaker_label_state != 0:
        return None
    if clause_slot != 1:  # HAS_SUBJ
        return None
    if letter_run_len < 2:
        return None
    if verb_agreement not in (VA_THOU, VA_THIRD_SG):
        return None
    if not word_buffer:
        return None

    last = word_buffer[-1].lower()
    vec = [0.0] * VOCAB_SIZE

    if verb_agreement == VA_THOU:
        if letter_run_len >= 4 and last == "s":
            # buffer ends in "s"; push "t" to form -st.
            if "t" in VOCAB_INDEX:
                vec[VOCAB_INDEX["t"]] += 2.4
            if " " in VOCAB_INDEX:
                vec[VOCAB_INDEX[" "]] -= 0.8
            return vec
        if letter_run_len >= 4 and last == "d":
            # buffer ends in "d"; push "s" then "t" to form -dst
            # (didst, couldst, wouldst, shouldst).
            if "s" in VOCAB_INDEX:
                vec[VOCAB_INDEX["s"]] += 1.3
            if " " in VOCAB_INDEX:
                vec[VOCAB_INDEX[" "]] -= 0.4
            return vec
        if letter_run_len >= 3 and last in _VOWELS:
            # buffer ends in vowel; push "s" to form -est.
            if "s" in VOCAB_INDEX:
                vec[VOCAB_INDEX["s"]] += 1.2
            if " " in VOCAB_INDEX:
                vec[VOCAB_INDEX[" "]] -= 0.40
            return vec
        return None

    # VA_THIRD_SG
    if letter_run_len >= 4 and last == "t":
        # buffer ends in "t"; push "h" to form -th.
        # Only if the penultimate char is a vowel (avoid -rt/-nt/-st).
        if len(word_buffer) >= 2 and word_buffer[-2].lower() in _VOWELS:
            if "h" in VOCAB_INDEX:
                vec[VOCAB_INDEX["h"]] += 1.25
            return vec
        return None
    if letter_run_len >= 3 and last in _VOWELS:
        # small boost to "s" (terminating -s form) and "t" (for -th).
        if "s" in VOCAB_INDEX:
            vec[VOCAB_INDEX["s"]] += 0.45
        if "t" in VOCAB_INDEX:
            vec[VOCAB_INDEX["t"]] += 0.25
        return vec
    return None
