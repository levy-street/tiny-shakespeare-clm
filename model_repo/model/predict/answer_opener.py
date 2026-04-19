"""Cross-turn answer-opener bias.

Reads `state.prev_turn_final_sent_type` (captured at the turn
boundary by pipeline/sentence.py) and biases the first letter of
the first word of the NEW turn toward openers that respond to the
prior speaker's sentence type:

  prev_turn_final_sent_type == SENT_INTERROG:
    The prior speaker asked a question. Current speaker is very
    likely to answer with Yes/No/Nay/Ay/Aye/Marry/Indeed/Truly/I/
    Not/Never/Well/Why/O. Boost those first letters.

  prev_turn_final_sent_type == SENT_EXCLAM:
    The prior speaker exclaimed. Current speaker often responds
    with Peace/Hold/Stay/Soft/Nay/O/Alas/Why/Fie. Boost those.

  prev_turn_final_sent_type == SENT_IMPER:
    The prior speaker gave a command. Current speaker often
    acknowledges with I/Yes/Ay/My/No/Nay/Well or pushes back
    with Why/Nay/What.

  prev_turn_final_sent_type in {SENT_DECL, SENT_UNKNOWN}:
    No particular answer-opener preference — return None.

Gates:
  - words_in_turn == 0 and sentences_in_turn == 0 (very first word
    of new turn)
  - speaker_label_state == 0 (outside speaker label FSM)
  - letter_run_len == 0 (word-start position)
  - word_buffer empty

These conditions make the bias fire exactly once per new turn,
at the first letter of the first word.

All biases from prior knowledge of Shakespeare's dialogue patterns.
No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


SENT_INTERROG = 2
SENT_EXCLAM = 3
SENT_IMPER = 4


# After a question: answer-style openers.
_AFTER_QUESTION: dict[str, float] = {
    # Yes/No/Nay/Ay family
    "Y": 0.55,  # Yes, Ye, Yea
    "N": 0.50,  # No, Nay, Not, Never
    "A": 0.48,  # Ay, Aye, Ah, Alas
    "I": 0.38,  # I, Indeed
    "M": 0.25,  # Marry, My, Madam, Methinks
    "T": 0.18,  # Truly, Troth, That
    "W": 0.18,  # Well, Why, What
    "O": 0.15,  # O, Oh
}


# After an exclamation: calming / surprise / counter-exclamation.
_AFTER_EXCLAM: dict[str, float] = {
    "P": 0.35,   # Peace
    "H": 0.32,   # Hold, Ha
    "S": 0.30,   # Soft, Stay, Sir
    "N": 0.28,   # Nay, No
    "O": 0.28,   # O
    "A": 0.25,   # Alas, Ah, Away
    "W": 0.22,   # Why, What, Well
    "F": 0.20,   # Fie, Faith
    "T": 0.18,   # Tush, Tut, True
    "Y": 0.15,   # Yea, Yes
}


# After an imperative: acknowledgement or pushback.
_AFTER_IMPER: dict[str, float] = {
    "I": 0.32,   # I (will/shall/am), Indeed
    "A": 0.28,   # Ay, Aye
    "Y": 0.28,   # Yes
    "M": 0.25,   # My (lord), Madam
    "N": 0.22,   # Nay, No, Not
    "W": 0.22,   # Well, Why, What
    "S": 0.18,   # Sir, So
    "T": 0.15,   # That, Thou
    "H": 0.12,   # How, He
}


def answer_opener_start_bias(
    prev_turn_final_sent_type: int,
    words_in_turn: int,
    sentences_in_turn: int,
    speaker_label_state: int,
    letter_run_len: int,
    word_buffer: str,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0 or word_buffer:
        return None
    # Must be the very first word of the turn.
    if words_in_turn != 0 or sentences_in_turn != 0:
        return None

    if prev_turn_final_sent_type == SENT_INTERROG:
        table = _AFTER_QUESTION
    elif prev_turn_final_sent_type == SENT_EXCLAM:
        table = _AFTER_EXCLAM
    elif prev_turn_final_sent_type == SENT_IMPER:
        table = _AFTER_IMPER
    else:
        return None

    vec = [0.0] * VOCAB_SIZE
    for ch, b in table.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += b
    return vec
