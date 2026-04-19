"""Answer-expectation bias.

Reads `state.pending_question_type` (set by pipeline/question_answer
on a `?` at end of the previous turn's question sentence, carried
across the turn boundary). At the first letter of the first word of
the responding turn, bias first letters toward class-specific answer
openers grounded in Shakespeare's Q-A idiom:

  ANS_YESNO  → "Ay", "Yes", "No", "Nay", "Indeed", "I", "Marry", "Troth"
               boost: A, Y, N, I, M, T

  ANS_WHAT   → "I", "That", "Nothing", "A", "The", "It", "Tis"
               boost: I, T, N, A

  ANS_WHERE  → "Here", "There", "In", "At", "On", "Beyond", "Within"
               boost: H, T, I, A, O, B, W

  ANS_WHEN   → "Anon", "Now", "Tomorrow", "Today", "Ere", "When", "Soon",
               "Tonight", "Yesterday"
               boost: A, N, T, E, W, S, Y

  ANS_WHY    → "Because", "For", "Since", "To", "That", "I", "My"
               boost: B, F, S, T, I, M

  ANS_HOW    → "Well", "Ill", "So", "Like", "By", "With", "Thus", "As"
               boost: W, I, S, L, B, T, A

  ANS_WHO    → "I", "Thou", "He", "She", "My", "The", "None", "A"
               boost: I, T, H, S, M, N, A

  ANS_WHICH  → "The", "That", "This", "These", "A", "All", "None"
               boost: T, A, N

Gates:
  - words_in_turn == 0 and sentences_in_turn == 0 (very first word
    of new turn)
  - speaker_label_state == 0 (outside speaker label FSM)
  - letter_run_len == 0 and word_buffer empty (word-start position)
  - pending_question_type != ANS_NONE

This bias layer complements `answer_opener` (which fires on any
INTERROG prior turn with a generic Yes/No/Ay/... table): when the
WH-class is known, we use a tighter class-specific distribution.

No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


ANS_NONE = 0
ANS_YESNO = 1
ANS_WHAT = 2
ANS_WHERE = 3
ANS_WHEN = 4
ANS_WHY = 5
ANS_HOW = 6
ANS_WHO = 7
ANS_WHICH = 8


# Per-class first-letter bias tables. Caps uppercase (sentence-opener
# is typically capitalized) and also give the lowercase a smaller
# share, since Shakespeare's editorial conventions occasionally open
# a response mid-line in lowercase.
_CLASS_TABLES: dict[int, dict[str, float]] = {
    ANS_YESNO: {
        "A": 0.60,  # Ay, Aye, Alas
        "Y": 0.55,  # Yes, Yea
        "N": 0.50,  # No, Nay, Not
        "I": 0.35,  # Indeed, I
        "M": 0.28,  # Marry, My
        "T": 0.25,  # Troth, That, Truly
        "O": 0.15,  # O, Oh
        "W": 0.12,  # Well, Why
    },
    ANS_WHAT: {
        "I": 0.40,  # I, It
        "T": 0.42,  # That, The, 'Tis, This
        "N": 0.32,  # Nothing, Naught, None
        "A": 0.30,  # A, An, All
        "M": 0.20,  # My, Marry
        "'": 0.18,  # 'Tis
    },
    ANS_WHERE: {
        "H": 0.55,  # Here, Hence
        "T": 0.45,  # There, Thither, To
        "I": 0.38,  # In, Into
        "A": 0.32,  # At, Away, Above
        "O": 0.28,  # On, Over, Out
        "B": 0.25,  # Beyond, Beneath, By, Below
        "W": 0.22,  # Within, Without, Where
        "N": 0.18,  # Near, Nigh
        "U": 0.15,  # Upon, Under
    },
    ANS_WHEN: {
        "A": 0.48,  # Anon, At, After, Always
        "N": 0.45,  # Now, Never, Next, No
        "T": 0.42,  # Tomorrow, Today, Tonight, To-morrow, This
        "E": 0.30,  # Ere, Even, Early
        "W": 0.25,  # When, While, Within
        "S": 0.25,  # Soon, Shortly, So
        "Y": 0.18,  # Yesterday, Yet
        "I": 0.15,  # In
        "P": 0.12,  # Presently
    },
    ANS_WHY: {
        "B": 0.50,  # Because
        "F": 0.42,  # For, 'Fore
        "S": 0.35,  # Since, So
        "T": 0.35,  # To, That, Therefore
        "I": 0.28,  # I
        "M": 0.22,  # My, Marry
        "W": 0.18,  # Why, Well
        "N": 0.15,  # Not, Nay, Nothing
        "A": 0.12,  # As
    },
    ANS_HOW: {
        "W": 0.50,  # Well, With, Why, What
        "I": 0.40,  # Ill, I, In
        "S": 0.38,  # So, Soft, Slowly
        "L": 0.30,  # Like, Lightly
        "B": 0.28,  # By, Boldly, Bravely
        "T": 0.28,  # Thus, Truly
        "A": 0.22,  # As, Alas
        "M": 0.18,  # My, Most
        "G": 0.15,  # Gently, Good
    },
    ANS_WHO: {
        "I": 0.50,  # I
        "T": 0.42,  # Thou, The, That
        "H": 0.35,  # He, Himself, Her
        "S": 0.32,  # She, Sir
        "M": 0.30,  # My, Madam, Master
        "N": 0.25,  # None, No, Nobody, No-one
        "A": 0.22,  # A, All
        "Y": 0.18,  # You, Your
        "O": 0.15,  # One, Our
    },
    ANS_WHICH: {
        "T": 0.55,  # The, That, This, These, Those
        "A": 0.38,  # A, An, All, Any
        "N": 0.30,  # None, Neither, No
        "B": 0.22,  # Both
        "E": 0.20,  # Either, Every
        "S": 0.18,  # Some
    },
}

# Small lowercase echo — scaled down because a new turn almost always
# starts with a capital. Applied only to letters that have a capital
# counterpart in the table.
_LOWER_SHARE: float = 0.18


def answer_expectation_start_bias(
    pending_question_type: int,
    words_in_turn: int,
    sentences_in_turn: int,
    speaker_label_state: int,
    letter_run_len: int,
    word_buffer: str,
) -> list[float] | None:
    if pending_question_type == ANS_NONE:
        return None
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0 or word_buffer:
        return None
    if words_in_turn != 0 or sentences_in_turn != 0:
        return None

    table = _CLASS_TABLES.get(pending_question_type)
    if table is None:
        return None

    vec = [0.0] * VOCAB_SIZE
    for ch, b in table.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += b
        # Lowercase echo for A..Z letters (skip "'" and other non-alpha).
        if len(ch) == 1 and "A" <= ch <= "Z":
            lo = ch.lower()
            lo_idx = VOCAB_INDEX.get(lo)
            if lo_idx is not None:
                vec[lo_idx] += b * _LOWER_SHARE
    return vec
