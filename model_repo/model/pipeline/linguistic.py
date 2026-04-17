"""Tier 2 — linguistic state updates.

This stage inspects the incoming token's character and updates the
linguistic features: character class of the last char, word run lengths,
newline context, speaker-label FSM, and sentence-start signal.

Character classes (bucket ids — see predict.py for the same enumeration):
  0  NEWLINE        '\n'
  1  SPACE          ' '
  2  UPPER          A–Z
  3  LOWER_VOWEL    a e i o u
  4  LOWER_CONS     other a–z
  5  APOSTROPHE     '
  6  PUNCT_END      . ? !
  7  PUNCT_MID      , ; :
  8  DASH           -
  9  OTHER          $ & digits 3
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB

NEWLINE = 0
SPACE = 1
UPPER = 2
LOWER_VOWEL = 3
LOWER_CONS = 4
APOSTROPHE = 5
PUNCT_END = 6
PUNCT_MID = 7
DASH = 8
OTHER = 9

N_CLASSES = 10

_VOWELS = set("aeiou")


def _classify(ch: str) -> int:
    if ch == "\n":
        return NEWLINE
    if ch == " ":
        return SPACE
    if "A" <= ch <= "Z":
        return UPPER
    if "a" <= ch <= "z":
        return LOWER_VOWEL if ch in _VOWELS else LOWER_CONS
    if ch == "'":
        return APOSTROPHE
    if ch in ".?!":
        return PUNCT_END
    if ch in ",;:":
        return PUNCT_MID
    if ch == "-":
        return DASH
    return OTHER


# Precompute class per vocab id.
CLASS_OF_TOKEN: list[int] = [_classify(ch) for ch in VOCAB]


def update_linguistic(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]
    cls = CLASS_OF_TOKEN[token_id]

    is_letter = cls in (UPPER, LOWER_VOWEL, LOWER_CONS)
    is_upper = cls == UPPER
    is_newline = cls == NEWLINE
    is_space = cls == SPACE

    # Run-length updates.
    letter_run_len = state.letter_run_len + 1 if is_letter else 0
    upper_run_len = state.upper_run_len + 1 if is_upper else 0

    consecutive_newlines = state.consecutive_newlines + 1 if is_newline else 0
    chars_since_newline = 0 if is_newline else state.chars_since_newline + 1
    chars_since_space = 0 if (is_space or is_newline) else state.chars_since_space + 1
    chars_since_sentence_end = (
        0 if cls == PUNCT_END else state.chars_since_sentence_end + 1
    )

    # Word completion.
    just_finished_word = (state.letter_run_len > 0) and not is_letter
    if is_letter:
        current_word_len = state.current_word_len + 1
    else:
        current_word_len = 0

    # Remember the tail of the word that just finished.
    if just_finished_word and state.last_char:
        # state.last_char is the last letter of the completed word.
        tail = state.last_char.lower()
        last_completed_word_tail = tail
    elif is_letter:
        last_completed_word_tail = state.last_completed_word_tail
    else:
        last_completed_word_tail = state.last_completed_word_tail

    # Speaker-label FSM.
    #   0 — default
    #   1 — just saw "\n\n", awaiting capital letter (or blank line)
    #   2 — in upper-case name
    #   3 — just saw ":" after upper name, expecting newline
    sp = state.speaker_label_state
    if is_newline and consecutive_newlines >= 2:
        sp_next = 1
    elif sp == 1:
        if is_upper:
            sp_next = 2
        elif is_newline:
            sp_next = 1  # still in blank-line region
        else:
            sp_next = 0
    elif sp == 2:
        if is_upper or is_space:
            # allow "KING HENRY IV" style multi-word labels
            sp_next = 2
        elif cls == PUNCT_MID and ch == ":":
            sp_next = 3
        else:
            sp_next = 0
    elif sp == 3:
        if is_newline:
            sp_next = 0  # label is closed by newline; go back to default
        else:
            sp_next = 0
    else:
        sp_next = 0

    sentence_start_pending = (
        consecutive_newlines >= 2
        or (is_space and state.chars_since_sentence_end <= 1)
    )

    last_is_vowel = ch.lower() in _VOWELS

    # word_buffer: accumulate letters (lowercased); reset on non-letter,
    # but treat apostrophe as part of the word ('tis, ne'er, o'er, 'em, 'd, 's).
    WORD_BUF_CAP = 16
    if is_letter:
        wb = (state.word_buffer + ch.lower())[-WORD_BUF_CAP:]
    elif ch == "'":
        wb = (state.word_buffer + "'")[-WORD_BUF_CAP:]
    else:
        wb = ""

    # speaker_buffer: active inside a speaker label (state 1/2), reset
    # when the label ends or we leave speaker-label territory.
    SPEAKER_BUF_CAP = 24
    if sp_next in (1, 2):
        if is_upper:
            sb = (state.speaker_buffer + ch)[-SPEAKER_BUF_CAP:]
        elif is_space and sp_next == 2:
            sb = (state.speaker_buffer + " ")[-SPEAKER_BUF_CAP:]
        else:
            sb = state.speaker_buffer
    else:
        sb = ""

    return state.model_copy(
        update={
            "prev_char": state.last_char,
            "last_char": ch,
            "last_char_class": cls,
            "prev_char_class": state.last_char_class,
            "letter_run_len": letter_run_len,
            "upper_run_len": upper_run_len,
            "consecutive_newlines": consecutive_newlines,
            "chars_since_newline": chars_since_newline,
            "chars_since_space": chars_since_space,
            "chars_since_sentence_end": chars_since_sentence_end,
            "just_finished_word": just_finished_word,
            "current_word_len": current_word_len,
            "last_completed_word_tail": last_completed_word_tail,
            "speaker_label_state": sp_next,
            "sentence_start_pending": sentence_start_pending,
            "last_is_vowel": last_is_vowel,
            "word_buffer": wb,
            "speaker_buffer": sb,
        }
    )
