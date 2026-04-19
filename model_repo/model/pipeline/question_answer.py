"""Question-answer discourse axis.

Tracks `pending_question_type`: a cross-turn discourse slot indicating
what kind of answer the *current* turn should open with, given that
the *previous* turn ended with a question.

Mechanism:

  - On `?` emission (last_char == "?" at PUNCT_END), classify the
    sentence that just ended by looking at `curr_sentence_first_word`
    (not yet cleared by update_sentence, which runs later). The
    classification yields an ANS_* code.

  - The code is carried through the turn boundary (update_sentence
    clears its own fields but does NOT touch pending_question_type).

  - On the *first word-completion* of any later turn, we clear
    pending_question_type back to ANS_NONE — the response opener has
    committed, the slot is consumed.

The predict layer `answer_expectation` reads `pending_question_type`
at the first letter of the first word of a new turn and biases the
first-letter distribution toward class-specific answer openers:

  ANS_YESNO  → Ay/Yes/No/Nay/Indeed/I/Marry/Troth
  ANS_WHAT   → I/That/Nothing/A/The/It/Tis
  ANS_WHERE  → Here/There/In/At/On/Beyond/Within
  ANS_WHEN   → Anon/Now/Tomorrow/Today/Ere/When/Soon/Tonight
  ANS_WHY    → Because/For/Since/To/That/I
  ANS_HOW    → Well/Ill/So/Like/By/With
  ANS_WHO    → I/Thou/He/She/My/The/None/A
  ANS_WHICH  → The/That/This/These/A/All

No corpus statistics. All signal from prior knowledge of Shakespeare's
dialogic Q-A patterns.

Must run AFTER update_linguistic (needs last_char / last_char_class)
and BEFORE update_sentence (which clears curr_sentence_first_word on
PUNCT_END).
"""

from __future__ import annotations

from ..state import ModelState
from .linguistic import PUNCT_END


ANS_NONE = 0
ANS_YESNO = 1
ANS_WHAT = 2
ANS_WHERE = 3
ANS_WHEN = 4
ANS_WHY = 5
ANS_HOW = 6
ANS_WHO = 7
ANS_WHICH = 8


# WH-word → answer class. Keyed lowercase (word_buffer is lowercased).
# Shakespeare-era WH forms included: whither, whence, wherefore.
_WH_TO_CLASS: dict[str, int] = {
    "what":      ANS_WHAT,
    "wherein":   ANS_WHAT,
    "whereof":   ANS_WHAT,
    "whereto":   ANS_WHAT,
    "whereon":   ANS_WHAT,

    "where":     ANS_WHERE,
    "whither":   ANS_WHERE,
    "whence":    ANS_WHERE,

    "when":      ANS_WHEN,

    "why":       ANS_WHY,
    "wherefore": ANS_WHY,

    "how":       ANS_HOW,

    "who":       ANS_WHO,
    "whom":      ANS_WHO,
    "whose":     ANS_WHO,

    "which":     ANS_WHICH,
}

# Auxiliary-led questions yield yes/no answers.
_AUX_STARTERS: frozenset[str] = frozenset({
    "is", "are", "was", "were", "am", "be",
    "do", "does", "did", "doth", "dost",
    "have", "has", "had", "hast", "hath",
    "can", "could", "canst", "couldst",
    "shall", "should", "shalt", "shouldst",
    "will", "would", "wilt", "wouldst",
    "may", "might", "mayst",
    "must", "art", "wert",
})


def _classify(first_word: str) -> int:
    """Classify a sentence's first word (lowercased) into an ANS_* class.

    Returns ANS_NONE if the first word is unknown / not a canonical
    question opener — we don't want to install a mis-classifier for
    declarative-style sentences that happen to end in "?".
    """
    if not first_word:
        return ANS_NONE
    w = first_word.lstrip("'")
    if w in _WH_TO_CLASS:
        return _WH_TO_CLASS[w]
    if w in _AUX_STARTERS:
        return ANS_YESNO
    return ANS_NONE


def update_question_answer(state: ModelState, token_id: int) -> ModelState:
    ch = state.last_char
    cls = state.last_char_class

    # 1. On "?" emission at sentence end: classify the just-closed
    # sentence from its first word. curr_sentence_first_word is still
    # live here (update_sentence runs later and will clear it).
    if cls == PUNCT_END and ch == "?":
        new_type = _classify(state.curr_sentence_first_word)
        if new_type != state.pending_question_type:
            return state.model_copy(
                update={"pending_question_type": new_type}
            )
        return state

    # 2. On ANY other sentence-end punctuation (period, exclamation)
    # — also clear the slot. A `? ... .` pair inside one turn means
    # the question was answered within the turn; no cross-turn
    # expectation remains.
    if cls == PUNCT_END and ch != "?":
        if state.pending_question_type != ANS_NONE:
            return state.model_copy(
                update={"pending_question_type": ANS_NONE}
            )
        return state

    # 3. On first word-completion of a turn that HAS a pending
    # question type: the response opener just landed. Consume the
    # slot — one shot only, so the subsequent words aren't biased.
    # "first word of a turn" = just_finished_word AND the word that
    # just completed is the opener (words_in_turn == 1 after its
    # completion counter increments in update_turn_progress, which
    # hasn't run yet, so here words_in_turn is still 0 at the moment
    # the first word completes).
    if (
        state.just_finished_word
        and state.pending_question_type != ANS_NONE
        and state.speaker_label_state == 0
        and state.words_in_turn == 0
        and state.sentences_in_turn == 0
    ):
        return state.model_copy(
            update={"pending_question_type": ANS_NONE}
        )

    return state
