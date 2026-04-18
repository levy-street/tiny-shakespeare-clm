"""Sentence-type FSM.

Classifies the sentence currently being written into one of four types
based on its first word. The classification, plus a running count of
completed words in the sentence, lets the predict layer bias toward the
right terminal punctuation when the sentence becomes overdue:

  - Declarative → period dominant
  - Interrogative → question mark dominant
  - Exclamative  → exclamation mark elevated

Classification happens exactly once per sentence — when the first word
completes. It resets on any PUNCT_END character.

Runs after `update_pos` so `last_completed_word` is fresh when the state
commits.
"""

from __future__ import annotations

from ..state import ModelState
from .linguistic import PUNCT_END

# WH-words and interrogative starters that indicate a question.
_WH_STARTERS: frozenset[str] = frozenset({
    "who", "whom", "whose", "what", "when", "where", "why", "how",
    "which", "whither", "whence", "wherefore", "whereof",
})

# Auxiliary/modal verbs that, when they open a sentence, indicate an
# inverted-auxiliary question (e.g. "Is he here?", "Shall we go?",
# "Canst thou tell me?", "Art thou mad?"). This is heuristic — some
# imperative openings ("Do be quiet.") are declaratives in disguise —
# so the bias applied downstream is softer than for WH-starters.
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

# Interjections / exclamative starters — almost always yield a ! or a
# strong-feeling . line.
_EXCLAM_STARTERS: frozenset[str] = frozenset({
    "o", "oh", "ah", "alas", "hark", "lo", "fie", "pshaw",
    "marry", "zounds", "away", "hence", "come",
    "welcome", "hail", "behold",
})


SENT_UNKNOWN = 0
SENT_DECL = 1
SENT_INTERROG = 2
SENT_EXCLAM = 3


def _classify_first_word(word: str) -> int:
    """Return a sentence-type tag from the lowercased first word."""
    if not word:
        return SENT_UNKNOWN
    w = word
    # Strip leading apostrophe (e.g. 'tis, 'gainst).
    core = w.lstrip("'")
    if core != w:
        # 'tis = it is → declarative default
        if core == "tis":
            return SENT_DECL
    if w in _WH_STARTERS:
        return SENT_INTERROG
    if w in _AUX_STARTERS:
        # Aux-starters lean interrogative but imperatives also use them
        # ("Be still.", "Come, peace."). Still, interrogative is a
        # reasonable prior for Shakespeare's dialogic register.
        return SENT_INTERROG
    if w in _EXCLAM_STARTERS:
        return SENT_EXCLAM
    return SENT_DECL


def update_sentence(state: ModelState, token_id: int) -> ModelState:
    # Two things to handle per step: punctuation reset and first-word
    # classification.
    ch = state.last_char  # already updated by linguistic stage
    cls = state.last_char_class

    # Reset on sentence-end punctuation — but save the just-finished
    # sentence's type into prev_sentence_type so downstream predict
    # layers can condition the NEXT sentence's opener on what kind of
    # sentence just ended. If we never classified (sentence was too
    # short), keep the prior prev_sentence_type rather than losing it.
    if cls == PUNCT_END:
        if state.sentence_type != SENT_UNKNOWN:
            saved = state.sentence_type
        else:
            saved = state.prev_sentence_type
        # Also elevate to EXCLAM if the terminal punctuation is "!"
        # and the classifier hadn't already caught it — "!" itself
        # is strong evidence of an exclamative sentence, whatever the
        # opener was.
        if ch == "!" and saved == SENT_DECL:
            saved = SENT_EXCLAM
        elif ch == "?" and saved == SENT_DECL:
            saved = SENT_INTERROG
        return state.model_copy(
            update={
                "sentence_type": SENT_UNKNOWN,
                "words_in_sentence": 0,
                "prev_sentence_type": saved,
            }
        )

    # Speaker-turn boundary: clear the cross-sentence memory; a new
    # speaker's first sentence should not inherit the prior speaker's
    # sentence-type context.
    if state.consecutive_newlines >= 2 and ch == "\n":
        return state.model_copy(
            update={
                "sentence_type": SENT_UNKNOWN,
                "words_in_sentence": 0,
                "prev_sentence_type": SENT_UNKNOWN,
            }
        )

    if not state.just_finished_word:
        return state

    word = state.last_completed_word
    # Don't let the speaker-label machinery feed the sentence classifier;
    # a speaker label's "word" is not a sentence-opening word.
    if state.speaker_label_state != 0:
        return state

    if state.words_in_sentence == 0:
        new_type = _classify_first_word(word)
    else:
        new_type = state.sentence_type
        # Fallback: if we never classified (e.g. first word was odd),
        # treat sentence as declarative by default.
        if new_type == SENT_UNKNOWN:
            new_type = SENT_DECL

    # Upgrade to EXCLAM if we see a strong-feeling marker mid-sentence.
    # This gently nudges toward "!" later. Only upgrade from DECL, never
    # downgrade INTERROG.
    if new_type == SENT_DECL and word in _EXCLAM_STARTERS:
        new_type = SENT_EXCLAM

    return state.model_copy(
        update={
            "sentence_type": new_type,
            "words_in_sentence": state.words_in_sentence + 1,
        }
    )
