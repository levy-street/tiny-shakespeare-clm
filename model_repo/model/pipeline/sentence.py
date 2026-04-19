"""Sentence-type FSM.

Classifies the sentence currently being written into one of five types
based on its first word. The classification, plus a running count of
completed words in the sentence, lets the predict layer bias toward the
right terminal punctuation when the sentence becomes overdue:

  - Declarative   → period dominant
  - Interrogative → question mark dominant
  - Exclamative   → exclamation mark elevated
  - Imperative    → exclamation mark dominant, period secondary, early
                    vocative comma favored, question mark suppressed

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
# strong-feeling . line. Kept as the original set; imperative-only
# classifiers (_IMPER_STARTERS) are a separate tier; EXCLAM wins in
# the classifier's if/else order so overlap is harmless.
_EXCLAM_STARTERS: frozenset[str] = frozenset({
    "o", "oh", "ah", "alas", "hark", "lo", "fie", "pshaw",
    "marry", "zounds", "away", "hence", "come",
    "welcome", "hail", "behold",
})

# Bare-verb imperative starters — Shakespeare's "Speak, sirrah",
# "Hark thee!", "Begone!". An imperative opener is a bare verb that
# almost ALWAYS commands rather than declares at sentence start. We
# keep the list tight: ambiguous words (say/tell/see/look/get/hear/
# stay/hold/rise/sit/stand/give/take/bring/keep/serve/...) are also
# common declarative/subjunctive openers and belong under DECL. The
# cost of mis-tagging a DECL as IMPER (via the end-punct ratio shift
# toward "!") is high, so we err toward inclusion only for verbs
# that are essentially imperative-only at sentence start.
_IMPER_STARTERS: frozenset[str] = frozenset({
    # Sensory / attention imperatives — archaic, command-only at
    # sentence-initial position. "hark" also lives in _EXCLAM_STARTERS;
    # EXCLAM check wins first so that path is uncontested. Keeping
    # "hark" here too is harmless.
    "hark", "listen", "hearken", "mark",
    # Speech-act imperatives — almost always commands at sentence open.
    "speak", "tell",
    # Observation / attention imperatives.
    "look",
    # Stopping / holding imperatives.
    "stay", "hold", "cease", "forbear",
    # Archaic dismissals — unambiguously imperative.
    "begone", "avaunt",
    # Exhortative — "Let us go", "Let me speak" always imperative.
    "let",
    # NOTE: "come" / "go" are ambiguous at sentence start ("Come the
    # appointed hour..." subjunctive declarative) and net-regress
    # BPC when tagged as imperative. "come" already tags EXCLAM via
    # _EXCLAM_STARTERS.
})


SENT_UNKNOWN = 0
SENT_DECL = 1
SENT_INTERROG = 2
SENT_EXCLAM = 3
SENT_IMPER = 4


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
    if w in _IMPER_STARTERS:
        return SENT_IMPER
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
        # Sentence-level anaphora bookkeeping: save the just-finished
        # sentence's first word into prev_sentence_first_word so the
        # NEXT sentence's first word can be compared against it. Leave
        # curr_sentence_first_word empty (the next sentence's first
        # word will fill it when that word completes).
        return state.model_copy(
            update={
                "sentence_type": SENT_UNKNOWN,
                "words_in_sentence": 0,
                "prev_sentence_type": saved,
                "prev_sentence_first_word": state.curr_sentence_first_word,
                "curr_sentence_first_word": "",
            }
        )

    # Speaker-turn boundary: clear the cross-sentence memory; a new
    # speaker's first sentence should not inherit the prior speaker's
    # sentence-type context. BUT — preserve the prior turn's final
    # sentence type in prev_turn_final_sent_type so cross-turn
    # answer-opener biases can fire on the next speaker's first word.
    if state.consecutive_newlines >= 2 and ch == "\n":
        # Capture the outgoing turn's last sentence type. Prefer
        # prev_sentence_type (set at the last ./?/! of the outgoing
        # turn); fall back to the in-progress sentence_type if the
        # turn ended without terminal punctuation. Only overwrite
        # the memory slot when we have a real value — otherwise
        # keep whatever was there (unlikely to matter).
        outgoing = state.prev_sentence_type
        if outgoing == SENT_UNKNOWN and state.sentence_type != SENT_UNKNOWN:
            outgoing = state.sentence_type
        return state.model_copy(
            update={
                "sentence_type": SENT_UNKNOWN,
                "words_in_sentence": 0,
                "prev_sentence_type": SENT_UNKNOWN,
                "prev_sentence_first_word": "",
                "curr_sentence_first_word": "",
                "sentence_anaphora_run": 0,
                "prev_turn_final_sent_type": outgoing,
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

    # Sentence-level anaphora tracking: at the first word of a sentence,
    # compare against the previous sentence's first word and update
    # sentence_anaphora_run accordingly.
    curr_first = state.curr_sentence_first_word
    anaphora_run = state.sentence_anaphora_run
    if state.words_in_sentence == 0:
        # This IS the first word of the current sentence.
        curr_first = word
        if state.prev_sentence_first_word and word == state.prev_sentence_first_word:
            anaphora_run = min(anaphora_run + 1, 4)
        else:
            anaphora_run = 0

    return state.model_copy(
        update={
            "sentence_type": new_type,
            "words_in_sentence": state.words_in_sentence + 1,
            "curr_sentence_first_word": curr_first,
            "sentence_anaphora_run": anaphora_run,
        }
    )
