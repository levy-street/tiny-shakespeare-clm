"""Tier 2 — word-form (morphological slot) expectation FSM.

Runs after `update_verb_agreement` so POS tags and clause/transitivity
state are already set. Maintains:

  - word_form_expectation: 0..6 enumeration (see schema docstring)
  - wfe_wait_words: words since expectation was set

Transitions at word completion:

  * Modal / infinitive-marker (to/shall/will/must/may/might/can/could/
    would/should/let/do/dost/does/did/didst/shalt/wilt/canst):
      → WFE_INFINITIVE, wait=0

  * Perfect-auxiliary (have/has/had/having/hath/hast):
      → WFE_PAST_PART, wait=0

  * Copula / be-form (is/am/are/was/were/be/been/being/art/wert):
      → WFE_ING_OR_PP, wait=0

  * Preposition "of" (NP-head specifically):
      → WFE_NOMINAL, wait=0

  * Comparative markers (more/less):
      → WFE_COMPARATIVE, wait=0

  * Superlative marker (most):
      → WFE_SUPERLATIVE, wait=0

Resolution / reset triggers:
  - VERB / VERB_ED / VERB_ING / NOUN / PROPER_NOUN / ADJECTIVE
    completion: expectation satisfied; reset to NONE.
  - ARTICLE / POSSESSIVE / ADJECTIVE / ADVERB / NUMBER / NEGATION
    between trigger and target: wait += 1 (pre-modifiers allowed).
  - CONJUNCTION: reset to NONE.
  - Sentence-end (. ? !), clausal break (,;:), speaker-turn (\\n\\n): reset.
  - wait >= 4: stale; reset.

All classification is from prior knowledge of English — no corpus stats.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB
from .pos import (
    POS_ADJECTIVE,
    POS_ADVERB,
    POS_ARTICLE,
    POS_AUX_VERB,
    POS_CONJUNCTION,
    POS_INTERJECTION,
    POS_MODAL,
    POS_NEGATION,
    POS_NOUN,
    POS_NUMBER,
    POS_POSSESSIVE,
    POS_PREPOSITION,
    POS_PRONOUN,
    POS_PROPER_NOUN,
    POS_UNKNOWN,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
    POS_WH,
)

WFE_NONE = 0
WFE_INFINITIVE = 1
WFE_PAST_PART = 2
WFE_ING_OR_PP = 3
WFE_NOMINAL = 4
WFE_COMPARATIVE = 5
WFE_SUPERLATIVE = 6


# Infinitive triggers — after these, a bare verb is expected.
_INF_TRIGGERS: frozenset[str] = frozenset({
    "to",
    "shall", "shalt",
    "will", "wilt",
    "must",
    "may", "mayst",
    "might",
    "can", "canst",
    "could", "couldst",
    "would", "wouldst",
    "should", "shouldst",
    "let",
    "do", "dost", "does", "did", "didst",
})

# Perfect-auxiliary triggers — past participle expected.
_PP_TRIGGERS: frozenset[str] = frozenset({
    "have", "has", "had", "having", "hath", "hast", "haste",
})

# Copula / be-form triggers — -ing or past participle (passive).
_COP_TRIGGERS: frozenset[str] = frozenset({
    "is", "am", "are", "was", "were", "be", "been", "being",
    "art", "wert",
})

# Nominal-expectation triggers — we pick specifically "of", because
# after "of" Shakespeare overwhelmingly supplies a head noun
# ("of love", "of death", "of war", "of our", "of his"). np_open
# already catches generic NP openers; this is finer than that.
_NOMINAL_TRIGGERS: frozenset[str] = frozenset({"of"})

# Comparative / superlative triggers.
_COMP_TRIGGERS: frozenset[str] = frozenset({"more", "less"})
_SUP_TRIGGERS: frozenset[str] = frozenset({"most"})


# Content-word POS tags that RESOLVE an expectation (the slot is filled).
_RESOLVE_POS: frozenset[int] = frozenset({
    POS_VERB, POS_VERB_ED, POS_VERB_ING,
    POS_NOUN, POS_PROPER_NOUN, POS_ADJECTIVE,
})

# Modifier POS tags that EXTEND the wait without resolving.
_EXTEND_POS: frozenset[int] = frozenset({
    POS_ARTICLE, POS_POSSESSIVE, POS_ADVERB,
    POS_NUMBER, POS_NEGATION,
})

# POS tags that RESET the expectation (different constituent took over).
_RESET_POS: frozenset[int] = frozenset({
    POS_CONJUNCTION, POS_INTERJECTION,
})


def update_word_form(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    wfe = state.word_form_expectation
    wait = state.wfe_wait_words

    # Hard resets on sentence/clausal/turn boundaries.
    if ch in ".?!":
        if wfe != WFE_NONE or wait != 0:
            return state.model_copy(
                update={"word_form_expectation": WFE_NONE, "wfe_wait_words": 0}
            )
        return state
    if state.consecutive_newlines >= 2 and ch == "\n":
        if wfe != WFE_NONE or wait != 0:
            return state.model_copy(
                update={"word_form_expectation": WFE_NONE, "wfe_wait_words": 0}
            )
        return state
    if ch in ",;:" and state.speaker_label_state == 0:
        if wfe != WFE_NONE or wait != 0:
            return state.model_copy(
                update={"word_form_expectation": WFE_NONE, "wfe_wait_words": 0}
            )
        return state

    if state.just_finished_word and state.last_completed_word:
        w = state.last_completed_word
        pos = state.last_word_pos

        # Check trigger sets first: a completed word can OPEN a new
        # expectation (overriding any previous expectation since only
        # one form is expected at a time).
        if w in _INF_TRIGGERS:
            wfe = WFE_INFINITIVE
            wait = 0
        elif w in _PP_TRIGGERS:
            wfe = WFE_PAST_PART
            wait = 0
        elif w in _COP_TRIGGERS:
            wfe = WFE_ING_OR_PP
            wait = 0
        elif w in _NOMINAL_TRIGGERS:
            wfe = WFE_NOMINAL
            wait = 0
        elif w in _COMP_TRIGGERS:
            wfe = WFE_COMPARATIVE
            wait = 0
        elif w in _SUP_TRIGGERS:
            wfe = WFE_SUPERLATIVE
            wait = 0
        # Not a trigger: check whether this completion resolves or
        # extends the current expectation.
        elif wfe != WFE_NONE:
            if pos in _RESOLVE_POS:
                # Verb/noun/adj just completed — expectation satisfied.
                wfe = WFE_NONE
                wait = 0
            elif pos in _RESET_POS:
                # Conjunction or interjection — different path.
                wfe = WFE_NONE
                wait = 0
            elif pos in _EXTEND_POS:
                # Pre-modifier: wait, don't reset yet.
                wait = min(wait + 1, 5)
                if wait >= 4:
                    wfe = WFE_NONE
                    wait = 0
            else:
                # Unknown / pronoun / other — decay wait.
                wait = min(wait + 1, 5)
                if wait >= 4:
                    wfe = WFE_NONE
                    wait = 0

    if (
        wfe != state.word_form_expectation
        or wait != state.wfe_wait_words
    ):
        return state.model_copy(
            update={
                "word_form_expectation": wfe,
                "wfe_wait_words": wait,
            }
        )
    return state
