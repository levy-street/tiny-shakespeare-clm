"""Tier 2: verb-complement class tracker.

Maintains `state.verb_complement_class` (VCC_*) and `state.vcc_wait_words`.
At word completion, if the word is a verb in one of the categorized
inventories, set the expectation; otherwise age the existing wait
and clear when a fitting complement appears or time runs out.

Categories (set by last_completed_word lookup):

  VCC_THAT  — mental / communication / speech verbs:
              say, said, saith, tell, told, speak, spoke,
              think, thought, know, knew, hear, heard,
              believe, wish, swear, vow, fear, hope, trust,
              doubt, pray, confess, affirm, promise, answer,
              cry, beg, protest, deny, grant, perceive
  VCC_PP    — motion / trajectory verbs:
              come, came, go, went, goes, goeth, went,
              run, ran, fly, flew, march, ride, rode,
              walk, walked, move, moved, travel, travelled,
              return, depart, leave, left, arrive, arrived,
              enter, exit, flee, fled
  VCC_PPART — aux / perfective auxiliaries:
              have, has, had, hath, hast, having
  VCC_INF   — modal / infinitive-licensing:
              shall, will, would, should, could, may, might,
              must, can, cannot, canst, canst, wilt, shalt,
              wouldst, shouldst, couldst, durst, dare
  VCC_PRED  — copula / linking:
              is, are, was, were, be, am, art, been, being,
              seem, seems, seemed, become, became, grew (=became)

Reset rules:
  * On ".?!" or ",;:" punctuation (update_basic_counters already
    set last_char_class; we peek at the emitted char indirectly).
  * On "\\n\\n" speaker-turn boundary.
  * When a fitting complement *opens* (e.g., "that" word after VCC_THAT;
    a preposition word after VCC_PP; a VERB_ED word after VCC_PPART).
  * When vcc_wait_words reaches a cap (5).

Runs AFTER `update_pos` so `last_completed_word` and `last_word_pos`
are fresh.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB
from .pos import (
    POS_PREPOSITION,
    POS_VERB,
    POS_VERB_ED,
    POS_VERB_ING,
)

VCC_NONE = 0
VCC_THAT = 1
VCC_PP = 2
VCC_PPART = 3
VCC_INF = 4
VCC_PRED = 5


_THAT_VERBS: frozenset[str] = frozenset({
    "say", "said", "says", "saith", "saidst", "sayest",
    "tell", "told", "tells", "telleth",
    "speak", "spoke", "speaks", "speaketh", "speaking",
    "think", "thinks", "thought", "thinketh",
    "know", "knew", "knows", "knoweth", "knowest",
    "hear", "heard", "hears", "heareth", "hearest",
    "believe", "believes", "believed",
    "wish", "wishes", "wished",
    "swear", "swore", "swears", "swearest",
    "vow", "vowed", "vows",
    "fear", "fears", "feared",
    "hope", "hoped", "hopes",
    "trust", "trusted", "trusts",
    "doubt", "doubts", "doubted",
    "pray", "prayed", "prays",
    "confess", "confessed", "confesseth",
    "affirm", "affirmed",
    "promise", "promised", "promises",
    "answer", "answered",
    "cry", "cried", "cries", "crieth",
    "beg", "begged", "begs",
    "protest", "protested",
    "deny", "denied", "denies",
    "grant", "granted", "grants",
    "perceive", "perceived", "perceives",
    "suppose", "supposed",
    "judge", "judged",
    "imagine", "imagined",
    "remember", "remembered",
    "forget", "forgot",
    "suspect", "suspected",
    "warn", "warned",
})

_PP_VERBS: frozenset[str] = frozenset({
    "come", "came", "comes", "cometh",
    "go", "goes", "goeth", "went",
    "run", "ran", "runs", "runneth",
    "fly", "flies", "flew", "flying",
    "march", "marched", "marches",
    "ride", "rode", "rides", "riding",
    "walk", "walked", "walks",
    "move", "moved", "moves", "moving",
    "travel", "travelled", "travels", "traveling",
    "return", "returned", "returns",
    "depart", "departed", "departs",
    "leave", "left", "leaves", "leaving",
    "arrive", "arrived", "arrives",
    "enter", "entered", "enters",
    "exit", "exited", "exits",
    "flee", "fled", "flees",
    "retreat", "retreated",
    "escape", "escaped",
    "fall", "fell", "falls",
    "climb", "climbed",
    "rise", "rose", "rises", "risen",
    "pass", "passed", "passes",
    "approach", "approached",
    "advance", "advanced",
    "retire", "retired",
    "head", "heads",  # as verb: "head to the gate"
    "hie", "hied",    # "hie thee to the chamber"
    "get", "got", "gets",  # "get thee gone" — motion imperative common
    "hasten", "hastened",
})

_PPART_VERBS: frozenset[str] = frozenset({
    "have", "has", "had", "hath", "hast", "having",
})

_INF_VERBS: frozenset[str] = frozenset({
    "shall", "will", "would", "should", "could", "may", "might",
    "must", "can", "cannot", "canst", "wilt", "shalt",
    "wouldst", "shouldst", "couldst", "durst", "dare",
    "do", "did", "does", "doth", "dost", "doest",  # when used as aux
})

_PRED_VERBS: frozenset[str] = frozenset({
    "is", "are", "was", "were", "be", "am", "art", "been", "being",
    "seem", "seems", "seemed", "seemeth",
    "become", "became", "becomes",
    "appear", "appeared", "appears", "appeareth",
    "remain", "remained", "remains",
    "prove", "proved", "proves",
    "grow", "grew", "grows", "groweth",  # "grew quiet"
})


_WAIT_CAP = 5


# Words that close an expectation — the complement arrived.
_THAT_CLOSERS: frozenset[str] = frozenset({
    "that", "whether", "if", "how", "why", "what", "where", "when",
    "who", "whom", "which", "because",
})
# For VCC_THAT, also a full NP opener (the, a, an, my, thy, his, her,
# our, their, this, these, those, some, any) or a pronoun closes it.
_NP_OPENERS: frozenset[str] = frozenset({
    "the", "a", "an", "my", "thy", "thine", "his", "her", "our",
    "your", "their", "this", "that", "these", "those", "some", "any",
    "no", "all", "such", "same", "one",
})
_PRONOUNS_OBJ: frozenset[str] = frozenset({
    "me", "thee", "him", "us", "you", "them", "it",
    "i", "thou", "he", "she", "we", "they",  # also subjects
})

# Prepositions — close PP expectation.
_PREPS: frozenset[str] = frozenset({
    "to", "from", "toward", "towards", "into", "in", "on", "upon",
    "at", "of", "with", "within", "without", "through", "across",
    "before", "after", "beyond", "beside", "among", "between",
    "above", "below", "under", "by", "for", "against", "near",
    "around", "about",
})

# Past participle / irregular past-participle forms for VCC_PPART.
_PAST_PARTICIPLES: frozenset[str] = frozenset({
    "seen", "done", "gone", "taken", "given", "spoken", "written",
    "broken", "chosen", "driven", "risen", "frozen", "stolen",
    "eaten", "fallen", "forgotten", "gotten", "hidden", "ridden",
    "shaken", "shown", "sworn", "thrown", "torn", "worn",
    "been", "become", "come",
    "fought", "taught", "brought", "bought", "caught", "thought",
    "sought", "wrought",
    "lost", "left", "kept", "slept", "wept", "crept", "swept",
    "felt", "dealt", "heard", "held", "made", "said", "sent",
    "paid", "laid", "met", "led", "fed", "bled", "read",
    "borne", "born", "sworn", "slain", "struck", "stricken",
    "drunk", "sunk", "sung", "rung", "flung", "hung", "clung",
    "brought", "crept", "taken",
    # Regular -ed forms: handled via POS tag check rather than list.
})


def update_verb_complement(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-turn boundary: clear.
    if ch == "\n" and state.consecutive_newlines >= 2:
        if state.verb_complement_class != VCC_NONE:
            return state.model_copy(update={
                "verb_complement_class": VCC_NONE,
                "vcc_wait_words": 0,
            })
        return state

    # Sentence-end / clause-end punctuation: clear.
    if ch in (".", "?", "!", ";", ":"):
        if state.verb_complement_class != VCC_NONE:
            return state.model_copy(update={
                "verb_complement_class": VCC_NONE,
                "vcc_wait_words": 0,
            })
        return state

    if not state.just_finished_word or not state.last_completed_word:
        return state

    word = state.last_completed_word.lower().strip("'")

    cur_vcc = state.verb_complement_class
    cur_wait = state.vcc_wait_words

    # Check if the CURRENT complement is satisfied by this just-completed
    # word — clear expectation if so.
    if cur_vcc != VCC_NONE:
        closed = False
        if cur_vcc == VCC_THAT:
            if (word in _THAT_CLOSERS
                or word in _NP_OPENERS
                or word in _PRONOUNS_OBJ):
                closed = True
        elif cur_vcc == VCC_PP:
            if word in _PREPS:
                closed = True
        elif cur_vcc == VCC_PPART:
            if (word in _PAST_PARTICIPLES
                or state.last_word_pos == POS_VERB_ED):
                closed = True
        elif cur_vcc == VCC_INF:
            # Bare verb (not aux) closes the infinitive slot.
            if state.last_word_pos == POS_VERB:
                closed = True
        elif cur_vcc == VCC_PRED:
            # Copula predicate filled by many POS — close on any content
            # word that isn't a determiner. Conservative: close on
            # ADJECTIVE / NOUN / VERB_ING / VERB_ED or an article-opened
            # NP will keep us going.
            if state.last_word_pos in (POS_VERB_ED, POS_VERB_ING):
                closed = True
            # Predicate nominal is hard to detect here; rely on wait.
        if closed:
            cur_vcc = VCC_NONE
            cur_wait = 0

    # Now, does THIS word START a new expectation?
    new_vcc = VCC_NONE
    if word in _PPART_VERBS:
        new_vcc = VCC_PPART
    elif word in _INF_VERBS:
        new_vcc = VCC_INF
    elif word in _PRED_VERBS:
        new_vcc = VCC_PRED
    elif word in _THAT_VERBS:
        new_vcc = VCC_THAT
    elif word in _PP_VERBS:
        new_vcc = VCC_PP

    # If a new expectation fires, it overrides any prior one (chain:
    # "I shall have spoken" — shall sets INF, have sets PPART).
    if new_vcc != VCC_NONE:
        if cur_vcc != new_vcc or cur_wait != 0:
            return state.model_copy(update={
                "verb_complement_class": new_vcc,
                "vcc_wait_words": 0,
            })
        return state

    # Otherwise age the existing expectation.
    if cur_vcc != VCC_NONE:
        new_wait = cur_wait + 1
        if new_wait >= _WAIT_CAP:
            if state.verb_complement_class != VCC_NONE or state.vcc_wait_words != 0:
                return state.model_copy(update={
                    "verb_complement_class": VCC_NONE,
                    "vcc_wait_words": 0,
                })
            return state
        if new_wait != state.vcc_wait_words or cur_vcc != state.verb_complement_class:
            return state.model_copy(update={
                "verb_complement_class": cur_vcc,
                "vcc_wait_words": new_wait,
            })

    # If we cleared an expectation above (closed=True) without setting a
    # new one, commit the clearing.
    if cur_vcc != state.verb_complement_class or cur_wait != state.vcc_wait_words:
        return state.model_copy(update={
            "verb_complement_class": cur_vcc,
            "vcc_wait_words": cur_wait,
        })

    return state
