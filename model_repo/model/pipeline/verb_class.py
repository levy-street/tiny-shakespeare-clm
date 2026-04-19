"""Tier 2 — verb semantic-class tracker.

Runs after `update_transitivity` so POS / transitivity / word-form
machinery is already settled. Maintains two fields:

  - verb_class: coarse 9-way semantic class (see state/schema.py)
  - vc_wait_words: words elapsed since class was set

Transitions at word completion:

  * A verb-family word (POS_VERB / POS_VERB_ED / POS_VERB_ING /
    POS_AUX_VERB) in one of the class-specific frozensets below:
      → verb_class = that class, vc_wait_words = 0

  * Otherwise, if verb_class != NONE:
      - NOUN / PROPER_NOUN / PRONOUN at vc_wait_words >= 1:
          the object slot just filled; we still keep the class
          alive briefly (drop to 0 at wait >= 3) because Shakespeare
          often extends with further object-class material.
      - CONJUNCTION / INTERJECTION / WH: reset (new constituent).
      - other: wait += 1, drop to NONE at wait >= 4.

Hard resets on:
  - sentence-end punctuation . ? !
  - clausal break , ; : (outside speaker labels)
  - speaker-turn boundary (\\n\\n)

All classification comes from prior English knowledge — no corpus
statistics. Verbs classified here overlap heavily with the transitive
list; they are re-partitioned into semantic groups.

Complements existing transitivity (boolean "DO expected") with a
*which-kind* signal that downstream predict layers can use to bias
first-letters of post-verb content words by semantic compatibility.
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

VC_NONE = 0
VC_PERCEPT = 1
VC_COGNITION = 2
VC_SPEECH = 3
VC_MOTION = 4
VC_GIVE_TAKE = 5
VC_VIOLENCE = 6
VC_EMOTION = 7
VC_BE_EXIST = 8


# Classified from prior English knowledge; forms include present,
# third-sing, past, participle, and Shakespearean -st / -eth endings.
_PERCEPT: frozenset[str] = frozenset({
    "see", "sees", "saw", "seen", "seest", "seeth",
    "hear", "hears", "heard", "hearst", "heareth",
    "feel", "feels", "felt", "feelest", "feeleth",
    "behold", "beholds", "beheld", "beholdst", "beholdeth",
    "watch", "watches", "watched",
    "observe", "observes", "observed",
    "witness", "witnesses", "witnessed",
    "mark", "marks", "marked",
    "spy", "spies", "spied",
    "find", "finds", "found",  # perceptual / cognitive bridge
    "look", "looks", "looked",
})

_COGNITION: frozenset[str] = frozenset({
    "know", "knows", "knew", "known", "knowest", "knoweth",
    "think", "thinks", "thought", "thinkst", "thinketh",
    "believe", "believes", "believed",
    "doubt", "doubts", "doubted", "doubtst", "doubteth",
    "suspect", "suspects", "suspected",
    "understand", "understands", "understood",
    "mean", "means", "meant",
    "intend", "intends", "intended",
    "consider", "considers", "considered",
    "remember", "remembers", "remembered",
    "forget", "forgets", "forgot", "forgotten",
    "deem", "deems", "deemed",
    "judge", "judges", "judged",
    "reckon", "reckons", "reckoned",
    "conceive", "conceives", "conceived",
    "perceive", "perceives", "perceived",
    "guess", "guesses", "guessed",
    "imagine", "imagines", "imagined",
})

_SPEECH: frozenset[str] = frozenset({
    "say", "says", "said", "sayest", "saith",
    "speak", "speaks", "spoke", "spoken", "speakst", "speaketh",
    "tell", "tells", "told", "tellst", "telleth",
    "ask", "asks", "asked",
    "answer", "answers", "answered",
    "call", "calls", "called", "callest", "calleth",
    "cry", "cries", "cried", "criest", "crieth",
    "name", "names", "named",
    "swear", "swears", "swore", "sworn",
    "promise", "promises", "promised",
    "command", "commands", "commanded",
    "beseech", "beseeches", "besought",
    "pray", "prays", "prayed",
    "thank", "thanks", "thanked",  # borderline emotion but mostly speech-act
    "declare", "declares", "declared",
    "bid", "bids", "bade", "bidden",
    "report", "reports", "reported",
    "confess", "confesses", "confessed",
    "deny", "denies", "denied",
    "vow", "vows", "vowed",
    "quoth",
    "sing", "sings", "sang", "sung",
    "speak'st",
})

_MOTION: frozenset[str] = frozenset({
    "go", "goes", "gone", "went", "goest", "goeth",
    "come", "comes", "came", "comest", "cometh",
    "follow", "follows", "followed",
    "lead", "leads", "led",
    "meet", "meets", "met",
    "seek", "seeks", "sought", "seekst", "seeketh",
    "hunt", "hunts", "hunted",
    "fly", "flies", "flew", "flown",
    "run", "runs", "ran",
    "walk", "walks", "walked",
    "ride", "rides", "rode", "ridden",
    "sail", "sails", "sailed",
    "march", "marches", "marched",
    "flee", "flees", "fled",
    "pursue", "pursues", "pursued",
    "return", "returns", "returned",
    "depart", "departs", "departed",
    "arrive", "arrives", "arrived",
    "enter", "enters", "entered",
    "climb", "climbs", "climbed",
    "wander", "wanders", "wandered",
})

_GIVE_TAKE: frozenset[str] = frozenset({
    "give", "gives", "gave", "given", "givest", "giveth",
    "take", "takes", "took", "taken", "takest", "taketh",
    "bring", "brings", "brought",
    "send", "sends", "sent",
    "offer", "offers", "offered",
    "keep", "keeps", "kept",
    "hold", "holds", "held", "holdst", "holdeth",
    "grant", "grants", "granted",
    "lend", "lends", "lent",
    "borrow", "borrows", "borrowed",
    "receive", "receives", "received",
    "present", "presents", "presented",
    "yield", "yields", "yielded",
    "pay", "pays", "paid",
    "fetch", "fetches", "fetched",
    "lose", "loses", "lost",
    "win", "wins", "won",
    "carry", "carries", "carried",
    "bear", "bears", "bore", "borne",
})

_VIOLENCE: frozenset[str] = frozenset({
    "kill", "kills", "killed",
    "slay", "slays", "slew", "slain",
    "strike", "strikes", "struck", "strikest", "striketh",
    "wound", "wounds", "wounded",
    "stab", "stabs", "stabbed",
    "beat", "beats", "beaten",
    "hurt", "hurts",
    "break", "breaks", "broke", "broken",
    "cut", "cuts",
    "hit", "hits",
    "tear", "tears", "tore", "torn",
    "crush", "crushes", "crushed",
    "smite", "smites", "smote", "smitten",
    "destroy", "destroys", "destroyed",
    "burn", "burns", "burned", "burnt",
    "bleed", "bleeds", "bled",
    "rend", "rends", "rent",
    "murder", "murders", "murdered",
    "attack", "attacks", "attacked",
    "pierce", "pierces", "pierced",
    "slaughter", "slaughters", "slaughtered",
    "execute", "executes", "executed",
    "hang", "hangs", "hanged", "hung",
    "fight", "fights", "fought",
})

_EMOTION: frozenset[str] = frozenset({
    "love", "loves", "loved", "lovest", "loveth",
    "hate", "hates", "hated",
    "fear", "fears", "feared", "fearst", "feareth",
    "curse", "curses", "cursed",
    "bless", "blesses", "blessed",
    "pity", "pities", "pitied",
    "praise", "praises", "praised",
    "mourn", "mourns", "mourned",
    "lament", "laments", "lamented",
    "rejoice", "rejoices", "rejoiced",
    "wish", "wishes", "wished",
    "want", "wants", "wanted",
    "need", "needs", "needed",
    "long", "longs", "longed",  # as in long-for
    "adore", "adores", "adored",
    "despise", "despises", "despised",
    "scorn", "scorns", "scorned",
    "honour", "honours", "honoured",
    "honor", "honors", "honored",
    "trust", "trusts", "trusted",
    "envy", "envies", "envied",
})

_BE_EXIST: frozenset[str] = frozenset({
    "is", "are", "was", "were", "be", "been", "being",
    "am", "art", "wert", "wast",
    "seem", "seems", "seemed", "seeming", "seemest", "seemeth",
    "become", "becomes", "became", "becoming",
    "appear", "appears", "appeared", "appearing",
    "prove", "proves", "proved", "proven",
    "remain", "remains", "remained",
    "stay", "stays", "stayed",  # copular-adjacent here
    "'tis",  # clitic; POS tagger may or may not see this
})


_CLASS_LOOKUP: tuple[tuple[frozenset[str], int], ...] = (
    (_BE_EXIST, VC_BE_EXIST),  # priority: be-forms win first
    (_PERCEPT, VC_PERCEPT),
    (_COGNITION, VC_COGNITION),
    (_SPEECH, VC_SPEECH),
    (_MOTION, VC_MOTION),
    (_GIVE_TAKE, VC_GIVE_TAKE),
    (_VIOLENCE, VC_VIOLENCE),
    (_EMOTION, VC_EMOTION),
)


_VERB_POS: frozenset[int] = frozenset({
    POS_VERB, POS_VERB_ED, POS_VERB_ING, POS_AUX_VERB,
})

_OBJECT_POS: frozenset[int] = frozenset({
    POS_NOUN, POS_PROPER_NOUN, POS_PRONOUN,
})

_RESET_POS: frozenset[int] = frozenset({
    POS_CONJUNCTION, POS_INTERJECTION,
})


def _classify(w: str) -> int:
    for s, cls in _CLASS_LOOKUP:
        if w in s:
            return cls
    return VC_NONE


def update_verb_class(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    vc = state.verb_class
    wait = state.vc_wait_words

    # Sentence-end / clausal / turn resets.
    if ch in ".?!":
        if vc != VC_NONE or wait != 0:
            return state.model_copy(
                update={"verb_class": VC_NONE, "vc_wait_words": 0}
            )
        return state
    if state.consecutive_newlines >= 2 and ch == "\n":
        if vc != VC_NONE or wait != 0:
            return state.model_copy(
                update={"verb_class": VC_NONE, "vc_wait_words": 0}
            )
        return state
    if ch in ",;:" and state.speaker_label_state == 0:
        if vc != VC_NONE or wait != 0:
            return state.model_copy(
                update={"verb_class": VC_NONE, "vc_wait_words": 0}
            )
        return state

    if state.just_finished_word and state.last_completed_word:
        w = state.last_completed_word
        pos = state.last_word_pos

        # Verb completion — classify; new verb overrides prior class.
        if pos in _VERB_POS:
            cls = _classify(w)
            if cls != VC_NONE:
                vc = cls
                wait = 0
            elif vc != VC_NONE:
                # Unclassified verb after a known one — mild decay.
                wait = min(wait + 1, 5)
                if wait >= 4:
                    vc = VC_NONE
                    wait = 0
        elif vc != VC_NONE:
            if pos in _RESET_POS:
                vc = VC_NONE
                wait = 0
            elif pos in _OBJECT_POS:
                # Object slot partially filled; keep class alive but
                # decay.
                wait = min(wait + 1, 5)
                if wait >= 3:
                    vc = VC_NONE
                    wait = 0
            else:
                # Modifiers / unknowns — gentle decay.
                wait = min(wait + 1, 5)
                if wait >= 4:
                    vc = VC_NONE
                    wait = 0

    if vc != state.verb_class or wait != state.vc_wait_words:
        return state.model_copy(
            update={"verb_class": vc, "vc_wait_words": wait}
        )
    return state
