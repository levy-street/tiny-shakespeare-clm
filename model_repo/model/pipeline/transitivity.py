"""Tier 2 — verb transitivity / object-expectation FSM.

Runs after `update_np_head` so `state.last_word_pos` and `state.np_open`
reflect what has just happened. Maintains two fields:

  - verb_transitivity: 0=NONE, 1=DO_EXPECTED, 2=COMP_EXPECTED
  - vt_wait_words: words elapsed since the expectation was set

Transitions at word completion:
  - transitive-class VERB / VERB_ED / VERB_ING (from known list):
       → VT_DO_EXPECTED, wait=0
  - linking AUX_VERB (be-forms) / select VERBs (seem/become):
       → VT_COMP_EXPECTED, wait=0
  - NOUN / PROPER_NOUN / PRONOUN → VT_NONE (object resolved)
  - intransitive verb completion (go, come, fall, rise, ...):
       → VT_NONE (no DO required)
  - ARTICLE / POSSESSIVE / ADJECTIVE / NUMBER / WH — these build the
       NP within the DO slot; if expectation active, wait+=1, no reset.
  - PREPOSITION / CONJUNCTION / INTERJECTION / NEGATION: reset to NONE
       (likely a different constituent took over).

Also resets on: ".?!" punctuation, ",;:" mid-clause punctuation,
speaker-turn boundary (\n\n), wait >= 4.

All classification comes from prior knowledge of English — no
corpus statistics.
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

VT_NONE = 0
VT_DO_EXPECTED = 1
VT_COMP_EXPECTED = 2


# Main-verb transitive class (takes a direct object NP).
# Past-tense / participle forms are listed explicitly since the POS
# tagger sends them through POS_VERB_ED.
_TRANSITIVE_VERBS: frozenset[str] = frozenset({
    # Cognition / perception
    "see", "sees", "seen", "saw",
    "know", "knows", "knew", "known",
    "hear", "hears", "heard",
    "feel", "feels", "felt",
    "find", "finds", "found",
    "think", "thinks", "thought",
    "fear", "fears", "feared",
    "love", "loves", "loved",
    "hate", "hates", "hated",
    "want", "wants", "wanted",
    "need", "needs", "needed",
    "wish", "wishes", "wished",
    "remember", "forget", "forgot", "forgotten",
    "mark", "marks", "marked",
    "behold", "beheld",
    "observe", "observes", "observed",
    "regard", "regards", "regarded",
    "suspect", "suspects", "suspected",
    "believe", "believes", "believed",
    "doubt", "doubts", "doubted",
    # Action / manipulation
    "take", "takes", "took", "taken",
    "make", "makes", "made",
    "give", "gives", "gave", "given",
    "bring", "brings", "brought",
    "send", "sends", "sent",
    "keep", "keeps", "kept",
    "hold", "holds", "held",
    "set", "sets",
    "put", "puts",
    "lay", "lays", "laid",
    "cast", "casts",
    "strike", "strikes", "struck",
    "hit", "hits",
    "cut", "cuts",
    "break", "breaks", "broke", "broken",
    "bear", "bears", "bore", "borne",
    "beat", "beats", "beaten",
    "kill", "kills", "killed",
    "slay", "slays", "slew", "slain",
    "wear", "wears", "wore", "worn",
    "tear", "tears", "tore", "torn",
    "draw", "draws", "drew", "drawn",
    "throw", "throws", "threw", "thrown",
    "raise", "raises", "raised",
    "lift", "lifts", "lifted",
    "lose", "loses", "lost",
    "win", "wins", "won",
    "buy", "buys", "bought",
    "sell", "sells", "sold",
    "spend", "spends", "spent",
    # Speech / social
    "tell", "tells", "told",
    "speak", "speaks", "spoke", "spoken",
    "say", "says", "said",
    "ask", "asks", "asked",
    "call", "calls", "called",
    "name", "names", "named",
    "greet", "greets", "greeted",
    "praise", "praises", "praised",
    "curse", "curses", "cursed",
    "command", "commands", "commanded",
    "beseech", "beseeches", "besought",
    "thank", "thanks", "thanked",
    "pardon", "pardons", "pardoned",
    "forgive", "forgives", "forgave", "forgiven",
    "trust", "trusts", "trusted",
    "promise", "promises", "promised",
    # Movement-with-object
    "follow", "follows", "followed",
    "lead", "leads", "led",
    "meet", "meets", "met",
    "seek", "seeks", "sought",
    "hunt", "hunts", "hunted",
    # Reading / writing
    "read", "reads",
    "write", "writes", "wrote", "written",
    # Using
    "use", "uses", "used",
    "serve", "serves", "served",
    # Swearing / performance
    "swear", "swears", "swore", "sworn",
    # Holding, guarding
    "guard", "guards", "guarded",
    "save", "saves", "saved",
    "protect", "protects", "protected",
    "defend", "defends", "defended",
    "answer", "answers", "answered",
    # Thinking
    "consider", "considers", "considered",
    "understand", "understood",
    "mean", "means", "meant",
    "intend", "intends", "intended",
    # Shared with intransitive; favor DO here
    "eat", "eats", "ate", "eaten",
    "watch", "watches", "watched",
})

# Intransitive verbs — do NOT set expectation.
_INTRANSITIVE_VERBS: frozenset[str] = frozenset({
    "go", "goes", "gone", "went",
    "come", "comes", "came",
    "fall", "falls", "fell", "fallen",
    "rise", "rises", "rose", "risen",
    "die", "dies", "died",
    "fly", "flies", "flew", "flown",
    "run", "runs", "ran",
    "sit", "sits", "sat",
    "stand", "stands", "stood",
    "stay", "stays", "stayed",
    "sleep", "sleeps", "slept",
    "wake", "wakes", "woke", "woken",
    "weep", "weeps", "wept",
    "laugh", "laughs", "laughed",
    "cry", "cries", "cried",
    "live", "lives", "lived",
    "swim", "swims", "swam",
    "arrive", "arrives", "arrived",
    "depart", "departs", "departed",
    "enter", "enters", "entered",
    "exit", "exits", "exited",
    "wait", "waits", "waited",
    "work", "works", "worked",  # can be trans too; default intrans
    "play", "plays", "played",  # same
    "grow", "grows", "grew", "grown",
    "shine", "shines", "shone",
    "tremble", "trembles", "trembled",
    "breathe", "breathes", "breathed",
    "sigh", "sighs", "sighed",
    "swoon", "swoons", "swooned",
    "faint", "faints", "fainted",
    "lie", "lies",  # "lie" as stay-prone; "lies" ambiguous
})

# Linking / copular verbs — take a predicative complement (adj or NP).
_LINKING_VERBS: frozenset[str] = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "am", "art",
    "seem", "seems", "seemed", "seeming",
    "become", "becomes", "became", "becoming",
    "appear", "appears", "appeared", "appearing",
    "prove", "proves", "proved", "proven",
    "remain", "remains", "remained",
    # Shakespearean: "thou art", "he is", often followed by adj/N
})


_RESOLVE_POS: frozenset[int] = frozenset({POS_NOUN, POS_PROPER_NOUN, POS_PRONOUN})
_EXTEND_POS: frozenset[int] = frozenset({
    POS_ARTICLE, POS_POSSESSIVE, POS_ADJECTIVE, POS_NUMBER, POS_WH, POS_ADVERB,
})
_RESET_POS: frozenset[int] = frozenset({
    POS_PREPOSITION, POS_CONJUNCTION, POS_INTERJECTION, POS_NEGATION,
})


def update_transitivity(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    vt = state.verb_transitivity
    wait = state.vt_wait_words

    # Reset on sentence-end, clausal break, speaker-turn boundary.
    if ch in ".?!":
        if vt != VT_NONE or wait != 0:
            return state.model_copy(
                update={"verb_transitivity": VT_NONE, "vt_wait_words": 0}
            )
        return state
    if state.consecutive_newlines >= 2 and ch == "\n":
        if vt != VT_NONE or wait != 0:
            return state.model_copy(
                update={"verb_transitivity": VT_NONE, "vt_wait_words": 0}
            )
        return state
    if ch in ",;:" and state.speaker_label_state == 0:
        if vt != VT_NONE or wait != 0:
            return state.model_copy(
                update={"verb_transitivity": VT_NONE, "vt_wait_words": 0}
            )
        return state

    if state.just_finished_word and state.last_completed_word:
        pos = state.last_word_pos
        w = state.last_completed_word

        # Verb completions — set expectation.
        if pos in (POS_VERB, POS_VERB_ED, POS_VERB_ING, POS_AUX_VERB):
            # Linking first (takes precedence for be-forms).
            if w in _LINKING_VERBS:
                vt = VT_COMP_EXPECTED
                wait = 0
            elif w in _TRANSITIVE_VERBS:
                vt = VT_DO_EXPECTED
                wait = 0
            elif w in _INTRANSITIVE_VERBS:
                vt = VT_NONE
                wait = 0
            else:
                # Unknown verb: if -ed/-ing, likely takes a DO if the verb
                # sense is typically transitive. Default: weak DO expectation.
                vt = VT_DO_EXPECTED
                wait = 0
        elif pos in _RESOLVE_POS:
            # Object (or subject of next clause) resolved.
            vt = VT_NONE
            wait = 0
        elif pos in _RESET_POS:
            # Prep/conj/interjection/negation — different constituent.
            vt = VT_NONE
            wait = 0
        elif vt != VT_NONE and pos in _EXTEND_POS:
            # Pre-head modifiers while we're waiting for the object.
            wait = min(wait + 1, 5)
            if wait >= 4:
                vt = VT_NONE
                wait = 0

    if vt != state.verb_transitivity or wait != state.vt_wait_words:
        return state.model_copy(
            update={"verb_transitivity": vt, "vt_wait_words": wait}
        )
    return state
