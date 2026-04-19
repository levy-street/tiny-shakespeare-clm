"""Tier 2 — sentence-level tense register tracker.

Classifies the FIRST finite verb of each sentence into PAST / PRESENT
/ FUTURE / UNSET, and maintains that classification across subsequent
word completions in the same sentence.

The classification uses hand-curated lists of Early Modern English and
modern verbs. No corpus statistics.

Update rules:
  - If the previous char was sentence-end punctuation (. ? !) or we
    just crossed a speaker-turn boundary, reset sentence_tense to 0.
  - At each just_finished_word:
    - If sentence_tense == 0 (still unset): attempt to classify the
      last_completed_word as a finite verb; if classifiable, set.
    - If sentence_tense != 0: increment sentence_tense_age.

Must run after update_basic_counters so last_completed_word,
just_finished_word, and chars_since_sentence_end are fresh.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


TENSE_UNSET = 0
TENSE_PAST = 1
TENSE_PRESENT = 2
TENSE_FUTURE = 3


# Suppletive past-tense verbs (irregular past forms from hand knowledge).
_PAST_IRREGULAR: frozenset[str] = frozenset({
    "was", "wast", "were", "wert", "had", "hadst",
    "did", "didst",
    "said", "saidst",
    "went", "wentst",
    "came", "camest",
    "saw", "sawest",
    "knew", "knewst",
    "gave", "gavest",
    "took", "tookst",
    "made", "madest",
    "told", "toldst",
    "heard", "heardst",
    "thought", "thoughtst",
    "fell", "fellst",
    "found", "foundst",
    "lost", "lostst",
    "wrote", "wrotest",
    "stood", "stoodst",
    "grew", "grewst",
    "threw", "threwst",
    "sat", "satst",
    "drew", "drewst",
    "bore", "borest",
    "tore", "torest",
    "wore", "worest",
    "spoke", "spakest", "spake",
    "rose", "rosest",
    "broke", "brokest", "brake",
    "slew", "slewst", "slept", "kept", "felt", "meant", "dealt",
    "held", "built", "sent", "meant", "left", "shot", "shook",
    "ran", "rang", "drank", "sang", "sank", "sprang", "swam",
    "wept", "swept", "bled", "fed", "led", "read", "spread",
    "forgot", "begot", "got", "gotst", "beheld", "beheldst",
    "stole", "bore", "chose", "wove", "froze", "arose",
    "outgrew", "forsook", "forbore", "forgot", "forgave",
    "forswore", "withdrew", "withstood", "understood",
    "began", "begat", "fought", "brought", "bought", "sought", "taught",
    "caught", "wrought", "hied", "shone", "struck", "stuck",
    "clad", "hid", "bid", "bade", "forbad", "forbade",
    "slid", "rid", "rode", "strode", "wrung", "stung",
    "flung", "clung", "hung", "sprung", "sung", "sunk",
    "wept", "leapt", "crept", "swept", "dwelt", "spelt",
    "burnt", "learnt", "dreamt", "leant", "knelt", "meant",
})

# Modal past (subjunctive-leaning):
_MODAL_PAST: frozenset[str] = frozenset({
    "would", "wouldst",
    "should", "shouldst",
    "could", "couldst",
    "might", "mightst",
    "must",  # always present-leaning, but stative — skip
})

# Present-tense markers (3sg / be-forms / aux).
_PRESENT_IRREGULAR: frozenset[str] = frozenset({
    "is", "am", "are", "art",
    "be", "been", "being",  # weak — but "be" is often infinitival
    "has", "hath", "have", "hast",
    "do", "does", "doth", "dost",
    "says", "saith",
    "goes", "goeth",
    "comes", "cometh",
    "sees", "seeth",
    "knows", "knoweth",
    "takes", "taketh",
    "makes", "maketh",
    "gives", "giveth",
    "speaks", "speaketh",
    "loves", "loveth",
    "hears", "heareth",
    "thinks", "thinketh",
    "feels", "feeleth",
    "holds", "holdeth",
    "finds", "findeth",
    "stands", "standeth",
    "tells", "telleth",
    "brings", "bringeth",
    "sends", "sendeth",
    "keeps", "keepeth",
    "leaves", "leaveth",
    "means", "meaneth",
    "lives", "liveth",
    "dies", "dieth",
    "seems", "seemeth",
})

# Future/modal aux (bare — subject + will/shall + base verb).
_FUTURE_MODAL: frozenset[str] = frozenset({
    "will", "wilt",
    "shall", "shalt",
})


def _classify_tense(word: str) -> int:
    """Return TENSE_{PAST,PRESENT,FUTURE,UNSET} for a lowercased word."""
    if not word:
        return TENSE_UNSET
    w = word.lower()
    # Strip leading/trailing apostrophes.
    if w.startswith("'"):
        w = w[1:]
    if w.endswith("'"):
        w = w[:-1]
    if not w:
        return TENSE_UNSET

    if w in _FUTURE_MODAL:
        return TENSE_FUTURE
    if w in _MODAL_PAST:
        return TENSE_PAST
    if w in _PAST_IRREGULAR:
        return TENSE_PAST
    if w in _PRESENT_IRREGULAR:
        return TENSE_PRESENT

    # Regular -ed past tense. Need length guard: "red", "fed", "led",
    # "bed" are not past. Require >= 4 letters and preceded by a
    # consonant for -ed (walked, loved, feared).
    if len(w) >= 4 and w.endswith("ed"):
        # Skip obvious non-verbs that coincidentally end -ed:
        # "red", "bed", "fed" are ≤ 3 letters and don't hit this.
        # "need", "feed", "seed" end -ed with a long-e vowel; we guard
        # against these by checking the char before -ed: if it's a
        # vowel (ee, ae, ie, oe, ue), skip classification.
        pre = w[-3]
        if pre not in "aeiou":
            return TENSE_PAST
        # -eed often a noun (seed, need, heed, speed, feed). Don't
        # classify as past to avoid mis-setting.
        return TENSE_UNSET

    # -eth suffix → Early Modern 3sg present (speaketh, loveth, hath).
    if len(w) >= 5 and w.endswith("eth"):
        return TENSE_PRESENT
    # -s with preceding consonant → possibly 3sg present (speaks, loves,
    # feels). But MANY -s words are plural nouns (lords, words, kings).
    # Only classify if sufficiently verb-shaped: length >= 4 AND
    # not in a very short list of obvious plurals. Skip for safety —
    # too noisy.

    return TENSE_UNSET


def update_tense(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-turn boundary: fresh slate.
    if ch == "\n" and state.consecutive_newlines >= 2:
        if state.sentence_tense != 0 or state.sentence_tense_age != 0:
            return state.model_copy(update={
                "sentence_tense": 0,
                "sentence_tense_age": 0,
            })
        return state

    # Sentence boundary: reset. `chars_since_sentence_end == 0`
    # indicates we just emitted . ? !.
    if state.chars_since_sentence_end == 0 and state.tokens_seen > 0:
        if state.sentence_tense != 0 or state.sentence_tense_age != 0:
            return state.model_copy(update={
                "sentence_tense": 0,
                "sentence_tense_age": 0,
            })
        return state

    # Skip speaker-label territory.
    if state.speaker_label_state != 0:
        return state

    # Only update at word completion.
    if not state.just_finished_word or not state.last_completed_word:
        return state

    # If tense already set, just age it.
    if state.sentence_tense != 0:
        new_age = state.sentence_tense_age + 1
        if new_age > 20:
            new_age = 20
        if new_age != state.sentence_tense_age:
            return state.model_copy(update={"sentence_tense_age": new_age})
        return state

    # Tense unset — try to classify.
    t = _classify_tense(state.last_completed_word)
    if t != 0:
        return state.model_copy(update={
            "sentence_tense": t,
            "sentence_tense_age": 0,
        })
    return state
