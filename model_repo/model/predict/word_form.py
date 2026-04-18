"""Predict layer — morphological-form (word_form_expectation) biases.

Two consumers of `state.word_form_expectation`:

1. `word_form_start_bias(wfe, wait, slt)` — at word-start, boost the
   first letters most common for words of the expected form.

2. `word_form_midword_bias(wfe, wait, buf, rlen, slt)` — mid-word,
   steer toward form-distinctive endings:
     - PAST_PART: favor "-en", "-n" suffixes; -ed/-d already in trie
     - ING_OR_PP: favor "-ing" suffix continuation
     - INFINITIVE: discourage -ing/-ed suffixes (bare form expected)

All biases return None (no-op) in speaker-label mode or when
expectation is NONE.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

WFE_NONE = 0
WFE_INFINITIVE = 1
WFE_PAST_PART = 2
WFE_ING_OR_PP = 3
WFE_NOMINAL = 4
WFE_COMPARATIVE = 5
WFE_SUPERLATIVE = 6


# ---- Word-start first-letter weights per expectation ----

# Bare-verb infinitive first-letter weights. Common Shakespearean
# infinitives: be, go, see, take, know, make, come, hear, speak,
# think, have, find, live, die, fall, rise, bear, say, tell, hold,
# give, get, keep, leave, let, stay, send, bring, read, run, weep,
# swear, show, wear.
_INF_STARTERS: dict[str, float] = {
    "b": 0.40,  # be, bear, bring, break
    "g": 0.35,  # go, give, get, grow, grant
    "s": 0.55,  # see, say, speak, send, show, stay, swear, stand, sleep, strike
    "t": 0.45,  # take, tell, teach, touch, try
    "k": 0.25,  # know, keep, kill
    "m": 0.35,  # make, meet, mean, mend
    "c": 0.25,  # come, call, cry, crave
    "h": 0.40,  # have, hear, hold, help, hate, hope
    "l": 0.35,  # live, love, leave, let, lend, lie, look
    "d": 0.28,  # die, do, deny, draw, dare
    "f": 0.25,  # fall, find, fear, feel, fly, follow
    "r": 0.25,  # rise, run, rest, read, rend, rule
    "w": 0.30,  # wait, walk, wear, weep, wish, win
    "p": 0.22,  # pray, part, pass, praise, please
    "a": 0.15,  # ask, act, answer, arrive
    "e": 0.12,  # enter, endure, envy
    "o": 0.10,  # offer, open, obey
}

# Past-participle first-letter weights. Strong Shakespearean forms:
# seen, slain, taken, given, gone, known, done, borne, worn, torn,
# drawn, thrown, broken, spoken, stolen, forgotten, forsaken, risen.
# Plus -ed/-d forms: loved, feared, killed, hated, pleased, grieved.
_PP_STARTERS: dict[str, float] = {
    "s": 0.50,  # seen, slain, spoken, stolen, sworn, sent, said, set
    "t": 0.35,  # taken, told, thrown, touched, tried
    "g": 0.45,  # given, gone, got, grown, gained
    "k": 0.22,  # known, killed, kept
    "b": 0.40,  # borne, broken, been, begun, brought, built
    "d": 0.28,  # done, died, drawn, dared
    "f": 0.30,  # forgotten, forsaken, found, fallen, feared
    "l": 0.28,  # lost, loved, lived, led, left
    "h": 0.30,  # heard, held, hurt, hated
    "m": 0.25,  # made, met, mourned
    "c": 0.25,  # come, called, crowned
    "w": 0.25,  # won, worn, wept, wrought, wished
    "r": 0.28,  # risen, run, read, ruled
    "p": 0.20,  # pleased, proved, passed, poured
}

# Present-participle + past-participle combined (copula context).
# Includes -ing candidates which may be slightly less common as heads.
_ING_PP_STARTERS: dict[str, float] = {
    "s": 0.35, "t": 0.25, "g": 0.28, "b": 0.30, "d": 0.22,
    "f": 0.22, "l": 0.25, "h": 0.22, "m": 0.20, "c": 0.22,
    "w": 0.22, "r": 0.22, "p": 0.20, "k": 0.15,
}

# Nominal (after "of") head-noun first-letter weights. Heavy hitters:
# love, death, war, honour, heart, life, time, God, heaven, fate,
# grace, blood, soul, spirit, kingdom, peace, wrath, fortune, might,
# name, lord, king, queen, truth, youth, age, song, state.
_NOMINAL_STARTERS: dict[str, float] = {
    "l": 0.55,  # love, life, lord, lady, light
    "h": 0.55,  # honour, heart, heaven, her, him, hope
    "d": 0.40,  # death, day, dread, duty
    "w": 0.40,  # war, world, wrath, woe, wonder
    "s": 0.45,  # soul, spirit, state, song, sun, storm, sky
    "t": 0.40,  # time, truth, throne, tears, thy
    "g": 0.35,  # grace, god, glory, gold, good
    "f": 0.40,  # fortune, fate, fire, force, friend, faith
    "k": 0.30,  # king, kingdom, knight
    "b": 0.30,  # blood, body, beauty, battle
    "m": 0.35,  # might, mind, mercy, marriage, music
    "p": 0.30,  # peace, pride, power, praise, passion
    "n": 0.30,  # name, night, nature, noble
    "y": 0.20,  # youth, yesterday
    "a": 0.25,  # age, arm, arms, art
    "e": 0.18,  # eye, ear, earth, England
    "r": 0.22,  # rage, right, realm, rose
    "c": 0.22,  # cause, country, crown, court, comfort
    # thy/thine/this/these/my (possessives) get a modest share too
    "o": 0.15,  # our, one
}

# Comparative/superlative: after "more"/"less"/"most" — adjective/adverb.
_COMP_STARTERS: dict[str, float] = {
    "g": 0.35,  # great, good, grievous, gentle
    "f": 0.35,  # fair, fine, fond, faithful, fearful
    "n": 0.30,  # noble, natural
    "d": 0.30,  # dear, deep, divine, dread
    "s": 0.30,  # sweet, sacred, strong, strange
    "t": 0.25,  # true, tender, terrible
    "b": 0.28,  # bright, bold, blessed, bitter, beautiful
    "h": 0.28,  # high, happy, holy, heavy, humble
    "w": 0.22,  # wise, worthy, weak, wondrous
    "m": 0.22,  # merry, mighty, meek, marvelous
    "l": 0.20,  # lovely, low, light
    "p": 0.20,  # pure, proud, poor, precious
    "r": 0.20,  # rare, rich, right, royal
    "k": 0.18,  # kind, keen
    "c": 0.20,  # clear, cold, cruel
    "e": 0.18,  # excellent, evil, earnest
}


def _build_vec(weights: dict[str, float], scale: float) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch, w in weights.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] = w * scale
    return vec


def word_form_start_bias(
    wfe: int,
    wfe_wait_words: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if wfe == WFE_NONE:
        return None
    # Stale expectation after a few modifiers — cap influence.
    if wfe_wait_words >= 4:
        return None

    if wfe == WFE_PAST_PART:
        # This is the highest-signal expectation: after have/has/had,
        # a past participle is very strongly expected.
        weights = _PP_STARTERS
        scale = 0.85
    else:
        # NOMINAL/INFINITIVE/ING_OR_PP/COMP/SUPER — all regressed
        # in isolation; gate off and keep state machine intact for
        # future refinements (e.g., mid-word ending biases).
        return None

    # Decay with wait: as modifiers stack up, the prior weakens.
    if wfe_wait_words == 1:
        scale *= 0.75
    elif wfe_wait_words == 2:
        scale *= 0.55
    elif wfe_wait_words == 3:
        scale *= 0.35

    return _build_vec(weights, scale)


# ---- Mid-word ending biases ----

# For PAST_PART, after a word buffer of length 2-3, boost letters that
# complete -en/-n/-ed endings.

# Mid-word PP-ending bias: biases toward -en/-n and -ed completions.
# Only fires when WFE_PAST_PART is active and buffer is 3+ letters.
#
# Key suffix shapes for past participles in Shakespeare:
#   -en:  seen, taken, given, gone→(not -en), broken, spoken, stolen,
#         borne, worn, torn, drawn, thrown, fallen, forgotten, forsaken,
#         risen, woken, chosen
#   -n (not after e): gone, slain, known, done, drawn, worn, torn, borne
#   -ed:  loved, feared, killed, hated, pleased, grieved, cursed

_PP_ENDING_2: dict[str, float] = {
    # After 2 letters, the 3rd-letter choice slightly favors suffix
    # progression: e.g., "ta" → "k" (taken), "gi" → "v" (given).
    # But at 2 letters, suffix bias is weak — many words have yet
    # to diverge from their base forms.
}

def word_form_midword_bias(
    wfe: int,
    wfe_wait_words: int,
    word_buffer: str,
    letter_run_len: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if wfe != WFE_PAST_PART:
        return None
    if wfe_wait_words >= 3:
        return None
    if letter_run_len < 3 or letter_run_len > 6:
        return None
    if not word_buffer:
        return None

    wb = word_buffer
    vec = [0.0] * VOCAB_SIZE

    # Common ending transitions (hand-specified from known Shakespeare
    # past participles). The premise: at 3-5 letters into a PP context,
    # the next letter is more likely to be part of a -en/-n or -ed
    # completion than in an average mid-word position.
    last = wb[-1] if wb else ""
    last2 = wb[-2:] if len(wb) >= 2 else ""

    # Boost "e" after many 3-letter PP stems (seen/take/giv/brok/spok...
    # trajectories land on "e" next):
    #   s-e-e: already at e
    #   t-a-k, t-o-o: → e
    #   g-i-v: → e
    #   b-o-r, b-r-o: → n/e
    #   s-p-o: → k (spoken); w-o-r: → n (worn)
    # Rough heuristic: after a vowel-consonant pair at letters 2-3,
    # tilt toward "e" (for -en).
    if letter_run_len == 3:
        # 3 letters in. Tilt toward "n" or "e" depending on the
        # stem's last letter.
        if last in "rlwvk":
            # e.g., "wor", "tor", "dra", "bor" → n (worn/torn/drawn/borne)
            vec[VOCAB_INDEX["n"]] += 0.65
        if last in "kbvz":
            # e.g., "tak", "brok", "giv", "spok", "sto" → e
            vec[VOCAB_INDEX["e"]] += 0.42
    elif letter_run_len == 4:
        # 4 letters in. Tilt toward "n" completing -en.
        if last == "e":
            vec[VOCAB_INDEX["n"]] += 0.80
        # Also tilt toward "d" for -ed completions on common stems.
        if last in "relod":
            vec[VOCAB_INDEX["d"]] += 0.40
    elif letter_run_len == 5:
        if last == "e":
            vec[VOCAB_INDEX["n"]] += 0.45

    # If no component bumped, return None to signal no-op.
    any_nonzero = any(v != 0.0 for v in vec)
    if not any_nonzero:
        return None
    return vec
