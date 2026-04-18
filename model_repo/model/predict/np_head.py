"""NP-head-expectation bias.

Reads `state.np_open` and `state.np_wait_words`. When np_open is True,
we're waiting for a head noun to resolve the current noun phrase.
At word-start, we:
  - boost first letters common to concrete nouns
  - boost first letters common to pre-head adjectives (milder)
  - penalize first letters common to new determiners / prepositions
    (don't nest: "of the" is fine — article after preposition — but
    "the of" / "of to" are pathological. Detection here fires after
    the article opens so np_wait_words is 0; we penalize only at
    wait >= 1 to allow the det→adj→noun chain.)

No corpus statistics — letter weights from typical Shakespeare noun
and function-word inventories.

Scale grows with np_wait_words: a long wait indicates drift, so
close pressure should rise.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Common first letters of concrete/animate nouns.
# Weights reflect how often the letter opens a vivid noun in the
# Shakespeare lexicon we expect (heart, hand, head, honour, hour,
# king, queen, knave, knight, light, love, life, lord, lady,
# man, maid, mouth, moon, master, name, night, nature, noble,
# soul, sword, sun, star, sea, stone, shadow, fair, face, fire,
# flower, friend, foe, father, blood, body, breast, breath, blade,
# crown, cloud, court, child, day, death, dagger, earth, eye, ear,
# god, ground, grave, time, tongue, truth, throne, tear, voice,
# virtue, villain, word, world, woman, way, war, wound, rose, rock,
# river, peace, pain, prince, queen).
_NOUN_FIRST_LETTERS: dict[str, float] = {
    "h": 1.0,  # heart/hand/head/honour/hour
    "l": 1.0,  # love/lord/lady/light/life/law/land
    "m": 1.0,  # man/maid/mouth/moon/master/mind
    "s": 1.1,  # soul/sword/sun/star/sea/stone/shadow/son/sister
    "f": 0.9,  # face/fire/flower/friend/father/foe/fall
    "w": 0.9,  # word/world/woman/way/war/wound/will
    "d": 0.8,  # day/death/dagger/daughter/duke/doubt
    "k": 0.7,  # king/knight/knave/knee
    "b": 0.9,  # blood/body/breast/breath/blade/book/beauty
    "c": 0.9,  # crown/cloud/court/child/cheek/country
    "p": 0.8,  # prince/peace/pain/part/power
    "e": 0.7,  # eye/ear/earth/enemy
    "g": 0.6,  # god/ground/grave/girl/grace
    "n": 0.5,  # name/night/nature/noble/news
    "t": 0.6,  # time/tongue/truth/throne/tear/thing
    "r": 0.6,  # rose/rock/river/rest/right/reason
    "v": 0.5,  # voice/virtue/villain/vow
    "q": 0.3,  # queen
    "j": 0.2,  # joy/judge
    "y": 0.3,  # youth/year
}

# Pre-head adjective first letters — these extend the NP without
# resolving it. Small positive weight (we want nouns more, but
# adjectives are fine at wait==0).
_ADJ_FIRST_LETTERS: dict[str, float] = {
    "g": 0.4,  # good/great/gentle/green/golden
    "s": 0.3,  # sweet/sacred/sad/silent/sure
    "f": 0.3,  # fair/false/foul/full
    "d": 0.3,  # dear/dark/deep/dead/divine
    "t": 0.2,  # true/tender
    "l": 0.2,  # little/long/last/late/low
    "h": 0.2,  # high/holy/happy/harsh
    "n": 0.2,  # noble/new/near/naked
    "p": 0.2,  # poor/pale/proud
    "b": 0.3,  # bright/brave/brief/bold/black/base
    "m": 0.2,  # mad/mere/meek
    "o": 0.2,  # old/own
    "y": 0.1,  # young
    "e": 0.2,  # every/evil
    "w": 0.2,  # wise/weak/weary/wild
    "r": 0.2,  # rich/rude/rare
}

# New-determiner/preposition starter letters to SOFT-penalize once
# np_wait_words >= 1 (we've already opened, don't re-open).
# o=of/our/one/on, t=the/to/this/that/these/those/thy/thine/there
# (also many nouns), i=in/into/if, a=a/an/at/after/about/as, m=my,
# h=his/her, u=upon/under, b=by/before/behind, w=with/within.
# We weight these gently because many also start nouns — only penalize
# function-word-dominant starters.
_FUNC_FIRST_LETTERS: dict[str, float] = {
    "o": 0.35,  # of (very common preposition leak)
    "u": 0.25,  # upon/unto/under
    "i": 0.20,  # in/into/if
}


def np_head_wordend_bias(
    np_open: bool,
    np_wait_words: int,
    word_buffer: str,
    letter_run_len: int,
    on_word_trie: bool,
    speaker_label_state: int,
    last_word_pos: int,
) -> list[float] | None:
    """At word-end on-trie, when np_open is True AND the buffer is
    a short function-word (determiner/possessive/preposition), heavily
    penalize sentence-enders, newline, and comma — an NP can't end
    a sentence or line without a head noun.
    """
    if speaker_label_state != 0:
        return None
    if not np_open:
        return None
    if letter_run_len < 1 or not on_word_trie:
        return None
    # Only fire when the buffer is itself a short word that would
    # NOT be a head noun — otherwise np_open would close.
    # We identify by letter count: most closed-class NP-openers are
    # 1-4 letters (a, an, the, of, to, in, on, at, by, my, thy,
    # his, her, our, your, their, its, this, that).
    if len(word_buffer) < 1 or len(word_buffer) > 4:
        return None
    vec = [0.0] * VOCAB_SIZE
    # Sentence-enders: strong penalty.
    pen_strong = 1.0 + 0.3 * min(np_wait_words, 3)
    for ch in ".?!":
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] -= pen_strong
    # Newline: medium penalty (line can enjamb through "the\n___"
    # but it's uncommon).
    if "\n" in VOCAB_INDEX:
        vec[VOCAB_INDEX["\n"]] -= 0.8
    # Comma / semicolon / colon: mild penalty — a clause-break
    # mid-NP is very rare.
    for ch in ",;:":
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] -= 0.5
    return vec


def np_head_start_bias(
    np_open: bool,
    np_wait_words: int,
    speaker_label_state: int,
    last_word_pos: int,
) -> list[float] | None:
    """Return a word-start bias when an NP head is being awaited."""
    if speaker_label_state != 0:
        return None
    if not np_open:
        return None
    # Scale: at wait=0 (just opened, "the ___"), boost adj/noun;
    # at wait=1 ("the good ___"), stronger noun preference over adj;
    # at wait >= 2, aggressive close.
    wait = min(np_wait_words, 3)
    # Noun-boost scales up with wait (need to resolve).
    noun_scale = 0.30 + 0.15 * wait
    # Adj-boost weakens with wait.
    adj_scale = 0.22 if wait == 0 else max(0.12 - 0.04 * wait, 0.02)
    # Penalty on re-openers — mild at wait=0, stronger at wait >= 1.
    func_scale = 0.3 + 0.2 * wait

    vec = [0.0] * VOCAB_SIZE
    for ch, w in _NOUN_FIRST_LETTERS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += noun_scale * w
    for ch, w in _ADJ_FIRST_LETTERS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += adj_scale * w
    for ch, w in _FUNC_FIRST_LETTERS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] -= func_scale * w
    return vec
