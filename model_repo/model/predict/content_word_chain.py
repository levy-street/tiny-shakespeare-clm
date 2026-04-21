"""Predict layer — push clause glue when content-word streak is long.

Reads `state.content_word_streak` (maintained by
`pipeline/content_word_chain.py`). When the streak has grown to 3+
consecutive content words, Shakespeare grammatically NEEDS either a
function word (preposition, conjunction, pronoun, determiner) to bind
them, or a clausal pause (comma, semicolon, colon, period).

Fires at word-start (letter_run_len == 0, post-space/post-punct) outside
speaker labels — boosts function-word-initial letters and penalizes
the most content-heavy starting letters.

Also, when the streak is large and we are at a just-finished-word
boundary that hasn't yet added a new space/letter, boost mid-clause
punctuation directly (comma, semicolon).

Streak length => behavior:
  0, 1, 2 — no-op (ordinary grammar).
  3       — mild bias toward function starts + comma.
  4       — stronger.
  5+      — dominant push.

This is the mirror / complement of `function_word_chain_bias` — together
they constrain grammatical breakdown from both directions.

No corpus statistics — the starter letters come from prior knowledge
of English / Shakespearean grammar.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Function-word first letters — values are relative weights for the
# starter push at word-start.
# of, to, in, on, or, a, an, and, as, at, any, all, am, are, art,
# with, we, what, which, when, where, who, would, will,
# by, but, before, because, between, beneath,
# for, from,
# no, not, nor, never, ne'er,
# unto, upon, until, under,
# he, him, his, her, hath, hast, had, has, have,
# you, your, ye, yon, yet,
# my, me, mine,
# she, shall, should, so, since,
# can, could, canst,
# do, does, did, doth, dost,
# ere, every, each,
# if, into, I, it,
# the, this, that, those, thy, their, than, then,
# our, of, on, or.
_FUNCTION_STARTS: dict[str, float] = {
    "o": 0.95,  # of, on, or, our
    "t": 0.80,  # to, the, this, that, those, thy, their, than, then
    "i": 0.90,  # in, into, if, I, it
    "w": 0.80,  # with, we, what, which, when, where, who, would, will
    "a": 0.85,  # a, an, and, as, at, any, all, am, are, art
    "b": 0.70,  # by, but, before, because, between, beneath
    "f": 0.65,  # for, from
    "n": 0.55,  # no, not, nor, never, ne'er
    "u": 0.65,  # unto, upon, until, under
    "h": 0.55,  # he, him, his, her, hath, hast, had, has, have
    "y": 0.60,  # you, your, yours, ye, yon, yet
    "m": 0.40,  # my, me, mine
    "s": 0.35,  # she, shall, should, so, since
    "c": 0.30,  # can, could, canst
    "d": 0.25,  # do, does, did, doth, dost
    "e": 0.25,  # ere, every, each
}

# Content-word first letters to suppress — the letters that dominate
# noun/verb/adjective openings (see function_word_chain for the
# inverse). We want LESS mass on these when the streak is too long.
_CONTENT_STARTS: dict[str, float] = {
    "s": 0.95,  # see/say/speak/soul/sun/son/sweet/sharp/strong
    "l": 0.95,  # love/life/lord/lady/light/long/little/loyal
    "h": 0.85,  # heart/hand/head/honour/heaven/hold/hot/high (also function h)
    "m": 0.85,  # man/mind/make/meet/move/mercy/mother/mighty
    "t": 0.60,  # take/tell/think/true/tongue (also function t)
    "g": 0.95,  # go/give/get/good/great/god/gold/grace/grief
    "f": 0.75,  # find/feel/fair/fear/friend/father (also function f)
    "d": 0.85,  # do/draw/day/death/dear/deep/dark/dread/dead
    "b": 0.70,  # bring/bear/body/blood/bone/bold/brave (also function b)
    "k": 0.90,  # know/king/keep/kin/knight/kind/keen
    "c": 0.80,  # come/call/care/child/crown/cold/clean
    "p": 0.75,  # pray/pass/power/prince/peace/pain/poor
    "r": 0.80,  # rise/rule/reason/right/red/rude/rank
    "w": 0.55,  # weep/wake/win/wish/word/world/war (also function w)
    "v": 0.55,  # voice/virtue/vile/vow/view/valiant
    "e": 0.35,  # earth/eye/end/enemy
    "n": 0.35,  # night/name/noble
}


def _build_vec(scale: float) -> list[float]:
    """Build word-start bias vector: + on function starts, - on content starts.

    Only penalize a content letter if the content weight exceeds the
    function weight for that letter — so we don't net-penalize letters
    that robustly begin function words (like 'o'/'i'/'a').
    """
    vec = [0.0] * VOCAB_SIZE

    for ch, w in _FUNCTION_STARTS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += scale * w
        up = ch.upper()
        if up in VOCAB_INDEX:
            # Mid-line cap rarely starts a function word.
            vec[VOCAB_INDEX[up]] += scale * w * 0.2

    for ch, w in _CONTENT_STARTS.items():
        if ch not in VOCAB_INDEX:
            continue
        fn_w = _FUNCTION_STARTS.get(ch, 0.0)
        if w > fn_w + 0.10:
            net_pen = -(w - fn_w) * scale * 0.50
            vec[VOCAB_INDEX[ch]] += net_pen
            up = ch.upper()
            if up in VOCAB_INDEX:
                vec[VOCAB_INDEX[up]] += net_pen * 0.3

    return vec


def _build_punct_vec(scale: float) -> list[float]:
    """At word boundary, boost mid-clause punctuation to let it close."""
    vec = [0.0] * VOCAB_SIZE
    for ch, w in ((",", 1.00), (";", 0.55), (":", 0.35), (".", 0.45)):
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += scale * w
    return vec


def content_word_chain_bias(
    content_word_streak: int,
    letter_run_len: int,
    last_char_class: int,
    speaker_label_state: int,
    words_in_sentence: int,
    just_finished_word: bool,
) -> list[float] | None:
    """Word-start bias: push function starts + mid-punct when content streak is 3+."""
    if speaker_label_state != 0:
        return None
    if content_word_streak < 2:
        return None

    n = content_word_streak
    if n == 2:
        # 2 content words already; nudge word 3 toward function/punct.
        # Gentle so that legitimate 3-content runs ("good night
        # sweet prince") still sample naturally.
        scale = 0.18
    elif n == 3:
        scale = 0.50
    elif n == 4:
        scale = 0.85
    elif n == 5:
        scale = 1.20
    else:  # 6+
        scale = 1.50

    # Case 1: at word-start (letter_run_len == 0, post-space/mid-punct).
    if letter_run_len == 0 and last_char_class in (1, 7):
        return _build_vec(scale)

    # Case 2: just finished a word, about to emit space or punct. Boost
    # comma/semi/period to close the streak before the next word.
    if just_finished_word and letter_run_len == 0:
        # only-space follows — handled by the default word-start gate;
        # here we boost the comma/etc as an alternative pathway.
        return _build_punct_vec(scale * 0.5)

    return None
