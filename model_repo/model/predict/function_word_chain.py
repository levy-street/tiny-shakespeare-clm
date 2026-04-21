"""Predict layer — boost content-word starts when function-word chain is long.

Reads `state.function_word_chain_len` (maintained by
`pipeline/function_word_chain.py`). When the chain has grown to 3+
consecutive function words, Shakespeare grammatically REQUIRES a
content word (noun, verb, adjective, adverb) to arrive soon. Boost
first letters that commonly begin content words and suppress first
letters of common function-word starters.

Fires at word-start (letter_run_len == 0, post-space/post-punct) outside
speaker labels. Scales with chain length so the pressure grows with each
additional function word.

Chain length => behavior:
  0, 1, 2 — no-op (ordinary grammar).
  3       — mild bias toward content starts.
  4       — stronger.
  5+      — dominant push, treating the chain as near-breakdown.

No corpus statistics — the starter letters come from prior knowledge
of English / Shakespearean vocabulary.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Common content-word first letters (noun/verb/adj combined). Values
# are relative weights; scaled at call time.
# Nouns: man, heart, love, life, day, death, king, lord, lady, light,
#        blood, body, bone, mind, soul, god, war, world, hand, head,
#        eye, voice, name, night, word, way, time, fire, friend, foe.
# Verbs: see, say, speak, take, tell, hold, make, know, think, come,
#        go, give, get, grow, find, fight, feel, bring, bear, break.
# Adjectives: good, fair, great, true, false, bold, sweet, dear, noble,
#        sick, gentle, honest, poor, rich, dead, fresh, cold, hot, sharp,
#        brave, dire, deep, dark, bright, long, short, strong, weak.
_CONTENT_STARTS: dict[str, float] = {
    "s": 1.00,  # see/say/speak/soul/sun/son/sweet/sharp/strong
    "l": 0.95,  # love/life/lord/lady/light/long/little/loyal
    "h": 0.95,  # heart/hand/head/honour/heaven/hold/hope/hot/high
    "m": 0.90,  # man/mind/make/meet/move/mercy/mother/mighty
    "t": 0.80,  # take/tell/think/true/tongue/tender/time/twain
    "g": 0.90,  # go/give/get/good/great/god/gold/grace/grief
    "f": 0.95,  # find/feel/fair/fear/friend/father/fine/false/fresh
    "d": 0.90,  # do/draw/day/death/dear/deep/dark/dread/dead
    "b": 0.90,  # bring/bear/body/blood/bone/bold/brave/bright/black
    "k": 0.85,  # know/king/keep/kin/knight/kind/keen
    "c": 0.80,  # come/call/care/child/crown/cold/clean/close
    "p": 0.65,  # pray/pass/power/prince/peace/pain/poor/proud/pale
    "r": 0.70,  # rise/rule/reason/right/red/rude/rank/raw
    "w": 0.70,  # weep/wake/win/wish/word/world/war/wise/weak/wild
    "e": 0.45,  # earth/eye/end/enemy/eager/empty/early/easy
    "n": 0.45,  # night/name/noble/noise/naked/near (note "no"/"nor"/"not" also n—soft)
    "v": 0.40,  # voice/virtue/vile/vow/view/valiant
    "j": 0.10,
    "q": 0.00,
    "x": -0.70,
    "z": -0.40,
}

# Common function-word first letters to suppress. "of/to/in/on/with/
# for/by/at/and/but/or/nor/the/a/an/my/thy/his/her/this/that/these/
# those/I/we/you/he/she/they/it/if/when/where/who/what".
_FUNCTION_STARTS: dict[str, float] = {
    "o": 0.85,  # of, on, or, our
    "t": 0.70,  # to, the, this, that, those, thy, their, than, then
    "i": 0.85,  # in, into, if, I, it
    "w": 0.75,  # with, we, what, which, when, where, who, would, will
    "a": 0.80,  # a, an, and, as, at, any, all, am, are, art
    "b": 0.65,  # by, but, before, because, between, beneath
    "f": 0.65,  # for, from
    "n": 0.60,  # no, not, nor, never, ne'er
    "u": 0.55,  # unto, upon, until, under
    "h": 0.55,  # he, him, his, her, hath, hast, had, has, have
    "y": 0.55,  # you, your, yours, ye, yon, yet
    "m": 0.45,  # my, me, mine
    "s": 0.45,  # she, shall, should, so, since
    "c": 0.35,  # can, could, canst
    "d": 0.35,  # do, does, did, doth, dost
    "e": 0.25,  # ere, every, each
}


def _build_vec(scale: float) -> list[float]:
    """Build a bias vector: + on content starts, - on function starts.

    Function-start penalty is applied only when the content weight for
    the same letter is LESS than the function weight (so we don't
    end up net-negative on letters that robustly begin content words).
    """
    vec = [0.0] * VOCAB_SIZE

    for ch, w in _CONTENT_STARTS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += scale * w
        up = ch.upper()
        if up in VOCAB_INDEX:
            # Mid-line cap rarely starts content — small share.
            vec[VOCAB_INDEX[up]] += scale * w * 0.3

    for ch, w in _FUNCTION_STARTS.items():
        if ch not in VOCAB_INDEX:
            continue
        content_w = _CONTENT_STARTS.get(ch, 0.0)
        # Only penalize where function dominates content.
        if w > content_w + 0.10:
            # Net push downward on that letter.
            net_pen = -(w - content_w) * scale * 0.55
            vec[VOCAB_INDEX[ch]] += net_pen
            up = ch.upper()
            if up in VOCAB_INDEX:
                vec[VOCAB_INDEX[up]] += net_pen * 0.3

    return vec


def function_word_chain_bias(
    function_word_chain_len: int,
    letter_run_len: int,
    last_char_class: int,
    speaker_label_state: int,
    words_in_sentence: int,
) -> list[float] | None:
    """Word-start bias: push content starts when function chain is 3+."""
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None
    # Gate on post-space / post-mid-punct word-start contexts.
    if last_char_class not in (1, 7):
        return None
    if function_word_chain_len < 4:
        return None
    # Skip very short sentences where a content word hasn't had a
    # chance to appear (e.g., "For I" opening — common fragment).
    if words_in_sentence < 3:
        return None

    n = function_word_chain_len
    if n == 3:
        scale = 0.10
    elif n == 4:
        scale = 0.25
    elif n == 5:
        scale = 0.45
    else:  # 6+
        scale = 0.65

    return _build_vec(scale)
