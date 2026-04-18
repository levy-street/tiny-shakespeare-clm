"""Verb-chain suppression layer.

When the recent POS history shows a main verb already filled in the
current clause slot (verb_chain_len >= 1), penalize first letters of
common main-verb starters at the next word-start. Prevents sample
patterns like "Sail roar endanger" — three main verbs in a row, which
virtually never occur in real Shakespeare.

Triggers at word-start, outside speaker-label territory, when
state.verb_chain_len >= 1 and the clause slot is HAS_VERB (2) or later.

Importantly: does NOT penalize AUX/MODAL starter letters, since
legitimate chains like "had gone", "would have seen", "dost wish"
stack an aux/modal onto a prior verb. Those are handled by the
verb_chain_len being transparent to AUX/MODAL.

All weights from prior knowledge of English — no corpus stats.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# First letters of common Shakespearean MAIN verbs (imperative/finite).
# These get penalized when another main verb just filled the slot.
# Weight per-letter is a rough relative frequency of main-verb starts.
_MAIN_VERB_STARTERS: dict[str, float] = {
    "s": 1.0,   # say/see/speak/stand/strike/sleep/send/seek
    "t": 0.9,   # take/tell/think/turn/touch/trust
    "l": 0.9,   # look/leave/love/live/lie/lay/learn/let
    "g": 0.8,   # go/give/get/grow/grant/guard/greet
    "m": 0.8,   # make/meet/mark/move/mourn
    "k": 0.8,   # know/kill/keep/kiss/kneel
    "f": 0.8,   # fall/find/feel/fight/fear/follow/fly/forget
    "b": 0.7,   # bring/beat/bear/break/build/bow/bleed/bid
    "r": 0.7,   # run/rise/rule/remain/remember/reach/render
    "c": 0.7,   # come/call/carry/catch/cast/close/clasp/cry
    "p": 0.6,   # put/pray/pass/praise/pluck/pour
    "w": 0.5,   # walk/weep/wake/wish/wait/wear/watch (also many function)
    "d": 0.4,   # draw/drink/drive/die (but many aux too — low)
    "h": 0.4,   # hold/heal/hear/hate/hang/hurl (but many aux too — low)
    "a": 0.4,   # answer/arrive/ask/attend
    "e": 0.4,   # eat/enter/end/entreat
    "j": 0.3,   # join/judge
    "u": 0.3,   # use/urge/utter
    "o": 0.3,   # open/offer/owe/order
}


def _build_bias() -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch, w in _MAIN_VERB_STARTERS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = -w
        up = ch.upper()
        if up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] = -w * 0.5
    return vec


_BIAS = _build_bias()


def verb_chain_bias(
    verb_chain_len: int,
    clause_slot: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a penalty vector over main-verb first letters, or None.

    Active only at word-start outside speaker labels when we just
    completed a main verb (verb_chain_len >= 1) and the clause has
    a verb slot filled (slot >= 2 HAS_VERB).
    """
    if speaker_label_state != 0:
        return None
    if verb_chain_len < 1:
        return None
    # Scale with chain length: 1 → mild nudge, 2+ → strong.
    # Kept mild at len=1 because legitimate "came and went", "rose and
    # fell" patterns still exist and we shouldn't crush them.
    if verb_chain_len == 1:
        scale = 0.12
    elif verb_chain_len == 2:
        scale = 0.40
    else:
        scale = 0.60
    return [x * scale for x in _BIAS]
