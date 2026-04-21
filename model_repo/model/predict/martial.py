"""Predict layer — martial / peaceful register word-start echo.

Reads `state.martial_charge` maintained by pipeline/martial.py.

When charge > +1.3 (scene is in battle / war-speech mode), gently
push first letters typical of martial vocabulary — the same vocabulary
that set the charge in the first place. This reinforces the scene's
lexical texture so the next content word is more likely to continue
the battlefield register.

When charge < -1.0 (scene is peaceful / pastoral), push peaceful
first letters instead.

Magnitudes are modest (0.10 - 0.22). The existing startword /
next_word / trie machinery still makes the concrete decision; this
layer only tilts the distribution toward texture-consistent choices.

No corpus statistics — the martial-starter and peaceful-starter
letter sets come from the same prior-knowledge vocabularies used to
compute the charge.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# First letters of the martial vocabulary (deduped).
# s — sword/steel/strike/slay/soldier/slain/shield/spear/siege/
#     stab/smite
# b — blood/battle/bones/blade/bow/bleed/breach/banner
# w — wound/war/warrior/weapon
# a — arms/armor/arrow/assault/axe/attack
# f — fight/foe/force/fray/fought
# k — kill/knight
# m — march/musket/murder/mail
# h — horse/host/helm
# c — captain/cavalry/cannon/combat/charge/conquer
# i — iron
# d — drum/dagger/duel/defeat
# t — troop/trumpet/triumph
# p — pike/poniard/pierce/power/platoon
# g — general/gore/gash/guns/gauntlet
# l — lance/legion
# e — enemy/ensign
# v — vanquish/victory
# r — rider/rapier/rank/retreat
_MARTIAL_STARTERS: str = "sbwfak"

# First letters of the peaceful vocabulary.
# p — peace/pleasure
# l — love/lute/light/lamb/laughter/lily
# g — gentle/garden/grove
# s — sweet/soft/sleep/slumber/smile/song/spring/summer
# k — kind/kiss
# m — mild/music/melody/morning/meadow/
# h — home/hearth/hope/heal
# f — flower/friend/friendship
# b — bed/bread/bird/bless/benediction/beloved
# r — rest/rose
# d — dove/delight/dawn
# t — tender
# q — quiet
# j — joy/joyful
# n — nightingale
# w — wine/wood
_PEACEFUL_STARTERS: str = "psglm"

_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"


def _letter_push_vec(letters: str, push_lower: float, push_upper: float) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    for ch in letters:
        idx_lo = VOCAB_INDEX.get(ch)
        if idx_lo is not None:
            vec[idx_lo] += push_lower
        idx_hi = VOCAB_INDEX.get(ch.upper())
        if idx_hi is not None:
            vec[idx_hi] += push_upper
    return vec


# Pre-build for martial (applied when charge is strongly positive).
_MARTIAL_VEC_STRONG = _letter_push_vec(_MARTIAL_STARTERS, 0.65, 0.32)
_MARTIAL_VEC_MILD = _letter_push_vec(_MARTIAL_STARTERS, 0.35, 0.17)

# Pre-build for peaceful (applied when charge is strongly negative).
_PEACEFUL_VEC_STRONG = _letter_push_vec(_PEACEFUL_STARTERS, 0.55, 0.27)
_PEACEFUL_VEC_MILD = _letter_push_vec(_PEACEFUL_STARTERS, 0.28, 0.14)


def martial_word_start_bias(
    martial_charge: float,
    letter_run_len: int,
    speaker_label_state: int,
    last_char_class: int,
    word_buffer: str,
) -> list[float] | None:
    # Gate: only at word-start, outside speaker labels, not mid-word.
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None
    if word_buffer:
        return None
    if last_char_class != 1:
        return None

    # Scale magnitude linearly with |charge|, capped. Below a small
    # dead-zone (|charge| < 0.35) no push — avoids noise near zero.
    if martial_charge > 0.35:
        # Positive (martial) side: saturate at charge >= 2.0.
        scale = min(martial_charge / 2.0, 1.0)
        if scale >= 0.65:
            return _MARTIAL_VEC_STRONG
        return _MARTIAL_VEC_MILD
    if martial_charge < -0.35:
        # Negative (peaceful) side: saturate at charge <= -1.5.
        scale = min(abs(martial_charge) / 1.5, 1.0)
        if scale >= 0.65:
            return _PEACEFUL_VEC_STRONG
        return _PEACEFUL_VEC_MILD
    return None
