"""Block back-to-back finite verbs inside a single clause.

Closes a specific concrete failure visible in samples:

    "Mourn contemplate,"        — two imperatives without a connector
    "wash enlarge her m's"      — two verbs back-to-back
    "Rest ya we dotage"          — imperative + pronoun + pronoun + verb
    "does titania Tumble"        — aux + proper + verb (verb-chain)

English never places two finite lexical verbs directly adjacent without
an intervening coordinator ("Mourn and contemplate"), conjunction
("Mourn, then contemplate"), or a non-finite form ("Mourn to
contemplate"). The existing `clause_skel` VERB_DONE penalty on verb-
starter letters is swamped by other signals because its penalty
formula compares against overlapping DET/PREP starter weights for the
same letters.

This layer fires ONLY when:
  * speaker_label_state == 0
  * letter_run_len == 0 (at word-start)
  * last_char_class is SPACE or NEWLINE (not post-comma/semicolon —
    a comma signals a list and allows verb enumeration)
  * clause_skel in {VERB_DONE, CLAUSE_DONE} — we already have a
    finite main verb in this clause
  * last_word_pos is one of POS_VERB / POS_VERB_ED / POS_VERB_ING —
    the PREVIOUS WORD was a lexical verb (not aux/modal — those
    legitimately chain to a verb as in "do speak", "will come")

It penalizes word-start letters that are heavily verb-specific —
letters that are common lexical-verb onsets but rare noun/det/prep/
pronoun onsets. Uppercase variants are penalized in parallel (for
line-initial imperative-verb starts after a \\n, which is where
speakers begin new verses).

The layer is a blocker, not a commit: it pushes probability away
from verb-starting letters without actively pushing toward any
specific non-verb path. Downstream layers (clause_skel, np_head,
syntactic_frame) already push toward legal complements.

No corpus statistics. Letter list is hand-graded from Early-Modern
English lexical-verb onsets with overlap analysis against other POS.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# POS tag codes — must match pipeline/pos.py.
POS_VERB = 16
POS_VERB_ING = 11
POS_VERB_ED = 12

# Linguistic class codes — must match pipeline/linguistic.py.
SPACE = 5
NEWLINE = 6

# Verb-specific onset letters — letters that begin many lexical verbs
# but relatively few nouns/dets/prons/preps. Weights are a relative
# "verb-lexicon-weight" based on prior knowledge of Early-Modern
# English. We deliberately exclude:
#   't' — thou/thee/that/thy/the/to (pronoun/det/prep heavy)
#   'a' — a/an/and/art (det/conj/aux heavy)
#   'i' — i/in/is/if (pronoun/prep/aux heavy)
#   'o' — o/of/on/oh (particle/prep heavy)
#   'w' — with/what/when/who (prep/wh heavy)
#   'h' — he/his/her/have/hath (pronoun/aux heavy)
#   'm' — my/me/may/must (poss/pronoun/modal heavy)
#   'y' — you/ye/your (pronoun heavy)
#   'n' — no/not/nor (neg/conj heavy)
#   'b' — but/be (conj/aux heavy)
#   's' — she/sir/so/shall (pronoun/voc/conj/modal heavy)
#   'd' — do/dost/doth/did (aux heavy)
_VERB_ONSETS: dict[str, float] = {
    "c": 0.90,   # come, call, care, carry, cast, charge, chose, cry
    "f": 0.90,   # fly, fight, fall, fear, feed, find, fire, follow
    "g": 0.90,   # go, give, grant, grow, gather, gaze, grieve
    "k": 0.85,   # know, keep, kiss, kill, kneel
    "l": 0.85,   # look, let, live, love, leave, lie, lead, learn
    "p": 0.75,   # pray, press, prove, put, pull, point, pass, pluck
    "r": 0.80,   # read, rise, run, render, remember, reveal, ride
    "v": 0.65,   # vow, vex, vanquish, venture, visit
}


def verb_chain_block_bias(
    letter_run_len: int,
    last_char_class: int,
    clause_skel: int,
    last_word_pos: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None
    if last_char_class not in (SPACE, NEWLINE):
        return None
    # Only fire in the "main-verb already seated" states. Not in
    # COMP_DUE (complement coming, not another verb) and not in EMPTY
    # (no verb has been seen yet — imperative start is legitimate).
    if clause_skel not in (3, 5):  # VERB_DONE, CLAUSE_DONE
        return None
    if last_word_pos not in (POS_VERB, POS_VERB_ING, POS_VERB_ED):
        return None

    # Strength: firm but not overwhelming. The signal is high-confidence
    # (both previous-POS and clause_skel agree), but BPC must survive
    # legitimate edge cases ("come speak to him") — rare but real.
    scale = 1.15

    vec = [0.0] * VOCAB_SIZE
    for ch, w in _VERB_ONSETS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] -= scale * w
        up = ch.upper()
        if up in VOCAB_INDEX:
            # Line-initial uppercase imperative chain ("Mourn\nContemplate,")
            # — penalize equally; also caught by line-opener POS memory.
            vec[VOCAB_INDEX[up]] -= scale * w * 0.85
    return vec
