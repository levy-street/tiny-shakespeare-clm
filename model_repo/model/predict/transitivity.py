"""Predict layer for verb transitivity / object expectation.

Reads state.verb_transitivity (VT_NONE / VT_DO_EXPECTED / VT_COMP_EXPECTED)
and state.vt_wait_words. At word-start outside speaker-label context
and when an expectation is active, bias first-letter choices toward
the family of words that could fill the expected role.

VT_DO_EXPECTED: push toward determiner/possessive starter letters
  (t=the/this/that/thy/thine/thou; m=my/mine; h=his/her/him; a=a/an/all;
  y=your; o=our) plus common noun starter letters (l=lord/love/life;
  s=sword/sun/sir; h=heart/hand/hell; f=fire/face/fool; w=word/world;
  b=blood/body/book; d=day/death; e=eye/ear/earth). Penalize
  preposition/conjunction starters that would defer the object
  (i/u/w for "into"/"upon"/"with", but we must be careful since
  these letters also lead many content words — keep penalties small).

VT_COMP_EXPECTED: push slightly toward adjective starters
  (g=good/great/gentle; f=fair/fine/false; s=sweet/strong/sad;
  d=dear/dead/deep; n=noble/new; t=true; l=like/little; m=mad/merry)
  plus determiners.

Scale fades as vt_wait_words grows (expectation weakens as the NP
is being assembled by pre-head modifiers).

No corpus statistics — weights come from prior knowledge.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

VT_NONE = 0
VT_DO_EXPECTED = 1
VT_COMP_EXPECTED = 2


# Starter letters for direct-object head / NP-opener after transitive verb.
# Values are relative weights, log-normalized into a bias.
_DO_STARTERS: dict[str, float] = {
    # Determiners / possessives (most likely NP opener after a V)
    "t": 1.6,   # the, this, that, thy, thine, thou, thee
    "m": 1.5,   # my, me, mine, more
    "h": 1.4,   # his, her, him, himself
    "a": 1.1,   # a, an, all
    "y": 1.0,   # your, yours, ye
    "o": 0.7,   # our, one
    # Common NP heads
    "l": 1.1,   # lord, love, life, light, lady, land
    "s": 1.2,   # sword, sir, son, sun, soul, self, self-, shame
    "f": 1.0,   # face, friend, fire, fool, father, fair
    "w": 1.0,   # world, word, war, woman, wife, way, woe
    "b": 0.9,   # blood, body, book, beauty, breath
    "d": 0.9,   # day, death, deed, doom, duty, doubt
    "e": 0.7,   # eye, ear, earth, end
    "n": 0.6,   # name, night, noise, nothing
    "p": 0.6,   # peace, power, place, prince
    "c": 0.7,   # court, cause, country, crown, care
    "g": 0.6,   # god, grace, grief, gold, ghost
    "k": 0.5,   # king, knight, knee
    "r": 0.5,   # room, rest, right, rose, rule
    # Pronouns are also legal objects.
    "i": 0.4,   # it
    "u": 0.1,   # us, upon (ambiguous; mild)
}

# Penalties for letters that tend to open non-object constituents
# (prepositions, conjunctions, aux verbs) — they defer the object.
_DO_ANTI_STARTERS: dict[str, float] = {
    # "and", "or", "but" — coordinating conjunctions
    # These rarely appear immediately after a transitive verb because
    # Shakespeare gives the object first. Mild penalty.
    # Note: Do not over-penalize "with", "for", "from", "in", "on",
    # "unto" since post-verb PPs ARE legal ("gave it to him").
    # We penalize only the clearest misfires.
}

# Complement (VT_COMP_EXPECTED) — adjective-heavy starters.
_COMP_STARTERS: dict[str, float] = {
    # Adjective openers
    "g": 1.3,   # good, great, gentle, grave, glad, golden, grievous
    "f": 1.3,   # fair, false, foul, full, faithful, fine, fit, faint
    "s": 1.2,   # sweet, strong, sad, sure, stern, sacred, soft, silly
    "d": 1.1,   # dead, dear, deep, dark, dull, damn, divine
    "n": 1.0,   # noble, new, near
    "t": 1.3,   # the, this, true, too
    "l": 1.0,   # like, little, lame, learn'd, light (adj), lost, loyal
    "m": 1.2,   # my, mine, mad, merry, mild, mortal
    "h": 1.1,   # his, her, happy, holy, hard, high, honest, human
    "a": 0.9,   # a, an, all, alive, afraid
    "w": 0.9,   # well, weary, wild, wise, white, wide
    "b": 0.8,   # bad, brave, bitter, bright, blind, blessed
    "p": 0.8,   # poor, plain, proud, pale, pure, perfect
    "c": 0.8,   # cold, cruel, chaste, clear, common
    "y": 0.7,   # young, your
    "o": 0.6,   # old, our, one, own
    "r": 0.6,   # rich, royal, right, ready, rude
    "e": 0.6,   # earnest, easy, evil, empty
    "u": 0.5,   # unkind, unknown
    "k": 0.5,   # kind, known
    "i": 0.6,   # it, ill, idle, innocent
}


def _build_vec(starters: dict[str, float], global_scale: float) -> list[float]:
    vec = [0.0] * VOCAB_SIZE
    # Baseline small penalty on letters NOT in starters, limited to letters
    # we have opinions on.
    total = sum(starters.values()) if starters else 1.0
    for ch, w in starters.items():
        if ch in VOCAB_INDEX:
            # Log-ratio against uniform-over-listed.
            frac = w / total
            bias = global_scale * (frac * 10.0 - 0.4)
            vec[VOCAB_INDEX[ch]] = bias
            up = ch.upper()
            if up in VOCAB_INDEX:
                vec[VOCAB_INDEX[up]] = bias * 0.55
    return vec


_DO_VEC = _build_vec(_DO_STARTERS, global_scale=0.12)
_COMP_VEC = _build_vec(_COMP_STARTERS, global_scale=0.08)


def transitivity_start_bias(
    verb_transitivity: int,
    vt_wait_words: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return first-letter bias vector when an object/complement is
    expected at the next word-start."""
    if speaker_label_state != 0:
        return None
    if verb_transitivity == VT_NONE:
        return None
    # Fade with wait words — NP is getting built, head letter becomes
    # less constrained.
    if vt_wait_words >= 3:
        return None
    fade = 1.0 if vt_wait_words == 0 else (0.7 if vt_wait_words == 1 else 0.45)

    if verb_transitivity == VT_DO_EXPECTED:
        base = _DO_VEC
    elif verb_transitivity == VT_COMP_EXPECTED:
        base = _COMP_VEC
    else:
        return None

    if fade == 1.0:
        return base
    return [x * fade for x in base]
