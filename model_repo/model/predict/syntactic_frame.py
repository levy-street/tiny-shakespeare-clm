"""Predict layer — syntactic-frame first-letter bias at word-start.

Reads `expected_next_role` and `frame_confidence` from state (set by
pipeline/syntactic_frame.py). Applies a first-letter bias at word-start
positions (letter_run_len == 0, outside speaker label) toward the
first letters of the projected role's typical vocabulary.

The bias is additive with `next_word`, `startword`, `startbigram`, and
the unigram/context layers. It's NOT a replacement — those layers
carry direct word-level knowledge; this layer contributes SYNTACTIC
ROLE coverage for transitions the existing layers miss (especially
three-word phrase completions where the third word's role is
determined by the two-word prefix rather than just the most recent
word).

First-letter maps are from prior knowledge of English vocabulary.
No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Frame enum (mirrored from pipeline/syntactic_frame.py).
FRAME_ANY = 0
FRAME_NOUN = 1
FRAME_ADJ_OR_NOUN = 2
FRAME_NOUN_ONLY = 3
FRAME_DET_OR_POSS = 4
FRAME_VERB_FAMILY = 5
FRAME_VERB_ONLY = 6
FRAME_PREP_OR_CONJ = 7
FRAME_OBJ = 8
FRAME_SUBJ = 9
FRAME_ADV_OR_PREP = 10


# Per-frame first-letter bias tables. Values are relative weights;
# dominant first letters get larger values. Applied as positive-only
# (no penalties) to avoid eating probability mass from other layers.
#
# Each table is lowercase; uppercase is applied at reduced weight
# elsewhere (see code below).

_NOUN_STARTS: dict[str, float] = {
    # Common Shakespeare nouns: heart, hand, head, love, life, lord,
    # king, queen, night, day, death, fate, faith, soul, sun, moon,
    # sword, tongue, eye, ear, word, time, world, mind, man, woman,
    # son, daughter, master, friend, foe, peace, war, blood, bone,
    # grave, tomb, body, breast, face, voice, song.
    "h": 1.00, "l": 0.85, "k": 0.55, "n": 0.55, "d": 0.60, "f": 0.60,
    "s": 0.90, "m": 0.70, "w": 0.65, "t": 0.70, "e": 0.40, "b": 0.60,
    "c": 0.55, "p": 0.50, "g": 0.45, "r": 0.35, "o": 0.25, "v": 0.15,
    "a": 0.25, "y": 0.10, "i": 0.10, "j": 0.05, "q": 0.05, "u": 0.10,
}

_ADJ_STARTS: dict[str, float] = {
    # fair, foul, good, great, gentle, grave, gracious, noble, dear,
    # sweet, sour, bold, brave, brief, wise, weary, cold, cruel, pale,
    # poor, proud, rich, royal, sad, still, strange, strong, true,
    # tender, true, vain, young, young.
    "f": 0.85, "g": 0.80, "d": 0.70, "s": 0.95, "b": 0.75, "w": 0.70,
    "n": 0.40, "p": 0.65, "t": 0.75, "c": 0.55, "r": 0.45, "l": 0.45,
    "m": 0.35, "h": 0.45, "k": 0.20, "v": 0.20, "o": 0.25, "e": 0.30,
    "a": 0.35, "u": 0.15, "y": 0.15, "i": 0.20, "j": 0.05, "q": 0.02,
}

_VERB_STARTS: dict[str, float] = {
    # Action / speech verbs: bear, bring, break, come, call, cast,
    # do, die, draw, fall, find, fight, give, go, gaze, grow, hate,
    # have, hear, hold, keep, kill, know, lay, leave, live, love,
    # look, make, mark, meet, move, pay, play, put, run, rise, see,
    # seek, send, speak, stand, stay, strike, take, tell, think,
    # turn, teach, walk, weep, want, wait.
    "b": 0.80, "c": 0.90, "d": 0.80, "f": 0.75, "g": 0.70, "h": 0.85,
    "k": 0.55, "l": 0.85, "m": 0.60, "p": 0.65, "r": 0.55, "s": 1.00,
    "t": 0.90, "w": 0.85, "e": 0.20, "n": 0.15, "o": 0.10, "a": 0.25,
    "u": 0.10, "v": 0.15, "y": 0.15, "i": 0.10, "j": 0.05, "q": 0.03,
}

_AUX_MODAL_STARTS: dict[str, float] = {
    # Auxiliaries: is, are, was, were, be, been, am, art, hath, has,
    # hast, doth, dost, do, did, have, had.
    # Modals: shall, should, shalt, shouldst, will, would, wilt,
    # wouldst, can, canst, could, couldst, may, mayst, might, must.
    "h": 1.00, "i": 0.80, "a": 0.90, "w": 1.00, "d": 0.85, "s": 0.95,
    "c": 0.85, "m": 0.90, "b": 0.75,
}

_DET_POSS_STARTS: dict[str, float] = {
    # Articles: the, a, an. Possessives: my, mine, thy, thine, his,
    # her, our, your, their.
    "t": 1.00, "a": 0.80, "m": 0.80, "h": 0.75, "o": 0.50, "y": 0.45,
    "n": 0.15,
}

_PREP_CONJ_STARTS: dict[str, float] = {
    # Prepositions: of, to, in, on, at, by, for, with, from, upon,
    # into, unto, through, against, beneath, above, below, within,
    # without, amid, among. Conjunctions: and, but, or, nor, yet,
    # so, if, though, although, because, since, when, while, that.
    "o": 0.75, "t": 0.85, "i": 0.70, "a": 0.75, "b": 0.75, "f": 0.70,
    "w": 0.90, "u": 0.35, "s": 0.60, "n": 0.30, "p": 0.30, "y": 0.25,
}

_SUBJ_STARTS: dict[str, float] = {
    # Subject pronouns: I, thou, he, she, we, ye, they, you, it.
    # Subject determiners / proper nouns at sentence start: the, a,
    # my, his, O, Good, Fair, (names).
    "i": 0.90, "t": 0.95, "h": 0.85, "s": 0.65, "w": 0.75, "y": 0.60,
    "m": 0.55, "o": 0.50, "a": 0.50, "g": 0.40, "f": 0.40, "b": 0.35,
    "n": 0.25, "e": 0.20, "l": 0.25, "p": 0.20, "r": 0.15, "d": 0.20,
    "c": 0.20, "k": 0.25, "v": 0.10, "j": 0.05, "q": 0.05,
}

_OBJ_STARTS: dict[str, float] = {
    # After a verb, common objects: him, her, them, me, thee, it
    # (pronouns), or det+noun starts (the, a, my, his, thy, our,
    # your, their), or proper nouns, or bare nouns.
    "t": 0.90, "a": 0.65, "m": 0.85, "h": 0.85, "o": 0.50, "y": 0.55,
    "i": 0.50, "s": 0.45, "w": 0.35, "l": 0.40, "d": 0.45, "b": 0.40,
    "c": 0.40, "f": 0.45, "g": 0.35, "p": 0.40, "r": 0.30, "n": 0.30,
    "e": 0.25, "k": 0.30, "v": 0.15, "u": 0.10, "j": 0.05, "q": 0.05,
}


_TABLES: dict[int, dict[str, float]] = {
    FRAME_NOUN: _NOUN_STARTS,
    FRAME_ADJ_OR_NOUN: {
        # Weighted mix of noun and adjective starts.
        ch: 0.6 * _NOUN_STARTS.get(ch, 0.0) + 0.4 * _ADJ_STARTS.get(ch, 0.0)
        for ch in set(_NOUN_STARTS) | set(_ADJ_STARTS)
    },
    FRAME_NOUN_ONLY: _NOUN_STARTS,
    FRAME_DET_OR_POSS: _DET_POSS_STARTS,
    FRAME_VERB_FAMILY: {
        # Verbs plus aux/modal.
        ch: 0.55 * _VERB_STARTS.get(ch, 0.0) + 0.45 * _AUX_MODAL_STARTS.get(ch, 0.0)
        for ch in set(_VERB_STARTS) | set(_AUX_MODAL_STARTS)
    },
    FRAME_VERB_ONLY: _VERB_STARTS,
    FRAME_PREP_OR_CONJ: _PREP_CONJ_STARTS,
    FRAME_OBJ: _OBJ_STARTS,
    FRAME_SUBJ: _SUBJ_STARTS,
    FRAME_ADV_OR_PREP: {
        # Adverb / PP starters.
        ch: 0.5 * _PREP_CONJ_STARTS.get(ch, 0.0) + 0.3 * _ADJ_STARTS.get(ch, 0.0)
        for ch in set(_PREP_CONJ_STARTS) | set(_ADJ_STARTS)
    },
}


# Overall scale — kept modest since this STACKS with next_word /
# startword / startbigram. Stronger would over-narrow the distribution.
_MAX_SCALE = 0.80


def syntactic_frame_start_bias(
    expected_next_role: int,
    frame_confidence: float,
    letter_run_len: int,
    speaker_label_state: int,
    last_char: str,
    consecutive_newlines: int,
) -> list[float] | None:
    """Return a positive-only first-letter bias vec for the projected
    role at word-start. None if no projection or no letter to bias.
    """
    if speaker_label_state != 0:
        return None
    if letter_run_len != 0:
        return None
    if expected_next_role == FRAME_ANY:
        return None
    if frame_confidence <= 0.0:
        return None
    # Only fire right after a space — word-start after non-space
    # (sentence-end punctuation, newline + capital) is handled by
    # separate sentence-opener biases.
    if last_char != " ":
        return None
    # At a speaker-label open (consecutive_newlines >= 2) we hand
    # off to speaker_trie.
    if consecutive_newlines >= 2:
        return None

    table = _TABLES.get(expected_next_role)
    if table is None:
        return None

    scale = _MAX_SCALE * frame_confidence
    if scale <= 0.0:
        return None

    vec = [0.0] * VOCAB_SIZE
    for ch, w in table.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
        # Uppercase is rare at non-sentence-start but not zero; add
        # at substantially reduced weight so we don't fight the
        # existing capital-letter sentence biases.
        up = ch.upper()
        if up != ch:
            upi = VOCAB_INDEX.get(up)
            if upi is not None:
                vec[upi] += w * scale * 0.20
    return vec
