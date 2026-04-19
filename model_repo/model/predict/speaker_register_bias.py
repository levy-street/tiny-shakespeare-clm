"""Predict consumer for `state.speaker_register`.

At word-start, apply a small categorical bias over first-letter choices
reflecting the register of the currently-speaking character. Keeps the
bias GENTLE — we're not trying to dominate the context-class / startword /
phrase_bigram layers that already shape word-start choice; we're adding
a soft Bayesian tilt.

Registers:
    1 TRAGIC_NOBLE    — thou/thee/hath/-est forms; abstract/philosophic vocab
                         starting letters ("s","l","t","b","m","n","f")
    2 COMIC_PROSE     — colloquial/direct; "i","y","w","g","a","o","n","f"
    3 ROYAL_FORMAL    — formal address; "m","s","l","t","o","c","g"
    4 VILLAIN         — dark/harsh diction; "b","d","h","m","n","s","c","r"
    5 LOVER_FEMININE  — love/sweetness; "l","h","m","s","f","d","t"
    6 SERVANT_BRIEF   — deferential/short; "m","y","s","g","n","t"
    7 SUPERNATURAL    — incantatory; "h","w","d","b","m","f","t","s"

No corpus statistics — just prior knowledge of each archetype's typical
vocabulary starts.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Per-register first-letter nudges. Values are LOG-BIAS magnitudes
# (positive = boost, negative = penalize). Keep modest — this runs
# ALONGSIDE startword, context_class, phrase_bigram, etc.
_TILT: dict[int, dict[str, float]] = {
    # TRAGIC_NOBLE — Hamlet, Lear, Macbeth. Philosophic/abstract,
    # often archaic.
    1: {
        "t": 0.20, "s": 0.18, "l": 0.14, "m": 0.12, "n": 0.12,
        "f": 0.10, "b": 0.08, "o": 0.10,
        "p": 0.08, "d": 0.07,
    },
    # COMIC_PROSE — Fool, Bottom, Launce. Colloquial, direct, often
    # 'i' / 'y' / 'a' openings.
    2: {
        "i": 0.18, "y": 0.14, "a": 0.14, "w": 0.12, "o": 0.10,
        "g": 0.10, "n": 0.08, "f": 0.08,
        "s": 0.06, "b": 0.06,
    },
    # ROYAL_FORMAL — King / Duke / Prince / Prospero. Elevated address.
    3: {
        "m": 0.18, "s": 0.12, "l": 0.12, "t": 0.12, "o": 0.12,
        "c": 0.08, "g": 0.08, "f": 0.07, "r": 0.07, "n": 0.06,
    },
    # VILLAIN — Iago, Edmund, Richard. Harsh diction, "blood", "damn", etc.
    4: {
        "b": 0.15, "d": 0.13, "h": 0.12, "m": 0.10, "n": 0.08,
        "c": 0.08, "r": 0.08, "s": 0.07, "w": 0.06,
    },
    # LOVER_FEMININE — Juliet, Viola, etc. Love/sweetness vocabulary.
    5: {
        "l": 0.18, "h": 0.14, "m": 0.12, "s": 0.12, "f": 0.10,
        "d": 0.08, "t": 0.08, "o": 0.08, "n": 0.07,
    },
    # SERVANT_BRIEF — Messenger, Citizen, Servant. Deferential short speech.
    6: {
        "m": 0.18, "y": 0.12, "s": 0.12, "g": 0.08, "n": 0.07,
        "t": 0.08, "h": 0.08, "w": 0.07,
    },
    # SUPERNATURAL — Ghost, Witch, Ariel, Puck. Incantatory.
    7: {
        "h": 0.15, "w": 0.12, "d": 0.12, "b": 0.10, "m": 0.08,
        "f": 0.08, "t": 0.08, "s": 0.07,
    },
}


def _build_vec(register: int) -> list[float]:
    tilt = _TILT.get(register)
    if tilt is None:
        return []
    vec = [0.0] * VOCAB_SIZE
    for ch, b in tilt.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] = b
        # Capital variant gets a slightly smaller share — useful for
        # sentence-first words.
        up = ch.upper()
        if up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] = b * 0.6
    return vec


# Precompute per-register vectors.
_VECS: dict[int, list[float]] = {
    r: _build_vec(r) for r in _TILT
}


def speaker_register_start_bias(
    speaker_register: int,
    register_age: int,
    letter_run_len: int,
    speaker_label_state: int,
) -> list[float] | None:
    """Return a word-start bias for the current speaker's register, or
    None when we shouldn't apply one (unknown register, inside speaker
    label, not at word start, or speaker just changed — let the label
    settle).
    """
    if letter_run_len != 0:
        return None
    if speaker_label_state != 0:
        return None
    # First couple tokens after a register change are the speaker label
    # itself plus a newline — don't tilt vocabulary yet.
    if register_age < 3:
        return None
    vec = _VECS.get(speaker_register)
    if not vec:
        return None
    return vec
