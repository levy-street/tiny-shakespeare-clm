"""Mid-departure gibberish-close bias.

Reads `state.mid_departure_extension` (see pipeline/mid_departure) —
active only when the current word departed the trie at position 3 or
4 and is now off-trie. This fills the gap in the existing
offtrie_depart and trie_recovery layers, both of which are quiet in
this exact regime.

Escalation schedule (keyed on extension length):

  ext == 1   : quiet — the word just left the trie; could be a
               legitimate unusual inflection. No bias.
  ext == 2   : mild word-end letter boost (e/s/d/t/n/h/r/y).
  ext == 3   : stronger word-end letter boost, light space/comma nudge.
  ext == 4   : word-end letters strong; space now moderately preferred.
  ext == 5   : terminator push dominant.
  ext >= 6   : hard push toward terminator.

We never apply the terminator push until extension >= 3 so that legit
inflection drift (readings -> readingly; wantoning; loved -> lov'd)
isn't clipped prematurely.

Suppression on rare letters j/q/x/z/v is active from ext >= 2 — these
almost never appear mid-word in real English beyond the first 3
letters.

No corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Schedule: (end_letter_scale, terminator_scale, gib_scale)
# Keyed by extension (index), starting at 0.
_SCHEDULE: list[tuple[float, float, float]] = [
    (0.00, 0.00, 0.00),  # 0
    (0.00, 0.00, 0.00),  # 1
    (0.00, 0.00, 0.00),  # 2
    (0.20, 0.00, 0.10),  # 3
    (0.40, 0.05, 0.20),  # 4
    (0.65, 0.18, 0.30),  # 5
    (0.90, 0.40, 0.45),  # 6
    (1.10, 0.70, 0.60),  # 7
    (1.25, 1.00, 0.75),  # 8
    (1.35, 1.35, 0.90),  # 9
    (1.45, 1.70, 1.00),  # 10+
]


_TERMINATORS: dict[str, float] = {
    " ": 1.00,
    ",": 0.55,
    ".": 0.45,
    ";": 0.30,
    ":": 0.20,
    "!": 0.32,
    "?": 0.28,
    "\n": 0.38,
}

_END_LETTERS: dict[str, float] = {
    "e": 1.00, "s": 0.95, "d": 0.85, "t": 0.80, "n": 0.70,
    "h": 0.55, "r": 0.70, "y": 0.55, "l": 0.45, "g": 0.35,
}

_GIB_LETTERS: dict[str, float] = {
    "j": -1.0, "q": -1.0, "x": -1.0, "z": -1.0,
    "v": -0.40, "w": -0.25, "k": -0.30,
}


def mid_departure_bias(
    mid_departure_extension: int,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if mid_departure_extension <= 1:
        return None

    idx = min(mid_departure_extension, len(_SCHEDULE) - 1)
    end_scale, term_scale, gib_scale = _SCHEDULE[idx]

    if end_scale == 0.0 and term_scale == 0.0 and gib_scale == 0.0:
        return None

    vec = [0.0] * VOCAB_SIZE

    for ch, w in _END_LETTERS.items():
        vi = VOCAB_INDEX.get(ch)
        if vi is not None:
            vec[vi] += end_scale * w

    if term_scale > 0.0:
        for ch, w in _TERMINATORS.items():
            vi = VOCAB_INDEX.get(ch)
            if vi is not None:
                vec[vi] += term_scale * w

    if gib_scale > 0.0:
        for ch, w in _GIB_LETTERS.items():
            vi = VOCAB_INDEX.get(ch)
            if vi is not None:
                vec[vi] += gib_scale * w  # w is negative

    return vec
