"""Oath-mode predict layer.

Reads the flow-tier `oath_mode` float [0, 1] (maintained by pipeline
flow) plus the structural context (last completed word, sentence
position, punctuation state) and tilts the distribution toward the
canonical oath-phrase continuations of Shakespearean idiom:

  - After "by" / "upon" / "my" with hot oath_mode, word-start letters
    shift toward oath-object vocabulary: heaven (h), troth (t), soul
    (s), faith (f), honour (h), life (l), word (w), sword (s), hand
    (h), God (g), blood (b), crown (c), grave (g).

  - After an oath object has just completed with oath_mode hot, a
    comma is a high-probability closure — "..., by my troth, ..."
    — so we boost "," at the token following the oath-object's
    word-ending space.

No corpus statistics; all content comes from Shakespeare idiom.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# Oath object words by first letter, for word-start bias after an
# oath-opener context. Values = soft weights; identical-first-letter
# objects share credit.
_OATH_OBJECT_LETTERS: dict[str, float] = {
    "h": 2.4,   # heaven, honour, hand, hell
    "t": 1.8,   # troth, tomb
    "s": 2.4,   # soul, sword, saints
    "f": 1.8,   # faith
    "l": 1.4,   # life
    "w": 1.4,   # word
    "g": 2.0,   # god, grave
    "b": 1.4,   # blood
    "c": 1.4,   # crown, cross
    "m": 1.2,   # mother, majesty
}

# Capital forms (for "By Heaven," / "Upon my Soul,") — the oath
# object is often capitalized mid-sentence as invocation. Weighted
# slightly less than lowercase because modern Shakespeare editions
# aren't consistent.
_OATH_OBJECT_UPPER: dict[str, float] = {
    k.upper(): v * 0.45 for k, v in _OATH_OBJECT_LETTERS.items()
}

_OATH_OPENER_WORDS: frozenset[str] = frozenset({
    "by", "upon", "my", "mine", "his", "thy", "our",
})

_OATH_OBJECT_COMPLETIONS: frozenset[str] = frozenset({
    "heaven", "heavens", "honour", "honor", "troth", "faith",
    "soul", "souls", "sword", "saints", "god", "gods",
    "life", "blood", "word", "hand", "grave", "crown", "cross",
    "mother", "father", "king", "queen", "majesty",
})


def oath_mode_start_bias(
    oath_mode: float,
    last_completed_word: str,
    speaker_label_state: int,
    letter_run_len: int,
    word_buffer: str,
) -> list[float] | None:
    """At word-start position: bias first letters toward oath-object
    vocabulary when oath_mode is warm and the immediately-preceding
    completed word is an oath-phrase opener ("by", "upon", "my",
    "mine", "his", "thy", "our").

    Fires only at the word-start position (letter_run_len == 0 and
    word_buffer empty) so it competes with START_BIAS and the
    start-bigram/trigram layers without interfering mid-word.
    """
    if speaker_label_state != 0:
        return None
    if oath_mode < 0.20:
        return None
    # Only at true word-start.
    if word_buffer:
        return None
    if letter_run_len != 0:
        return None
    if not last_completed_word:
        return None
    if last_completed_word.lower() not in _OATH_OPENER_WORDS:
        return None

    # Scale: grows with mode intensity.
    scale = min(1.0, max(0.0, (oath_mode - 0.15) / 0.50))
    if scale <= 0.0:
        return None

    vec = [0.0] * VOCAB_SIZE
    for ch, w in _OATH_OBJECT_LETTERS.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
    for ch, w in _OATH_OBJECT_UPPER.items():
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += w * scale
    return vec


def oath_mode_close_bias(
    oath_mode: float,
    last_completed_word: str,
    speaker_label_state: int,
    word_buffer: str,
    letter_run_len: int,
    chars_since_sentence_end: int,
) -> list[float] | None:
    """Just after completing an oath-object word with oath_mode warm,
    bias the next token toward "," (formula closure) and "." (hard
    closure). Fires only at word-boundary position (word_buffer
    empty, letter_run_len == 0) when the previous completed word is
    a canonical oath-object.
    """
    if speaker_label_state != 0:
        return None
    if oath_mode < 0.30:
        return None
    if word_buffer:
        return None
    if letter_run_len != 0:
        return None
    if not last_completed_word:
        return None
    if last_completed_word.lower() not in _OATH_OBJECT_COMPLETIONS:
        return None

    scale = min(1.0, max(0.0, (oath_mode - 0.25) / 0.55))
    if scale <= 0.0:
        return None

    vec = [0.0] * VOCAB_SIZE
    if "," in VOCAB_INDEX:
        vec[VOCAB_INDEX[","]] += 0.85 * scale
    if "." in VOCAB_INDEX:
        vec[VOCAB_INDEX["."]] += 0.35 * scale
    if "!" in VOCAB_INDEX:
        vec[VOCAB_INDEX["!"]] += 0.35 * scale
    # Mild suppression of " " — we already emitted the space; the
    # next *content* char prefers the punct, not another space.
    # (Not applied: we're at position after a space, so " " is
    # contextually unlikely anyway.)
    return vec
