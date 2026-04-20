"""Syllable-saturation off-trie termination pressure layer.

Reads `state.syllables_in_word` (a prosody-tier integer tracking how
many vowel-clusters the current word has accumulated) alongside
off-trie state.

Real Shakespeare word lengths peak at 1-2 syllables; 3-syllable words
are common ("virtuous", "gentleman"); 4-syllable are rarer ("honesty",
"innocence", "majesty"); 5+ syllables are rare outside specific
terms ("multitudinous", "indissoluble"). When the CURRENT mid-word
buffer has already accumulated 3+ syllables AND the word is off the
known-word trie AND the word has drifted (letters_off_trie >= 2),
the model is almost certainly extending letter-ngram noise into a
phantom polysyllabic word. Push terminators hard; penalize any further
letter extension.

Gates:
  * speaker_label_state == 0 — speaker labels aren't natural English
    words.
  * not on_word_trie — on-trie polysyllabics are legitimate.
  * letters_off_trie >= 2 — short off-trie departures (morphology /
    rare names) don't trigger; sustained drift does.
  * syllables_in_word >= 3 — below this, the word is short enough
    that ordinary off-trie layers suffice.

Pressure scales with BOTH syllable count and letter_run_len, giving
a two-dimensional escalation: a 3-syllable 7-letter off-trie word
gets moderate pressure; a 5-syllable 12-letter off-trie word gets
near-hard-cap pressure.

No corpus statistics — the syllable thresholds come from prior
knowledge of English word-length distribution.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_TERMINATORS: tuple[tuple[str, float], ...] = (
    (" ", 1.0),
    (",", 0.55),
    (".", 0.45),
    (";", 0.35),
    (":", 0.25),
    ("\n", 0.40),
    ("!", 0.30),
    ("?", 0.30),
)

_LOWER = "abcdefghijklmnopqrstuvwxyz"
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def syllable_saturation_bias(
    syllables_in_word: int,
    letter_run_len: int,
    letters_off_trie: int,
    on_word_trie: bool,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if on_word_trie:
        return None
    if letters_off_trie < 2:
        return None
    if syllables_in_word < 3:
        return None
    # Minimum buffer length; below this the syllable count is unlikely
    # to reflect real syllable saturation.
    if letter_run_len < 6:
        return None

    # Syllable-count escalation. Conservative: 3-syllable off-trie
    # can still be legitimate morphology ("commander", "continuing").
    if syllables_in_word >= 5:
        syl_scale = 0.9
    elif syllables_in_word == 4:
        syl_scale = 0.5
    else:  # == 3
        syl_scale = 0.0  # disabled at 3 — too many real words

    if syl_scale <= 0.0:
        return None

    # Letter-run-length escalation (on top of syllable count).
    if letter_run_len >= 11:
        len_scale = 1.2
    elif letter_run_len >= 9:
        len_scale = 0.9
    elif letter_run_len >= 8:
        len_scale = 0.7
    else:  # 6-7
        len_scale = 0.5

    scale = syl_scale * len_scale

    vec = [0.0] * VOCAB_SIZE
    for t, w in _TERMINATORS:
        idx = VOCAB_INDEX.get(t)
        if idx is not None:
            vec[idx] += w * scale
    # Slight penalty across all letter continuations to redirect mass
    # toward terminators.
    letter_pen = -0.18 * scale
    for ch in _LOWER:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += letter_pen
    upper_pen = -0.28 * scale
    for ch in _UPPER:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += upper_pen
    # Apostrophe mildly penalized — 'd / 's / 'll extensions on a
    # 3-syllable gibberish word are almost always wrong.
    idx = VOCAB_INDEX.get("'")
    if idx is not None:
        vec[idx] += -0.12 * scale
    return vec
