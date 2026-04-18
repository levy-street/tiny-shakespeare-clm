"""Word-trie drift recovery — escalate terminator bias when we've
drifted past a viable word-end.

Reads `state.has_seen_complete` and `state.letters_past_complete`:

  - `has_seen_complete` is True iff at some point during the current
    word's growth, the buffer equaled a member of COMPLETE_WORDS.
  - `letters_past_complete` is the number of letters written since
    that valid-stop position (0 when the buffer IS currently a
    complete word).

When `letters_past_complete >= 1`, we know we had a viable stop N
letters ago and kept writing. The longer we drift past that stop,
the more likely we're building gibberish. This layer:

  1. Boosts word-terminator tokens (space, comma, period) so the
     sampler/argmax is pulled toward ending the word.
  2. Boosts common word-ending letters (e, s, d, t, n, h, r, y, g)
     so if another letter IS emitted, it's at least a plausible
     word-ending letter that might extend to a real word.
  3. Penalizes uncommon mid-word letters that would only extend
     the gibberish further.

The bias is ZERO until `has_seen_complete` is True — this avoids
penalizing short prefixes like "th", "wh", "sh", "str" where the
word is clearly still growing. It only fires once we've passed a
real word-form.

All weights are hand-chosen from prior knowledge of English word-
ending distribution — no corpus statistics.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# Common word-ending letters (English / Shakespeare). Pulled from
# prior knowledge: -e, -s, -d, -t, -n, -h, -r, -y, -l, -g, -w.
_WORD_END_LETTERS: dict[str, float] = {
    "e": 1.0,   # -e (silent), -le, -te, -ne, -re, -ce, -ge, -se
    "s": 1.0,   # plural, 3sg verb
    "d": 0.9,   # -ed, -d
    "t": 0.8,   # -t, -st, -nt, -ct
    "n": 0.75,  # -n, -on, -en, -un, -in, -an
    "h": 0.65,  # -th, -sh, -ch, -gh, -oh, -ah
    "r": 0.70,  # -er, -or, -ar, -ir, -ur
    "y": 0.70,  # -y, -ly, -ty, -ry, -ny
    "l": 0.55,  # -l, -ll, -al
    "g": 0.45,  # -g, -ng, -ing
    "w": 0.35,  # -w, -ow, -aw, -ew
}

# Letters that are VERY unlikely to extend a word past a viable
# stop. If we had a complete word and the next letter is one of
# these, we're building gibberish.
_GIBBERISH_LETTERS: dict[str, float] = {
    "j": -1.0, "q": -1.0, "x": -1.0, "z": -1.0,
    "b": -0.25, "c": -0.20, "f": -0.30, "k": -0.30,
    "m": -0.15, "p": -0.20, "v": -0.35,
}


def trie_recovery_bias(
    has_seen_complete: bool,
    letters_past_complete: int,
    letters_off_trie: int = 0,
) -> list[float] | None:
    """Return a bias vector to push toward word-end when drifting.

    Two escalation axes, whichever is stronger wins:
      (a) letters_past_complete — drifted past a known-complete form.
          Safer: we KNOW there was a real word we could have ended
          at, so extending further is suspicious.
      (b) letters_off_trie — drifted past all known-word prefixes.
          Catches gibberish that never reached a complete form
          ("naitagomo" — never had a complete prefix).

    Returns None when both axes are 0 (no drift signal).
    """
    # Effective drift signal: use whichever is stronger.
    past = letters_past_complete if has_seen_complete else 0
    off = letters_off_trie
    n = max(past, off)
    if n <= 0:
        return None

    # The off-trie axis is noisier (some real words aren't in our
    # word_trie) so we apply a dampener when past is 0.
    is_off_only = (past == 0 and off > 0)
    damp = 0.55 if is_off_only else 1.0

    # Gibberish-letter penalty is cheap: rare letters (j/q/x/z/v) don't
    # appear mid-word in real corpus words, so penalizing them costs
    # nothing on legit escapes from the word_trie. Terminator pressure
    # is expensive: biasing toward " " when the corpus has a real word
    # continuation costs BPC. Escalate gib hard, terminator very gently.
    if n <= 2:
        term_boost = 0.0
        end_letter_scale = 0.0
        gib_scale = 0.0
    elif n == 3:
        term_boost = 0.0
        end_letter_scale = 0.0
        gib_scale = 0.20
    elif n == 4:
        term_boost = 0.0
        end_letter_scale = 0.0
        gib_scale = 0.40
    elif n == 5:
        term_boost = 0.0
        end_letter_scale = 0.0
        gib_scale = 0.65
    elif n == 6:
        term_boost = 0.0
        end_letter_scale = 0.0
        gib_scale = 0.90
    else:
        term_boost = 0.0
        end_letter_scale = 0.0
        gib_scale = 1.25

    term_boost *= damp
    end_letter_scale *= damp
    gib_scale *= damp

    if term_boost == 0.0 and end_letter_scale == 0.0 and gib_scale == 0.0:
        return None

    vec = [0.0] * VOCAB_SIZE

    # Terminators: space is the primary word-ender. Comma, period,
    # and newline are also valid endings but less generic.
    if " " in VOCAB_INDEX:
        vec[VOCAB_INDEX[" "]] += term_boost
    if "," in VOCAB_INDEX:
        vec[VOCAB_INDEX[","]] += term_boost * 0.55
    if "." in VOCAB_INDEX:
        vec[VOCAB_INDEX["."]] += term_boost * 0.40
    if ";" in VOCAB_INDEX:
        vec[VOCAB_INDEX[";"]] += term_boost * 0.30
    if "\n" in VOCAB_INDEX:
        vec[VOCAB_INDEX["\n"]] += term_boost * 0.35

    # Word-ending letters get a smaller boost.
    for ch, w in _WORD_END_LETTERS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += end_letter_scale * w

    # Penalize gibberish-extending letters.
    for ch, w in _GIBBERISH_LETTERS.items():
        if ch in VOCAB_INDEX:
            vec[VOCAB_INDEX[ch]] += gib_scale * w

    return vec
