"""Contraction-tail closure pressure.

When the current word contains a mid-word apostrophe and the letters
emitted after the apostrophe have NOT yet formed a valid elision tail
(tracked by `contraction_tail_ok`), we want to gently discourage
further non-closing letters and prefer either:

  (a) a closing letter that completes the tail (handled by
      `apostrophe_elision_bias` — it already biases the correct
      single- and double-letter closers), or

  (b) a word terminator (space / punctuation) — occasionally a tail
      will "close" via the word ending even if no canonical closer
      letter appears (e.g. rare archaic elisions).

This layer adds an orthogonal signal: once `contraction_tail_ok` is
True AFTER having had an apostrophe this word (i.e. the tail closed),
gently prefer terminating the word over continuing. In Shakespeare
contracted forms like "thou'lt", "know'st", "could'st", "we've",
"'tis" reliably end soon after the closer lands.

Gates:
  * speaker_label_state == 0        — not inside a SPEAKER: label
  * had_apostrophe_this_word        — only matters for contracted words
  * contraction_tail_ok             — tail is closed (valid form achieved)
  * letters_since_apostrophe in 2..5  — at or past the typical closer
  * word_buffer non-empty

Scales: deliberately gentle. +0.30 on space / word-enders,
-0.18 on uppercase letters (near-impossible mid-word),
-0.10 on lowercase letters.

No corpus statistics — purely a morphological-closure prior.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


_LOWER = "abcdefghijklmnopqrstuvwxyz"
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_TERMINATORS = (" ", ",", ".", ";", ":", "!", "?", "\n")


def contraction_close_block_bias(
    had_apostrophe_this_word: bool,
    contraction_tail_ok: bool,
    letters_since_apostrophe: int,
    word_buffer: str,
    speaker_label_state: int,
) -> list[float] | None:
    if speaker_label_state != 0:
        return None
    if not had_apostrophe_this_word:
        return None
    if not contraction_tail_ok:
        return None
    if letters_since_apostrophe < 2 or letters_since_apostrophe > 5:
        return None
    if not word_buffer:
        return None

    # Scale tapers with position: earliest closure (pos 1, lsa=2) can
    # still get legitimately extended (e.g. "could'st" → "could'st's"
    # is not a word but "we'll" → word-end is dominant). Later
    # positions already had their chance — stronger close pressure.
    lsa = letters_since_apostrophe
    if lsa == 2:
        term_w, upper_w, lower_w = 0.22, -0.14, -0.07
    elif lsa == 3:
        term_w, upper_w, lower_w = 0.30, -0.18, -0.10
    elif lsa == 4:
        term_w, upper_w, lower_w = 0.40, -0.22, -0.13
    else:  # lsa == 5
        term_w, upper_w, lower_w = 0.50, -0.26, -0.16

    vec = [0.0] * VOCAB_SIZE
    for t in _TERMINATORS:
        idx = VOCAB_INDEX.get(t)
        if idx is not None:
            vec[idx] += term_w
    for ch in _UPPER:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += upper_w
    for ch in _LOWER:
        idx = VOCAB_INDEX.get(ch)
        if idx is not None:
            vec[idx] += lower_w
    return vec
