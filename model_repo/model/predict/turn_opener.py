"""Turn-opener first-letter bias.

When we're at the very first word of a new speaker turn (words_in_turn
== 0 AND sentences_in_turn == 0) and at a letter-start position outside
speaker-label territory, tilt next-letter choice toward the cluster of
letters that disproportionately opens Shakespearean turns.

Turn-opener cluster (from prior knowledge, not corpus counts):
  - O     : O, Oh — the paradigmatic turn-open exclamation
  - A     : Alas, Ah, Ay — classic turn-open interjections
  - N     : Nay, No, Now — response / denial openers
  - W     : Why, What, Well, Where — question / response openers
  - M     : My, Marry — vocative / oath openers
  - P     : Pray, Peace, Prithee — polite / plea openers
  - G     : Good, God — vocative / oath openers
  - T     : Thou, Thy, The, This, That — direct address
  - F     : Fie, Faith, For — interjection / continuation
  - Y     : Yet, Ye, Yes — response openers

These are additive on top of the generic sentence-start caps boost.
Scale is modest (0.2–0.5) because existing is_sentence_start already
provides a general capital nudge; this tips the distribution WITHIN
those capitals toward the subset favored at turn-open.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE


# letter → boost (applied to the uppercase form; a smaller share to
# lowercase in case the turn opens with a lower-cased character, e.g.
# following an opening single-quote/apostrophe).
#
# Only letters that are DISPROPORTIONATELY turn-open get a boost;
# letters common to any sentence-start (T, I, S, C, B, D, L) are left
# to the generic line_start_caps block above. Adding them here just
# re-spends the same probability mass twice.
_OPENER_CAPS: dict[str, float] = {
    "O": 0.60,  # O / Oh — the paradigmatic exclamation opener
    "A": 0.35,  # Alas / Ah / Ay
    "N": 0.30,  # Nay / No / Now
    "W": 0.30,  # Why / What / Well
    "M": 0.25,  # My / Marry
    "P": 0.20,  # Pray / Peace / Prithee
    "G": 0.20,  # Good / God
    "F": 0.18,  # Fie / Faith
    "Y": 0.18,  # Yea / Yes / Ye
}


def turn_opener_start_bias() -> list[float]:
    """Per-VOCAB log-bias vector for turn-opener first letter."""
    vec = [0.0] * VOCAB_SIZE
    for up, b in _OPENER_CAPS.items():
        if up in VOCAB_INDEX:
            vec[VOCAB_INDEX[up]] += b
        low = up.lower()
        if low in VOCAB_INDEX:
            vec[VOCAB_INDEX[low]] += b * 0.35
    return vec


TURN_OPENER_START_BIAS: list[float] = turn_opener_start_bias()
