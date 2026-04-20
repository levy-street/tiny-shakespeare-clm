"""Archaic-density predict bias.

Reads `state.archaic_density` (0..1 float maintained by
pipeline/archaic_density.py) and tilts the distribution toward
archaic forms when density is hot.

Two application positions:

  A) Mid-word suffix tilt.
     When the word_buffer is at a position where an archaic
     -th / -st / -lt suffix is plausible and archaic density is
     high, boost the archaic continuation letter. Complements
     existing verb-agreement logic (which fires only when the
     clause subject is explicitly "thou") by adding a
     density-driven nudge even when the subject is implicit.

     Specific cases (each fires at a specific word_buffer value):
       "ha"  -> "s" (hast) +0.6 * d; "t" (hath) also plausible but
                agreement handles finer choice
       "do"  -> "t" (doth) +0.5 * d; "s" (dost) +0.3 * d
       "wi"  -> "l" (wilt/will) ok; after "wil" push "t" (wilt)
                +0.5 * d
       "sh"  -> "a" (shall/shalt) ok; after "sha" push "l" then
                "t"
     Only fires when density >= 0.25.

  B) Word-start tilt at sentence-start.
     When letter_run_len == 0 and we're about to open a new word
     (inside an active sentence, not inside speaker label, not
     immediately after punctuation-only), boost letters that open
     archaic pronouns / adverbs / fillers:
       t (thou / thee / thy / thine), h (hath / hast / hither /
       hence), y (ye / yon / yonder), m (methinks / marry),
       p (prithee), w (wherefore / whence / whither), a (anon /
       aye / alack), e (ere), o (oft)
     Weight scaled by density; caps individual bumps at ~0.4 even
     at density=1 to avoid overwhelming other structural layers.

Gated to speaker_label_state == 0 and consecutive_newlines <= 1.
No corpus statistics — bias shape from Early Modern English idiom.
"""

from __future__ import annotations

from ..vocab import VOCAB_INDEX, VOCAB_SIZE

# Word-start letter bumps (applied with weight density * factor).
# Kept small — the first-letter class distribution is already heavily
# shaped by many existing layers (startword, unigram, register, etc.).
# Archaic density nudges it only slightly.
_START_LETTERS: dict[str, float] = {
    "t": 0.12,   # thou/thee/thy/thine
    "h": 0.09,   # hath/hast/hither/hence
    "y": 0.07,   # ye/yon/yonder
    "m": 0.05,   # methinks/marry
    "p": 0.05,   # prithee
    "w": 0.06,   # wherefore/whence/whither
}

# Capital-letter variant for sentence-initial / post-punct position.
_START_UPPER: dict[str, float] = {
    k.upper(): v * 0.55 for k, v in _START_LETTERS.items()
}


def archaic_density_bias(
    archaic_density: float,
    word_buffer: str,
    letter_run_len: int,
    speaker_label_state: int,
    consecutive_newlines: int,
    chars_since_sentence_end: int,
    last_char_class: int,
) -> list[float] | None:
    if archaic_density < 0.20:
        return None
    if speaker_label_state != 0:
        return None
    if consecutive_newlines >= 2:
        return None

    d = archaic_density
    vec = [0.0] * VOCAB_SIZE
    applied = False

    # --- Mid-word suffix tilt ---
    if letter_run_len >= 2 and d >= 0.25:
        wb = word_buffer
        # "ha" -> archaic completions hath/hast
        if wb == "ha":
            for letter, w in (("s", 0.50), ("t", 0.40)):
                idx = VOCAB_INDEX.get(letter)
                if idx is not None:
                    vec[idx] += w * d
                    applied = True
        # "do" -> doth/dost
        elif wb == "do":
            for letter, w in (("t", 0.45), ("s", 0.35)):
                idx = VOCAB_INDEX.get(letter)
                if idx is not None:
                    vec[idx] += w * d
                    applied = True
        # "wil" -> wilt
        elif wb == "wil":
            idx = VOCAB_INDEX.get("t")
            if idx is not None:
                vec[idx] += 0.45 * d
                applied = True
        # "sha" -> shalt (after l)
        elif wb == "shal":
            idx = VOCAB_INDEX.get("t")
            if idx is not None:
                vec[idx] += 0.40 * d
                applied = True
        # "ar" -> art (2nd sing)
        elif wb == "ar":
            idx = VOCAB_INDEX.get("t")
            if idx is not None:
                vec[idx] += 0.30 * d
                applied = True
        # "has" -> hast (over "has/have")
        elif wb == "has":
            idx = VOCAB_INDEX.get("t")
            if idx is not None:
                vec[idx] += 0.38 * d
                applied = True

    # --- Word-start tilt ---
    # Require higher density to fire word-start; this position is
    # already heavily shaped by other layers.
    if letter_run_len == 0 and d >= 0.35:
        # Uppercase at sentence-start / post-punct (last_char_class
        # PUNCT_END=7 or NEWLINE=6 or post-turn-break); lowercase
        # otherwise.
        # Heuristic: chars_since_sentence_end <= 2 or last_char_class in
        # {NEWLINE, PUNCT_END} → uppercase context.
        sent_start = (
            chars_since_sentence_end <= 2
            or last_char_class in (6, 7)
        )
        if sent_start:
            for letter, w in _START_UPPER.items():
                idx = VOCAB_INDEX.get(letter)
                if idx is not None:
                    vec[idx] += w * d
                    applied = True
        # Always apply lowercase bumps — they're valid inside a
        # sentence and also as a soft prior even at sentence-start
        # (lowercased verse-continuation lines).
        for letter, w in _START_LETTERS.items():
            idx = VOCAB_INDEX.get(letter)
            if idx is not None:
                vec[idx] += w * d
                applied = True

    if not applied:
        return None
    return vec
