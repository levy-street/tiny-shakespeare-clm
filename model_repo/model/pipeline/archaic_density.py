"""Archaic-density Tier-3 flow register.

Maintains `state.archaic_density` — a float in [0, 1] that tracks how
archaic the current speaker's diction feels, smoothed over completed
words.

Evidence sources (all at `just_finished_word`):
  - Archaic pronouns: thou, thee, thy, thine, ye
  - Archaic aux verbs: hath, doth, hast, dost, wilt, shalt, art
  - Archaic adverbs / discourse markers: ere, oft, anon, hither,
    thither, whither, whence, hence, thence, yon, yonder,
    methinks, prithee, wherefore, forsooth, troth, marry, aye, nay,
    fie, hark, lo, alack, alas
  - Apostrophe-elided forms: 'tis, 'twas, 'twere, 'twixt, 'gainst,
    o'er, e'er, ne'er, 'neath (detected via had_apos + short form)

Updates:
  - On every just_finished_word:
      density *= 0.88  (decay — fades over ~8-10 non-archaic words)
      if hit:  density += 0.28
      clamp to [0.0, 1.0]
  - On turn boundary (consecutive_newlines >= 2): reset to 0.0.

Initialized to 0.0 (at text start we're in unknown register).

Runs after update_basic_counters / update_linguistic so
last_completed_word, just_finished_word, and had_apos are current.
Runs near the end of the pipeline but BEFORE update_flow (which
computes composite flow signals) so downstream consumers see the
updated density.

No corpus statistics — archaic lexicon and decay/bump magnitudes
are from prior knowledge of Early Modern English register.
"""

from __future__ import annotations

from ..state import ModelState

_ARCHAIC_WORDS: frozenset[str] = frozenset({
    # Pronouns
    "thou", "thee", "thy", "thine", "ye",
    # Archaic aux / finite verbs
    "hath", "doth", "hast", "dost", "wilt", "shalt", "art",
    # Archaic adverbs
    "ere", "oft", "anon",
    "hither", "thither", "whither",
    "whence", "hence", "thence",
    "yon", "yonder",
    # Archaic discourse markers / interjections (contextually
    # archaic-coded)
    "methinks", "prithee", "wherefore",
    "forsooth", "troth", "marry",
    "aye", "nay", "fie", "lo", "alack", "alas",
    "hark",
    # Archaic short verbs / past forms (occasional)
    "quoth", "durst", "spake", "oped",
})

# Elided-apostrophe short forms that count as archaic evidence.
# Detected at word completion AFTER stripping leading apostrophe
# (word_cap_apos handles leading apostrophe → had_apos flag).
_ELIDED_ARCHAIC: frozenset[str] = frozenset({
    # Leading-apostrophe forms (we strip the leading ' before lookup)
    "tis", "twas", "twere", "twill", "twixt", "gainst",
    "neath",
})

# Internal-apostrophe archaic contractions — check raw lowercased form.
_INTERNAL_APOS_ARCHAIC: frozenset[str] = frozenset({
    "o'er", "e'er", "ne'er", "i'faith", "'gainst",
})

_DECAY = 0.88
_BUMP = 0.28


def update_archaic_density(state: ModelState, token_id: int) -> ModelState:
    # Turn boundary: hard reset.
    if state.consecutive_newlines >= 2:
        if state.archaic_density != 0.0:
            return state.model_copy(update={"archaic_density": 0.0})
        return state

    if not state.just_finished_word:
        return state

    lcw = (state.last_completed_word or "").lower()
    if not lcw:
        return state

    # Strip leading apostrophe for elided-form detection; don't strip
    # trailing (preserves "'tis" → "tis", "hast'" → "hast" which is
    # fine because archaic set is lowercase bare forms).
    core = lcw.lstrip("'").rstrip("'")
    hit = core in _ARCHAIC_WORDS
    if not hit:
        # Elided-apostrophe archaic: 'tis, 'twas, etc.
        if "'" in lcw and core in _ELIDED_ARCHAIC:
            hit = True
        # Internal-apostrophe archaic: o'er, e'er, ne'er.
        elif lcw in _INTERNAL_APOS_ARCHAIC:
            hit = True

    # Decay then bump.
    new_density = state.archaic_density * _DECAY
    if hit:
        new_density += _BUMP
    if new_density > 1.0:
        new_density = 1.0
    elif new_density < 0.0:
        new_density = 0.0

    # Snap to 0 at very small values to keep state changes minimal.
    if new_density < 1e-3:
        new_density = 0.0

    if abs(new_density - state.archaic_density) < 1e-6:
        return state
    return state.model_copy(update={"archaic_density": new_density})
