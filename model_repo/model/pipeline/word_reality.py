"""Pipeline stage — classify the most recently completed word.

At each token that boundaries a word (`just_finished_word == True`),
classify `last_completed_word` along a 3-way axis:

  REAL (1)       — the word is a recognized English/Shakespearean form.
                    Criteria: has_seen_complete AND no phonotactic red
                    flags AND no illegal bigrams/trigrams AND
                    letters_off_trie <= 1.
  PLAUSIBLE (2)  — off-trie but phonotactically sane; could be an
                    inflected / archaic / proper-noun variant.
  GIBBERISH (3)  — one or more phonotactic red flags, illegal bigrams,
                    illegal trigrams, or a long unsupported off-trie run.

Maintain per-turn and per-sentence counts so downstream predict layers
can bias toward recovery (sentence-close, safe word-start) when garbage
accumulates.

Placement requirement: runs AFTER `update_linguistic` (which sets
`just_finished_word` and `last_completed_word`), but BEFORE
`update_word_shape` and `update_phonotactic` — those reset
`word_red_flags`, `bad_bigram_count`, `bad_trigram_count` on the
boundary char. Our classification reads those pre-reset values.

No corpus statistics. Classification thresholds derive from prior
phonotactic knowledge of English (already encoded in the phonotactic /
word_shape counters this stage consumes).
"""

from __future__ import annotations

from ..state import ModelState
from .linguistic import PUNCT_END


def _classify(
    word: str,
    has_seen_complete: bool,
    letters_off_trie: int,
    word_red_flags: int,
    bad_bigram_count: int,
    bad_trigram_count: int,
) -> int:
    """Return 0 (skip), 1 (real), 2 (plausible), or 3 (gibberish)."""
    # Strip apostrophes for length counting.
    alpha = [c for c in word if c.isalpha()]
    n = len(alpha)
    if n < 2:
        return 0  # too short to classify

    # GIBBERISH — hard phonotactic signals.
    # An illegal trigram is a near-certain gibberish marker (three
    # consonants that form no legal English cluster, or three vowels
    # outside the small attested set).
    if bad_trigram_count >= 1:
        return 3
    # Two illegal bigrams, or one bigram plus at least one red flag.
    if bad_bigram_count >= 2:
        return 3
    if bad_bigram_count >= 1 and word_red_flags >= 1:
        return 3
    # Three or more red flags — stacked phonotactic warnings.
    if word_red_flags >= 3:
        return 3
    # Unsupported long off-trie extension: we never matched a complete
    # word anywhere along the way, yet the word is long and far off-trie.
    if (not has_seen_complete) and n >= 7 and letters_off_trie >= 5:
        return 3
    # Drifted far past last complete form in a long word.
    if n >= 8 and letters_off_trie >= 6:
        return 3

    # REAL — strict criteria: known word, no red flags at all.
    if (
        has_seen_complete
        and letters_off_trie <= 1
        and word_red_flags == 0
        and bad_bigram_count == 0
        and bad_trigram_count == 0
    ):
        return 1

    # Otherwise PLAUSIBLE — off-trie extension of a real prefix,
    # or a short word with zero trie match but no red flags.
    return 2


def update_word_reality(state: ModelState, token_id: int) -> ModelState:
    # Inside a speaker label — do not account. But still apply turn
    # reset when we cross into a new turn (consecutive_newlines >= 2 at
    # the moment linguistic set speaker_label_state=1 is captured via
    # the newline→label transition, but we keep it simple: reset turn
    # counters whenever speaker_label_state becomes 1 from 0).
    if state.speaker_label_state != 0:
        # Reset per-turn on entry to a new speaker-label region.
        if (
            state.turn_gibberish_count != 0
            or state.turn_real_count != 0
            or state.sentence_gibberish_count != 0
            or state.sentence_real_count != 0
            or state.last_word_reality != 0
            or state.recent_word_realities
        ):
            return state.model_copy(update={
                "turn_gibberish_count": 0,
                "turn_real_count": 0,
                "sentence_gibberish_count": 0,
                "sentence_real_count": 0,
                "last_word_reality": 0,
                "recent_word_realities": (),
            })
        return state

    updates: dict = {}

    # Classify the just-finished word (if any).
    if state.just_finished_word and state.last_completed_word:
        reality = _classify(
            state.last_completed_word,
            state.has_seen_complete,
            state.letters_off_trie,
            state.word_red_flags,
            state.bad_bigram_count,
            state.bad_trigram_count,
        )
        if reality != 0:
            updates["last_word_reality"] = reality
            # Roll recent window.
            updates["recent_word_realities"] = (
                (reality,) + state.recent_word_realities
            )[:4]
            if reality == 3:
                updates["turn_gibberish_count"] = min(
                    state.turn_gibberish_count + 1, 12
                )
                updates["sentence_gibberish_count"] = min(
                    state.sentence_gibberish_count + 1, 8
                )
            elif reality == 1:
                updates["turn_real_count"] = min(
                    state.turn_real_count + 1, 30
                )
                updates["sentence_real_count"] = min(
                    state.sentence_real_count + 1, 20
                )

    # Sentence-end reset (PUNCT_END: . ? !). Clear sentence counters.
    # Also lightly bleed the turn counter — each sentence end gives the
    # predict layer a chance to relax. But keep most of the history.
    if state.last_char_class == PUNCT_END:
        updates["sentence_gibberish_count"] = 0
        updates["sentence_real_count"] = 0

    if updates:
        return state.model_copy(update=updates)
    return state
