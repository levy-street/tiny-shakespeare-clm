"""Tier 2 — vowel-cluster letter tracker (mirror of coda_tracker).

Maintains `vowel_run_letters`: the lowercase string of vowel letters
emitted since the most recent consonant or word boundary within the
current word.

Purpose: lets the vowel_cluster_bias predict layer see the LITERAL
vowel cluster in progress (not just a count) so it can check whether
a given next-vowel would form a phonotactically legal English
vowel cluster.

Rules:
  - Reset to "" on word boundary (non-letter, non-apostrophe char).
  - Reset to "" on any consonant letter.
  - Reset to "" in speaker-label territory (proper names have loose
    phonotactics; we don't want to penalize novel name sequences).
  - Append the lowercased vowel for a/e/i/o/u unconditionally.
  - Append 'y' ONLY when a strict vowel has already appeared in the
    current word (word-internal y acts as a vowel — "may", "toy").
    When y is word-initial with no prior vowel ("yes", "you"), it
    acts as consonant; don't treat it as vowel — reset.
  - Apostrophe: pass through unchanged (contractions invisible).
  - Cap stored length at 4 — deeper than any legal English cluster.

Runs after `update_flow` (which maintains `vowels_in_word`) so the
updated count is available.
"""

from __future__ import annotations

from ..state import ModelState
from ..vocab import VOCAB


_LETTERS: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
_STRICT_VOWELS: frozenset[str] = frozenset("aeiouAEIOU")


def _is_letter(ch: str) -> bool:
    return ch in _LETTERS


def update_vowel_run(state: ModelState, token_id: int) -> ModelState:
    ch = VOCAB[token_id]

    # Speaker-label: reset.
    if state.speaker_label_state != 0:
        if state.vowel_run_letters:
            return state.model_copy(update={"vowel_run_letters": ""})
        return state

    # Apostrophe: invisible to vowel-run tracking.
    if ch == "'":
        return state

    # Non-letter: word boundary, reset.
    if not _is_letter(ch):
        if state.vowel_run_letters:
            return state.model_copy(update={"vowel_run_letters": ""})
        return state

    ch_low = ch.lower()

    # Strict vowel: append.
    if ch_low in ("a", "e", "i", "o", "u"):
        new_run = state.vowel_run_letters + ch_low
        if len(new_run) > 4:
            new_run = new_run[-4:]
        if new_run == state.vowel_run_letters:
            return state
        return state.model_copy(update={"vowel_run_letters": new_run})

    # y: treat as vowel when a strict vowel has preceded in the word.
    # Otherwise (word-initial y like "yes"), treat as consonant → reset.
    if ch_low == "y":
        # `state.vowels_in_word` at this point reflects the word's
        # vowel count INCLUDING the current y only if flow.py already
        # incremented it. To get the pre-this-char count, we look at
        # whether any strict vowel appeared earlier in the buffer:
        # easiest is to check state.vowels_in_word >= 1 AND the last
        # char isn't the word-initial position. Since flow.py runs
        # first and increments on vowel, we can't distinguish here.
        # Use vowels_in_word directly: after flow runs, if y was the
        # only vowel so far, vowels_in_word == 1 but the run is empty
        # (y word-initial). We want the prior-vowel-present signal.
        # A clean proxy: check word_buffer prior content for a strict
        # vowel. state.word_buffer at this stage already includes this
        # y? Depends on pipeline order.  Conservatively: if
        # vowels_in_word >= 2, definitely a prior strict vowel existed
        # (this y + >=1 earlier). If vowels_in_word == 1, either this y
        # is the first vowel (word-initial) or the count reflects a
        # prior strict vowel that already fired. Use word_buffer.
        buf = state.word_buffer
        # If the buffer prior to appending this char contains any
        # strict vowel, y acts as vowel. Pipeline order may append the
        # char to word_buffer BEFORE this stage — strip the trailing
        # 'y' if present.
        prior = buf[:-1] if buf.endswith(("y", "Y")) else buf
        has_prior_vowel = any(c in _STRICT_VOWELS for c in prior)
        if has_prior_vowel:
            new_run = state.vowel_run_letters + "y"
            if len(new_run) > 4:
                new_run = new_run[-4:]
            if new_run == state.vowel_run_letters:
                return state
            return state.model_copy(update={"vowel_run_letters": new_run})
        # y as consonant: reset.
        if state.vowel_run_letters:
            return state.model_copy(update={"vowel_run_letters": ""})
        return state

    # Any other letter is a consonant: reset.
    if state.vowel_run_letters:
        return state.model_copy(update={"vowel_run_letters": ""})
    return state
