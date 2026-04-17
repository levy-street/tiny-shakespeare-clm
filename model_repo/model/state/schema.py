"""The rolling state schema.

Every field has a default so `ModelState()` is the well-defined initial
state. The state is frozen (immutable); `advance()` returns a new state via
`model_copy(update=...)`.

Fields are organized into three tiers. The tiers are conceptual guidance,
not Python-enforced — the optimizer should feel free to add fields to
whichever tier they belong in.


# Tier 1 — Base

Unconditional bookkeeping that every subsequent stage can rely on:
how many tokens have been seen, what the previous token was, recent
context, etc. Updated by `pipeline.counters.update_basic_counters`.


# Tier 2 — Linguistic

High-level structural features you could point at with a linguistics
term. Examples:
  - clause_depth, chars_since_period, chars_since_comma
  - word_position (how deep into the current word we are)
  - sentence_type (declarative / interrogative / imperative / exclamative)
  - speaker_label_state (FSM over "\n\nNAME:\n" patterns)
  - verse_mode, syllable_position_in_line, iambic_phase
  - morphology markers (is the current suffix -ing? -ed? -ly?)
  - syntactic role of the most recent completed word

Updated by the linguistic stage of the pipeline. These are fields any
NLP textbook would recognize.


# Tier 3 — Flow

Moody, stylistic, rhythmic features — harder to define precisely,
sometimes overlapping with linguistic state but captured along a
different axis. Examples:
  - register_tension (formal ↔ colloquial)
  - emotional_arc (rising ↔ falling)
  - cadence (staccato ↔ flowing)
  - imagery_density (abstract ↔ vivid)
  - urgency (languid ↔ frantic)
  - formality_drift
  - vowel_saturation

Updated by the flow stage of the pipeline. These fields encode
something a reader feels before they can name. They may not map to
clean categories; continuous floats and soft flags are welcome.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelState(BaseModel):
    model_config = ConfigDict(frozen=True)

    # --- Tier 1: base ---
    tokens_seen: int = 0
    last_token_id: int = -1  # -1 sentinel: no token observed yet
    last_char: str = ""  # "" before any token; otherwise single char

    # --- Tier 2: linguistic ---
    # Character-class bucket of the last emitted character. See
    # pipeline.linguistic for the enumeration.
    last_char_class: int = 0
    # Bucket before last (for bigram-of-bigrams context).
    prev_char_class: int = 0
    # Length of the run of consecutive letters ending at last char
    # (0 if last char is not a letter). This is "word position + 1".
    letter_run_len: int = 0
    # Length of run of consecutive uppercase letters ending at last char.
    upper_run_len: int = 0
    # Consecutive newlines immediately preceding the cursor. 0 if last
    # char was not \n.
    consecutive_newlines: int = 0
    # Chars since the last newline (how deep into current line).
    chars_since_newline: int = 0
    # Chars since the last space or newline (rough word-start distance).
    chars_since_space: int = 0
    # Chars since the last sentence-ending punctuation (. ? !).
    chars_since_sentence_end: int = 0
    # Did we just finish a completed word? (letter run ended with non-letter)
    just_finished_word: bool = False
    # Length of the word that just finished (or the current word so far).
    current_word_len: int = 0
    # The last completed word's final letter (lowercase) — hint for
    # predicting the character that follows a space.
    last_completed_word_tail: str = ""
    # Speaker-label FSM state:
    #   0 — not in / just after speaker label
    #   1 — right after "\n\n", expecting capital letter
    #   2 — inside upper-case speaker label (UPPERS [space UPPERS]*)
    #   3 — just past ":" at end of speaker label; newline expected
    speaker_label_state: int = 0
    # Whether last char was start-of-line context that we treat as
    # "beginning of a sentence" (post double-newline or post ". ").
    sentence_start_pending: bool = False
    # Is last char a vowel (aeiouAEIOU)?
    last_is_vowel: bool = False

    # --- Tier 3: flow ---
    # (add fields here)
