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
    prev_char: str = ""  # char before last_char (for trigram context)

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
    # Chars since the last comma, semicolon, or colon. Shakespeare
    # tends to use a comma every 12-25 chars; this helps the predict
    # layer time the next clausal pause.
    chars_since_comma: int = 0
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
    #   2 — inside speaker label (UPPERCASE or Mixed-Case)
    #   3 — just past ":" at end of speaker label; newline expected
    speaker_label_state: int = 0
    # True once we've seen a lowercase letter in state 2 — lets the
    # predict layer know this is a Mixed-Case label like "First Citizen"
    # rather than an all-caps label like "HAMLET".
    speaker_label_saw_lower: bool = False
    # Whether last char was start-of-line context that we treat as
    # "beginning of a sentence" (post double-newline or post ". ").
    sentence_start_pending: bool = False
    # Is last char a vowel (aeiouAEIOU)?
    last_is_vowel: bool = False
    # The current partially-written word (lowercased, up to a cap) since
    # the last non-letter character. Used to bias word completions.
    word_buffer: str = ""
    # Buffer of upper-case characters (and internal spaces) accumulated
    # while inside a speaker label. Used to bias toward known names.
    speaker_buffer: str = ""
    # The last fully completed word (lowercased, including trailing
    # apostrophe-suffixes like 'tis). Used for next-word bias.
    last_completed_word: str = ""

    # --- Tier 3: flow ---
    # Is the current word_buffer still a prefix of some known word? False
    # when we have no buffer or when the buffer has drifted off-trie.
    on_word_trie: bool = True
    # Rough line-length floor threshold bucket — encodes "is the line
    # already long enough to plausibly end?". Derived from chars_since_newline.
    # 0: short (<20), 1: medium (20-34), 2: long (35-49), 3: overlong (>=50)
    line_length_bucket: int = 0
    # Distance (in chars) since the last sentence-end punctuation, bucketed.
    # 0: very recent (<40), 1: due (40-79), 2: overdue (>=80)
    sent_distance_bucket: int = 0
    # True when the last completed word is a short closed-class word
    # (pronoun, auxiliary, preposition) — next word is likely content.
    after_function_word: bool = False
    # True when we're inside what looks like a prose paragraph (a newline
    # follows something other than a speaker-label/blank separator).
    in_prose_line: bool = False
    # Coarse POS tag of `last_completed_word`. See pipeline/pos.py for
    # the enumeration. 0 = UNKNOWN.
    last_word_pos: int = 0
    # POS tag of the word before last_completed_word — lets downstream
    # layers look at a two-word POS context.
    prev_word_pos: int = 0
    # Length (in chars) of the previous line. Updated exactly when we
    # emit a \n. Used to distinguish verse lines (typically short,
    # ~30-50 chars) from prose lines (typically 60+ chars). This lets
    # the predict layer know whether the next line is likely to start
    # with a capital (verse) or may continue in lowercase (prose wrap).
    prev_line_length: int = 0
    # Length of the line before the previous one — for smoothing.
    prev_prev_line_length: int = 0
    # Number of letters written in the current word since the buffer first
    # went off the known-word trie. 0 when still on-trie (or no word).
    # Grows fast when we've drifted into nonsense territory.
    letters_off_trie: int = 0
    # Number of consecutive consonant letters since the last vowel in
    # the current word. Real English words rarely allow 4+ consecutive
    # consonants (even "strength" tops out at "str"). Resets on vowel,
    # word break, or 'y' (treated as vowel-ish for phonotactic purposes).
    consonants_since_vowel: int = 0
    # Number of vowels observed in the current word. Words with 6+
    # letters and 0 vowels are implausible; we use this to force a
    # vowel when the word has gone too long without one.
    vowels_in_word: int = 0
    # Number of consecutive vowels since the last consonant in the
    # current word. Resets on consonant or word break.
    vowels_since_consonant: int = 0

    # --- Tier 2/3: sentence-type FSM ---
    # Classification of the current sentence in progress:
    #   0 = UNKNOWN (not yet classified; before first word of sentence)
    #   1 = DECLARATIVE (default; ends in .)
    #   2 = INTERROGATIVE (WH-question or aux-inversion; ends in ?)
    #   3 = EXCLAMATIVE (begins with O, Alas, How, etc.; often ends in !)
    # Updated by pipeline/sentence.py after a word completes at sentence
    # position 0. Reset to UNKNOWN on sentence-ending punctuation.
    sentence_type: int = 0
    # Number of completed words since the last sentence-end punctuation
    # (or since the start of the text). Used to detect "first word of
    # sentence" for type classification.
    words_in_sentence: int = 0
