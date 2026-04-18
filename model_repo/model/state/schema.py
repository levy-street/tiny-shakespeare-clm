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
    # The word before last_completed_word. Enables two-word memory for
    # 3-gram-style word-level formulas ("I pray thee", "O my lord",
    # "by my troth", "good my lord", "I have been", etc.). Set at word
    # completion, holds steady between completions.
    prev_completed_word: str = ""
    # Up to 4 most-recently-completed *content* words (NOUN, VERB,
    # VERB_ING, VERB_ED, ADJECTIVE, ADVERB, PROPER_NOUN, or UNKNOWN),
    # most-recent first. Function words (articles, pronouns, aux,
    # prepositions, conjunctions, etc.) are filtered out. This gives
    # the predict layer a rolling *content* memory that outlives the
    # two-word cursor, enabling topical-coherence biases (dark/war/
    # death clusters vs love/tender clusters vs royal/court clusters)
    # to persist across function-word scaffolding. Updated by
    # `pipeline/pos.py` at word completion.
    content_words: tuple[str, ...] = ()

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

    # --- Tier 3: verse-mode flow ---
    # Rolling score in [-3, +3] estimating whether we're inside a verse
    # passage (positive) vs. a prose passage (negative). Updated each
    # time a line completes, based on that line's length and a decay
    # toward zero. Verse lines are typically 25-55 chars with ~95% of
    # lines falling under 55; prose lines frequently exceed 60 and
    # commonly run 70-90. Downstream: verse_mode strengthens line-end
    # biases at ~30-45 chars and slightly penalizes overflow.
    verse_score: float = 0.0

    # --- Tier 3: prosody / syllable tracking ---
    # Rough syllable count in the current line, measured as
    # consonant->vowel transitions (vowel clusters). Reset on newline.
    # Shakespeare's iambic pentameter usually hits 10 syllables per line
    # (11 for feminine endings). Line-end prediction should spike at
    # syllables_in_line >= 9 in verse mode.
    syllables_in_line: int = 0
    # Syllable count in the current word only. Reset on word boundary.
    # Most English words are 1-3 syllables; 4+ is plausible but rare.
    # Used to distinguish "word ending soon" vs "word still growing".
    syllables_in_word: int = 0
    # True iff the last character is part of a vowel cluster (a/e/i/o/u,
    # lower or upper). Used by the prosody stage to detect consonant->
    # vowel transitions (the start of a new syllable).
    in_vowel_group: bool = False
    # Length of the previous line in syllables (set when a newline ends
    # a non-blank line). Helps calibrate whether the next line is also
    # expected to be verse-length.
    prev_line_syllables: int = 0

    # --- Tier 2: clause-structure tracking ---
    # Count of clausal breaks (commas/semicolons/colons, excluding those
    # inside speaker labels) since the last sentence-end punctuation.
    # Shakespeare sentences typically have 1-3 clauses before closing;
    # at 4+ clauses, sentence-end becomes highly overdue.
    clauses_in_sentence: int = 0
    # True iff the current clause was opened by a subordinating
    # conjunction ("that", "which", "who", "when", "where", "while",
    # "if", "though", "because", "unless", "as"). Reset on
    # sentence-ending punctuation or on a new comma/semicolon.
    in_dependent_clause: bool = False
    # The dominant subject pronoun of the current clause/sentence, if
    # any ("i", "thou", "he", "she", "it", "we", "ye", "you", "they").
    # Used for verb-agreement-aware mid-word bias (e.g., after "thou"
    # the form "hast" is expected, not "have"). Reset on sentence end.
    subject_pronoun: str = ""

    # --- Tier 3: register / texture ---
    # A rolling [0, 1] float estimating how *archaic* the recent register
    # has been. Bumped by archaic words like "thou", "hath", "doth",
    # "ere", "anon", "prithee", "whence", "forsooth", "'tis", "'twas",
    # and by archaic-contraction apostrophes; decays slowly per
    # completed word. Captures the scene's archaic-texture feel — not
    # counters. Consumed at word-start to bias archaic vs. modern
    # word-initial letters. This is a genuine flow/register signal:
    # lexical archaicness is self-reinforcing in Shakespeare (a speaker
    # who just said "prithee" is far more likely to say "thou" next
    # than one who just said "you" and "your").
    archaic_density: float = 0.0

    # A rolling [0, 1] float estimating *emotional intensity* — bumped
    # by "!" / "?" marks, "O"/"Oh" vocatives, and emotional
    # interjections (alas, fie, ah, alack, ay). Decays per completed
    # word. Consumed by predict to favor "!" at sentence end, boost
    # "\n" after strong exclamation, and favor emotional follow-on
    # words. Emotional outbursts cluster in Shakespeare: once a
    # speaker cries "O!" they tend to repeat — this field captures
    # that texture.
    emotional_intensity: float = 0.0

    # --- Tier 2: clause slot state machine ---
    # Coarse syntactic-slot tracker for the current clause:
    #   0 = FRESH       — sentence start / post-clause-break; expect
    #                     subject, interjection, or WH-question word.
    #   1 = HAS_SUBJ    — saw subject-like element (pronoun, proper
    #                     noun, determiner+noun); expect aux/modal/
    #                     verb or adjective.
    #   2 = HAS_VERB    — saw aux/modal/verb; expect object (noun,
    #                     pronoun, adjective, participle, preposition).
    #   3 = POST_OBJ    — saw object/complement; clause is complete;
    #                     expect sentence end, conjunction, or clausal
    #                     break.
    # Transitions are driven by the POS of last_completed_word. This
    # gives the predict layer a real syntactic-position prior that
    # tells it "after a subject, a verb is expected; after a verb,
    # a determiner/noun is expected" — which the existing n-gram
    # layers don't see.
    clause_slot: int = 0
    # Number of completed words since the last verb-like (VERB,
    # AUX_VERB, MODAL, VERB_ING, VERB_ED) word was seen. Reset to 0
    # on a verb-ish word completion and at sentence-end punctuation.
    # High values indicate the clause has been wandering through
    # function words and content without reaching a verb — a strong
    # syntactic signal that a verb is overdue.
    words_since_verb: int = 0

    # True when the last completed word was a vocative-prefix adjective
    # (good, sweet, gentle, fair, dear, poor, noble) AND the previous
    # word was also a possessive-like word ("my", "thy", "good") OR a
    # sentence break. Signals that a vocative noun (lord, sir, madam,
    # lady, friend, master) is imminent. Reset on any clause break,
    # verb, or non-adjective word. This captures a distinctive
    # Shakespearean construction: ", my dear lord," / ", good sir,".
    vocative_expectation: bool = False

    # --- Tier 3: tonal texture ---
    # --- Tier 2/3: speaker memory across turns ---
    # Uppercase canonical label of the currently speaking character,
    # captured at the moment the speaker label closes with ":". Holds
    # steady throughout the dialogue body of that turn. Reset to "" at
    # initialization. Updated by the linguistic stage exactly at the
    # 2->3 speaker-label FSM transition (when the ":" arrives).
    last_speaker_label: str = ""
    # Tuple of up to 4 most-recently-seen distinct speaker labels,
    # most-recent first. The current speaker is element [0]. Shakespeare
    # scenes usually have 2-4 recurring speakers; knowing who has
    # spoken recently lets the predict layer (a) strongly boost
    # recently-seen names at the next speaker label, and (b) penalize
    # immediate self-repetition (a speaker is very unlikely to produce
    # two adjacent speaker labels with their own name).
    recent_speakers: tuple[str, ...] = ()

    # A rolling float in [-1, +1] tracking the dark/heavy vs
    # light/hopeful tonal texture of the emerging text. Shakespeare's
    # scenes have strong tonal coherence — once "blood" and "death"
    # appear, more dark lexicon follows; once "love" and "sweet"
    # appear, more tender lexicon follows. This field bleeds that
    # register through word boundaries.
    #
    # Bumps per completed word by the word's tonal class:
    #   STRONG_DARK  (death, blood, grief, murder, hell, ...)  : -0.30
    #   MILD_DARK    (cold, pale, dim, weary, sick, ...)       : -0.12
    #   MILD_LIGHT   (bright, gentle, soft, warm, ...)         : +0.12
    #   STRONG_LIGHT (love, joy, bliss, fair, sweet, ...)      : +0.30
    # Decays toward 0 at 0.96 per completed word.
    # Resets a fraction on speaker change (consecutive_newlines >= 2).
    # Consumed by predict.tonal.word_start_bias at word-starts to shift
    # next-word first-letter mass toward the in-register lexicon.
    tonal_weight: float = 0.0

    # A rolling [0, 1] float tracking *imagery density* — how much
    # sensory, concrete, embodied lexicon has appeared recently.
    # Distinct axis from tonal_weight (dark/light valence) and
    # archaic_density (formal/archaic register): imagery is about
    # whether the text is painting pictures (blood, sword, moon,
    # eye, hand, rose, crown, shadow, blade, dagger, fire, storm)
    # vs. speaking abstractly (thought, matter, case, cause,
    # reason, purpose, sake, manner).
    #
    # Shakespeare's imagistic passages cluster: once a scene turns
    # toward sensory language, more sensory language follows. This
    # field bleeds that texture through the function-word scaffolding
    # the same way tonal_weight does, but tracks a different axis
    # and selects different next-letter priors.
    #
    # Consumed by predict.imagery.word_start_bias at word-starts
    # outside speaker labels.
    imagery_density: float = 0.0

    # --- Tier 3: second-person addressing register ---
    # Running scalar in [-3, +3] tracking whether the current speaker's
    # turn has been addressing in the thou-register (+) or you-register
    # (-). Shakespeare's characters usually pick one and stay in it
    # within a turn — thou signals intimacy / condescension / emotion,
    # you signals formal respect. Updated at word completion when a
    # 2nd-person pronoun (thou/thee/thy/thine/thyself vs you/your/
    # yours/yourself/ye) is observed; decays toward 0 each word.
    # Dampened (but not reset) on speaker-turn change — the next
    # speaker may inherit or flip.
    #
    # Consumed by predict to boost the matching series of 2nd-person
    # pronouns at word-start once the register is established.
    addressing_register: float = 0.0

    # --- Tier 2: formulaic-phrase progress ---
    # Current node ID in a precomputed trie of common multi-word
    # Shakespeare formulas ("I pray thee", "good my lord", "by my
    # troth", "thou shalt not", "I do beseech thee", etc.). 0 = root
    # (not currently inside any recognized formula match).
    #
    # Updated at word completion by pipeline/formula.py:
    #   - if the completed word advances the current node in the trie,
    #     descend deeper;
    #   - else if it starts a fresh formula at root, jump there;
    #   - else reset to 0.
    # Also resets on sentence-end punctuation and speaker-turn changes.
    #
    # Consumed by predict/formula.py to bias the first letter (and mid-
    # word continuation letter) of the next word toward expected
    # completions — giving a real multi-word lookahead that the
    # two-word phrase_bigram cannot see.
    formula_node: int = 0
