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
    # The word before prev_completed_word. Gives the predict layer a
    # true 3-word lookback — enabling phrase-trigram biases keyed on
    # (w3, w2, w1) -> first-letter of next word. Without this the
    # model can only see 2 words back, which misses common Shakespeare
    # 4-grams like "I pray thee tell", "to be or not", "my good lord
    # I". Updated at every word completion (3-slot shift).
    prev_prev_completed_word: str = ""
    # Rolling tuple of the last up-to-5 completed words, most-recent
    # first. Generalizes prev/last/prev_prev to any-length lookback.
    # Future layers can index to look at 4 or 5 words back (e.g.
    # detecting "of X of Y" parallelism, or identifying completion
    # targets of long formulaic sequences). Reset fields reset this
    # to () where appropriate (e.g., speaker-turn boundary).
    recent_completed_words: tuple[str, ...] = ()
    # Rolling tuple of the last up-to-6 completed word LENGTHS (in
    # letters, not counting apostrophe suffixes), most-recent LAST.
    # This is a cadence/rhythm memory orthogonal to the word identity
    # tuple: it answers "have we been in a run of short staccato
    # monosyllables, or a stretch of polysyllabic declamation?"
    #
    # Shakespeare famously alternates between monosyllabic runs (for
    # emphasis: "To be or not to be", "Out, out, brief candle") and
    # polysyllabic cadences (for elaboration: "The multitudinous
    # seas incarnadine"). After several very short words a longer
    # content word often follows; after a long polysyllabic word the
    # next word is typically short. This field lets a predict layer
    # observe that prosodic rhythm and bias first-letter and word-end
    # pressure to match the expected next-word length.
    #
    # Cap 6. Consumed by predict/word_length_cadence.py at word-start.
    # No reset on turn boundary — rhythm persists across speakers.
    recent_word_lengths: tuple[int, ...] = ()
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
    # Number of COMPLETE known words that still start with word_buffer.
    # 0 = no known word remains possible; 1 = exactly one completion; >1
    # = many. Graded complement to on_word_trie. 0 when word_buffer
    # is empty. Updated by pipeline/word_matches.py.
    trie_match_count: int = 0
    # `trie_match_count` on the previous token. Lets downstream predict
    # layers detect the exact character that dropped the count to 0
    # (and react strongly the next step).
    prev_trie_match_count: int = 0
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
    # --- Tier 2: coordinator parallelism (X and Y, X or Y, X nor Y) ---
    # When a coordinating conjunction ("and", "or", "nor") completes, we
    # record the POS tag of the word that sat immediately BEFORE the
    # conjunction (the "X" of "X and Y"). This is the POS that is
    # strongly expected to echo onto the NEXT word ("Y") — Shakespeare
    # heavily favors parallel POS across coordinators:
    #   - "fair and foul"     (adj + adj)
    #   - "night and day"     (noun + noun)
    #   - "Romeo and Juliet"  (proper + proper)
    #   - "thou and I"        (pronoun + pronoun)
    #   - "to live or die"    (verb-infin + verb-infin)
    # 0 = inactive. Set on the tick the coordinator word completes;
    # consumed (cleared) when the NEXT word completes, or earlier on
    # sentence-end / turn-boundary. A separate flag `coord_echo_pending`
    # is True for the narrow word-start slot where the bias should fire.
    coord_echo_pos: int = 0
    # True at word-start positions between the coordinator's trailing
    # space and the first letter of the following word. The predict
    # layer reads this + coord_echo_pos and biases the first letter of
    # the coming word toward typical starters of the echoed POS class,
    # plus an alliterative first-letter echo and case-echo.
    coord_echo_pending: bool = False
    # First letter (lowercased) of the word that sat immediately before
    # the coordinator. Used to push alliterative echo on the first
    # letter of the word following the coordinator — Shakespeare's
    # coordinate pairs are frequently alliterative: "fair and foul",
    # "kith and kin", "beck and call", "tooth and nail", "day and
    # dark". "" when no echo is active.
    coord_echo_first_letter: str = ""
    # True iff the pre-coord word was capitalized (proper-noun-like,
    # e.g. "Romeo"). When True, the word following the coordinator is
    # overwhelmingly likely to also be capitalized ("Romeo and
    # Juliet", "Cassio and Iago"). The predict layer reads this to
    # push upper-case A-Z at the next word-start.
    coord_echo_was_capital: bool = False
    # Helper flag: snapshot of `current_word_started_cap` captured at
    # the moment the PRECEDING word completed. Lets us detect the
    # mid-sentence-cap (proper-noun-like) status of the word BEFORE
    # the currently-completing word — needed at the tick when a
    # coordinator ("and") completes so we know whether the pre-coord
    # word was a proper-noun-like capital.
    coord_prev_word_started_cap: bool = False
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
    # The letter_run_len at the moment the current word FIRST left the
    # word-trie. 0 while the word is still on the trie (or no word).
    # Unlike letters_off_trie (which tracks HOW FAR we've drifted),
    # this records WHERE THE DEPARTURE HAPPENED. A late departure
    # (>= 5) means we had a solid real prefix that's now being extended
    # into nonsense — the trie knows no word of this shape. An early
    # departure (1-2) means the word was gibberish from the start.
    # Predict layers use this to modulate word-end pressure and
    # gibberish-letter penalty with a much sharper signal than the
    # current-letters-off count alone.
    offtrie_depart_pos: int = 0

    # --- Tier 2: mid-departure extension length ---
    # Counts letters written since the current word drifted off the
    # word-trie, but ONLY for words whose departure happened at
    # position 3 or 4 (i.e., had a plausible 3-4 letter prefix but
    # then stepped off). This is the exact regime the existing
    # offtrie_depart_bias layer leaves unhandled (it returns None for
    # depart_pos <= 4), and also the regime trie_recovery handles
    # only weakly (its term_boost is 0). Real gibberish samples like
    # "etustartea", "Fulfilm", "iegeohce" depart at positions 3-4 and
    # then drift 4-6 more letters before terminating.
    #
    # Semantics:
    #   Active (>= 1) iff on_word_trie == False, letters_off_trie >= 1,
    #                  and offtrie_depart_pos in {3, 4}.
    #   Value = letters written past the departure point
    #   (= letters_off_trie for this regime).
    #   0 whenever any condition is false: on-trie, early-dep (1-2),
    #   late-dep (>= 5 — already covered by other layers), or no word.
    #   Reset to 0 at word completion.
    #
    # Consumed by predict/mid_departure.py to apply the missing
    # terminator-plus-end-letter pressure in the 5-10 char gibberish
    # window.
    mid_departure_extension: int = 0

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
    # Classification of the PREVIOUS sentence, preserved across the
    # sentence boundary. Saved when PUNCT_END fires (before sentence_type
    # is reset). Lets predict condition the first letter of a new
    # sentence on what kind of sentence just ended:
    #   - After ? (INTERROG): a response sentence typically starts with
    #     a declarative or an interjection ("Ay,", "No,", "I ...");
    #     another wh/aux opener is less likely.
    #   - After ! (EXCLAM): emotional momentum — another exclamative
    #     opener (O/Ah/Alas) is elevated, as is a first-person declarative.
    #   - After . (DECL): neutral default.
    # Reset to SENT_UNKNOWN on speaker-turn boundary.
    prev_sentence_type: int = 0

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

    # --- Tier 3: caesura (mid-line pause) tracking ---
    # Syllable position (within the current line) at which the last
    # mid-line punctuation break fired (comma, semicolon, colon not in
    # speaker label, em-dash). -1 means no caesura has fired in the
    # current line yet. Reset to -1 on newline.
    #
    # Why: Shakespeare's iambic pentameter typically has a caesura at
    # syllable 4, 5, or 6 — the mid-line balance pause. Prose lacks a
    # fixed caesura but still breaks on mid-clause punctuation. Tracking
    # whether the current line has received its caesura, and where,
    # lets the predict layer (a) push a comma/semicolon up at word-end
    # when syllables_in_line is in the caesura-expected range AND no
    # caesura has fired yet (in verse); and (b) suppress a second
    # mid-line break right after a caesura just fired.
    caesura_syllable: int = -1
    # True iff the current line has already received a mid-line break.
    # Reset to False on newline.
    has_caesura_this_line: bool = False

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

    # A rolling [0, 1] float estimating *meditative register* — the
    # philosophical / inward-gazing texture of speech. Bumped by
    # abstract-mental vocabulary (think, thought, mind, soul, spirit,
    # dream, doubt, wonder, question, nature, reason, conscience,
    # truth, memory, etc.) and weakly by subjunctive/conditional
    # framings. Decays per completed word. Distinct from
    # emotional_intensity (which is reactive/outward) and from
    # imagery_density (which is corporeal/sensory) — this captures
    # the Hamlet-soliloquy feel vs. the battlefield-cry feel.
    # Consumed at word-start to bias toward meditative-lexicon
    # first letters (t, m, s, d, w, n, r) and against concrete
    # battlefield-continuation letters when high.
    meditative_register: float = 0.0

    # A rolling float in [-1.0, +1.0] capturing the *confessional-
    # intimacy* ↔ *public-declamation* register polarity.
    #
    #   High positive (≈ +0.4 … +1.0): the speaker is in a confessional/
    #     soliloquy/intimate-address register. Signaled by 1sg pronouns
    #     (I / me / my / mine), interior-state verbs (think, feel,
    #     fear, hope, suspect, doubt, remember, wish, dream), intimate
    #     2sg address (thou / thee / thy / thine), tender vocabulary
    #     (heart, soul, breath), contractions (I'm, 'tis, ne'er),
    #     sigh-interjections ("alas", "ah", "O").
    #
    #   High negative (≈ -0.4 … -1.0): the speaker is in a public/
    #     oratorical/ceremonial register. Signaled by 1pl/2pl pronouns
    #     (we / our / you / your / ye), plural vocatives (lords,
    #     friends, gentlemen, countrymen, masters, sirs), imperative
    #     commands (hear / behold / mark / attend / come / go), titles
    #     and honorifics (majesty, grace, highness, excellence, lord),
    #     ceremonial openers (now / hear / witness).
    #
    # Bumps happen at word completion based on the completed word. Decays
    # toward 0 at 0.93 per completed word. Resets toward 0 on speaker-
    # turn boundary (a new speaker inherits none of the prior register).
    #
    # Distinct from every existing register:
    #   - fury / gravitas / tenderness / lament / doubt: emotional tones;
    #     a speech can be confessional-AND-tender or public-AND-grave.
    #   - addressing_register (thou vs. you): pronoun-form commitment,
    #     not audience size. Thou can be confessional (lover) or public
    #     (king addressing subject).
    #   - archaic_density: vocabulary age, orthogonal to audience register.
    #
    # Consumed by predict/confessional.py at word-start outside speaker
    # labels to tilt first-letter mass toward in-register lexicon.
    confessional_intimacy: float = 0.0

    # --- Tier 2/3: antithesis / rhetorical contrast ---
    # Shakespeare's signature antithesis structures — "not X but Y",
    # "to be or not to be", "neither A nor B", "more X than Y" —
    # create a two-part prosody that the model can exploit. After a
    # contrast-opener word ("not", "nor", "neither", "either",
    # "rather", "more", "less", "either") appears, the next pivot
    # word ("but", "or", "nor", "than", "yet") is elevated, and
    # after the pivot the complement half is expected.
    #
    # antithesis_state:
    #   0 = NONE          — no active contrast
    #   1 = OPENER_SEEN   — an opener word has fired; pivot expected
    #                       within a few words
    #   2 = PIVOTED       — pivot word has fired; we're in the
    #                       contrast complement half
    # Reset on sentence-end punctuation (. ? !) and on speaker-turn
    # boundary (consecutive_newlines >= 2 AND last_char == "\n").
    antithesis_state: int = 0
    # Completed-words since the opener fired. Used to decay the
    # OPENER_SEEN state back to NONE after ~6 words without a pivot
    # (the contrast never materialized). Zero in state NONE / PIVOTED
    # opener_age reset.
    antithesis_words_since_opener: int = 0
    # Completed-words since the pivot fired. Lets the predict layer
    # know how deep into the complement half we are — closing
    # punctuation becomes more likely after 3-5 complement words.
    # Zero unless antithesis_state == PIVOTED.
    antithesis_words_since_pivot: int = 0

    # Specific opener-type that fired. Lets the predict layer bias
    # the EXACT paired pivot letter rather than a generic "any pivot"
    # set. English correlatives pair:
    #   0 = NONE            — no specific opener
    #   1 = NEITHER         → expects "nor"       (letter n)
    #   2 = EITHER          → expects "or"        (letter o)
    #   3 = BOTH            → expects "and"       (letter a)
    #   4 = MORE_LESS       → expects "than"      (letter t)
    #   5 = NOT             → expects "but"       (letter b)
    #   6 = WHETHER         → expects "or"        (letter o)
    #   7 = THOUGH_ALBEIT   → expects "yet"       (letter y)
    #   8 = RATHER          → expects "than"      (letter t)
    # Generic antithesis_state already provides a catch-all bias; this
    # type enables a sharper, more targeted pivot-letter elevation.
    antithesis_opener_type: int = 0

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

    # --- Tier 2: subordinate-clause depth tracker (new axis) ---
    # Depth of currently-open subordinate / relative clauses inside
    # the current sentence. Increments when a subordinator (that,
    # which, who, whom, whose, where, when, while, whilst, though,
    # although, if, unless, because, till, until, since, as, ere,
    # lest, whereas) completes in a position where it opens a
    # dependent clause. Decrements when the dependent clause closes
    # (detected via next-comma-that-isn't-nested or sentence-end).
    # Hard-capped at 3; reset to 0 on sentence-end punctuation and
    # on speaker-turn boundaries.
    #
    # Why this matters: Shakespeare sentences are often multi-clausal:
    # "The king WHO loves her WHEN she sings IS happy." Without
    # tracking subordinate depth, clause_slot can't distinguish
    # "verb in relative clause" (doesn't close main clause) from
    # "main-clause verb" (does). A model that knows subord_depth > 0
    # can keep clause_slot = HAS_VERB open for the main clause even
    # after a dependent verb lands, and can resist premature sentence
    # termination inside deep nesting.
    subord_depth: int = 0
    # Words since the last subordinator was emitted. Grows per word
    # completed while subord_depth > 0; resets on subordinator entry
    # and on sentence-end. Useful for deciding when a subordinate
    # clause "should close" — 4+ words in and no verb yet means the
    # clause is malformed or closing soon.
    subord_words_since_open: int = 0
    # clause_slot at the moment the most-recent subordinate clause
    # opened (so we can restore context when it closes). Saved as a
    # small stack flattened into a single int (3 bits per level,
    # cap depth 3): lowest 3 bits = level 1, next 3 = level 2, etc.
    subord_slot_stack: int = 0

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
    # Tuple of up to 7 most-recently-seen distinct speaker labels,
    # most-recent first. The current speaker is element [0]. Shakespeare
    # scenes usually have 4-6 recurring speakers; the seven-slot
    # capacity holds the full cast of an ensemble scene while still
    # excluding stale cross-scene speakers. Knowing who has spoken
    # recently lets the predict layer (a) strongly boost recently-seen
    # names at the next speaker label, and (b) penalize immediate
    # self-repetition (a speaker is very unlikely to produce two
    # adjacent speaker labels with their own name).
    recent_speakers: tuple[str, ...] = ()
    # Categorical register of the current speaker (recent_speakers[0]),
    # derived from a hand-curated map over canonical Shakespeare character
    # names. Used by predict consumers to condition word-start vocabulary:
    #   0 UNKNOWN         — no match / empty turn
    #   1 TRAGIC_NOBLE    — Hamlet, Lear, Macbeth, Othello, Romeo, Brutus
    #   2 COMIC_PROSE     — Fool, Launce, Bottom, Dogberry, Touchstone
    #   3 ROYAL_FORMAL    — Henry, Richard, Edward, Duke, Prince, Caesar
    #   4 VILLAIN         — Iago, Edmund, Richard III, Aaron, Angelo
    #   5 LOVER_FEMININE  — Juliet, Viola, Rosalind, Portia, Miranda, Desdemona
    #   6 SERVANT_BRIEF   — Messenger, Servant, Citizen, Officer, Soldier
    #   7 SUPERNATURAL    — Ghost, Witch, Oracle, Fairy, Ariel, Puck
    speaker_register: int = 0
    # --- Tier 2/3: play-family lock ---
    # Shakespeare scenes never mix characters from different plays. Our
    # samples do: a HAMLET-prefixed turn will introduce NERISSA (Merchant
    # of Venice), LEONATO (Much Ado), AUMERLE (Richard II), etc. This
    # field captures WHICH PLAY-FAMILY the current scene belongs to,
    # inferred from `last_speaker_label` / `recent_speakers`:
    #   0 UNKNOWN           — no family-defining speaker seen yet
    #   1 HAMLET_DANE       — Hamlet, Horatio, Ophelia, Polonius, Laertes,
    #                         Claudius, Gertrude, Fortinbras, Rosencrantz,
    #                         Guildenstern, Osric, Marcellus, Bernardo
    #   2 ROMAN             — Caesar, Brutus, Cassius, Antony, Cleopatra,
    #                         Coriolanus, Menenius, Volumnia, Aufidius,
    #                         Titus, Tamora, Aaron, Enobarbus, Octavius
    #   3 ENGLISH_HISTORY   — Henry, Richard, Hal, Hotspur, Falstaff,
    #                         York, Warwick, Gloucester, Buckingham,
    #                         Bolingbroke, Northumberland, Aumerle,
    #                         Mowbray, Percy, Margaret, Catesby, Stanley
    #   4 OTHER_TRAGEDY     — Lear, Cordelia, Edmund, Macbeth, Banquo,
    #                         Othello, Iago, Desdemona, Romeo, Juliet,
    #                         Mercutio, Tybalt, Friar, Timon, Apemantus
    #   5 COMEDY_PROSE      — Beatrice, Benedick, Leonato, Portia,
    #                         Nerissa, Bassanio, Shylock, Rosalind, Celia,
    #                         Orsino, Viola, Malvolio, Bottom, Puck,
    #                         Petruchio, Launce, Dogberry, Touchstone,
    #                         Demetrius, Hermia, Helena, Oberon, Titania
    #   6 ROMANCE           — Prospero, Miranda, Caliban, Ariel, Leontes,
    #                         Hermione, Perdita, Autolycus, Cymbeline,
    #                         Imogen, Posthumus, Pericles, Marina
    #
    # Lock semantics: updated by pipeline/play_family.py at the moment
    # a new speaker label closes (FSM 2→3 with ":"). If the new speaker
    # maps to a family, it OVERWRITES (most recent wins — Shakespeare
    # scene-changes can happen). Unknown speakers (SERVANT, MESSENGER,
    # LORD, CITIZEN, GENTLEMAN, etc.) are NEUTRAL — they do not
    # overwrite. Cleared only at simulation start (default 0).
    #
    # Consumed by predict/play_family.py at speaker_label_state in
    # {1, 2} to tilt letter distributions toward in-family speaker
    # names and away from out-of-family ones.
    play_family: int = 0
    # Number of tokens since speaker_register was last updated. Lets a
    # consumer taper the bias strength early in a turn (first few tokens
    # are usually the speaker label itself).
    register_age: int = 0
    # --- Tier 2: thou/you register commit (Early Modern address form) ---
    # In Early Modern English a speaker chooses between the T-form
    # (thou/thee/thy/thine/thyself — singular, familiar, intimate or
    # condescending) and the V-form (you/your/yours/ye — plural or
    # polite-singular) for their addressee. Once a speaker commits
    # within a turn, mixing is JARRING — "Thou art mad. You look
    # pale." is ungrammatical Shakespeare.
    #
    # State codes:
    #   0 UNCOMMITTED — no 2nd-person pronoun seen this turn
    #   1 T_COMMIT     — thou / thee / thy / thine / thyself seen;
    #                    also implicit via -st auxiliary forms
    #                    (hast, didst, wilt, shalt, canst, art, wert)
    #   2 V_COMMIT     — you / your / yours / ye seen
    #
    # Reset rule: on turn boundary (consecutive_newlines >= 2). Once
    # committed, stays committed for the rest of the turn, giving
    # downstream consumers a TURN-LEVEL prior that reinforces the
    # clause-level verb_agreement signal across sentence breaks.
    #
    # Consumed by predict.register_commit_bias at word-start to tilt
    # 2nd-person-pronoun leading letters (t/T vs y/Y) in favor of
    # the committed register.
    thou_thee_commit: int = 0
    # Rolling tuple of the last 4 completed sentences' types (most-recent
    # LAST). Each entry is a sentence_type integer (SENT_DECL / INTEROG /
    # EXCLAM / IMPER / UNKNOWN). Unlike prev_sentence_type (1-back), this
    # captures DISCOURSE-LEVEL rhythm: three-questions-in-a-row, two-
    # declaratives-then-exclamation, etc. — patterns that shape a speaker's
    # voice over multiple sentences and that 1-back memory can't see.
    # Reset on turn boundary (new speaker).
    recent_sentence_types: tuple[int, ...] = ()
    # Parallel rolling tuple of the last 4 completed sentences' word
    # counts (most-recent LAST). Captures sentence-LENGTH rhythm —
    # short-short-long staccato vs long-sentence declamatory mode.
    # Reset on turn boundary.
    recent_sentence_lengths: tuple[int, ...] = ()

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

    # --- Tier 2: line-starter anaphora tracking ---
    # Count of completed words so far on the current line. Reset to 0
    # on newline. Used to detect "this is the first word of the line"
    # (transitions to 1 on the space that terminates the first word).
    words_completed_on_line: int = 0
    # Rolling tuple of the last 3 first-words-of-line. Each time the
    # first word of a line completes, append; oldest is dropped.
    # Captures anaphoric patterns — "Now is... / Now are...",
    # "O, that... / O, that...", "And... / And...".
    # Empty tuple when we haven't seen 3 line-starters yet.
    #
    # Consumed by predict.anaphora at line-start positions: when the
    # letters agree across the tuple, boost the shared starter letter
    # at the next line-start.
    recent_line_starters: tuple[str, ...] = ()

    # --- Tier 2: line-opener POS pattern memory ---
    # Rolling tuple of up to 4 POS tags of the FIRST word of each of
    # the most recent completed lines (most-recent LAST). Captured at
    # the same moment as `recent_line_starters` — the moment a word
    # completes while `words_completed_on_line` transitions 0 → 1.
    #
    # Motivation: anaphora bias currently fires only on matching
    # line-starter WORDS. Verse anaphora also operates at the POS
    # level — Shakespeare opens successive lines with the same POS
    # class even when the actual word differs ("I know... I cannot...
    # I would..."  = three PRONOUN openers; "Hard... Sharp... Cold..."
    # = three ADJECTIVE openers). A POS-level opener memory lets the
    # predict layer boost openers of the same class without requiring
    # a literal word match, which the existing word-tuple bias cannot
    # do when it only has one or zero matching letters.
    #
    # Reset rule: cleared on speaker-turn change (consecutive_newlines
    # >= 2) to avoid letting one speaker's rhythmic anaphora leak into
    # the next speaker's opening.
    recent_line_opener_pos: tuple[int, ...] = ()

    # --- Tier 2: line-TERMINAL word memory (mirror of recent_line_starters) ---
    # Rolling tuple of up to 4 completed-word lowercased forms that
    # TERMINATED recent verse-plausible lines, most-recent first. This
    # is the mirror of `recent_line_starters`: that field tracks line
    # BEGINNINGS for anaphora; this field tracks line ENDINGS for
    # EPISTROPHE (word-identity rhyme) and closing-word parallelism.
    #
    # Shakespeare uses epistrophe rhetorically:
    #   "I'll have my bond, and therefore speak no more.
    #    I will not be made a soft and dull-eyed fool,
    #    To shake the head, relent, and sigh, and yield
    #    To Christian intercessors. Follow not;
    #    I'll have no speaking; I will have my bond."
    # Note the repeated "bond" at line-end. Or in couplets, the SAME
    # ending word sometimes echoes ("... vain ... vain ... pain").
    #
    # Captured at newline when:
    #   - consecutive_newlines == 1 (not a blank-line turn boundary)
    #   - prev_line_length in [1, 80] (not a huge prose run)
    #   - the line didn't end in ":" (speaker labels don't epistrophize)
    #   - last_completed_word is non-empty and all-lowercase letters
    #     (skip proper nouns, numerals, etc.)
    #
    # Reset on speaker-turn change (blank-line block).
    #
    # Consumed by predict/line_end_echo.py: at a word-start late in a
    # verse-plausible line (syllables_until_line_end small, line_length
    # moderate, meter_confidence non-trivial), boost the first letter
    # of each remembered line-ender to support rhetorical closing-word
    # recurrence. Mild strength — epistrophe is a stylistic choice
    # Shakespeare uses occasionally, not a dense prior.
    recent_line_end_words: tuple[str, ...] = ()

    # --- Tier 2/3: cross-turn rhythm memory (stichomythia axis) ---
    # Rolling tuple of up to 4 recent COMPLETED speaker-turn line-counts,
    # most-recent first. A "turn" is the block between two blank-line
    # separators. Captured at the same moment dialogue_adjacency takes
    # its snapshot (the consecutive_newlines 1→2 transition).
    #
    # This complements dialogue_adjacency's 1-back snapshot: we hold a
    # short HISTORY of turn shapes so predict layers can see patterns
    # across multiple exchanges, not just the immediately prior turn.
    # Examples the rolling tuple captures that 1-back can't:
    #   (1, 1, 1) → stichomythia: three terse turns in a row → the
    #       current turn is likely also a short quick retort.
    #   (8, 7, 6) → sustained declamatory exchange → current turn
    #       more likely to run multi-line.
    #   (12, 1) → monologue followed by one-word reaction → current
    #       turn is likely the start of a fresh topic swing.
    recent_turn_line_counts: tuple[int, ...] = ()

    # Categorical derivation from recent_turn_line_counts, computed in
    # the same stage that updates it:
    #   0 UNKNOWN   — fewer than 2 completed turns in history, or
    #                 ambiguous pattern.
    #   1 RAPID     — last 2+ turns each had ≤ 2 lines (rapid exchange,
    #                 stichomythia). Boosts early turn-end terminators.
    #   2 SUSTAINED — last completed turn had ≥ 6 lines (declamatory
    #                 mode). Suppresses early turn-end during current
    #                 turn's first sentence.
    stichomythia_mode: int = 0

    # --- Tier 2/3: within-turn line-word-count cadence ---
    # Rolling tuple of up to 3 recently-completed body-line word counts,
    # most-recent first. Captured at the body-newline boundary
    # (consecutive_newlines becomes 1) BEFORE turn_progress resets the
    # in-line counter. Never captures the turn-terminator blank line.
    #
    # Used by predict/line_word_cadence.py to derive a target length
    # for the in-progress line: when a speaker has just completed 2-3
    # lines of ~7 words each, the current line likely targets ~7 words
    # too, so as line_word_count approaches the running mean, boost the
    # newline terminator; below mean, softly suppress it.
    #
    # Complements `recent_turn_line_counts` (cross-turn shape) and
    # `prev_line_length` (in chars, single value) by adding a
    # *per-line word-count* history at turn-internal scale.
    #
    # Reset on speaker-turn boundary (consecutive_newlines >= 2).
    recent_line_word_counts: tuple[int, ...] = ()

    # Count of completed words in the CURRENT body line, since the last
    # body \n (consecutive_newlines == 1) or since the turn started.
    # Incremented at just_finished_word when speaker_label_state == 0
    # and consecutive_newlines == 0. Reset to 0 at body \n and at
    # turn boundary (consecutive_newlines >= 2).
    line_word_count: int = 0

    # --- Tier 3 FLOW: archaic-density texture ---
    # A smoothed float in [0.0, 1.0] tracking how archaic the
    # speaker's current diction feels. This is a true *flow* register —
    # it tries to capture idiom-texture rather than a structural
    # fact. Decays with a per-word multiplicative factor and bumps up
    # on each archaic-lexicon hit (thou/thee/thy/thine, hath/doth/
    # hast/dost, wilt/shalt/art, ere/oft/anon, hither/thither/whither/
    # whence/hence/thence, yon/yonder, methinks/prithee/wherefore/
    # forsooth/troth/marry/aye/nay, and apostrophe-elided 'tis/'twas/
    # 'twere/'gainst/o'er).
    #
    # Shakespeare speakers are reasonably consistent within a
    # passage: once a character starts using "thou hast", they tend
    # to continue using archaic forms; once another speaker is
    # established in modern usage, they stay modern. This field
    # captures that autocorrelation without committing to a hard
    # categorical split (which speaker_register already does
    # on a coarse categorical axis).
    #
    # Updated by pipeline/archaic_density.py on each completed word;
    # reset to 0.0 on turn boundary (consecutive_newlines >= 2).
    # Consumed by predict/archaic_density.py to tilt mid-word
    # suffix choices and word-start letters when density is hot.
    archaic_density: float = 0.0

    # --- Tier 2: short-range word-repetition memory ---
    # Tuple of up to 6 completed-word lowercased forms, most-recent
    # first, since the last strong boundary. Reset on sentence-ending
    # punctuation (. ? !) and on speaker-turn change
    # (consecutive_newlines >= 2). Used at the next word-start to
    # suppress echo-loop pathology: samples frequently drift into
    # "there there there" / "hear hear hear" because mid-word
    # content-repeat bias pulls toward a word that was already said.
    # This field lets the predict layer apply a growing first-letter
    # penalty for words that have already been emitted in this clause.
    recent_clause_words: tuple[str, ...] = ()

    # --- Tier 2: word-trie drift recovery ---
    # Tracks whether the current word-in-progress has, at any point
    # during its growth, equaled a member of COMPLETE_WORDS — i.e.
    # at some earlier letter position we could have emitted a space
    # and produced a real word. Reset to False at word boundary.
    #
    # Motivation: samples reveal that a sizable fraction of emitted
    # "words" are letter-garbage ("oonshul", "naitagomo", "ristotb").
    # These happen because the word_trie keeps extending on-trie even
    # past complete words, and once the buffer drifts off-trie the
    # letter-n-gram bias keeps emitting plausible-sounding letters
    # indefinitely. There's no back-pressure that says "you had a
    # valid word three letters ago — stop now".
    has_seen_complete: bool = False
    # Number of letters emitted since the current word_buffer was
    # last a member of COMPLETE_WORDS. 0 when wb itself is complete,
    # or when no complete prefix has been reached yet in this word.
    # Growing value signals we've drifted past a viable stop point;
    # consumed by `predict/trie_recovery.py` to escalate terminator
    # bias as drift grows.
    letters_past_complete: int = 0

    # --- Tier 2/3: rhyme position / line-tail memory ---
    # Last 3 letters (lowercased) of the most recently completed
    # non-empty verse-plausible line. Captured at the "\n" that closed
    # the line, iff the line itself had letters. Empty string when
    # there is no previous line (start of text, speaker-turn change,
    # or the line before was blank or a speaker-label).
    #
    # Shakespeare's verse uses couplets (AA) to close scenes and
    # sonnets; his quatrains use ABAB. A model with no memory of the
    # previous line's ending letter can never produce a rhyme by
    # construction — it can only stumble onto one. This field lets
    # the predict layer, when near line-end in verse mode, bias the
    # current word's completion toward letters that match the
    # previous tail — a letter-level proxy for rhyme.
    prev_line_tail: str = ""
    # The line-tail before prev_line_tail, for ABAB scheme detection.
    # Same reset rules.
    prev_prev_line_tail: str = ""
    # Rolling 3-char buffer of the most recent letters on the CURRENT
    # line (lowercased, non-letters ignored). Used to capture
    # prev_line_tail at the moment the line closes with \n. Reset on
    # newline (the line closed) and on speaker-turn change.
    line_tail_buffer: str = ""
    # Consecutive count of verse-plausible lines (length 15-55 chars,
    # non-empty, non-label) ending at the most recent \n. Helps the
    # rhyme predict layer gate: don't bias toward rhyme in prose
    # passages or at the first line of a turn.
    verse_line_run: int = 0

    # --- Tier 3: enjambment / line-flow texture ---
    # Rolling [0, 1] float — higher = the recent verse lines have been
    # ENJAMBED (ran over to the next line without terminal punctuation);
    # lower = the recent lines have been END-STOPPED (closed with
    # ., ?, !, :, ;, or , before the newline).
    #
    # Shakespeare's verse texture shifts between sections of
    # end-stopped closed-couplet rhythm ("And this our life exempt
    # from public haunt / Finds tongues in trees, ...") and sections
    # of enjambed speech-like flow ("Now is the winter of our
    # discontent / Made glorious summer by this sun of York"). The
    # current line's expected ending depends heavily on which
    # texture is active.
    #
    # Updated on newline closing a non-empty, non-speaker-label line:
    #   If the char immediately before the \n was a letter / y-vowel
    #     → line was enjambed; pull density up by ENJAMBMENT_UP.
    #   Else (. , ; : ? ! - etc.) → line was end-stopped; pull density
    #     down by ENJAMBMENT_DOWN.
    # Reset to 0.5 on speaker-turn boundary (consecutive_newlines >= 2).
    # Initializes to 0.5 (neutral) so downstream consumers can center-
    # normalize without systematically biasing one direction before any
    # verse lines have landed.
    enjambment_density: float = 0.5
    # Whether the LAST closed verse-plausible line was enjambed (True)
    # or end-stopped (False). A 1-bit companion to the rolling density
    # — the immediately preceding line is the strongest local anchor
    # for the current line's expected closure. Reset on speaker-turn.
    prev_line_enjambed: bool = False

    # --- Tier 3: polysyllable / word-length flow texture ---
    # Rolling [0, 1] EMA of how "polysyllabic" the recent words have
    # been. A polysyllabic word is estimated as one of:
    #   * length >= 7 characters
    #   * vowel count >= 3
    # Updated at the moment a word closes (space/punct/newline after a
    # letter run), outside speaker-label territory, skipping very short
    # fragments (< 2 letters) which are mostly single-character
    # fillers / fragments.
    #
    # Signal: Shakespeare shifts between plain-speech passages (mostly
    # monosyllables — "To be or not to be") and Latinate/elaborate
    # passages (polysyllabic — "philosophical", "providence",
    # "multitudinous"). A predict consumer can lean into the current
    # rhythm by nudging mid-word extension (letter vs space) based on
    # where the density sits.
    #
    # Initialized at 0.5 so downstream consumers can center-normalize.
    # Reset to 0.5 on speaker-turn boundary.
    polysyllable_density: float = 0.5

    # --- Tier 2/3: addressee / vocative memory ---
    # The most recent vocative noun used (lowercased) to address the
    # interlocutor within the current speaker turn. Empty when none
    # has been recorded. Reset on speaker-turn boundary
    # (consecutive_newlines >= 2). Captured at word completion when
    # the completed word matches a known vocative-noun class AND the
    # word immediately preceding it was a vocative-lead ("my", "thy",
    # "good", "dear", ...) — the diagnostic two-word construction.
    #
    # Motivation: Shakespeare speakers are remarkably consistent in
    # which addressee-noun they use within a single turn. A speaker
    # who opens with "my lord" will say "my lord" again ten lines
    # later, not "my friend". The existing vocative_expectation flag
    # knows a vocative is imminent but has no memory of WHICH noun —
    # so the predict layer biases the same generic l/s/m/f/p letter
    # set every time. This field adds the memory.
    last_vocative: str = ""
    # Count of vocative-noun mentions observed this turn. 0 at turn
    # start; bumped each time last_vocative is updated. Used as a
    # confidence gauge: 2+ mentions of the same noun is very strong
    # evidence that future vocatives in this turn will be that noun.
    turn_vocative_count: int = 0

    # --- Tier 2/3: dialogue-turn progress ---
    # Position within the current speaker's turn (between consecutive
    # speaker labels). Shakespeare's turns have internal structure the
    # model currently can't see:
    #   - First sentence of a turn is the OPENER. Common starters:
    #     interjections (O, Alas, Why, Nay, Ay, Well), vocatives
    #     ("My lord,..."), or direct questions. Later sentences rarely
    #     open the same way.
    #   - First word of the first line often carries a high density
    #     of turn-opener patterns.
    #   - Long verse turns (~10+ lines) often wind down with a rhyming
    #     couplet.
    #
    # These fields track a positional axis the model is blind to today.
    # `last_speaker_label` says WHO is speaking; these say HOW FAR into
    # that speaker's turn we are.
    #
    # Reset rule: all three reset to 0 when consecutive_newlines >= 2
    # (a between-turn blank line — the canonical turn boundary).
    # Updates:
    #   - words_in_turn  += 1 at each just_finished_word outside a
    #                        speaker-label (speaker_label_state == 0)
    #   - sentences_in_turn += 1 at each sentence-end punct outside
    #                            a speaker-label
    #   - lines_in_turn  += 1 at each \n whose consecutive_newlines==1
    #                        outside a speaker-label (a body newline,
    #                        not the turn-boundary \n)
    words_in_turn: int = 0
    sentences_in_turn: int = 0
    lines_in_turn: int = 0

    # --- Tier 3: turn emphasis texture ---
    # Counts of specific sentence-end punctuation tokens in the current
    # speaker turn. Together with `sentences_in_turn` they describe the
    # emphatic shape of the turn: an "!"-heavy turn (exclamations) vs a
    # "?"-heavy turn (interrogative cascade) vs a "."-heavy turn (even
    # declarative). Captures speaker TEXTURE: some Shakespeare turns are
    # rapid-fire exclamations ("O villain! O most damned villain!"),
    # some are a cascade of questions ("Is it possible? Is it so? Canst
    # thou believe it?"), some are composed declaratives. The real text
    # shows strong within-turn autocorrelation: once a speaker has
    # produced two "!"-ending sentences, another "!"-ending sentence is
    # more likely than baseline.
    #
    # Reset to 0 on turn boundary (consecutive_newlines >= 2).
    # Incremented at the emission of the corresponding punct, outside
    # speaker-label territory, by pipeline/turn.py.
    #
    # Consumed by predict (sentence-end-position bias) to nudge the
    # next sentence-end-punct choice toward the turn's dominant mode
    # and by sentence-start bias (interjection/WH openers).
    turn_exclam_count: int = 0
    turn_question_count: int = 0

    # --- Tier 2/3: turn pronoun profile (soliloquy vs direct-address) ---
    # Shakespeare's turns have a strong and very distinctive pronoun
    # signature that the model has not been tracking. A SOLILOQUY (Hamlet
    # alone on stage, Lear on the heath) is I-HEAVY: "I know not why I am
    # so sad"; a DIRECT-ADDRESS harangue (Henry's Crispin speech, a
    # lover's plea) is YOU-HEAVY: "thou canst not speak of what thou
    # dost not feel". A third category (NARRATIVE / REPORTAGE) has low
    # first/second-person density — "The king is dead, the prince has
    # fled".
    #
    # These modes shape:
    #   - Sentence openers ("I do...", "Thou hast...", "There came...")
    #   - Content vocabulary (soliloquy: abstract / reflective;
    #     direct-address: imperatives, epithets; narrative: past-tense
    #     event verbs)
    #   - Punctuation rhythm (soliloquy pauses; direct-address
    #     exclaims)
    #
    # Fields:
    #   turn_i_pronouns   — count of "i" / "my" / "me" / "mine" /
    #                       "myself" completions in current turn.
    #   turn_you_pronouns — count of "thou"/"thee"/"thy"/"thine"/
    #                       "you"/"ye"/"your"/"yours"/
    #                       "thyself"/"yourself" completions in turn.
    #   turn_pronoun_mode — classified mode after enough evidence:
    #     0 = insufficient evidence (< 3 total 1st/2nd person pronouns)
    #     1 = I-dominant (soliloquy-ish):  i >= 3 AND i >= 2*you
    #     2 = you-dominant (direct-address): you >= 3 AND you >= 2*i
    #     3 = mixed (both >= 2 and ratio within 2x): dialogue
    #
    # Reset all three on turn boundary (consecutive_newlines >= 2).
    # Maintained by pipeline/turn_pronoun.py which reads just_finished_word
    # and last_completed_word. Consumed by predict/turn_pronoun_bias.py.
    turn_i_pronouns: int = 0
    turn_you_pronouns: int = 0
    turn_pronoun_mode: int = 0

    # --- Tier 2/3: dialogue adjacency memory ---
    # A snapshot of the PREVIOUS (just-closed) turn's shape, preserved
    # across the turn boundary so the current turn's opening can react
    # to it. Existing turn_* fields are reset at each boundary; these
    # carry over.
    #
    # Intuition: in Shakespeare, what a speaker opens with is strongly
    # conditioned on what the PRIOR speaker just said. A question
    # ("Is it so?") is answered by an opener like "Ay", "No", "Nay",
    # "I am", "It is", "Marry", "'Tis", "Sir". An exclaim ("O gods!")
    # is often echoed by another interjection ("Alas", "O", "Ha", "Fie").
    # A short terse turn (stichomythia) invites another short retort.
    # Same speaker twice in a row (prev_turn_speaker == current speaker)
    # is unusual and signals a continuation / stage direction.
    #
    # Fields are snapshotted at the transition into consecutive_newlines
    # == 2 (the first blank-newline that marks turn close), BEFORE
    # update_turn_progress resets the in-turn counters. Implementation:
    # update_dialogue_adjacency runs immediately before
    # update_turn_progress in PIPELINE.
    #
    # current_turn_final_char: the last non-whitespace content character
    # emitted inside the still-open turn. Updated continuously inside a
    # turn body (speaker_label_state == 0 and last_char is neither space
    # nor newline). Used to determine the turn's final punctuation at
    # turn-close time.
    current_turn_final_char: str = ""
    # prev_turn_final_punct: one of "", ".", "?", "!", ",", ";", ":", "-"
    # — the final non-whitespace char of the just-closed turn.
    prev_turn_final_punct: str = ""
    prev_turn_word_count: int = 0
    prev_turn_sentence_count: int = 0
    prev_turn_line_count: int = 0
    prev_turn_exclam_count: int = 0
    prev_turn_question_count: int = 0
    # Whether the speaker of the prev turn is the same speaker as the
    # turn about to open — compared against last_speaker_label at the
    # moment a new turn's label closes. Unused for now but reserved.
    prev_turn_speaker_label: str = ""
    # Monotonic count of turns observed — useful for first-turn guards.
    turns_closed: int = 0

    # --- Tier 2/3: cross-turn content echo ---
    # Snapshot of the previous speaker's turn_content_cache at the
    # moment that turn closed (cn == 2). Up to 6 distinct content words
    # (nouns/verbs/adjectives/adverbs/proper nouns) that the prior
    # speaker just used, most-recent first. Lets the new speaker's
    # opening words echo what the prior speaker said — a classic
    # dialogue dynamic:
    #     A: "Where is the king?"
    #     B: "The king is dead."
    # without this memory, once the turn boundary crosses, the echo
    # signal is lost. turn_content_cache resets on turn open (cn>=2)
    # to a fresh speaker's own thematic spine; prev_turn_content_tail
    # carries the adjacency.
    #
    # Updated in update_dialogue_adjacency at cn == 2 by snapshotting
    # turn_content_cache[:6] before update_turn_content would reset it.
    # Reset on subsequent turn closes (overwritten each time).
    prev_turn_content_tail: tuple[str, ...] = ()

    # --- Tier 3: turn content echo memory ---
    # A rolling cache of up to 10 content words (NOUN, VERB, VERB_ING,
    # VERB_ED, ADJECTIVE, ADVERB, PROPER_NOUN) emitted in the current
    # speaker turn, most-recent first, with duplicates removed so the
    # cache captures DISTINCT thematic words, not a token stream.
    #
    # This is TURN-scoped (reset on speaker-turn boundary), in contrast
    # to `content_words` which is a short global 4-word rolling buffer
    # without turn awareness. Captures the Shakespearean pattern where
    # a single turn circles back to its key nouns / verbs / images
    # ("honour", "king", "blood", "death", "sweet", "love") multiple
    # times — the thematic spine of a speech.
    #
    # Updated by pipeline/turn_content.py at word completion when
    # last_word_pos is a content tag AND the word is >=3 chars. Reset
    # to () at turn boundary (consecutive_newlines >= 2).
    #
    # Consumed by predict/turn_content_echo.py to:
    #   - at word-start: boost the first letter of cached words so the
    #     speaker is slightly more likely to reach back for a thematic
    #     word already said (a soft anaphora/repetition pull)
    #   - at mid-word: when word_buffer is a prefix of a cached word
    #     (and the buffer already differs from the most-recently-said
    #     word to avoid immediate verbatim echo), boost the continuing
    #     letter toward that cached completion — makes the model finish
    #     a mid-word into a thematically-relevant known word.
    turn_content_cache: tuple[str, ...] = ()

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

    # --- Tier 3: cadence (staccato ↔ flowing) ---
    # Rolling float in [-1, +1] tracking whether the recent text has
    # been *staccato* (-1; short words, many commas/semicolons, tight
    # punctuation — "Stay, villain, hold!") or *flowing* (+1; long
    # words, long clauses, enjambed lines — "The multitudinous seas
    # incarnadine"). This is a genuine texture/feel axis, distinct
    # from cadence-adjacent structural fields (chars_since_comma
    # measures *distance*, not *feel*).
    #
    # Bumps per completed word + per clausal punctuation:
    #   short word (≤ 3 letters):       -0.08 (staccato pull)
    #   long word (≥ 7 letters):        +0.12 (flowing pull)
    #   very long word (≥ 10 letters):  +0.08 extra
    #   clausal comma/semicolon:        -0.14 (staccato pull)
    #   sentence-end punctuation:        mild neutral decay
    # Decays toward 0 at 0.95 per completed word.
    # Speaker turn reset: multiply by 0.4 (carryover dampened).
    #
    # Consumed by predict.cadence: at word-end positions, modulate
    # the comma/space balance (staccato → commas, flowing → space).
    # Small magnitude bias, scaled by |cadence|.
    cadence: float = 0.0

    # --- Tier 3: ornament density (ornate ↔ spare texture) ---
    # Rolling [0, 1] float tracking whether the recent text has been
    # *ornate* (heavy with adjectives / adverbs stacked before head
    # nouns — "good sweet gentle rose", "most fair dear lord",
    # "noble and valiant captain") vs. *spare* (few pre-modifiers,
    # direct predication — "I am dead", "go now", "he hath fled").
    #
    # Distinct from tonal_weight (valence), imagery_density (sensory),
    # and archaic_density (register): ornament tracks *how decorated*
    # the noun phrases have been. Shakespeare alternates between
    # highly ornate passages (royal speeches, love scenes, soliloquies)
    # and spare passages (action, urgent dialogue, terse commands).
    # Once a speaker is in an ornate groove, more adjectives follow;
    # once they're in spare mode, direct diction follows.
    #
    # Bumps per completed word:
    #   ADJECTIVE:  +0.18 (ornament)
    #   ADVERB:     +0.08 (ornament-ish)
    #   NOUN / PROPER_NOUN:  -0.10 (noun consumed; resets a bit)
    #   VERB / AUX / MODAL:  -0.06 (action mode, spare)
    # Decays toward 0 at 0.96 per completed word.
    # Sentence-end: *0.85 (partial reset).
    # Speaker turn: *0.4 (dampen).
    #
    # Consumed by predict.ornament at word-start: when np_open AND
    # ornament_density is high, push harder toward noun head (resolve
    # the NP); when low, allow adjective modifiers. Also shapes
    # cadence-like decisions about whether to insert another adjective
    # before the head noun.
    ornament_density: float = 0.0

    # --- Tier 3: monosyllabic-run momentum (percussive-rhythm texture) ---
    # Integer counter of the number of consecutive 1-syllable words most
    # recently completed. Captures one of Shakespeare's most distinctive
    # textural modes: the drumbeat of stacked monosyllables —
    #   "To be, or not to be, that is the question"
    #   "Words, words, words"
    #   "Out, out, brief candle"
    #   "This above all: to thine own self be true"
    #   "Now is the winter of our discontent"
    #
    # Monosyllabic runs coincide with heightened rhetorical force —
    # gnomic lines, epigrams, soliloquy climaxes, urgent dialogue —
    # and have sharply different statistics from mixed-syllable prose:
    #   * Next word is far more likely to also be monosyllabic (the
    #     percussive momentum reinforces itself).
    #   * First letters of next words concentrate on a small set:
    #     t/a/b/w/h/i/n/o/s/m/y/d/f/g/l — the head letters of the
    #     closed-class function words plus the most common 1-syllable
    #     content verbs (be/go/do/say/see/know/come/love/die/live).
    #   * Low likelihood of Latinate polysyllables starting q/x/z/j,
    #     or of heavy consonant clusters starting str-/spr-/scr-.
    #
    # Updated in pipeline/flow.py on `just_finished_word`:
    #   * If the completed word's syllable count (counted via vowel
    #     groups) is 1: increment (cap at 12).
    #   * Else: reset to 0.
    # Reset on:
    #   * Sentence-end punctuation (. ? !).
    #   * Speaker-turn boundary (consecutive_newlines >= 2).
    #   * Speaker label transition.
    # Preserved across:
    #   * Newlines within a turn (so enjambed monosyllabic lines sustain).
    #   * Comma / semicolon / colon breaks (so "To be, or not to be"
    #     keeps climbing across the comma).
    #
    # Consumed by predict/rhythm.py at word-start:
    #   * When run >= 3, boost first letters of the short-word cluster
    #     above (additive, small magnitude).
    #   * When run >= 5, also penalize polysyllable-leader clusters
    #     (q, x, z, j, and consonant-heavy starts of >8-letter latinate
    #     words).
    monosyllabic_run: int = 0

    # --- Tier 3: urgency_tempo — action-speed / frantic-vs-languid texture ---
    # Rolling [0, 1] float capturing the *tempo* of the unfolding scene:
    # is the speaker in commanding / pursuing / fleeing mode (frantic,
    # imperatives, "!") or in reflective / ceremonial mode (languid,
    # long phrasing, measured cadence)?
    #
    # This is distinct from:
    #   * cadence (staccato ↔ flowing — driven by CLAUSE length): cadence
    #     measures phrase-length rhythm; urgency measures ACTION-demand.
    #     A long-winded exclamation "O come, thou sweet and gentle friend!"
    #     is flowing but urgent. A short reflective "I think so." is
    #     staccato but languid.
    #   * invocation_mode (rhetorical declamation): invocation is about
    #     grand apostrophe; urgency is about hurry / action.
    #   * emotional_intensity (overall emotive): emotion may be sorrow
    #     or awe (languid); urgency is specifically "go/do/now".
    #
    # Bumps on completed word:
    #   * Hurry adverbs (now, anon, straight, quick, quickly, hence,
    #     hither, haste, swift, swiftly, presently, soon, fast, speedy,
    #     speedily, fly, flee, hie) — strong.
    #   * Imperative action verbs when first word of sentence (come, go,
    #     stand, stay, hold, strike, run, speak, tell, hark, look, rouse,
    #     away, up, down, forth, off) — strong.
    #   * Motion / pursuit verbs (chase, pursue, seize, catch, rush,
    #     follow, attack, defend) — moderate.
    # Bumps on punctuation:
    #   * "!" — strong bump (a shout is inherently urgent).
    #   * Short-sentence close (≤ 4 words since last sentence-end) — mild.
    # Dampers:
    #   * Decay 0.94 per completed word (fast fade when scene stays calm).
    #   * Long words (≥ 10 letters) subtract a little (polysyllabic
    #     Latinate = ceremonial, not urgent).
    #   * Speaker-turn boundary (consecutive_newlines ≥ 2, \n) — *0.35.
    #
    # Consumed by predict/urgency.py at word-end / sentence-close positions.
    urgency_tempo: float = 0.0

    # --- Tier 2: per-word phonotactic red-flag accumulator ---
    # Count of phonotactic "red flags" observed in the current word
    # buffer so far. A red flag is a non-English-like substructure
    # that a legit word would very rarely contain:
    #   - a consonant-cluster reaching 4+ (ccccccc)
    #   - a vowel-sequence reaching 3+ (eee, uou — except well-known
    #     triphthongs)
    #   - a rare letter (j/q/x/z) at mid-word position (position > 0)
    #     that isn't part of an attested digraph (qu)
    #   - an off-trie letter emission after the buffer had already
    #     reached a valid complete word
    # Each flag fires once per event boundary, so a single word with
    # "cccc + vvv" accumulates 2 flags.
    #
    # Reset to 0 at word boundary (any non-letter character).
    #
    # Motivation: existing fields track state-at-latest-letter
    # (consonants_since_vowel, vowels_since_consonant, letters_off_trie).
    # These reset as the word normalizes. A word that had "rntr"
    # followed by a vowel then continues doesn't remember that it
    # started weird. The red-flag counter persists across the word,
    # letting predict know "this word has already failed phonotactics
    # twice, aggressively push it to close now."
    word_red_flags: int = 0
    # Bookkeeping: was the previous consonants_since_vowel >= 4
    # (to detect the moment the cluster maxes out)? Prevents double-
    # counting the same cluster.
    red_flag_cluster_fired: bool = False
    red_flag_vowel_fired: bool = False

    # --- Tier 2: phonotactic illegal-bigram count ---
    # Count of letter-pair bigrams within the current word that are
    # phonotactically illegal in English — pairs that virtually never
    # appear adjacent inside a real English/Shakespearean word.
    # Examples: "tv", "dq", "vs" at word-medial, "jn", "xd", "qk"…
    #
    # This catches a class of gibberish that the existing red-flag
    # counters miss: words like "etvsudqted" or "iaegofag" don't have
    # 4+ consonant clusters or 3+ vowel runs, and their rare-letter
    # flags are few, yet they contain outright illegal pairs. One or
    # two illegal bigrams inside a short word is a near-certain
    # gibberish signal.
    #
    # Reset on word boundary. Runs after update_linguistic so that
    # `word_buffer` already contains the incoming letter and its
    # predecessor.
    bad_bigram_count: int = 0

    # Count of 3-letter sequences within the current word whose
    # consonant structure is impossible in English — e.g., three
    # consonants in a row that don't form a legal onset (scr/spl/
    # spr/str/shr/thr) or coda (nct/rst/mpt/pts/…). The bigram
    # check misses trigrams like "glr" in "claitaglrt" or "rsn"
    # in "tarrsnrach" because each 2-letter pair happens to occur
    # elsewhere in English, but the specific 3-letter cluster is
    # phonotactically impossible.
    #
    # Reset on word boundary, same gating rules as bad_bigram_count.
    bad_trigram_count: int = 0

    # --- Tier 2/3: within-line alliteration memory ---
    # The lowercase first letter currently being alliterated on this
    # line (a content-word's first letter). "" when no alliteration
    # run is active (e.g., fresh line, or last content word started
    # with a different letter than the running alliteration).
    line_alliteration_letter: str = ""
    # Number of consecutive content-words on this line (since the last
    # newline) whose first letter matches line_alliteration_letter. A
    # value >= 2 signals active alliteration — the predict layer then
    # nudges the next word's first letter toward the same character.
    # Function words (articles, possessives, prepositions, conjunctions,
    # aux verbs, modals, pronouns) are transparent: they neither
    # advance nor break the run.
    line_alliteration_run: int = 0

    # --- Tier 2: clause nesting depth (subordinators) ---
    # Count of subordinating-conjunction openings since the last
    # sentence-end punctuation. Each "that"/"which"/"who"/"when"/
    # "where"/"while"/"if"/"though"/"because"/"since"/"as"/"unless"
    # at a plausible clause-opening position increments this counter.
    # Reset to 0 on sentence-end (. ? !) and on speaker-turn boundary.
    # Capped at 3 for downstream stability.
    #
    # Motivation: `clauses_in_sentence` counts ALL clausal breaks
    # (commas, semicolons, colons) but doesn't distinguish:
    #   1. a list-like parallel clause ("A, B, and C")
    #   2. a subordinate-clause nesting ("I know that which thou hast
    #      said when he whose name...")
    # A model with no handle on depth wanders indefinitely through
    # subordinate clauses without knowing when to return to the
    # main clause. This field captures true nesting depth so the
    # predict layer can escalate sentence-end pressure when we're
    # 2-3 subordinators deep.
    clause_depth: int = 0
    # Number of completed words since the most recent subordinator
    # that incremented clause_depth. Reset to 0 on subordinator,
    # sentence-end, and speaker-turn. Grows as we linger deep in
    # a subordinate clause — a long words_in_subordinate at depth
    # 2+ is a strong signal that sentence-close is overdue.
    words_in_subordinate: int = 0

    # --- Tier 2: NP-head expectation ---
    # True when the most recent noun-phrase opener (article,
    # possessive, or preposition) has been emitted, and no head noun
    # has been resolved yet. We're waiting for a head noun, possibly
    # after pre-modifiers (adjectives). Set to True on ARTICLE,
    # POSSESSIVE, or PREPOSITION word completion; set to False when:
    #   - a NOUN / PROPER_NOUN / PRONOUN completes (head noun found),
    #   - a VERB / AUX_VERB / MODAL / VERB_ING / VERB_ED completes
    #     (verb consumed slot; NP abandoned),
    #   - a CONJUNCTION / INTERJECTION / NEGATION completes,
    #   - sentence-ending punctuation fires,
    #   - speaker-turn boundary.
    #
    # Motivation: clause_slot tracks whole-clause structure (has
    # subject? has verb?). np_open tracks the finer-grained PHRASE
    # structure: after "of", "the", "my", a head noun is expected —
    # and the predict layer should heavily favor noun/adjective
    # first letters at word-start while np_open, and strongly
    # penalize sentence-enders and another determiner/preposition.
    # Without this, the model produces "winter of to" sequences —
    # preposition directly following preposition with nothing in
    # between, which virtually never happens in real English.
    np_open: bool = False
    # Number of completed words since np_open became True (or the
    # most recent re-opener). 0 on the word that set it. Grows
    # through adjectives; caps at 5.
    np_wait_words: int = 0

    # --- Tier 2: enjambment / line-end punctuation class ---
    # Classification of the final non-newline character of the
    # previous line (the char immediately before its closing \n).
    #   0 = no previous non-empty line yet (start of text / blank gap)
    #   1 = hard punctuation (. ! ?)             — clean sentence close
    #   2 = soft punctuation (, ; :)             — mid-clause pause / label
    #   3 = letter (a-z, A-Z)                    — enjambment: word cut by
    #                                              a prose-wrap; next line
    #                                              likely continues lowercase
    #   4 = other (apostrophe, dash, digit, ...)  — treat as continuation
    #
    # Captured by update_linguistic at the moment a \n lands and
    # chars_since_newline > 0. Held steady across blank lines.
    #
    # Motivation: the existing line-start logic boosts capital letters
    # on any prev_line_length in [1, 80], but Shakespeare's PROSE lines
    # wrap mid-phrase and the continuation line starts LOWERCASE
    # ("considering\nhow honour would..."). Without this distinction,
    # the model invents phantom new sentences at every prose wrap.
    # This field lets the predict layer condition the capital-boost
    # on whether the prev line ended cleanly vs. was enjambed.
    prev_line_final_class: int = 0

    # --- Tier 2: subject-verb agreement expectation ---
    # Once a subject has been identified in the current clause, the
    # expected morphology of the upcoming main verb is largely fixed
    # in Early Modern English. This field tells the predict layer
    # what morphology to reward when clause_slot == HAS_SUBJ and the
    # next word is likely a verb.
    #
    #   0  VA_NONE       — no current subject-agreement expectation
    #                      (e.g. clause is FRESH, or subject is missing)
    #   1  VA_THOU       — 2nd person singular archaic subject
    #                      ("thou", "thee" in marked positions).
    #                      Upcoming verb tends to end in "-st" or
    #                      "-est" ("thou art", "thou hast", "thou
    #                      knowest", "thou speakest", "thou didst").
    #   2  VA_THIRD_SG   — 3rd person singular: "he", "she", "it",
    #                      "who", any proper noun, or any ordinary
    #                      singular noun-phrase subject. Verb ends
    #                      in "-s", "-es" (modern) or "-th", "-eth"
    #                      (archaic: "hath", "doth", "saith", "hath",
    #                      "loveth"). Auxiliaries "is", "was", "has",
    #                      "had", "does" fit.
    #   3  VA_FIRST_SG   — "I". Verb is base form or with archaic
    #                      "-e" (I am, I do, I see, I prithee).
    #   4  VA_PLURAL     — "we", "they", "you", "ye" or a plural
    #                      noun-phrase. Verb is base form ("we are",
    #                      "they do", "you know", "we love").
    #   5  VA_IMPERATIVE — clause opened with a bare verb (after
    #                      O/Alas/Ah/Come/Go/See/Hear/Speak — no
    #                      preceding subject). Verb is base form.
    #
    # Set by update_verb_agreement (runs after update_clause_slot) at
    # the word completion that fills the subject slot. Reset to
    # VA_NONE on sentence-end punctuation or on a CONJUNCTION that
    # resets the clause.
    verb_agreement: int = 0

    # --- Tier 2: anaphoric pronoun referent tracking ---
    # Once a named referent has entered the discourse (a proper noun
    # or a distinctive role-noun like "king" / "lord" / "queen"), we
    # track its grammatical gender so subsequent pronouns can be
    # predicted. Shakespeare's character-tagged dialogue has strong
    # referent continuity: after "the king entered, he ...", the
    # pronoun "he" (not "she") is very likely. This field captures
    # that continuity across sentences within a speaker turn.
    #
    #   0  REF_NONE   — no tracked referent
    #   1  REF_MALE   — last tracked referent is masculine
    #                  (he/him/his; king, lord, duke, sir, father,
    #                  son, brother, prince, knight, master, boy,
    #                  man, friar, proper male names)
    #   2  REF_FEMALE — feminine (she/her; queen, lady, madam, sister,
    #                  mother, daughter, wife, maid, nurse, mistress,
    #                  proper female names)
    #   3  REF_NEUTER — inanimate or abstract (it; heart, soul, love,
    #                  sword, crown, throne, etc.) — tracked mainly
    #                  to avoid falsely boosting he/she for inanimates
    #   4  REF_PLURAL — group (they/them/their; lords, soldiers,
    #                  friends, gentlemen, ladies)
    #
    # Updated by update_referent (new stage after speaker_memory) at
    # word completion when a new discourse-significant noun is seen.
    # Decays to REF_NONE on speaker-turn change (new last_speaker_label).
    # Persists across sentence boundaries within a turn.
    referent_gender: int = 0
    # Light decay counter: how many words since the referent was last
    # confirmed (either by reintroduction or by a matching pronoun).
    # Used to weaken the bias as the referent stales.
    referent_staleness: int = 0

    # --- Tier 2: verb transitivity / object-expectation ---
    # After a transitive main verb, Shakespeare almost always supplies
    # a direct object — a noun phrase (article/possessive + optional
    # modifiers + head noun) or a clausal complement. The existing
    # clause_slot FSM collapses this into HAS_VERB → POST_OBJ on ANY
    # next word, which is far too coarse: after "I fear", the model
    # treats "nait" (gibberish) the same as "the storm". A real
    # verb-transitivity axis says: after "fear", we expect a determiner
    # or noun starter — NOT a preposition, conjunction, or bare verb.
    #
    # Values:
    #   0 VT_NONE          — no active object expectation
    #   1 VT_DO_EXPECTED   — transitive/ditransitive verb just completed;
    #                        a direct object (NP) is the expected next
    #                        constituent. Example: "love", "kill", "see",
    #                        "take", "make", "give", "bring", "find".
    #   2 VT_COMP_EXPECTED — linking/copula verb just completed; a
    #                        predicative complement (adjective or NP) is
    #                        expected. Example: "is", "art", "was",
    #                        "seem", "become". Overlaps with VT_DO but
    #                        biases toward adjectives more than determiners.
    #
    # Set by update_transitivity (new stage after np_head) when a
    # verb POS completes AND the word is in the transitive/linking
    # class. Reset to VT_NONE on:
    #   - NOUN / PROPER_NOUN / PRONOUN completion (object resolved),
    #   - sentence-end punctuation,
    #   - comma/semicolon/colon (clausal break),
    #   - CONJUNCTION (parallel clause coordination),
    #   - speaker-turn boundary,
    #   - vt_wait_words >= 4 (expectation staled out).
    #
    # Consumed by predict/transitivity.py at word-start to strongly
    # boost determiner/noun starter letters (t, m, h, a, y, o, s)
    # and penalize preposition/conjunction/aux starter letters that
    # would defer the object indefinitely.
    verb_transitivity: int = 0
    # Words elapsed since verb_transitivity last became non-NONE. 0 on
    # the transition-setting word. Grows through adjectives/numbers/
    # articles/possessives as we build the NP. Caps at 5.
    vt_wait_words: int = 0

    # --- Tier 3: sonority level (phonetic texture) ---
    # Rolling [-1, +1] float tracking the phonetic texture of the
    # recent text, updated on every letter emission. Positive =
    # melodic/sonorant (vowels, liquids l/m/n/r, approximants w/y);
    # negative = percussive (hard stops k/t/p/g/b/d, sibilants z/x/j/q).
    # This is a flow-level *feel* axis — Shakespeare's lyric passages
    # ("When shall we three meet again / In thunder, lightning, or
    # in rain?") cluster sonorant phonemes, while violent/urgent
    # passages ("Strike! Kill! Drag!") cluster stops.
    #
    # Distinct from cadence (staccato/flowing tempo) and imagery
    # (sensory lexicon) — sonority is about *sound*, the actual
    # phonetic color of the letters being emitted. Bleeds through
    # word boundaries; tends to be self-reinforcing (once a
    # passage is melodic, writers keep the melody).
    #
    # Bumps per emitted letter:
    #   vowels (a/e/i/o/u):       +0.035
    #   liquids (l/m/n/r):        +0.020
    #   approximants (w/y):       +0.015
    #   voiceless fricatives:     +0.005 (f/h/s)
    #   voiced consonants:        -0.010 (v/c)
    #   hard stops (k/t/p):       -0.025
    #   voiced stops (g/b/d):     -0.018
    #   rare harsh (j/q/x/z):     -0.035
    # Decay of 0.985 per letter (multiplicative) applied after bump.
    # On non-letter characters: decay 0.97 (faster fade between words).
    # On speaker-turn boundary: *0.30.
    #
    # Consumed by predict/sonority.py at mid-word positions to nudge
    # the next letter toward in-register phonemes.
    sonority_level: float = 0.0

    # --- Tier 3: invocation mode (rhetorical / declamatory texture) ---
    # Rolling [0, 1] float tracking whether the current speaker is in
    # *invocation mode* — the grand, oratorical, apostrophe-driven
    # texture that characterizes Shakespeare's high rhetorical passages
    # (Gaunt's "This royal throne of kings...", Hamlet's "O, what a
    # rogue and peasant slave am I!", Henry V's "O for a muse of
    # fire..."). Distinct from:
    #   - archaic_density (register of individual words)
    #   - emotional_intensity (outburst heat, short-decay)
    #   - tonal_weight (dark vs light valence)
    #   - imagery_density (sensory vs abstract)
    #   - cadence (staccato vs flowing)
    #   - ornament_density (adjective stacking)
    #
    # Invocation mode captures whether the SYNTAX-LEVEL voice is in
    # declamatory address mode, separate from what lexicon is in use.
    # Signals that raise it:
    #   - Sentence begins with "o"/"oh"/"alas"/"ah"/"hark"/"behold"/
    #     "hail"/"hear"/"lo"/"heavens" (+0.45) — canonical invocation
    #     openers
    #   - Sentence ends with "!" (+0.25) — invocation tends to chain
    #   - WH-rhetorical opener ("what"/"why"/"whence"/"wherefore"/
    #     "how"/"when") at sentence-start (+0.18)
    #   - Vocative noun completed while mode > 0.2 (+0.10) — reinforces
    #
    # Decay: 0.92 per completed word (mode is medium-persistence —
    # it lingers across 1-2 sentences but fades over a long passage).
    # Sentence-end: *0.92 (mild attenuation at punctuation).
    # Speaker turn: *0.25 (new speaker may or may not inherit).
    #
    # Consumed by predict/invocation.py at:
    #   - sentence starts: boost "O", "A" (Alas/Ah), "H" (Hark/Hail/
    #     Hear/How), "W" (What/Why/Whence), "L" (Lo)
    #   - sentence-end punctuation choice: boost "!" over "."
    #   - in-invocation word starts: boost vocative-lead letters
    #     (m=my, t=thy, g=good/gentle, s=sweet/sacred, n=noble, d=dear)
    invocation_mode: float = 0.0

    # --- Tier 2: word-form expectation (morphological slot) ---
    # When an auxiliary, modal, preposition, or quantifier just
    # completed, the NEXT content word is grammatically constrained in
    # form. This field gives the predict layer a morphology-aware
    # prior beyond what POS tags alone capture:
    #
    #   0 WFE_NONE        — no form expectation
    #   1 WFE_INFINITIVE  — bare verb expected. Triggered by:
    #                        to, shall, shalt, will, wilt, must, may,
    #                        mayst, might, can, canst, could, couldst,
    #                        would, wouldst, should, shouldst, let,
    #                        do, dost, does, did, didst.
    #                        Preferred: base verbs (be, go, see, take,
    #                        know, make, come, hear, speak, think, have,
    #                        find, live, die, fall, rise, bear).
    #   2 WFE_PAST_PART   — past participle expected. Triggered by:
    #                        have, has, had, having, hath, hast.
    #                        Preferred: -n/-en endings (seen, slain,
    #                        taken, given, gone, known, done, borne,
    #                        worn, torn, drawn, thrown, broken, spoken,
    #                        stolen) and -ed/-d ("loved, feared, killed").
    #   3 WFE_ING_OR_PP   — ambiguous progressive/passive context after
    #                        is/am/are/was/were/be/been/being. Either
    #                        -ing present participle ("is going, was
    #                        sleeping") or past participle ("is slain,
    #                        was taken"). Weaker bias, favoring both.
    #   4 WFE_NOMINAL     — NP head expected after "of" (strongest of
    #                        prepositions: "of love/death/war/honour").
    #   5 WFE_COMPARATIVE — after more/less → adjective/adverb expected.
    #   6 WFE_SUPERLATIVE — after most → adjective expected.
    #
    # Reset rules:
    #   - Resolved on VERB/VERB_ED/VERB_ING/NOUN/ADJECTIVE completion
    #     when the completion matches the expected form.
    #   - Escalate wait on ADVERB/NEGATION/ARTICLE/POSSESSIVE/ADJECTIVE
    #     (pre-modifiers allowed).
    #   - Full reset on sentence-end, clausal break, speaker-turn,
    #     conjunction, or wait >= 4 without resolution.
    #
    # This is a morphology-aware parallel to transitivity/clause_slot —
    # those track syntactic position; this tracks morphological form.
    word_form_expectation: int = 0
    wfe_wait_words: int = 0

    # --- Tier 2: speaker-label off-trie run-length ---
    # Count of consecutive letters added to `speaker_buffer` (while the
    # speaker-label FSM is in state 1 or 2) that took the buffer OFF
    # the known-speaker-prefix trie. 0 whenever the buffer is currently
    # a prefix of at least one canonical name (or we're not in a
    # speaker label). Monotonically grows as we extend into gibberish.
    #
    # Motivation: sample diagnostics showed phantom speaker labels
    # like "ZM SYMPATH:" emerging because once the FSM enters state 2
    # the only escape is ":" (or a rare-class char), and the existing
    # off-trie drift bias BOOSTED ":" closure regardless of how much
    # gibberish had accumulated. The run counter gives predict a way
    # to distinguish a plausible minor-character name (short off-trie
    # run: 1-2 letters) from run-away gibberish (5+ letters off-trie),
    # and flip the ":" bias from boost to penalty as the run grows —
    # pushing the model to escape the phantom label via newline.
    speaker_label_offtrie_run: int = 0

    # --- Tier 2: speaker-buffer vowel count ---
    # Count of vowels (A, E, I, O, U — uppercase, since speaker_buffer
    # is uppercased) present in the current speaker_buffer. Resets to 0
    # whenever speaker_label_state leaves {1, 2}.
    #
    # Motivation: phantom speaker labels like "TCK:" emerge when the
    # FSM runs through 3 consonant characters with no vowel and then
    # the ":" boost fires. Real Shakespeare speaker labels ALWAYS
    # contain at least one vowel — there is no consonant-only name.
    # This field lets a predict layer apply an extreme penalty to ":"
    # and to further consonant letters when buffer length >= 2 and
    # vowel count == 0, forcing the model to either emit a vowel
    # (completing what might be a legitimate prefix) or escape via
    # newline. Unlike speaker_label_offtrie_run (which measures drift
    # from the known-name trie), this measures a phonotactic
    # impossibility that applies even to unknown names.
    speaker_buffer_vowels: int = 0

    # --- Tier 2: speaker-trie next-char legality flags ---
    # Computed by `pipeline/speaker_strict.py` from `speaker_buffer`
    # using `predict.speaker_trie.SPEAKER_TRIE_NEXTS`. Flags whether
    # the current speaker_buffer prefix legally permits a space, colon,
    # or any letter as the next character in a canonical speaker label.
    # Consumed by `predict/speaker_label_strict.py` to hard-penalize
    # tokens that would emit space/colon/apostrophe/newline/opposite-case
    # letters when those aren't legal continuations. Without this, the
    # existing speaker_trie_bias only penalizes same-case letters outside
    # the next-set, letting malformed labels like "MOUNT tssayl:" slip
    # through because " " (space) and ":" get 0 bias at a prefix where
    # they're not trie-legal.
    #
    # All three False when outside a speaker label (state != 1 and != 2),
    # or when the speaker_buffer has drifted off-trie. When off-trie,
    # the predict layer pushes hard toward same-case letters (recovery)
    # and away from terminators.
    speaker_trie_on_trie: bool = False
    speaker_trie_space_valid: bool = False
    speaker_trie_colon_valid: bool = False

    # --- Tier 2: recent POS trigram (content-word history) ---
    # Rolling tuple of the last up-to-4 POS tags of completed words,
    # most-recent first. Unlike `last_word_pos` / `prev_word_pos` /
    # `prev_prev_word_pos` (which are positional, shift each word
    # regardless of POS type), this tuple is optionally *filtered*:
    # entries of class "transparent" (INTERJECTION, CONJUNCTION,
    # NEGATION, WH, ARTICLE, POSSESSIVE, PRONOUN) are skipped so the
    # tuple stores the *content backbone* (verbs, nouns, adjectives,
    # adverbs, aux, modal, preposition). This gives predict a
    # content-trigram view of the recent syntactic skeleton, which
    # the flat last/prev/prev_prev fields can't express once a
    # function word intervenes.
    #
    # Example:
    #   input text: "I do not know the way"
    #   flat sequence: PRON, AUX, NEG, VERB, ART, NOUN
    #   filtered:                      AUX, VERB, NOUN
    # Now at next word, the recent_pos_backbone lets a predict layer
    # know the local backbone is (NOUN, VERB, AUX) — providing a
    # content-sensitive next-POS prior over "what continuation makes
    # sense here".
    #
    # Cap at 4 entries (sufficient for a tight content-window lookback).
    # Resets to () on sentence-end punctuation and on speaker-turn change.
    recent_pos_backbone: tuple[int, ...] = ()

    # Third POS slot — completes the trigram context last/prev/prev_prev.
    # Allows predict layers to see a true three-word POS lookback
    # (shifted at every word completion, not filtered). Complements
    # recent_pos_backbone (which filters function words) by giving a
    # strict positional trigram.
    prev_prev_word_pos: int = 0

    # Count of consecutive verb-class words (VERB, VERB_ING, VERB_ED)
    # completed in a row, treating AUX_VERB, MODAL, ADVERB, and
    # NEGATION as *transparent* (they don't reset but don't count).
    # A value >= 1 means the previous main-verb position is filled;
    # >=2 is a chain like "Sail roar" — very rarely grammatical.
    # Reset by any non-verb non-transparent content word (NOUN,
    # PROPER_NOUN, PRONOUN, ADJECTIVE, ARTICLE, POSSESSIVE,
    # PREPOSITION, CONJUNCTION, INTERJECTION, WH, NUMBER) and by
    # sentence-end punctuation and speaker-turn boundary.
    #
    # Consumed by predict/verb_chain.py at word-start: when the
    # counter is already >=1, penalize first letters of common main
    # verbs to prevent ungrammatical verb-after-verb-after-verb
    # chains. AUX/MODAL starters are not penalized (legitimate
    # "had gone", "would have seen" chains).
    verb_chain_len: int = 0

    # Count of consecutive function-class words completed in a row
    # without a content word intervening. Function classes counted:
    # ARTICLE, POSSESSIVE, PRONOUN, PREPOSITION, CONJUNCTION, WH,
    # MODAL, AUX_VERB, NEGATION, INTERJECTION, NUMBER. Content classes
    # (NOUN, PROPER_NOUN, VERB, VERB_ED, VERB_ING, ADJECTIVE, ADVERB)
    # reset the counter to 0.
    #
    # A value >= 3 means three function words in a row — a red flag
    # for grammatical breakdown ("of your to and you"). Consumed by
    # predict/function_word_chain.py to push content-word starts
    # when the chain grows long.
    #
    # Resets on sentence-end punctuation and on speaker-turn boundary.
    function_word_chain_len: int = 0

    # Count of consecutive content-class words completed in a row without
    # a function word (or clause-boundary punctuation) intervening.
    # Content classes counted: NOUN, PROPER_NOUN, VERB, VERB_ED,
    # VERB_ING, ADJECTIVE, ADVERB. Function classes reset the counter
    # to 0.
    #
    # A value >= 3 means three content words in a row — a red flag for
    # the "noun-pileup" degenerate sampling mode Shakespeare almost
    # never produces (e.g. "the last noon drymudrted Kinsmen"). Real
    # English / Shakespeare almost always interleaves a preposition,
    # conjunction, pronoun, determiner, or clause-boundary punctuation
    # after 2 content words. Consumed by predict/content_word_chain.py
    # to push function-word starts / clause-close punctuation when the
    # streak grows.
    #
    # Resets on sentence-end punctuation, on mid-clause punctuation
    # (comma, semicolon, colon, dash), and on speaker-turn boundary.
    content_word_streak: int = 0

    # --- Tier 2: CLAUSE SKELETON FSM ---
    # Encodes the proposition-building state of the current clause.
    # A clause is a stretch of text between clause-boundary markers
    # (period, question mark, exclamation, semicolon, colon, comma,
    # em-dash, newline-turn-break, and completed coordinator
    # "and"/"or"/"but"/"nor"/"yet"/"so" that opens a new clause).
    #
    # Codes (most-recent-first transitions):
    #   0 EMPTY        — fresh clause, nothing committed yet
    #   1 SUBJ_OPEN    — an NP opener (DET/POSS/ADJ) has appeared but
    #                    the head noun hasn't closed yet
    #   2 SUBJ_DONE    — a subject head (NOUN/PROPER_NOUN/PRONOUN)
    #                    has completed; predicate (verb) is now owed
    #   3 VERB_DONE    — a finite verb (VERB/VERB_ED/VERB_ING +
    #                    optional preceding AUX/MODAL) has completed;
    #                    object/complement or adjunct is now expected
    #   4 COMP_DUE     — verb took an OBJ/PRED complement position;
    #                    an NP/PP/predicate-adj is owed
    #   5 CLAUSE_DONE  — predicate filled (object NP head OR predicate
    #                    adj OR intransitive verb + 1 adjunct); clause
    #                    is ready to close — boost terminators, allow
    #                    conjunction to open next clause
    #
    # Ages with each just_finished_word. Resets to EMPTY on PUNCT_MID
    # (", ; :"), PUNCT_END (". ? !"), speaker-turn boundary, entry into
    # speaker label, and on coordinator completion (POS_CONJUNCTION
    # and/or/nor/but/yet/so). Clause-age is tracked separately to
    # allow predict layers to escalate pressure as a clause drags on.
    #
    # Consumed by predict/clause_skel.py at word-start: bias next-word
    # first-letter toward the POS class expected by the current state.
    clause_skel: int = 0
    clause_skel_age: int = 0  # completed-word count since last reset

    # --- Tier 2: list-parallelism structure ---
    # Shakespeare uses "X, Y, and Z" / "nor A nor B" / "by heaven,
    # by earth, by all ..." heavily. Once a comma-separated list is
    # underway, the first letter/POS of subsequent items is often
    # parallel to prior items; and after 2+ commas in a clause, the
    # conjunction "and"/"or"/"nor"/"but" becomes very likely as the
    # penultimate element.
    #
    # Count of commas/semicolons/colons since the last sentence-end
    # punctuation — a proxy for list-progression depth. Resets on
    # PUNCT_END and on speaker-turn boundary (consecutive_newlines
    # >= 2).
    commas_since_sent_end: int = 0
    # First letter (lowercased) of the most recent "list-item" word —
    # the first word whose start immediately follows a comma (possibly
    # with an intervening space). Captured at word-start, committed
    # when the word completes. Empty string when no list item pending.
    list_last_item_first_letter: str = ""
    # True during the first word after a comma — the upcoming letters
    # are the "item start" position. Cleared once that word completes.
    list_item_pending: bool = False
    # Count of consecutive list items whose first letter matched the
    # previous list item's first letter. >=2 signals an alliterative
    # parallel list ("hand to hand, heart to heart, hope to hope");
    # boosts the same starter letter on future list items and on
    # continuation. Resets on sentence-end punctuation.
    list_parallel_run: int = 0
    # POS of the first "list item" word (the word that started right
    # after the FIRST comma of this sentence). Used to bias future
    # items toward the same POS class — most Shakespearean lists are
    # NOUN-NOUN-NOUN or ADJ-ADJ-ADJ or VERB-VERB-VERB. 0 when no
    # list item recorded yet in this sentence.
    list_first_item_pos: int = 0

    # --- Tier 3: scene-topic tracker (semantic cluster memory) ---
    # Rolling activation vector over 8 semantic clusters, each a
    # non-negative float representing how recently/strongly that
    # cluster has been invoked by the emerging text:
    #   0 TOPIC_WAR       — sword, blood, battle, foe, arms, steel,
    #                       slain, fight, siege, wound, strike, war,
    #                       field, spear, arrow, soldier, captain,
    #                       victory, defeat, banner
    #   1 TOPIC_LOVE      — love, heart, dear, kiss, sweet, rose,
    #                       charm, beauty, eye, cheek, mistress, bride
    #   2 TOPIC_DEATH     — death, grave, tomb, die, dead, corpse,
    #                       bury, grave, soul, ghost, dust, rot,
    #                       mourn, funeral, pale
    #   3 TOPIC_ROYALTY   — king, queen, crown, throne, prince, duke,
    #                       royal, sceptre, lord, lady, noble, court,
    #                       majesty, sovereign, reign, realm, subject
    #   4 TOPIC_NATURE    — sun, moon, stars, wind, rain, sea, sky,
    #                       flower, tree, bird, field, storm, morn,
    #                       night, day, shore, leaf, fire, earth
    #   5 TOPIC_BODY      — hand, eye, face, lip, tongue, cheek, arm,
    #                       breast, head, foot, heart (overlaps love),
    #                       tears, blood (overlaps war)
    #   6 TOPIC_FAITH     — god, heaven, hell, soul, prayer, sin,
    #                       holy, faith, grace, mercy, angel, devil,
    #                       sacred, spirit, church
    #   7 TOPIC_FORTUNE   — fate, chance, luck, fortune, star, doom,
    #                       destiny, providence, hap, wheel, time
    #
    # Updated by pipeline/topic_tracker.py at word completion:
    #   - word in cluster k → bump scene_topics[k] by +1.0
    #   - all clusters decay by 0.90 per completed word
    #   - speaker-turn boundary multiplies all by 0.35
    # Capped at 4.0 per cluster.
    #
    # Consumed by predict/scene_topic.py at word-start: the dominant
    # cluster (if any significantly above others) biases the next
    # word's first letter toward the cluster's characteristic starter
    # letters. This bleeds semantic coherence through function-word
    # scaffolding — once "sword" and "blood" appear, the next content
    # word is more likely "foe"/"steel"/"wound" than "rose"/"charm".
    scene_topics: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # --- Tier 2: sentence-level anaphora ---
    # First completed word (lowercased) of the PREVIOUS sentence of
    # this turn — captured at the moment the previous sentence closes.
    # Empty "" if no previous sentence, or at speaker-turn start.
    # Used by predict to detect and reinforce inter-sentence anaphora:
    # Shakespeare frequently chains sentences that begin with the same
    # word:
    #   "And so he dies. And so his name is gone. And so I weep."
    #   "Let me not to the marriage... Let me say..."
    #   "O, that ... O, that ... O, that ..."
    #   "When I was young... When I had seen... When I had lost..."
    # Reset on speaker-turn boundary (consecutive_newlines >= 2).
    prev_sentence_first_word: str = ""
    # First completed word of the CURRENT sentence (set when that word
    # completes; otherwise "" until first word of a sentence completes).
    # Reset to "" on sentence-end punctuation (the first word of the
    # next sentence fills it). Also reset on speaker-turn boundary.
    curr_sentence_first_word: str = ""
    # Count of consecutive sentences in the current turn that started
    # with the same first word as the one before them. 0 at turn start;
    # reset on mismatch or on speaker-turn boundary. >=1 signals an
    # active sentence-anaphora pattern.
    sentence_anaphora_run: int = 0

    # --- Tier 3: doubt register (assertion ↔ uncertainty texture) ---
    # Rolling [-1, +1] float tracking whether the recent text is in
    # DOUBTING mode (+) — "perhaps", "methinks", "haply", "belike",
    # "may", "might", "perchance", "if it be so", "I wonder",
    # rhetorical questions — vs ASSERTIVE mode (-) — "verily",
    # "surely", "certain", "indeed", "I know", "I am", "it is",
    # imperatives, resolute exclamations.
    #
    # Distinct from:
    #   - emotional_intensity: heat, not commitment-direction
    #   - tonal_weight: valence (dark/light), not epistemic stance
    #   - invocation_mode: rhetorical declamation, orthogonal to doubt
    #   - sentence_type: per-sentence syntactic type, not a rolling
    #     cross-sentence register
    #
    # Shakespeare's long monologues arc along this axis: Hamlet opens
    # in doubt and moves toward resolution; Lady Macbeth opens in
    # resolution and collapses into doubt. This field captures that
    # arc.
    #
    # Bumps per completed word (additive):
    #   doubt words  (+): perhaps, perchance, belike, methinks,
    #                     haply, may, might, maybe, peradventure,
    #                     seem, seems, seemed      : +0.14 each
    #   doubt-light  (+): if, whether, or         : +0.05 each
    #   certain words(-): verily, surely, indeed, truly, certain,
    #                     doubtless, no-doubt      : -0.14 each
    #   strong knows (-): know, knows, known, knew : -0.08 each
    #   imperatives  (-): go, come, speak, hear, look, stay (when
    #                     appearing at start of a sentence)   : -0.06
    # Punctuation:
    #   "?" end-of-sentence : +0.06  (question = doubt spike)
    #   "!" end-of-sentence : -0.06  (exclamation = assertion spike)
    # Decay: 0.93 per completed word (multiplicative toward 0).
    # Clip: [-1.0, +1.0].
    # Speaker turn: *0.3  (new speaker may inherit faint register).
    #
    # Consumed by predict.doubt at:
    #   - word-start: in doubt mode boost p (perhaps, perchance, peradventure),
    #                 m (may, might, methinks), h (haply), b (belike), s (seem);
    #                 in assertion mode boost v (verily), s (surely/so),
    #                 i (indeed/I), k (know), t (truly/the/that).
    #   - sentence-end: in doubt mode boost "?"; in assertion mode boost "!"/"."
    doubt_register: float = 0.0

    # --- Tier 2: proper-noun slot ---
    # Encodes how strongly a capitalized proper noun is expected at
    # the next word-start:
    #   0 PN_NONE   — no signal.
    #   1 PN_MILD   — after a vocative-lead adjective/possessive
    #                 ("my", "thy", "good", "dear", "sweet", "fair",
    #                  "noble", "brave") or "O"/"Oh".
    #   2 PN_STRONG — after a title noun ("lord", "sir", "king",
    #                  "queen", "lady", "saint", ...) or a vocative-
    #                  delimiter punctuation (",", ";", ":").
    #   3 PN_QUOTE  — after a reported-speech lead ("said", "quoth",
    #                  "cried", ...). Direct-quote capital follows.
    # One-shot: cleared on the first letter of the next word, on
    # sentence-end punctuation, and on speaker-turn boundary.
    # Consumed by predict.proper_noun at word-start:
    #   PN_NONE mid-sentence → gentle penalty on A-Z (phantom-cap
    #   guard); PN_STRONG/QUOTE → mild boost on capital starts.
    proper_noun_slot: int = 0

    # --- Tier 2: line-break propriety ---
    # A coarse "is this a legal place to end a verse line?" tracker,
    # computed each token from existing syntactic-position fields
    # (np_open, clause_slot, letter_run_len, word_buffer, chars_since_comma).
    #
    #   0 BREAK_DEEP_MID_PHRASE — mid-word, or an NP is open (article/
    #                             preposition/possessive unresolved), or
    #                             the clause has no verb yet. Line break
    #                             here is ungrammatical; suppress \n.
    #   1 BREAK_WEAK            — post-verb but NP still pending head,
    #                             or general mid-clause. Line break is
    #                             tolerated but not ideal.
    #   2 BREAK_PHRASE_END      — at a complete word AND clause has a
    #                             verb AND no NP is open. Natural place
    #                             to break a verse line.
    #   3 BREAK_CLAUSE_END      — just after a clause-break punctuation
    #                             (, ; :) — the strongest line-break
    #                             signal short of sentence-end.
    #
    # Why this captures something new: the existing verse-line signals
    # (verse_score, chars_since_newline, prev_line_length) are all
    # position/scoring — they say "a break is overdue" but not "this
    # is a grammatical place to break". Samples show the model ending
    # lines mid-NP ("wade certain\nEngage yearly", "Is adieu\n"), which
    # real Shakespeare rarely does. Predict consumers can suppress \n
    # at propriety 0/1 and keep it at 2/3.
    line_break_propriety: int = 0

    # --- Tier 2: verb semantic class of the current clause's main verb ---
    # A coarse 9-way classification of the verb that was most recently
    # seen in the current clause. Enables post-verb predictions that
    # depend on the verb's *meaning*, not just its presence.
    # Values:
    #   0 VC_NONE        — no recent verb / reset by sentence/clause break
    #   1 VC_PERCEPT     — see/hear/feel/behold/witness/watch/observe…
    #   2 VC_COGNITION   — know/think/believe/doubt/suspect/understand…
    #   3 VC_SPEECH      — tell/say/speak/ask/call/swear/promise/answer
    #   4 VC_MOTION      — go/come/follow/lead/seek/meet/fly/hunt…
    #   5 VC_GIVE_TAKE   — give/take/bring/send/offer/keep/hold…
    #   6 VC_VIOLENCE    — kill/slay/strike/wound/stab/hurt/beat/break
    #   7 VC_EMOTION     — love/hate/fear/curse/bless/thank/praise…
    #   8 VC_BE_EXIST    — is/are/be/seem/become/appear/remain/prove
    #
    # Why this captures something new: transitivity tracks WHETHER an
    # object is expected; verb_class tracks what KIND of object. The
    # predict layer can use this to bias first-letters toward object
    # nouns semantically compatible with the verb — e.g. after VIOLENCE
    # verbs, person-pronoun / person-noun starters; after BE_EXIST,
    # article/adjective starters. This is a semantic axis n-grams
    # fundamentally can't compute.
    verb_class: int = 0
    # Words elapsed since verb_class was set. Resets with class. Used
    # to decay the bias as the object slot falls further out of reach.
    vc_wait_words: int = 0

    # --- Tier 2/3: cross-turn prior-speaker sentence-type memory ---
    # The sentence type of the last sentence the PREVIOUS speaker
    # uttered before the turn boundary. Captured by pipeline/sentence.py
    # at the moment the turn boundary (\n\n) clears prev_sentence_type.
    # Values match SENT_UNKNOWN/DECL/INTERROG/EXCLAM/IMPER.
    #
    # Why this captures something new: existing turn-opener biases
    # fire when words_in_turn == 0 but they don't know WHAT the prior
    # speaker said. When the prior speaker ended with "?", the current
    # speaker is likely answering — opener should favour "Yes/No/Nay/
    # Ay/Marry/Indeed/Truly/I/Not/Never/Well". When the prior speaker
    # ended with "!", the current speaker often responds with a
    # surprise / calming interjection ("Peace/Hold/Stay/Soft").
    # When the prior speaker ended with "." (declarative), no
    # particular answer-opener preference applies.
    # Reset to 0 at the next turn boundary (so only the IMMEDIATELY
    # preceding turn's final sentence is remembered).
    prev_turn_final_sent_type: int = 0

    # --- Tier 2: negation scope tracker ---
    # Count of negation-class words completed inside the current
    # sentence. Negation class is:
    #   not, no, nay, never, none, nothing, naught, nought, nor,
    #   neither, n't (contracted enclitic on an auxiliary: cannot,
    #   don't, hasn't — detected as a word ending in "n't" or a
    #   word equal to "cannot").
    # Reset to 0 on sentence-end punctuation (. ? !) and on
    # speaker-turn boundary (consecutive_newlines >= 2). Capped at 5.
    #
    # Why this captures something new: Shakespeare's negations chain
    # and balance in characteristic ways:
    #   "nor X nor Y"             — once "nor" fires, another "nor"
    #                                is very likely in the next 1-4
    #                                words.
    #   "not X but Y"             — "not" often attracts a later
    #                                "but".
    #   "neither X nor Y"         — "neither" almost always preludes
    #                                a "nor".
    #   "no, nor..."              — answer-opener "no" often pairs
    #                                with a follow-on "nor"/"not".
    #   "never X, never Y"        — parallel "never" patterns.
    # No existing state field (verb_agreement, clause_slot, subord,
    # list_structure, repetition) tracks clause-level negation scope
    # or the identity of the triggering negation word. Predict layers
    # reading this can sharpen word-start letter priors for "n" (nor,
    # never, no, nothing) and "b" (but) while negation is live.
    negation_count: int = 0
    # Words completed since the most recent negation word fired.
    # Set to 0 at the moment the negation completes. Increments on
    # each subsequent just_finished_word. Capped at 8. Reset to 0
    # on sentence-end / speaker-turn.
    words_since_negation: int = 0
    # The most recent negation word (lowercased) inside this sentence,
    # or "" if no active negation. Distinguishes "nor" (which most
    # strongly attracts another "nor") from "not"/"never"/"no" (which
    # attract "but"/"nor" more broadly). Reset to "" on sentence-end /
    # speaker-turn.
    last_negation_word: str = ""

    # --- Tier 2: pronoun case slot (syntactic role for upcoming pronoun) ---
    # When the next word is a pronoun, its case is determined by the
    # slot it fills. Shakespeare (EME) is strict about this:
    #   SUBJECT  (nominative):  I, thou, ye, he, she, we, they, who
    #   OBJECT   (accusative):  me, thee, you, him, her, us, them, whom
    #   POSSESS  (genitive):    my/mine, thy/thine, his, her, our, your,
    #                            their, whose
    #
    # No existing field identifies which case to expect. clause_slot
    # tells us FRESH/HAS_SUBJ/HAS_VERB/POST_OBJ (too coarse), and
    # verb_transitivity fires object-slot only after specific verbs.
    # This field fires for ANY slot — after ANY preposition, after
    # ANY transitive verb, at clause start — and specifies the
    # expected pronoun case. Values:
    #
    #   0  CASE_NONE — no active case expectation
    #   1  CASE_SUBJ — nominative slot. Expected pronoun first letters:
    #                   I, h (he), s (she), t (thou/they), w (we), y (ye)
    #   2  CASE_OBJ  — accusative slot. Expected first letters:
    #                   m (me), t (thee/them), h (him/her),
    #                   u (us), y (you)
    #
    # Triggers:
    #   CASE_SUBJ:
    #     - clause_slot becomes FRESH (post-sentence-break) → 1
    #     - after a subordinator / conjunction opens a new clause
    #       (last_completed_word == coordinating conj AND clause_slot
    #       was POST_OBJ or a sentence-start hasn't been reached)
    #   CASE_OBJ:
    #     - last_completed_word POS == PREPOSITION → 2
    #     - verb_transitivity == VT_DO_EXPECTED on a fresh completion
    #
    # Resets to CASE_NONE on:
    #   - pronoun / noun / proper-noun / adjective / article completion
    #     (the slot got filled or NP modifier began)
    #   - sentence-end punctuation
    #   - speaker-turn boundary
    #   - wait >= 3 (expectation stale)
    case_slot: int = 0
    # Words elapsed since case_slot became non-NONE; 0 on trigger.
    # Capped at 5. Reset with case_slot.
    case_wait_words: int = 0

    # --- Tier 3: lament register (grief / plaintive texture) ---
    # A rolling [0, 1] float tracking whether the recent text is in
    # lament mode — the specific moaning/mourning texture of grief
    # passages ("Alas, poor Yorick!", "O my dear father!", "Woe is
    # me, to have seen what I have seen"). This is distinct from
    # other flow axes:
    #   - tonal_weight (dark/light valence):   lament IS typically
    #                                          dark, but tonal_weight
    #                                          covers many dark things
    #                                          (violence, disease, war)
    #                                          that aren't lament.
    #   - emotional_intensity (heat/outburst): lament is soft,
    #                                          drawn-out grief — the
    #                                          opposite of outburst heat.
    #   - invocation_mode (declamatory):        grand rhetorical
    #                                          declamation may be
    #                                          lament-adjacent but
    #                                          also includes praise/
    #                                          exhortation.
    #   - doubt_register:                       orthogonal axis.
    #
    # Bumps per completed word (additive, clipped to [0, 1]):
    #   Lament-core     (+0.18 each):
    #       alas, alack, woe, sorrow, grief, weep, wept, sigh, sighs,
    #       tears, mourn, lament, pity, piteous, wretched, wretch,
    #       forlorn, doleful, sorrowful, sad
    #   Lament-halo     (+0.10 each):
    #       heart, heavy, dead, death, dying, lost, loss, poor,
    #       cursed, cursed, dread, pain
    #   Anti-lament     (-0.15 each):
    #       joy, mirth, merry, glad, laugh, laughed, smile, smiled,
    #       happy, gay, sport, revel
    #   Anti-lament-mild (-0.05 each):
    #       strike, march, charge, fight, seize, slay (action verbs
    #       — lament is contemplative, not kinetic)
    # Sentence-end "!": -0.02 (lament tends to moan rather than shout)
    # Decay: 0.92 per just_finished_word.
    # Speaker turn: *0.35 (some carry-over; new character usually shifts
    #                      register).
    #
    # Consumed by predict/lament.py at word-start: when register >=
    # 0.35, boost grief-lexicon first letters (a, w, s, g, h, t, m,
    # p, d, l) and modestly lift "O" at sentence-start (apostrophe
    # of grief: "O woe!", "O grief!", "O heaven!").
    lament_register: float = 0.0

    # --- Tier 3: tenderness register (love / romance texture) ---
    # A rolling [0, 1] float tracking whether the recent text is in
    # tenderness mode — the soft, romantic, caressing texture of
    # Shakespeare's love scenes ("my dear lady", "sweet love", "fair
    # flower", "gentle rose", "O my beloved").
    #
    # Distinct from other flow axes:
    #   - tonal_weight (dark/light):         light-valence covers many
    #                                        cheerful things (mirth, joy,
    #                                        victory); tenderness is
    #                                        specifically the caressing,
    #                                        endearing lexicon.
    #   - imagery_density (concrete/abstract): tenderness often carries
    #                                        vivid sensory imagery but
    #                                        so do violence scenes.
    #   - emotional_intensity (outburst):    tenderness is quiet warmth.
    #   - lament_register:                    opposite pole (grief).
    #   - invocation_mode:                    declamatory, not intimate.
    #
    # Bumps per completed word:
    #   Tender-core     (+0.15 each):
    #       love, loves, loved, loving, lover, lovers, beloved,
    #       sweet, sweets, sweetly, dear, dearest, darling,
    #       fair, fairer, fairest, beauty, beauteous,
    #       kiss, kisses, kissed, gentle, gently, mild,
    #       tender, tenderly, soft, softly, fond, kind, kindly,
    #       charming, angel, angels, flower, rose, bosom
    #   Tender-halo     (+0.08 each):
    #       cheek, cheeks, eye, eyes, lip, lips,
    #       heart (shared with lament — mild halo in tenderness),
    #       bright, blossom, delight, delights, grace,
    #       heaven, true, mine
    #   Anti-tender     (-0.12 each):
    #       war, wars, blood, sword, swords, arms (weapons),
    #       battle, slain, kill, strike, strikes, rage, hate,
    #       wrath, fury, curse, cursed, foe, enemy, foul,
    #       rotten, venom
    #   Anti-tender-mild (-0.04 each):
    #       death, dread, dread, grief, woe, sorrow, tears (overlap
    #       with lament — anti-correlated)
    # Decay: 0.93 per just_finished_word.
    # Speaker turn: *0.40.
    #
    # Consumed by predict/tenderness.py at word-start: when register
    # >= 0.35, boost tenderness-lexicon first letters (l, s, f, d, g,
    # k, m, t, b, r) and mildly lift "O" at sentence-start (apostrophe
    # of love: "O my love!", "O sweet!", "O beauteous!").
    tenderness_register: float = 0.0

    # --- Tier 2/3: scene drift detector (gibberish-streak safety net) ---
    # Counts CONSECUTIVE words that completed OFF the word-trie. A
    # structural quality signal distinct from per-word letters_off_trie
    # (which resets when the word completes). When this streak grows,
    # samples are in a runaway-gibberish regime:
    #
    #   "rseanhalgmiefsem se turnce arcahil IBiesfe steed eata"
    #
    # Real Shakespeare's word-trie misses are mostly isolated (a rare
    # inflection here, an archaic form there). Three or more consecutive
    # off-trie completions almost certainly indicate the letter-n-gram
    # runaway we want to interrupt with aggressive recovery.
    #
    # Update rule (in pipeline/drift.py, after update_basic_counters
    # and word_trie state are fresh):
    #   * On just_finished_word:
    #       - If the *completed* word was on the trie (last_completed_word
    #         in COMPLETE_WORDS): reset drift_streak to 0.
    #       - Else: drift_streak = min(drift_streak + 1, 8).
    #   * On "\n\n" turn boundary: reset to 0 (a fresh speaker starts
    #     clean even after gibberish).
    #
    # Consumed by predict/drift_recovery.py: when drift_streak >= 2,
    # apply an increasingly aggressive recovery bias at word-start
    # that boosts very common English first letters (t, a, i, o, h, w,
    # b, s, f, m) and suppresses rare starters (x, z, j, q) — pulling
    # the sampler back toward known-word territory.
    drift_streak: int = 0

    # --- Tier 2: verb-complement class expectation ---
    # When a verb completes, mark what KIND of complement it expects.
    # This is finer-grained than `verb_transitivity` (3-way: NONE /
    # DO_EXPECTED / COMP_EXPECTED): mental verbs want that-clauses,
    # motion verbs want PPs, auxiliaries want past participles,
    # perception verbs want NP-or-clause. Each needs a different
    # word-start bias on the IMMEDIATELY following word.
    #
    # Values (verb_complement_class):
    #   VCC_NONE = 0   — no active expectation
    #   VCC_THAT  = 1  — mental/communication verb: expect "that",
    #                    or direct quote/clause opener, or NP (when
    #                    the verb also licenses a bare NP: "I know him")
    #   VCC_PP    = 2  — motion verb: expect preposition (to/from/
    #                    toward/into/through/upon)
    #   VCC_PPART = 3  — auxiliary "have/hath/had/hast": expect past
    #                    participle (verb_ed or irregular: seen, done,
    #                    gone, taken, fought, spoken)
    #   VCC_INF   = 4  — modal / "to": expect bare infinitive verb
    #   VCC_PRED  = 5  — copula "is/are/was/were/be/am/art": expect
    #                    predicate ADJ, NP, or VERB_ING (progressive)
    #
    # Set at verb-completion (on just_finished_word, when
    # last_completed_word is in the appropriate verb inventory).
    # Aged each subsequent just_finished_word; reset when a fitting
    # complement appears or when wait exceeds a small budget.
    verb_complement_class: int = 0
    vcc_wait_words: int = 0

    # --- Tier 2: parenthetical-dash scope tracking ---
    # Shakespeare uses "--" (374× in train) as a mid-sentence
    # parenthetical break, often followed by a newline and a
    # capitalised new-clause opener ("--\nFor, look you..." /
    # "--\nBut...").  Nothing in the existing state tracks whether
    # we're inside an unclosed "--" aside, so predict layers can't
    # condition on "we just opened a dash" vs "we just closed one".
    #
    # `in_dash_aside` flips True on the second '-' of a "--" run and
    # flips False on:
    #   * another "--" (closing pair),
    #   * sentence-end punctuation ('.?!'),
    #   * a speaker-turn boundary (consecutive_newlines >= 2).
    #
    # `chars_since_dash_open` counts characters emitted since the
    # opening '--' was completed; used by predict layers to decay
    # the after-dash bias.
    #
    # `words_since_dash_open` counts completed words since dash open;
    # dash asides are typically short (1–5 words), so a high count is
    # a signal the aside is winding down.
    in_dash_aside: bool = False
    chars_since_dash_open: int = 0
    words_since_dash_open: int = 0

    # --- Tier 2: proper-noun scene rolodex ---
    # Whether the currently-building word's first letter was an
    # uppercase letter. Reset to False at word termination; set to
    # True when the first character of a new letter run is UPPER.
    # Lets us decide at word-completion whether this was a
    # capitalized word (proper noun / sentence-starter / vocative).
    current_word_started_cap: bool = False
    # Rolling tuple of recently-seen CAPITALIZED content words
    # (lowercased for lookup). Excludes:
    #   - sentence-first words (capital is forced by position)
    #   - speaker-label words (tracked separately)
    #   - 1-letter capitals ("I", "O")
    #   - common capitalized interjections ("Ay", "Nay")
    # Up to 10 entries, most-recent first. Shakespeare reuses proper
    # nouns heavily once introduced ("Rome", "Volsces", "Coriolanus",
    # "Antium", "Northumberland"); this field lets predict bias
    # future word-starts and mid-word letters toward these recurring
    # names instead of guessing fresh each time.
    proper_nouns_seen: tuple[str, ...] = ()

    # --- Tier 2/3: imperative chain counter ---
    # Tracks the number of CONSECUTIVE imperative sentences that have
    # just closed in the current speaker turn. Shakespeare's dramatic
    # pacing — especially at moments of crisis or command — features
    # rapid imperative chains:
    #
    #   "Speak! Attend! Mark!"
    #   "Go! Fly! Away!"
    #   "Hold! Peace! Stand forth!"
    #   "Bring torches! Call the guard! Rouse all the house!"
    #
    # After 2+ consecutive imperatives, the next sentence is much more
    # likely to open with another imperative-head verb than with a
    # declarative opener. This field captures that short-range
    # momentum so the predict layer can reinforce it.
    #
    # Update rules (applied at sentence-end punctuation):
    #   - If saved sentence_type == SENT_IMPER: count += 1
    #   - Otherwise: count reset to 0
    #   - On speaker-turn boundary (\n\n): reset to 0
    # No decay beyond the reset — the signal is structural (either
    # we're in a chain or we've broken it).
    #
    # Consumed by predict/imperative_chain.py at sentence-start when
    # count >= 2 (post two closed imperatives, before the next one's
    # first word): boost imperative-opener capitals (G=Go/Give, C=Come/
    # Call, S=Speak/Stand/Stay/See, T=Tell/Take, H=Hear/Hold/Hark/Help,
    # A=Away/Attend, L=Let/Look/Live, M=Mark, B=Be/Begone/Bring/Behold,
    # F=Fly/Forbear/Follow, P=Peace, O=Open/Out, W=Watch).
    imperative_chain_count: int = 0

    # --- Tier 3: gravitas register (moral / philosophical weight) ---
    # A rolling [0, 1] float tracking whether the recent discourse
    # carries moral / philosophical / cosmic weight — distinct from
    # grief (lament_register), love (tenderness_register), darkness
    # (tonal_weight), or declamation (invocation_mode).
    #
    # Shakespeare's heightened meditations — Hamlet's soliloquies,
    # Lear on the heath, Brutus in the forum — share a specific
    # lexicon of abstract nouns (honour, duty, virtue, conscience,
    # soul, heaven, fate, nature, truth, justice, reason) that signal
    # a character is weighing ethics and cosmos. When this register
    # is active, the next content word is far more likely to come
    # from that abstract-philosophical cloud than from the concrete
    # one (table, street, bread). This field carries that register
    # across function-word scaffolding.
    #
    # Bumps per completed word (applied at just_finished_word):
    #   Gravitas-core     (+0.14 each):
    #       honour, honor, virtue, virtuous, soul, souls,
    #       duty, duties, conscience, truth, justice, reason,
    #       fate, fates, doom, mortal, mortals, immortal,
    #       heaven, heavens, earth, nature, god, gods, divine,
    #       eternal, perpetual, everlasting
    #   Gravitas-halo     (+0.07 each):
    #       honest, honesty, faith, faithful, faithless, sin, sins,
    #       holy, sacred, blest, blessed, cursed, grace, mercy,
    #       pity, shame, glory, power, powers, spirit, spirits,
    #       right, wrong, deed, deeds, life, death, world, worlds,
    #       time, times, will (noun), peace, war, crown
    #   Anti-gravitas     (-0.05 each):
    #       drink, drinks, cup, ale, meat, bread, bed, sleep,
    #       laugh, laughs, jest, fool, merry, sport
    # Decay: 0.94 per just_finished_word.
    # Speaker turn: *0.50 (new speaker may inherit partially).
    #
    # Consumed by predict/gravitas.py at word-start: when register
    # >= 0.25, boost gravitas-lexicon first letters (h=honour/heaven,
    # v=virtue, s=soul/sin/sacred, d=duty/doom/divine, t=truth/time,
    # c=conscience/crown, j=justice, m=mortal/mercy, e=earth/eternal,
    # f=fate/faith, r=reason/right, g=god/glory, p=power/pity) and
    # lift "O" / "Oh" at sentence-start (gravitas apostrophe).
    gravitas_register: float = 0.0
    # --- Tier 3 flow: fury register — rage / wrath / curse texture ---
    # A rolling float in [0, 1] tracking the rage / wrath / cursing
    # register of the current speech. Distinct from:
    #   - tonal_weight (dark vs light scene events — external)
    #   - gravitas    (moral / philosophical weight — sober, not angry)
    #   - lament      (grief — mournful, not angry)
    #   - tenderness  (love — opposite polarity of fury)
    # Fury is angry speech FROM the speaker TOWARD someone or something:
    # curses, threats, insults, imprecations. Characteristic Shakespeare:
    # Lear's storm, Timon's misanthropy, Mercutio's "plague o' both your
    # houses", Iago's asides.
    #
    # Bumps on completed words:
    #   STRONG_FURY (rage, wrath, fury, damn, curse, hell, plague,
    #                villain, knave, traitor, slave-as-insult, vile,
    #                foul, wretch, fiend, viper, poison, venom, devil):
    #                                              +0.22
    #   MILD_FURY   (hate, hated, strike, kill, blood-for-vengeance,
    #                scorn, spite, shame-on-you, bastard, rascal,
    #                cur, dog-as-insult, rogue, rot, burn-in-hell):
    #                                              +0.10
    #   COUNTER     (peace, love, sweet, gentle, kind, fair, soft,
    #                calm, mercy, forgive): -0.06 (cools the register)
    #   "!" completed boosts +0.08 when fury > 0 (exclamation as anger
    #                 amplifier — neutral when fury already 0).
    #
    # Decay: multiply by 0.94 per completed word (faster decay than
    # gravitas: anger is punchier and should not linger).
    # Speaker-turn: multiply by 0.20 (mostly reset — a new speaker
    # rarely inherits the prior speaker's rage except in an ongoing
    # argument, where the next speaker's own words will quickly re-raise
    # it).
    #
    # Consumed by predict/fury.py at word-start and word-end to:
    #   - boost fury-lexicon starter letters (d, h, w, v, c, r, f)
    #   - boost "!" over "." at sentence-end when fury > 0.35
    #   - mildly discourage tender-lexicon starters (l, s)
    fury_register: float = 0.0

    # --- Tier 2: per-line coherence tracking ---
    # Counts of words completed on the CURRENT line, bucketed by
    # trie status. "On-trie" means the word as completed is a member
    # of COMPLETE_WORDS (i.e., a recognizable English/Shakespearean
    # word). "Off-trie" means the word drifted off the trie before
    # completion — i.e., it's a letter-n-gram hallucination with
    # no vocabulary grounding.
    #
    # Why this is a structural signal:
    #   Real Shakespeare lines are overwhelmingly composed of real
    #   words. The sampler's dominant quality defect is lines like
    #   "Yield, iegeohbe awaostr. observe bear ghost" where most
    #   tokens are garbage. Per-word drift_streak captures whether
    #   we're mid-run of gibberish, but NOT whether the current line
    #   has already become unsalvageable. A line with 2+ off-trie
    #   words and only a token or two of real vocabulary is dead
    #   weight — the predict layer should push hard to end it with
    #   a newline (close the line and restart on a cleaner basis).
    #
    # Update rule (in pipeline/line_coherence.py, runs AFTER
    # update_basic_counters and update_drift):
    #   - On every just_finished_word: classify last_completed_word
    #     against COMPLETE_WORDS. Increment on-trie or off-trie count.
    #   - Reset BOTH counters to 0 on newline (chars_since_newline == 0).
    #
    # Consumed by predict/line_coherence.py at word-end-on-trie
    # positions AND at off-trie-with-short-buffer positions:
    #   - If line_offtrie_words >= 2 AND line_ontrie_words <= 1:
    #     line is failing — boost \n (end the line now).
    #   - If line_ontrie_words >= 3 AND line_offtrie_words == 0:
    #     line is healthy — mild anti-newline nudge so a well-formed
    #     line can breathe out to natural length.
    line_ontrie_words: int = 0
    line_offtrie_words: int = 0

    # --- Tier 2: sentence-level tense register ---
    # Captures the tense / modality of the FIRST finite verb seen in
    # the current sentence. Once set, it biases later verb-like
    # word-ends in the same sentence toward tense-consistent suffix
    # endings so the sentence doesn't mix past and present arbitrarily
    # ("I walked to the market and am ..." — off-register).
    #
    # Values:
    #   0  TENSE_UNSET   — no finite verb classified yet in this sentence.
    #   1  TENSE_PAST    — first finite verb was past-tense:
    #                      - ended in -ed / -d (walked, loved, feared)
    #                      - suppletive past (was, were, had, did, said,
    #                        went, came, saw, knew, gave, took, made,
    #                        told, heard, thought, fell, found, lost,
    #                        wrote, stood, grew, threw, sat, drew, bore,
    #                        tore, wore, spoke, spake, rose, broke)
    #                      - modal past (would, should, could, might)
    #   2  TENSE_PRESENT — first finite verb was present-tense:
    #                      - 3sg -s / -eth / -s (speaks, speaketh, loves)
    #                      - be forms (is, am, are, art)
    #                      - have forms (have, has, hath)
    #                      - do forms (do, does, dost, doth)
    #                      - base present after "we/they/you/I"
    #   3  TENSE_FUTURE  — first finite verb was modal future:
    #                      - will, shall (+ bare verb)
    #
    # Update rule (pipeline/tense.py):
    #   - On sentence-end punctuation (. ? !): reset to TENSE_UNSET.
    #   - On speaker-turn boundary (\n\n): reset to TENSE_UNSET.
    #   - On every just_finished_word where sentence_tense == UNSET:
    #     attempt to classify last_completed_word. If classifiable,
    #     set the tense. Otherwise leave UNSET.
    #
    # Age counter `sentence_tense_age` tracks completed words since
    # the tense was set — decays influence of the bias for very long
    # sentences where tense can naturally shift in dependent clauses.
    #
    # Consumed by predict/tense.py at word-start and at verb-shaped
    # word-ends: tilts letter choices toward tense-consistent suffix
    # trajectories (-ed / -eth / -s / -ing branch selection).
    sentence_tense: int = 0
    sentence_tense_age: int = 0

    # --- Tier 3: martial / battlefield register ---
    # A rolling float in roughly [-2, +3] that tracks whether the
    # recent content words have been MARTIAL (sword, blood, wound,
    # arm, steel, strike, war, battle, soldier, captain, kill, slain,
    # iron, fight, march, pike, shield, lance, horse, trumpet, drum,
    # conquer, vanquish, foe, enemy, armor, helmet, banner, siege,
    # breach, assault) or PEACEFUL / PASTORAL (peace, love, sleep,
    # rest, home, gentle, kind, soft, mild, bed, bread, song, music,
    # flower, garden, grove, dove, lamb).
    #
    #   +3  = heavy martial texture (battle scene, history-play
    #         crisis, Macbeth/Othello/Antony & Cleopatra war talk)
    #    0  = neutral
    #   -2  = peaceful / pastoral (Midsummer wood-talk, lovers'
    #         chamber scenes, benediction)
    #
    # Update rule (pipeline/martial.py, at word-completion):
    #   - Look up last_completed_word in curated martial / peaceful
    #     word sets (prior knowledge, not corpus statistics).
    #     Martial words add +0.70, peaceful words add −0.40. Clamped
    #     to [-2.0, +3.0].
    #   - All words decay the value by ×0.93 per word so influence
    #     fades if the topic drifts.
    #   - Reset to 0 on speaker-turn boundary (\n\n).
    #
    # Consumed by predict/martial.py at word-start positions (outside
    # speaker labels, post-space/newline): when martial_charge > +1.3,
    # boosts first letters of martial-starter vocabulary (s/b/w/a/f/
    # k/m/h/i/c — sword/blood/wound/arms/fight/kill/march/horse/iron/
    # captain); when < -1.0, boosts first letters of peaceful
    # vocabulary (p/l/g/s/f/k/m/h — peace/love/gentle/soft/flower/
    # kind/mild/home).
    #
    # Distinct from:
    #   - fury (emotional rage/curse register, not battlefield lexicon)
    #   - sensory_charge (broad corporeal-vs-abstract axis, covers
    #     lyric imagery too — roses, dew, night, breeze)
    #   - gravitas (moral weight, honor, oath — cognate but different
    #     lexicon)
    # Martial is specifically the BATTLEFIELD / ARMS lexicon and the
    # peaceful counter-pole is its negation, not generic tenderness.
    martial_charge: float = 0.0

    # --- Tier 3: sensory charge (corporeal ↔ abstract axis) ---
    # A rolling float in roughly [-3, +3] that tracks whether recent
    # completed content words have been CORPOREAL / SENSORY (body
    # parts, elements, weapons, weather, blood, tears, flame, sword,
    # heart, eye, hand, night, storm, moon, sun, grave, fire) or
    # ABSTRACT / DISCURSIVE (cause, matter, reason, question, truth,
    # justice, honour, virtue, duty, purpose, sense, fault, doubt).
    #
    #   +3  = heavy sensory / corporeal texture (tragic lyric, battle
    #         speech, imagery-soaked verse)
    #    0  = neutral
    #   -3  = heavy discursive / reasoning texture (court argument,
    #         deliberation, deliberative prose)
    #
    # Update rule (pipeline/sensory_charge.py, at word-completion):
    #   - Look up last_completed_word in curated sensory / abstract
    #     word sets (prior knowledge, not corpus-derived). Sensory
    #     words add +0.6, abstract words add −0.5 — clamped to
    #     [-3, +3]. All words (including function words) decay the
    #     value by a small factor (×0.93) so influence fades.
    #   - Reset to 0 on speaker-turn boundary (\n\n).
    #
    # Consumed by predict/sensory_charge.py at word-start positions
    # (outside speaker-label territory, post space/newline): when
    # charge > +1.0, first-letters of sensory vocabulary get a boost
    # (b/e/h/f/s/t/n/g — blood/eye/heart/fire/sword/tears/night/grave);
    # when charge < −1.0, first-letters of abstract/discursive
    # vocabulary get a boost (c/m/r/q/t/j/h/v/d — cause/matter/reason/
    # question/truth/justice/honour/virtue/duty). Neutral charge =
    # no bias. This makes lyric passages continue their corporeal
    # register, and prose-argument passages continue their abstract
    # register, rather than drifting between modes mid-passage.
    sensory_charge: float = 0.0

    # -----------------------------------------------------------
    # Word integrity monitor — targeting gibberish word-runs.
    # -----------------------------------------------------------
    # Real Shakespeare contains no gibberish words. Our samples do:
    # sequences like "etustarse", "daetfaanwetfimnly", "rotxouddfser"
    # appear when the letter-run drifts off the word-trie and the
    # n-gram backoff keeps producing phonotactic-ish noise. The
    # existing `drift_streak` / `offtrie_depart` track departure from
    # the trie but don't evaluate the shape of the buffer *itself*.
    #
    # word_integrity is a running score in [0.0, 1.0] measuring how
    # word-shaped the current word_buffer is. It combines:
    #   - Presence of a vowel: true words always have one within the
    #     first 4 letters (except mono-consonantal "sh", "ps" etc).
    #   - Longest consonant run: real English words rarely have 4+
    #     consonant runs mid-word.
    #   - Recent consonant drought since last vowel: if we're 4+
    #     letters past the last vowel, the buffer is almost certainly
    #     broken.
    #   - Trie match: buffer is a prefix of a known word (already
    #     reflected in on_word_trie / trie_match_count, but we fold
    #     the signal in here too).
    #
    # Updated in pipeline/word_integrity.py at every character. Resets
    # to 1.0 when word_buffer is empty (word just ended).
    #
    # Consumed by predict/word_integrity.py — when integrity is low
    # (< 0.5) AND letter_run_len >= 4, strongly boost terminator
    # characters (space, comma, period, semicolon, newline) so the
    # model bails out of the nonsense run. When integrity is high
    # OR letter_run_len < 4, no bias (normal behavior).
    word_integrity: float = 1.0
    # Has the current word_buffer contained any vowel (a/e/i/o/u/y)?
    # Resets to False when word_buffer empties.
    buffer_has_vowel: bool = False
    # Position of the most recent vowel within word_buffer (1-indexed,
    # 0 means "no vowel yet"). Resets when word_buffer empties.
    buffer_last_vowel_pos: int = 0
    # Length of the current trailing consonant run (letters since the
    # last vowel in word_buffer). Resets to 0 at each vowel, growing
    # by 1 per consonant. Does not increment on apostrophe.
    buffer_consonant_run: int = 0

    # --- Discourse tier: expected_answer_type ---------------------------
    # Cross-turn discourse link. When the previous turn ended with a
    # question whose first sentence-opener was a WH-word, the *next*
    # turn's opening word is tightly constrained to answer-type
    # vocabulary. E.g., "Where art thou?" → "Here", "In the garden";
    # "Why dost thou weep?" → "Because", "For", "Since"; "Art thou
    # well?" → "Ay", "No", "Indeed". This field carries the expected
    # answer class across the turn boundary so the predict layer can
    # sharpen the opener.
    #
    # Values:
    #   0 ANS_NONE        (no pending answer)
    #   1 ANS_YESNO       (aux-led yes/no question: is/art/hast/dost/...)
    #   2 ANS_WHAT        (what / wherein)
    #   3 ANS_WHERE       (where / whither / whence)
    #   4 ANS_WHEN        (when)
    #   5 ANS_WHY         (why / wherefore)
    #   6 ANS_HOW         (how)
    #   7 ANS_WHO         (who / whom / whose)
    #   8 ANS_WHICH       (which / what + [noun])
    #
    # Set on `?` emission at end of sentence (pipeline/question_answer
    # detects the WH-class from curr_sentence_first_word); carried
    # across the double-newline turn boundary; cleared on the first
    # word-completion of the response turn.
    pending_question_type: int = 0

    # --- Flow tier: oath_mode -------------------------------------------
    # [0, 1] rolling field capturing solemn-oath texture. Bumps on oath
    # openers ("swear", "swore", "sworn", "oath", "pledge", "vow",
    # "troth", "faith"), on "by"/"upon" at sentence start or after
    # comma, and on reinforcing oath objects ("heaven", "soul", "God",
    # "honour") when the mode is already warm. Decays per completed
    # word. Damped across speaker turns.
    #
    # Unlike invocation_mode (declamatory voice) or emotional_intensity
    # (generic heat), oath_mode specifically captures the formulaic
    # promise / curse texture that punctuates Shakespeare — "by my
    # troth", "upon my soul", "God save the king", "I swear it by my
    # sword". The predict layer uses it to sharpen word-start biases
    # toward oath-object vocabulary after "by" / "upon" / "my" when
    # the mode is hot, and toward a closing comma once the oath-object
    # has been completed.
    oath_mode: float = 0.0

    # --- Tier 2: syntactic-frame role projection ---
    # `expected_next_role` names the syntactic role we project for the
    # NEXT word to complete, given the current two-word POS context
    # (last_word_pos, prev_word_pos), clause_slot, and np_open. The
    # motivation is that existing per-word POS-tagging is BACKWARD-
    # LOOKING (tag the word that just completed); trigram failures in
    # samples come from having no forward expectation at word-start.
    # A role projection lets the predict layer sharpen first-letter
    # mass toward a few plausible next roles instead of the full
    # 19-way POS distribution.
    #
    # Values are a small frame-role enum (see
    # pipeline/syntactic_frame.py for the canonical list):
    #   0 = FRAME_ANY        — no strong projection (default)
    #   1 = FRAME_NOUN       — bare noun
    #   2 = FRAME_ADJ_OR_NOUN — adjective or bare noun
    #   3 = FRAME_NOUN_ONLY  — strongly noun (DET+ADJ → NOUN, POSS+ADJ → NOUN)
    #   4 = FRAME_DET_OR_POSS — article/possessive at phrase start
    #   5 = FRAME_VERB_FAMILY — verb, aux, modal (after subject pronoun)
    #   6 = FRAME_VERB_ONLY  — main verb (after modal/aux)
    #   7 = FRAME_PREP_OR_CONJ — preposition / conjunction (after verb+obj)
    #   8 = FRAME_OBJ        — noun/pronoun/det (object position after verb)
    #   9 = FRAME_SUBJ       — subject pronoun / det / proper noun
    #  10 = FRAME_ADV_OR_PREP — adverbial / prepositional phrase start
    #
    # Resets to FRAME_ANY on sentence-end, turn boundary, and whenever
    # the FSM can't confidently project (e.g., after an unknown-POS
    # word).
    expected_next_role: int = 0
    # Confidence in the projection, in [0.0, 1.0]. Low conf → weak
    # predict-layer bias; high conf → sharp first-letter bias. Allows
    # the predict consumer to scale its push cleanly.
    frame_confidence: float = 0.0

    # -- Conditional / concessive discourse FSM (apodosis expectation)
    #
    # Shakespeare (and English generally) builds conditional and
    # concessive sentences with a PROTASIS → APODOSIS structure:
    # "If thou lovest me, then say so", "Though he be honest, he is
    # rash". The protasis opens with a subordinator (if / though /
    # when / since / unless / lest / albeit / whereas) and CLOSES
    # typically with a comma. The APODOSIS — the main clause — then
    # follows, almost always with a subject pronoun, a modal, a bare
    # imperative verb, or the adverbs "then"/"so".
    #
    # No existing state fires a signal specifically at the start of
    # the apodosis. subord_depth tracks nesting but doesn't project
    # what kind of constituent must come AFTER the subord closes.
    # conditional_mode closes that gap.
    #
    # Values:
    #   0 = NONE        — not inside a conditional/concessive structure
    #   1 = PROTASIS    — opened with a subordinator; no comma yet
    #   2 = APODOSIS    — protasis-closing comma has fired; awaiting
    #                    main-clause opener (pronoun / modal / imperative)
    #   3 = RESOLVED    — the main clause has started; register the
    #                    fact (some predict layers may still want to
    #                    know we came from a conditional)
    #
    # Resets to NONE on sentence-end (. ? !) and on turn boundary.
    conditional_mode: int = 0
    # The subordinator that opened the protasis (0 = none).
    #   1 = if, 2 = though, 3 = when, 4 = since, 5 = unless,
    #   6 = lest, 7 = whereas, 8 = albeit, 9 = although, 10 = while
    conditional_opener: int = 0
    # Words emitted since the conditional_mode last changed. Useful
    # for scale decay (apodosis bias should be strongest at its very
    # first word and decay thereafter).
    conditional_age: int = 0

    # --- Intra-sentence clause-parallelism ------------------------
    # Shakespeare often builds parallel clause structures within a
    # single sentence:
    #   "I came, I saw, I conquered."
    #   "She is fair, she is wise, she is true."
    #   "Speak soft, speak low, speak truly."
    # When two consecutive clauses (separated by comma or semicolon)
    # open with the same first word (or first letter), the pattern
    # tends to continue for at least one more clause. No existing
    # state tracks clause-level openers within a sentence — anaphora
    # tracks LINE-starters (newline-boundary anchored), and list_
    # structure tracks list items but is a coarser FSM.
    #
    # At the start of each clause WITHIN a sentence (post-comma or
    # semicolon), we record the first-letter of that clause-opener
    # word. At the next clause-start, the predict layer can consult
    # this to nudge the new opener's first letter toward echoing.
    #
    # Fields:
    #  - clause_opener_letter: first letter of the CURRENT clause's
    #    opener word (empty until the first word completes).
    #  - prev_clause_opener_letter: first letter of the PREVIOUS
    #    clause's opener word within the current sentence.
    #  - clauses_in_sentence_index: which clause we're in (0 for
    #    sentence-opening, 1 after first comma, etc.). Helps scale
    #    the echo pressure — strongest at clause index 2+.
    #
    # All three reset on sentence-end (. ? !) and on turn boundary.
    clause_opener_letter: str = ""
    prev_clause_opener_letter: str = ""
    clauses_in_sentence_index: int = 0

    # --- Word-orthographic integrity (apostrophe + cap state) ---------
    # Shakespeare's word-buffers are tightly constrained orthographically
    # within a single word:
    #   * Once a word has started lowercase, no uppercase letter ever
    #     appears until the word ends (punctuation or space).
    #   * Once a word has emitted its capital first letter, subsequent
    #     letters are nearly always lowercase — mid-word caps only in
    #     rare compound proper nouns (O'Neill, McBeth) and even those
    #     use an apostrophe or prefix before the second cap.
    #   * After an apostrophe inside a word, the following letter is
    #     drawn from a very small set: s / d / t / l / r / v / e / n / m
    #     (covering 's, 'd, 't, 'll, 're, 've, 'er, 'en, 'em and a few
    #     archaic forms like 'twas, 'twere, o'er). Any other letter
    #     after an apostrophe is almost certainly gibberish.
    #
    # Fields:
    #  - letters_since_apostrophe: distance (in letters) since the last
    #    apostrophe IN the current word_buffer. 0 = no apostrophe yet in
    #    this word; 1 = just emitted an apostrophe, next char is the
    #    first letter after it; 2 = one letter past apostrophe; etc.
    #    Resets to 0 whenever word_buffer empties.
    #  - had_apostrophe_this_word: has the current word contained an
    #    apostrophe? True between apostrophe emission and word end.
    letters_since_apostrophe: int = 0
    had_apostrophe_this_word: bool = False

    # --- Tier 2/3: iambic meter tracking ---
    # Shakespeare's dramatic verse is dominantly iambic pentameter:
    # ten-syllable lines with an alternating weak–STRONG foot pattern
    # (xX xX xX xX xX). Content words (nouns, verbs, adjectives) tend
    # to start on the STRONG ictus; monosyllabic function words (a,
    # the, to, of, and, but, in, on, with) tend to occupy the WEAK
    # offbeats. Line-end pressure spikes at syllable 10 (masculine
    # close) and syllable 11 (feminine ending).
    #
    # These fields expose that structure to predict layers so the next
    # word-start can be biased by expected stress, not just by syntactic
    # role.
    #
    # Fields:
    #  - meter_confidence: rolling [0.0, 1.0] estimate that the current
    #    passage is iambic verse. Bumped when a closing line lands in
    #    the 9–11 syllable window; decayed when a line overshoots (>13)
    #    or undershoots (<6 on a non-blank line). Also decayed per word
    #    so long prose passages drift the confidence to 0 even if we
    #    stopped seeing explicit line signals.
    #  - expected_stress: 0 (weak / offbeat) or 1 (strong / ictus) —
    #    the predicted metrical weight of the NEXT syllable onset,
    #    assuming iambic. Computed from `syllables_in_line` parity:
    #    iambic pentameter has strong beats at syllables 2, 4, 6, 8, 10
    #    (1-indexed). Next syllable index is `syllables_in_line + 1`;
    #    if that is even → STRONG, else WEAK.
    #  - syllables_until_line_end: projected syllables remaining before
    #    a pentameter line-end closure. Clamped to [0, 10]. Zero means
    #    a line-end is immediately plausible (syllables_in_line >= 10).
    #    Only meaningful when `meter_confidence` is elevated.
    #
    # Consumed by:
    #  - predict/meter.py — word-start bias tilting toward content-word
    #    opener letters on strong beats and function-word opener letters
    #    on weak beats when meter_confidence is committed.
    meter_confidence: float = 0.0
    expected_stress: int = 0
    syllables_until_line_end: int = 10

    # --- Tier 2/3: coarse semantic noun-class tagging ---
    # Addresses cross-word semantic drift ("throne of treasure", "my
    # mother is niece"): each transition is locally grammatical but
    # successive content words belong to incompatible semantic
    # frames. 12-class tagger lives in state/noun_classes.py.
    #
    # - last_noun_class: id of the most recently matched noun class;
    #   persists across intervening non-noun words so the bias can
    #   span a short N → (function-word)* → N phrase. 0 = no recent
    #   noun match, or memory cleared.
    # - noun_class_age: completed-words since last_noun_class was
    #   (re-)set. 0 immediately after the noun; incremented on every
    #   subsequent word completion; memory cleared at age >= 8.
    #
    # Consumed by predict/noun_class.py — gated to fire only at
    # word-start immediately after a preposition / possessive /
    # article / conjunction, where the upcoming content word's
    # semantic field really matters ("throne OF ___", "my ROYAL ___").
    last_noun_class: int = 0
    noun_class_age: int = 0

    # --- Tier 2: sentence backbone tracking ---
    # A well-formed English sentence needs a SUBJECT and a FINITE VERB.
    # Real Shakespeare nearly always has both before a terminal
    # punctuation. Tracking this gives the predict layer a principled
    # reason to suppress ".", "?", "!" when the sentence so far has no
    # verb (or no subject), and to mildly elevate them once the
    # backbone is present and the sentence has sufficient material.
    #
    # Detection heuristics:
    #  - Subject: POS_PRONOUN, POS_POSSESSIVE (then noun), POS_ARTICLE
    #    (then any word), POS_PROPER_NOUN, or a content noun at
    #    sentence-initial position.
    #  - Finite verb: POS_AUX_VERB, POS_MODAL, POS_VERB, or POS_VERB_ED
    #    at a position where it's acting as the main verb (i.e., not
    #    just following another verb as a participle). Heuristic: any
    #    of these counts as a finite verb.
    #
    # Fields:
    #  - sentence_has_subject: True once we've seen a subject candidate
    #    since the last sentence-end punctuation.
    #  - sentence_has_verb: True once we've seen a finite-verb
    #    candidate since the last sentence-end punctuation.
    # Both reset at sentence-end and at speaker-turn boundary.
    #
    # Consumed by predict/sentence_backbone.py to bias terminal
    # punctuation at word-end decision points.
    sentence_has_subject: bool = False
    sentence_has_verb: bool = False

    # --- Tier 2: sentence_pressure — signed completion-readiness score ---
    # Negative (= "keep going") when the current sentence is structurally
    # incomplete: missing subject/verb, inside an open NP waiting for a
    # head, inside a subord clause, or the last completed word is a
    # function word (conjunction / preposition / article / possessive /
    # aux / modal) that demands a following word.
    #
    # Positive (= "ready to close") when the sentence has a full backbone
    # (subject AND finite verb) and has already run long.
    #
    # Roughly in [-2.0, 2.0]. Updated every token by
    # pipeline/sentence_pressure.py (after np_head, clause_slot, subord,
    # pos, sentence_backbone). Consumed by predict/sentence_pressure.py
    # to suppress terminators (\n, . ? ! ; :) when pressure is strongly
    # negative, and to give a small positive bump to . when pressure is
    # strongly positive.
    #
    # Structural extension to sentence_backbone_bias (which only fires
    # on . ? ! at 5+ words with no verb) — this layer covers the full
    # range of structural incompleteness, operates at every word-end,
    # and crucially also suppresses newlines (a known sample pain point
    # where mid-clause \n produces fragment lines).
    sentence_pressure: float = 0.0

    # --- Tier 2/3: sentence-scoped semantic field lock ---
    # Once a sentence has introduced TWO content nouns of the same
    # noun_class (e.g. two BODY nouns: "heart" and "tongue"), that
    # noun_class is LOCKED as the sentence's semantic field. At
    # subsequent word-start positions within the same sentence, the
    # predict layer can tilt the first-letter distribution toward
    # letters that begin in-field noun / adjective words.
    #
    # This is orthogonal to last_noun_class (single-step bias) and
    # scene_topic (cross-turn topic) — it tracks within-sentence
    # semantic field stability, which real Shakespeare nearly always
    # respects at the clause level: "His heart, his tongue, his very
    # pulse" (all BODY); "The crown, the throne, the sceptre" (all
    # ROYALTY); "Sorrow, grief, despair" (all EMOTION).
    #
    # Fields:
    #  - sentence_sem_field: the locked class id, or 0 = none.
    #  - sentence_sem_strength: number of in-field noun hits seen this
    #    sentence (cap 3). Lock engages at strength >= 2.
    # Both reset on sentence-end and speaker-turn boundary.
    sentence_sem_field: int = 0
    sentence_sem_strength: int = 0

    # --- Tier 2/3: sentence-scoped syllable budget ---
    # Complements syllables_in_line (prosody.py) which tracks syllable
    # count per newline-bounded line. A SENTENCE in Shakespeare often
    # spans multiple lines of verse (run-on couplets) or is confined
    # to one long prose breath. Tracking syllables per sentence
    # captures a *rhythmic breath* that the per-line counter misses.
    #
    # Parallel sentences within a turn tend to equalize in syllable
    # length (Shakespeare's balanced rhetoric: "To be or not to be /
    # that is the question" — two ~10-syllable halves). When the
    # current sentence's syllable count approaches the rolling
    # average of the last two sentences, a terminator becomes
    # rhythmically appropriate; when it overshoots significantly,
    # the terminator bias strengthens.
    #
    # Fields:
    #  - syllables_in_sentence: running count within the current
    #    sentence (resets on PUNCT_END and speaker-turn boundary)
    #  - prev_sentence_syllables: captured at PUNCT_END of most
    #    recent closed sentence. Reset on speaker-turn boundary so
    #    each turn's rhythm is local.
    #  - prev_prev_sentence_syllables: one step further back. Having
    #    TWO peers gives a stable local average (k=2 is Shakespeare's
    #    typical parallel-pair reference).
    syllables_in_sentence: int = 0
    prev_sentence_syllables: int = 0
    prev_prev_sentence_syllables: int = 0

    # --- Tier 3 flow: mirth register ---
    # Comic / merry / festive texture. Rolling scalar in [0.0, 1.0]
    # that RISES on mirth-class lexicon (merry, laugh, jest, fool,
    # feast, song, revel, holiday, wedding, play, cheer, happy) and
    # FALLS on grief/fury/gravitas lexicon (grief, sorrow, death,
    # wrath, doom, hell). Decays per completed word so that a single
    # mirthful word doesn't commit the register for a whole turn;
    # sustained mirth requires multiple hits. Mostly reset across
    # speaker turns (soft reset — retain a small residue so a
    # following-on speaker in the same comic scene can carry some
    # momentum).
    #
    # Orthogonal to existing flow axes:
    #   - lament_register:   grief (opposite polarity but weaker signal)
    #   - tenderness:        love (different positive register)
    #   - fury:              rage (opposite polarity)
    #   - gravitas:          moral weight (sober, not merry)
    #   - tonal_weight:      scene dark/light (external; mirth is
    #                        interpersonal tone)
    #
    # Read by predict/mirth.py to gently tilt word-start letters
    # toward mirth-lexicon starters when the register is elevated.
    mirth_register: float = 0.0

    # --- Tier 3: apostrophe / figurative-address mode ---
    # A rhetorical-figure axis distinct from:
    #   - vocative_expectation (adjective-slot before addressee noun)
    #   - last_addressee / recent_addressees (character in scene)
    #   - speaker_register (formal/informal commitment)
    #   - doubt / lament / fury / etc. (emotional register)
    #
    # Apostrophe is the Shakespearean figure of ADDRESSING a non-present,
    # abstract, or inanimate entity: "O Fortune!", "Ye gods!", "O love!",
    # "O night!", "O death!", "O earth!", "O hell!", "O heart!", "Sweet
    # love, remember!", "Come, night! Come, Romeo!". It is a LYRIC mode —
    # its lexicon is abstract-nouns, its verbs are imperatives directed
    # at the apostrophized entity, its rhythm is exclamatory, and it
    # tolerates long expansive clauses with interjections.
    #
    # Representation:
    #   apostrophe_mode (int 0..3)
    #     0 — off / normal discourse. Default.
    #     1 — primed. Sentence opened with an "O"/"Oh"/"Ah"/"Alas" and
    #         we are in the vocative expansion (first 1-3 words after
    #         the invocation particle).
    #     2 — active. An abstract/figurative entity has been named as
    #         the apostrophe target (heaven, fortune, love, night,
    #         death, hell, earth, heart, gods, stars, time, soul,
    #         beauty, grief, nature). Lexicon bias should fire.
    #     3 — locked. A second apostrophe cue (another "O ___" or a
    #         follow-up imperative to the same target) has reinforced.
    #         Full effect.
    #
    # Update rules (pipeline/apostrophe.py):
    #   - Entry to mode 1 fires when:
    #       * Sentence-start (chars_since_sentence_end small and
    #         words_in_sentence == 0-1) AND
    #       * last_completed_word in {"o","oh","ah","alas","ye",
    #         "yea","hark","lo","fie","alack"}.
    #     "O" alone is ambiguous (could be interjection filler), so we
    #     use the follow-up noun to promote 1 -> 2.
    #   - Promotion 1 -> 2 fires when the next completed content word
    #     is in the APOSTROPHE_TARGETS lexicon (abstract / figurative
    #     addressable nouns). See pipeline for list.
    #   - Promotion 2 -> 3 fires on a SECOND apostrophe invocation
    #     within the same sentence/turn, or on an imperative verb
    #     directed at the target ("come night, hide thyself", "speak,
    #     heaven").
    #   - Reset to 0 on:
    #       * turn boundary (consecutive_newlines >= 2), OR
    #       * sentence-end PUNCT_END followed by 2+ completed non-
    #         apostrophe sentences, OR
    #       * explicit address to a CONCRETE character ("my lord",
    #         "Hamlet, ..., ") which shifts address-target away from
    #         abstract.
    #
    # Read by predict layers (future) to:
    #   - Boost abstract-noun word-starters after "O"/"Ye"/"Alas".
    #   - Boost imperative verbs after apostrophe_mode >= 2.
    #   - Prefer "!" over "." at sentence-end when mode >= 2.
    #   - Tolerate longer lines / resist premature newline when locked.
    apostrophe_mode: int = 0
    # Words consumed since the most recent apostrophe cue. Used by the
    # pipeline to decay mode 1 -> 0 if the target noun doesn't land
    # within a small window (the "O" was just filler, not apostrophe).
    # Resets to 0 each time mode transitions upward.
    apostrophe_words_since_cue: int = 0
    # Which apostrophe target (lowercased lexicon entry) is currently
    # being addressed, if any. Empty string when no target locked.
    # Informs downstream lexicon biasing by target family (celestial:
    # heaven/stars/sun/moon; mortality: death/grave/fate; affective:
    # love/heart/beauty; natural: night/earth/sea).
    apostrophe_target: str = ""

    # --- Capital-required-at-next-word-start gate ---
    #
    # A structural axis distinct from the many soft caps-pushes scattered
    # through compose.py: a single categorical signal capturing "the
    # next word MUST start with a capital letter because the orthographic
    # convention of Shakespeare's text demands it here, not because of
    # any rolling register/topic bias."
    #
    # Modes:
    #   0  NONE   — no capital required; the default everywhere mid-line
    #              and mid-word.
    #   1  SENTENCE_START — post ". "/"? "/"! "/"\n\n" with no speaker
    #              label active. Strong cap requirement; real Shakespeare
    #              always capitalizes these.
    #   2  VERSE_LINE  — single-\n terminated a verse-length line that
    #              was NOT enjambed (ended on punct or function-word ish).
    #              Verse convention: every new line begins capital.
    #   3  POST_LABEL  — just past the ":\n" at the end of a speaker
    #              label. Dialogue almost always opens with a capital.
    #   4  TURN_START  — post "\n\n" with nothing else yet. Speaker label
    #              is coming; hard caps required.
    #
    # Only valid at the instant letter_run_len == 0 AND we're about to
    # emit a letter. Cleared to NONE once any character is consumed (the
    # decision is made at that one letter position).
    #
    # Computed by pipeline/cap_required.py which reads last_char,
    # consecutive_newlines, prev_line_length, prev_line_final_class,
    # sentence_start_pending, speaker_label_state, prev_char_class.
    #
    # Read by predict/cap_required.py which applies a much sharper
    # UPPER-vs-lower bias than the scattered inline logic can — at the
    # sample-noise regime where the inline +1.2/-0.5 sometimes loses
    # to the ~2.5-nat unigram advantage of lowercase.
    cap_required_mode: int = 0

    # --- Committed-next-word identity commitment ---
    #
    # When the bigram/trigram context strongly predicts a specific next
    # word (e.g. "my good ___" → "lord"; "I pray ___" → "thee"; "to be
    # or not to ___" → "be"), commit to that word's IDENTITY and let
    # downstream predict layers bias each subsequent letter toward the
    # target's letters rather than re-deciding letter-by-letter via
    # independent n-gram signals.
    #
    # Without this, the chain of per-letter biases produces gibberish
    # even when the word-level n-gram context is near-deterministic: the
    # first letter lands correctly (from next_word / phrase bigram), but
    # letter 2, 3, 4, ... are picked independently by letter-ngram
    # momentum and drift into nonsense shapes like "etustartec".
    #
    # State:
    #   committed_word    — lowercase target ("" if no commit)
    #   committed_word_pos — how many of its letters have been emitted
    #
    # Lifecycle:
    #   - At word-start (post-space), `pipeline/word_commit.py` consults
    #     recent_completed_words + sentence/line context. If a hand-
    #     curated trigger matches unambiguously, sets committed_word to
    #     the target and committed_word_pos to 0.
    #   - At each letter step, if last emitted char matches
    #     committed_word[committed_word_pos], pos += 1.
    #   - On any letter mismatch, or on word completion (space/punct),
    #     or on speaker-turn/sentence boundaries, commit clears.
    #
    # Predict layer reads (committed_word, committed_word_pos,
    # letter_run_len) and applies a strong +boost on the target letter.
    # No corpus stats — targets are hand-written Shakespeare formulas.
    committed_word: str = ""
    committed_word_pos: int = 0

    # --- Word-reality running memory ---
    #
    # Classification of the most recently completed word as one of:
    #   0 — unset / too short / skipped (inside speaker label, 1-letter
    #       word, etc.)
    #   1 — REAL: known word (was on the word-trie with has_seen_complete)
    #       and no phonotactic flags.
    #   2 — PLAUSIBLE: off-trie but phonotactically sane — inflected form
    #       or archaic variant (e.g., "feard'st", "unwaxed").
    #   3 — GIBBERISH: has phonotactic red flags, illegal bigrams/trigrams,
    #       or unsupported long off-trie runs ("pytsaninsao").
    #
    # Populated at `just_finished_word` by `pipeline/word_reality.py`,
    # which runs EARLY in the pipeline so it can read the pre-reset
    # values of `word_red_flags`, `bad_bigram_count`, `bad_trigram_count`,
    # `letters_off_trie`, `has_seen_complete` (these counters all reset
    # later in the cycle on the word-boundary char).
    last_word_reality: int = 0
    # Rolling window of the last 4 word-reality classifications,
    # most-recent FIRST. Used by predict layers to read the trend
    # ("two of the last three were gibberish").
    recent_word_realities: tuple[int, ...] = ()
    # Per-turn gibberish and real counts. Reset on speaker-turn change
    # (consecutive_newlines >= 2). The count is a saturating accumulator
    # of GIBBERISH classifications; the real count is for contrast.
    turn_gibberish_count: int = 0
    turn_real_count: int = 0
    # Per-sentence gibberish count. Reset on sentence-end punctuation
    # (PUNCT_END: . ? !). Finer-grained than the turn counter — lets
    # the predict layer back off once the sentence closes and a fresh
    # sentence starts.
    sentence_gibberish_count: int = 0
    sentence_real_count: int = 0

    # --- Phrase-slot FSM ---
    #
    # Targeted fix for the sample-quality failure mode:
    #   "the man little evilly to and when" — after a determiner, the
    #   next content word is uncontrolled (noun, adjective, adverb, verb
    #   equally likely), producing ungrammatical noun-phrases.
    #
    # States (tracked across completed words within a sentence):
    #   0 — NEUTRAL: sentence-start, post-verb, post-punct, or post-
    #       conjunction. Any category can follow.
    #   1 — POST_DET: just saw article/possessive/wh-determiner. Next
    #       open-class word should be ADJECTIVE or NOUN, not verb/
    #       adverb/modal/another-determiner.
    #   2 — POST_ADJ: just saw an adjective within an NP. Next should
    #       continue with ADJ or NOUN, not a verb.
    #   3 — POST_NOUN: just saw the head noun of an NP. Next should be
    #       PREPOSITION / VERB / CONJUNCTION / terminator — NOT another
    #       determiner or adjective (those would start a new NP).
    #
    # Lifecycle:
    #   - Updated by `pipeline/phrase_slot.py` after `pipeline/pos.py`
    #     on each just_finished_word.
    #   - Reset to 0 on PUNCT_END and on speaker-turn change.
    #
    # Consumed by a predict layer at word-start (letter_run_len == 0,
    # last_char_class SPACE) that biases the next word's first letter
    # toward slot-appropriate POS openers.
    phrase_slot: int = 0
    # How many consecutive tokens we've remained in the current non-
    # NEUTRAL slot. Rises when we stay POST_DET / POST_ADJ for more
    # than one word (e.g. "the fair" = slot 2, len 1; "the fair young"
    # = slot 2, len 2). When len gets large in POST_DET/POST_ADJ, the
    # predict layer should VERY strongly demand a noun closer.
    phrase_slot_len: int = 0

    # --- Word-ending shape score ---
    #
    # Structural detector for the "drymudrtee / cojiunr / ineddseh"
    # failure mode: long letter-runs that are off-trie and whose tail
    # doesn't look like any real English word-ending, but whose local
    # bigrams are legal enough to dodge the existing phonotactic
    # close-out (which requires ≥2 violations).
    #
    # Values:
    #   2 — the buffer is IN the complete-word trie (terminating here
    #       yields a real word). Strongest "OK to end" signal.
    #   1 — the buffer's tail matches a canonical English word-final
    #       pattern (e.g., -ing, -ed, -ly, -er, -tion, -ness, -ful,
    #       -ous, -able, -ment, -ity, -ance, -ence, -ish, -ship,
    #       -dom, -hood, -like, -ward, -wise, -ness-less, common
    #       CVC / VCe codas). Usable-as-ending.
    #   0 — neither. Terminating now would yield a word-shaped
    #       nonsense fragment ("drymudrt", "cojiunr", "ineddseh").
    #
    # Lifecycle:
    #   - Updated by `pipeline/word_ending_shape.py` on every letter.
    #   - Resets to 0 on word boundary (non-letter, non-apostrophe).
    #
    # Consumed by `predict/word_ending_shape.py`: when
    # letter_run_len >= 5 AND on_word_trie == False AND
    # word_ending_shape_score == 0, strongly push termination and
    # suppress additional letters. The conjunction of "off-trie" AND
    # "no valid ending pattern" is the discriminator that separates
    # gibberish drift from legitimate long words the trie just
    # doesn't know about (which tend to preserve English morphology).
    word_ending_shape_score: int = 0
