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
