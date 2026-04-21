"""The sequence of state-update stages that `advance` threads a token through.

Each stage is a pure function `(state, token_id) -> state`. Stages run in
order: later stages see the field updates made by earlier stages, so a
downstream stage can condition its behavior on what an upstream stage
decided. This is the "depth" the pipeline provides — a chain of inspections
and updates within a single token's advance.

Add or split stages by editing PIPELINE. Keep each stage focused on one
cohesive concern so another stage can read its output and react.
"""

from __future__ import annotations

from typing import Callable

from ..state import ModelState
from .addressee import update_addressee
from .alliteration import update_alliteration
from .apostrophe import update_apostrophe
from .anaphora import update_anaphora
from .line_opener_pos import update_line_opener_pos
from .antithesis import update_antithesis
from .case_slot import update_case_slot
from .clause import update_clause
from .clause_slot import update_clause_slot
from .conditional import update_conditional
from .confessional import update_confessional
from .clause_parallel import update_clause_parallel
from .coord import update_coord
from .line_break import update_line_break
from .line_coherence import update_line_coherence
from .line_offtrie_streak import update_line_offtrie_streak
from .line_end_memory import update_line_end_memory
from .verb_agreement import update_verb_agreement
from .np_head import update_np_head
from .prep_governor import update_prep_governor
from .transitivity import update_transitivity
from .verb_class import update_verb_class
from .verb_complement import update_verb_complement
from .word_cap_apos import update_word_cap_apos
from .contraction_tail import update_contraction_tail
from .word_commit import update_word_commit
from .word_ending_shape import update_word_ending_shape
from .word_form import update_word_form
from .word_integrity import update_word_integrity
from .word_matches import update_word_matches
from .word_reality import update_word_reality
from .counters import update_basic_counters
from .dash_aside import update_dash_aside
from .dialogue_adjacency import update_dialogue_adjacency
from .doubt import update_doubt
from .drift import update_drift
from .flow import update_flow
from .formula import update_formula
from .function_word_chain import update_function_word_chain
from .content_word_chain import update_content_word_streak
from .clause_skel import update_clause_skel
from .fury import update_fury
from .lament import update_lament
from .martial import update_martial
from .mirth import update_mirth
from .linguistic import update_linguistic
from .mid_departure import update_mid_departure
from .list_structure import update_list_structure
from .negation import update_negation
from .noun_class import update_noun_class
from .play_family import update_play_family
from .phrase_slot import update_phrase_slot
from .pos import update_pos
from .proper_noun import update_proper_noun
from .proper_noun_memory import update_proper_noun_memory
from .turn_rolodex import update_turn_rolodex
from .question_answer import update_question_answer
from .caesura import update_caesura
from .cap_required import update_cap_required
from .meter import update_meter
from .prosody import update_prosody
from .referent import update_referent
from .register_commit import update_register_commit
from .repetition import update_repetition
from .subord import update_subord
from .syntactic_frame import update_syntactic_frame
from .tense import update_tense
from .tenderness import update_tenderness
from .gravitas import update_gravitas
from .word_shape import update_word_shape
from .enjambment import update_enjambment
from .phonotactic import update_phonotactic
from .polysyllable import update_polysyllable
from .rhyme import update_rhyme
from .sensory_charge import update_sensory_charge
from .sentence import update_sentence
from .sentence_backbone import update_sentence_backbone
from .sentence_pressure import update_sentence_pressure
from .sentence_sem import update_sentence_sem
from .sentence_syllables import update_sentence_syllables
from .speaker_memory import update_speaker_memory
from .speaker_register import update_speaker_register
from .speaker_offtrie import update_speaker_offtrie
from .speaker_strict import update_speaker_strict
from .speaker_vowels import update_speaker_vowels
from .speaker_cons_run import update_speaker_cons_run
from .topic_tracker import update_topic_tracker
from .turn import update_turn_progress
from .turn_content import update_turn_content
from .turn_pronoun import update_turn_pronoun
from .turn_shape import update_turn_shape
from .line_word_cadence import update_line_word_cadence
from .archaic_density import update_archaic_density
from .valence import update_valence
from .vocative import update_vocative

Stage = Callable[[ModelState, int], ModelState]

PIPELINE: list[Stage] = [
    update_basic_counters,  # Tier 1: base bookkeeping
    update_dash_aside,      # Tier 2: parenthetical-dash scope tracking
    update_linguistic,      # Tier 2: linguistic structure
    # NOTE: word_reality runs AFTER linguistic (which sets
    # just_finished_word / last_completed_word) and BEFORE word_shape /
    # phonotactic (which reset word_red_flags / bad_bigram_count /
    # bad_trigram_count on the boundary char). It reads those pre-reset
    # values to classify the just-finished word.
    update_word_reality,    # Tier 2/3: classify completed word (real/plausible/gibberish), maintain per-turn/-sentence counts
    update_word_matches,    # Tier 2: graded trie-completion count for word_buffer
    update_word_cap_apos,   # Tier 2: apostrophe-in-word position + had_apos flag
    update_contraction_tail,  # Tier 2: is the current post-apostrophe tail a valid elision closer?
    update_word_integrity,  # Tier 2/3: per-char word-shape plausibility monitor
    update_word_ending_shape,  # Tier 2: 0/1/2 score — would the current buffer terminate as a recognizable English word shape?
    update_mid_departure,   # Tier 2: mid-departure (pos 3-4) extension length
    update_drift,           # Tier 2/3: consecutive-off-trie word streak
    update_line_coherence,  # Tier 2: per-line on-trie vs off-trie word counts
    update_line_offtrie_streak,  # Tier 2: consecutive off-trie streak within line (resets on on-trie word)
    update_speaker_offtrie, # Tier 2: speaker-buffer off-trie run
    update_speaker_strict,  # Tier 2: speaker-trie next-char legality flags (on_trie / space_valid / colon_valid)
    update_speaker_vowels,  # Tier 2: speaker-buffer vowel count
    update_speaker_cons_run,  # Tier 2: speaker-buffer trailing consonant-run
    update_pos,             # Tier 2: POS tag of last completed word
    update_function_word_chain,  # Tier 2: count of consecutive function-class words — resets on content word
    update_content_word_streak,   # Tier 2: count of consecutive content-class words — mirror of function_word_chain, resets on function word / mid-punct
    update_coord,           # Tier 2: coordinator-parallelism echo (X and Y) — runs before proper_noun_memory so current_word_started_cap is still live
    update_phrase_slot,     # Tier 2: noun-phrase slot FSM (NEUTRAL/POST_DET/POST_ADJ/POST_NOUN) — runs after update_pos
    update_clause_skel,     # Tier 2: clause-skeleton FSM (EMPTY/SUBJ_OPEN/SUBJ_DONE/VERB_DONE/COMP_DUE/CLAUSE_DONE) — runs after phrase_slot
    update_noun_class,      # Tier 2/3: coarse semantic noun-class (KINSHIP/BODY/ROYALTY/...)
    update_proper_noun,     # Tier 2: proper-noun expectation slot
    update_proper_noun_memory,  # Tier 2: rolodex of recent capitalized words
    update_turn_rolodex,    # Tier 2: turn-scoped proper-noun rolodex (resets at blank-line; mirrors proper_nouns_seen head)
    update_list_structure,  # Tier 2: list-parallelism progress
    update_antithesis,      # Tier 2/3: antithesis / rhetorical-contrast state
    update_repetition,      # Tier 2: short-range word-repetition memory
    update_formula,         # Tier 2: formulaic-phrase trie position
    update_word_commit,     # Tier 2: commit to next-word identity when formula uniquely predicts it
    update_question_answer, # Tier 3: cross-turn WH-class answer expectation
    update_sentence,        # Tier 2/3: sentence-type FSM
    update_sentence_backbone,  # Tier 2: subject + finite-verb presence per sentence
    update_sentence_sem,    # Tier 2/3: sentence-scoped semantic-field lock
    update_clause,          # Tier 2: clause-structure (clauses, subj pronoun)
    update_clause_slot,     # Tier 2: syntactic-slot state machine
    update_subord,          # Tier 2: subordinate-clause depth tracker
    update_conditional,     # Tier 2: conditional/concessive protasis→apodosis FSM
    update_clause_parallel, # Tier 2: intra-sentence clause-parallelism opener echo
    update_negation,        # Tier 2: negation-scope polarity tracker
    update_verb_agreement,  # Tier 2: subject-verb agreement expectation
    update_tense,           # Tier 2: sentence-level tense register
    update_np_head,         # Tier 2: NP-head expectation (np_open, np_wait_words)
    update_prep_governor,   # Tier 2: prep→prep / prep→prep-word blocker flag
    update_sentence_pressure,  # Tier 2: signed completion-readiness score (reads subj/verb/np_open/subord/last_word_pos)
    update_syntactic_frame, # Tier 2: forward role projection for next word
    update_transitivity,    # Tier 2: verb transitivity / object-expectation
    update_case_slot,       # Tier 2: pronoun case slot (SUBJ/OBJ)
    update_verb_class,      # Tier 2: verb semantic class (9-way)
    update_verb_complement, # Tier 2: verb-complement class expectation
    update_word_form,       # Tier 2: morphological-form expectation FSM
    update_vocative,        # Tier 2: vocative-expectation flag
    update_addressee,       # Tier 2/3: vocative-noun memory
    update_speaker_memory,  # Tier 2/3: recent-speakers rolling window
    update_speaker_register, # Tier 2/3: categorical register from speaker name
    update_play_family,     # Tier 2/3: play-family lock from speaker name (runs AFTER speaker_memory which writes last_speaker_label)
    update_register_commit, # Tier 2: thou/you address-form commit (per-turn)
    update_referent,        # Tier 2: anaphoric referent gender tracking
    update_topic_tracker,   # Tier 3: scene-topic semantic cluster memory
    update_doubt,           # Tier 3: doubt/assertion register texture
    update_lament,          # Tier 3: lament/grief texture register
    update_tenderness,      # Tier 3: tenderness/love texture register
    update_gravitas,        # Tier 3: gravitas/moral-weight texture register
    update_fury,            # Tier 3: fury/rage/curse texture register
    update_mirth,           # Tier 3: mirth/merry/comic texture register
    update_apostrophe,      # Tier 3: figurative-address mode (O Fortune!)
    update_confessional,    # Tier 3: confessional vs public register
    update_sensory_charge,  # Tier 3: corporeal ↔ abstract charge (lyric vs argument register)
    update_martial,         # Tier 3: martial ↔ peaceful register (battlefield lexicon)
    update_valence,         # Tier 3 FLOW: emotional valence (positive ↔ negative diction polarity)
    update_dialogue_adjacency,  # Tier 2/3: snapshot prev-turn shape before turn counters reset
    update_turn_shape,      # Tier 2/3: cross-turn rhythm tuple + stichomythia_mode (must run BEFORE turn_progress reset)
    update_turn_pronoun,    # Tier 2/3: per-turn 1st/2nd person pronoun profile (soliloquy/direct-address/mixed)
    update_turn_progress,   # Tier 2/3: words/sentences/lines in current turn
    update_line_word_cadence,  # Tier 2/3: per-line word-count history within turn (cadence)
    update_archaic_density, # Tier 3 FLOW: rolling archaic-lexicon density
    update_turn_content,    # Tier 3: per-turn content-word echo cache
    update_anaphora,        # Tier 2: line-starter anaphora tracking
    update_line_opener_pos, # Tier 2: line-opener POS pattern memory
    update_alliteration,    # Tier 2/3: within-line alliteration memory
    update_rhyme,           # Tier 2/3: line-tail rhyme memory
    update_line_end_memory, # Tier 2: line-TERMINAL word memory (epistrophe)
    update_enjambment,      # Tier 3: enjambed vs. end-stopped line density
    update_polysyllable,    # Tier 3: polysyllable density rolling memory
    update_sentence_syllables,  # Tier 2/3: per-sentence syllable counter (runs before prosody)
    update_prosody,         # Tier 3: syllable / cadence tracking
    update_meter,           # Tier 2/3: iambic meter (expected_stress, confidence)
    update_caesura,         # Tier 3: mid-line pause (caesura) tracking
    update_word_shape,      # Tier 2: per-word phonotactic red-flag counter
    update_phonotactic,     # Tier 2: illegal letter-bigram count within current word
    update_line_break,      # Tier 2: syntactic line-break propriety
    update_cap_required,    # Tier 2: capital-required-next-word gate (runs near end so prev_line_length/prev_line_final_class are current)
    update_flow,            # Tier 3: flow / mood / cadence
]

__all__ = ["PIPELINE", "Stage"]
