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
from .anaphora import update_anaphora
from .line_opener_pos import update_line_opener_pos
from .antithesis import update_antithesis
from .case_slot import update_case_slot
from .clause import update_clause
from .clause_slot import update_clause_slot
from .conditional import update_conditional
from .confessional import update_confessional
from .clause_parallel import update_clause_parallel
from .line_break import update_line_break
from .line_coherence import update_line_coherence
from .verb_agreement import update_verb_agreement
from .np_head import update_np_head
from .transitivity import update_transitivity
from .verb_class import update_verb_class
from .verb_complement import update_verb_complement
from .word_cap_apos import update_word_cap_apos
from .word_form import update_word_form
from .word_integrity import update_word_integrity
from .word_matches import update_word_matches
from .counters import update_basic_counters
from .dash_aside import update_dash_aside
from .dialogue_adjacency import update_dialogue_adjacency
from .doubt import update_doubt
from .drift import update_drift
from .flow import update_flow
from .formula import update_formula
from .fury import update_fury
from .lament import update_lament
from .linguistic import update_linguistic
from .mid_departure import update_mid_departure
from .list_structure import update_list_structure
from .negation import update_negation
from .noun_class import update_noun_class
from .pos import update_pos
from .proper_noun import update_proper_noun
from .proper_noun_memory import update_proper_noun_memory
from .question_answer import update_question_answer
from .caesura import update_caesura
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
from .speaker_memory import update_speaker_memory
from .speaker_register import update_speaker_register
from .speaker_offtrie import update_speaker_offtrie
from .speaker_vowels import update_speaker_vowels
from .topic_tracker import update_topic_tracker
from .turn import update_turn_progress
from .turn_content import update_turn_content
from .vocative import update_vocative

Stage = Callable[[ModelState, int], ModelState]

PIPELINE: list[Stage] = [
    update_basic_counters,  # Tier 1: base bookkeeping
    update_dash_aside,      # Tier 2: parenthetical-dash scope tracking
    update_linguistic,      # Tier 2: linguistic structure
    update_word_matches,    # Tier 2: graded trie-completion count for word_buffer
    update_word_cap_apos,   # Tier 2: apostrophe-in-word position + had_apos flag
    update_word_integrity,  # Tier 2/3: per-char word-shape plausibility monitor
    update_mid_departure,   # Tier 2: mid-departure (pos 3-4) extension length
    update_drift,           # Tier 2/3: consecutive-off-trie word streak
    update_line_coherence,  # Tier 2: per-line on-trie vs off-trie word counts
    update_speaker_offtrie, # Tier 2: speaker-buffer off-trie run
    update_speaker_vowels,  # Tier 2: speaker-buffer vowel count
    update_pos,             # Tier 2: POS tag of last completed word
    update_noun_class,      # Tier 2/3: coarse semantic noun-class (KINSHIP/BODY/ROYALTY/...)
    update_proper_noun,     # Tier 2: proper-noun expectation slot
    update_proper_noun_memory,  # Tier 2: rolodex of recent capitalized words
    update_list_structure,  # Tier 2: list-parallelism progress
    update_antithesis,      # Tier 2/3: antithesis / rhetorical-contrast state
    update_repetition,      # Tier 2: short-range word-repetition memory
    update_formula,         # Tier 2: formulaic-phrase trie position
    update_question_answer, # Tier 3: cross-turn WH-class answer expectation
    update_sentence,        # Tier 2/3: sentence-type FSM
    update_sentence_backbone,  # Tier 2: subject + finite-verb presence per sentence
    update_clause,          # Tier 2: clause-structure (clauses, subj pronoun)
    update_clause_slot,     # Tier 2: syntactic-slot state machine
    update_subord,          # Tier 2: subordinate-clause depth tracker
    update_conditional,     # Tier 2: conditional/concessive protasis→apodosis FSM
    update_clause_parallel, # Tier 2: intra-sentence clause-parallelism opener echo
    update_negation,        # Tier 2: negation-scope polarity tracker
    update_verb_agreement,  # Tier 2: subject-verb agreement expectation
    update_tense,           # Tier 2: sentence-level tense register
    update_np_head,         # Tier 2: NP-head expectation (np_open, np_wait_words)
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
    update_register_commit, # Tier 2: thou/you address-form commit (per-turn)
    update_referent,        # Tier 2: anaphoric referent gender tracking
    update_topic_tracker,   # Tier 3: scene-topic semantic cluster memory
    update_doubt,           # Tier 3: doubt/assertion register texture
    update_lament,          # Tier 3: lament/grief texture register
    update_tenderness,      # Tier 3: tenderness/love texture register
    update_gravitas,        # Tier 3: gravitas/moral-weight texture register
    update_fury,            # Tier 3: fury/rage/curse texture register
    update_confessional,    # Tier 3: confessional vs public register
    update_sensory_charge,  # Tier 3: corporeal ↔ abstract charge (lyric vs argument register)
    update_dialogue_adjacency,  # Tier 2/3: snapshot prev-turn shape before turn counters reset
    update_turn_progress,   # Tier 2/3: words/sentences/lines in current turn
    update_turn_content,    # Tier 3: per-turn content-word echo cache
    update_anaphora,        # Tier 2: line-starter anaphora tracking
    update_line_opener_pos, # Tier 2: line-opener POS pattern memory
    update_alliteration,    # Tier 2/3: within-line alliteration memory
    update_rhyme,           # Tier 2/3: line-tail rhyme memory
    update_enjambment,      # Tier 3: enjambed vs. end-stopped line density
    update_polysyllable,    # Tier 3: polysyllable density rolling memory
    update_prosody,         # Tier 3: syllable / cadence tracking
    update_meter,           # Tier 2/3: iambic meter (expected_stress, confidence)
    update_caesura,         # Tier 3: mid-line pause (caesura) tracking
    update_word_shape,      # Tier 2: per-word phonotactic red-flag counter
    update_phonotactic,     # Tier 2: illegal letter-bigram count within current word
    update_line_break,      # Tier 2: syntactic line-break propriety
    update_flow,            # Tier 3: flow / mood / cadence
]

__all__ = ["PIPELINE", "Stage"]
