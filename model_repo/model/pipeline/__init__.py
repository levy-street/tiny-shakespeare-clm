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
from .case_slot import update_case_slot
from .clause import update_clause
from .clause_slot import update_clause_slot
from .line_break import update_line_break
from .verb_agreement import update_verb_agreement
from .np_head import update_np_head
from .transitivity import update_transitivity
from .verb_class import update_verb_class
from .word_form import update_word_form
from .counters import update_basic_counters
from .doubt import update_doubt
from .drift import update_drift
from .flow import update_flow
from .formula import update_formula
from .lament import update_lament
from .linguistic import update_linguistic
from .list_structure import update_list_structure
from .negation import update_negation
from .pos import update_pos
from .proper_noun import update_proper_noun
from .caesura import update_caesura
from .prosody import update_prosody
from .referent import update_referent
from .repetition import update_repetition
from .subord import update_subord
from .tenderness import update_tenderness
from .word_shape import update_word_shape
from .rhyme import update_rhyme
from .sentence import update_sentence
from .speaker_memory import update_speaker_memory
from .speaker_offtrie import update_speaker_offtrie
from .topic_tracker import update_topic_tracker
from .turn import update_turn_progress
from .vocative import update_vocative

Stage = Callable[[ModelState, int], ModelState]

PIPELINE: list[Stage] = [
    update_basic_counters,  # Tier 1: base bookkeeping
    update_linguistic,      # Tier 2: linguistic structure
    update_drift,           # Tier 2/3: consecutive-off-trie word streak
    update_speaker_offtrie, # Tier 2: speaker-buffer off-trie run
    update_pos,             # Tier 2: POS tag of last completed word
    update_proper_noun,     # Tier 2: proper-noun expectation slot
    update_list_structure,  # Tier 2: list-parallelism progress
    update_repetition,      # Tier 2: short-range word-repetition memory
    update_formula,         # Tier 2: formulaic-phrase trie position
    update_sentence,        # Tier 2/3: sentence-type FSM
    update_clause,          # Tier 2: clause-structure (clauses, subj pronoun)
    update_clause_slot,     # Tier 2: syntactic-slot state machine
    update_subord,          # Tier 2: subordinate-clause depth tracker
    update_negation,        # Tier 2: negation-scope polarity tracker
    update_verb_agreement,  # Tier 2: subject-verb agreement expectation
    update_np_head,         # Tier 2: NP-head expectation (np_open, np_wait_words)
    update_transitivity,    # Tier 2: verb transitivity / object-expectation
    update_case_slot,       # Tier 2: pronoun case slot (SUBJ/OBJ)
    update_verb_class,      # Tier 2: verb semantic class (9-way)
    update_word_form,       # Tier 2: morphological-form expectation FSM
    update_vocative,        # Tier 2: vocative-expectation flag
    update_addressee,       # Tier 2/3: vocative-noun memory
    update_speaker_memory,  # Tier 2/3: recent-speakers rolling window
    update_referent,        # Tier 2: anaphoric referent gender tracking
    update_topic_tracker,   # Tier 3: scene-topic semantic cluster memory
    update_doubt,           # Tier 3: doubt/assertion register texture
    update_lament,          # Tier 3: lament/grief texture register
    update_tenderness,      # Tier 3: tenderness/love texture register
    update_turn_progress,   # Tier 2/3: words/sentences/lines in current turn
    update_anaphora,        # Tier 2: line-starter anaphora tracking
    update_alliteration,    # Tier 2/3: within-line alliteration memory
    update_rhyme,           # Tier 2/3: line-tail rhyme memory
    update_prosody,         # Tier 3: syllable / cadence tracking
    update_caesura,         # Tier 3: mid-line pause (caesura) tracking
    update_word_shape,      # Tier 2: per-word phonotactic red-flag counter
    update_line_break,      # Tier 2: syntactic line-break propriety
    update_flow,            # Tier 3: flow / mood / cadence
]

__all__ = ["PIPELINE", "Stage"]
