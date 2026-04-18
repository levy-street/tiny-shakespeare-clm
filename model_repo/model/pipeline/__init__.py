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
from .anaphora import update_anaphora
from .clause import update_clause
from .clause_slot import update_clause_slot
from .verb_agreement import update_verb_agreement
from .np_head import update_np_head
from .counters import update_basic_counters
from .flow import update_flow
from .formula import update_formula
from .linguistic import update_linguistic
from .pos import update_pos
from .prosody import update_prosody
from .referent import update_referent
from .repetition import update_repetition
from .word_shape import update_word_shape
from .rhyme import update_rhyme
from .sentence import update_sentence
from .speaker_memory import update_speaker_memory
from .speaker_offtrie import update_speaker_offtrie
from .turn import update_turn_progress
from .vocative import update_vocative

Stage = Callable[[ModelState, int], ModelState]

PIPELINE: list[Stage] = [
    update_basic_counters,  # Tier 1: base bookkeeping
    update_linguistic,      # Tier 2: linguistic structure
    update_speaker_offtrie, # Tier 2: speaker-buffer off-trie run
    update_pos,             # Tier 2: POS tag of last completed word
    update_repetition,      # Tier 2: short-range word-repetition memory
    update_formula,         # Tier 2: formulaic-phrase trie position
    update_sentence,        # Tier 2/3: sentence-type FSM
    update_clause,          # Tier 2: clause-structure (clauses, subj pronoun)
    update_clause_slot,     # Tier 2: syntactic-slot state machine
    update_verb_agreement,  # Tier 2: subject-verb agreement expectation
    update_np_head,         # Tier 2: NP-head expectation (np_open, np_wait_words)
    update_vocative,        # Tier 2: vocative-expectation flag
    update_addressee,       # Tier 2/3: vocative-noun memory
    update_speaker_memory,  # Tier 2/3: recent-speakers rolling window
    update_referent,        # Tier 2: anaphoric referent gender tracking
    update_turn_progress,   # Tier 2/3: words/sentences/lines in current turn
    update_anaphora,        # Tier 2: line-starter anaphora tracking
    update_rhyme,           # Tier 2/3: line-tail rhyme memory
    update_prosody,         # Tier 3: syllable / cadence tracking
    update_word_shape,      # Tier 2: per-word phonotactic red-flag counter
    update_flow,            # Tier 3: flow / mood / cadence
]

__all__ = ["PIPELINE", "Stage"]
