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
from .anaphora import update_anaphora
from .clause import update_clause
from .clause_slot import update_clause_slot
from .counters import update_basic_counters
from .flow import update_flow
from .formula import update_formula
from .linguistic import update_linguistic
from .pos import update_pos
from .prosody import update_prosody
from .sentence import update_sentence
from .speaker_memory import update_speaker_memory
from .vocative import update_vocative

Stage = Callable[[ModelState, int], ModelState]

PIPELINE: list[Stage] = [
    update_basic_counters,  # Tier 1: base bookkeeping
    update_linguistic,      # Tier 2: linguistic structure
    update_pos,             # Tier 2: POS tag of last completed word
    update_formula,         # Tier 2: formulaic-phrase trie position
    update_sentence,        # Tier 2/3: sentence-type FSM
    update_clause,          # Tier 2: clause-structure (clauses, subj pronoun)
    update_clause_slot,     # Tier 2: syntactic-slot state machine
    update_vocative,        # Tier 2: vocative-expectation flag
    update_speaker_memory,  # Tier 2/3: recent-speakers rolling window
    update_anaphora,        # Tier 2: line-starter anaphora tracking
    update_prosody,         # Tier 3: syllable / cadence tracking
    update_flow,            # Tier 3: flow / mood / cadence
]

__all__ = ["PIPELINE", "Stage"]
